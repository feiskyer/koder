"""Harness auth subcommand flows."""

from __future__ import annotations

import asyncio
import json
import logging
import numbers
import os
import sys
import time
from contextlib import suppress
from dataclasses import replace
from datetime import datetime
from typing import Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from koder_agent.auth.base import OAuthTokens
from koder_agent.auth.callback_server import CallbackResult, run_oauth_flow
from koder_agent.auth.client_integration import async_refresh_token
from koder_agent.auth.constants import SUPPORTED_PROVIDERS, TOKEN_EXPIRY_BUFFER_MS
from koder_agent.auth.providers import get_provider
from koder_agent.auth.token_storage import TokenStorage, get_token_storage
from koder_agent.utils.async_tasks import await_owned_task, run_sync_owned

# Env var used as a fallback source for headless token ingestion.
AUTH_TOKEN_ENV = "KODER_AUTH_TOKEN"
# Access tokens ingested via stdin/env have no refresh flow; give them a long
# nominal lifetime so status does not immediately report them as expired.
STDIN_TOKEN_LIFETIME_MS = 365 * 24 * 60 * 60 * 1000

logger = logging.getLogger(__name__)
console = Console()

PROVIDER_DESCRIPTIONS: Dict[str, str] = {
    "google": "Gemini CLI (free with Google account)",
    "claude": "Claude Max subscription",
    "chatgpt": "ChatGPT Plus/Pro subscription",
    "antigravity": "Antigravity (Gemini 3 + Claude models)",
}

GITHUB_COPILOT_PROVIDER_ID = "github_copilot"
GITHUB_COPILOT_PROVIDER_ALIASES = {
    GITHUB_COPILOT_PROVIDER_ID,
    "github-copilot",
    "copilot",
}


def _normalize_provider_id(provider_id: str) -> str:
    provider = provider_id.strip().lower()
    if provider in GITHUB_COPILOT_PROVIDER_ALIASES:
        return GITHUB_COPILOT_PROVIDER_ID
    return provider


async def handle_login(provider_id: str, timeout: float = 300) -> bool:
    provider_id = _normalize_provider_id(provider_id)
    if provider_id == GITHUB_COPILOT_PROVIDER_ID:
        return await handle_github_copilot_login(timeout=timeout)

    if provider_id not in SUPPORTED_PROVIDERS:
        console.print(
            f"[red]Error:[/red] Unknown provider '{provider_id}'. "
            f"Supported: {', '.join([*SUPPORTED_PROVIDERS, GITHUB_COPILOT_PROVIDER_ID])}"
        )
        return False

    console.print(f"\n[bold]Authenticating with {provider_id}...[/bold]\n")
    try:
        provider = get_provider(provider_id)
        auth_url, verifier = provider.get_authorization_url()

        if provider_id == "claude":
            result = await _handle_manual_code_flow(auth_url, timeout)
        else:
            result = await run_oauth_flow(
                auth_url,
                port=provider.callback_port,
                callback_path=provider.callback_path,
                timeout=timeout,
            )

        if not result.success:
            console.print(
                f"[red]Authentication failed:[/red] {result.error}\n"
                f"{result.error_description or ''}"
            )
            return False

        console.print("Exchanging authorization code for tokens...")
        exchange_result = await provider.exchange_code(result.code, verifier)
        if not exchange_result.success:
            console.print(f"[red]Token exchange failed:[/red] {exchange_result.error}")
            return False

        tokens = exchange_result.tokens
        console.print("Fetching available models...")
        models, _ = await provider.list_models(tokens.access_token)
        tokens.models = models
        tokens.models_fetched_at = int(time.time() * 1000)

        storage = await run_sync_owned(get_token_storage)
        await run_sync_owned(storage.save, tokens)

        email = tokens.email or "Unknown"
        description = PROVIDER_DESCRIPTIONS.get(provider_id, "")
        success_msg = (
            f"[green]Successfully authenticated![/green]\n\n"
            f"Provider: {provider_id} ({description})\n"
            f"Account: {email}\n"
        )
        if models:
            success_msg += f"\n[bold]Available Models ({len(models)}):[/bold]\n"
            display_models = models[:5]
            success_msg += ", ".join(display_models)
            if len(models) > 5:
                success_msg += f"\n  +{len(models) - 5} more"
        else:
            success_msg += "\n[dim]No models found[/dim]"

        console.print(Panel(success_msg, title="Authentication Complete", border_style="green"))
        return True
    except Exception as exc:
        console.print(f"[red]Error during authentication:[/red] {exc}")
        return False


def _resolve_ingested_token(token_arg: Optional[str]) -> Optional[str]:
    """Resolve a token from the --token argument, stdin, or KODER_AUTH_TOKEN.

    ``--token -`` reads a single token from stdin. Any other ``--token`` value
    is used literally. When ``--token`` is absent, the ``KODER_AUTH_TOKEN``
    environment variable is used as a fallback.
    """
    if token_arg == "-":
        stdin_value = sys.stdin.read()
        stripped = stdin_value.strip()
        return stripped or None
    if token_arg:
        return token_arg.strip() or None
    env_value = os.environ.get(AUTH_TOKEN_ENV)
    if env_value and env_value.strip():
        return env_value.strip()
    return None


def handle_token_login(provider_id: str, token: str) -> bool:
    """Persist a directly-provided access token without a browser flow."""
    provider_id = _normalize_provider_id(provider_id)
    if provider_id == GITHUB_COPILOT_PROVIDER_ID:
        console.print(
            "[yellow]GitHub Copilot tokens are managed by LiteLLM and cannot be "
            "ingested directly. Run `koder auth login github_copilot`.[/yellow]"
        )
        return False

    tokens = OAuthTokens(
        provider=provider_id,
        access_token=token,
        refresh_token="",
        expires_at=int(time.time() * 1000) + STDIN_TOKEN_LIFETIME_MS,
    )
    storage = get_token_storage()
    storage.save(tokens)
    console.print(
        Panel(
            f"[green]Token saved for {provider_id}[/green]\nSource: stdin/env (no browser flow)",
            title="Authentication Complete",
            border_style="green",
        )
    )
    return True


async def handle_github_copilot_login(timeout: float = 300) -> bool:
    """Run owned device login with the installed SDK's headers and cache paths."""
    console.print("\n[bold]Authenticating with github_copilot...[/bold]\n")
    try:
        from koder_agent.auth.github_copilot import login

        def show_device_code(uri: str, code: str) -> None:
            console.print(Text(f"Please visit {uri} and enter code {code} to authenticate."))

        result = await login(timeout, on_device_code=show_device_code)
        console.print(
            Panel(
                "[green]Successfully authenticated![/green]\n\n"
                "Provider: github_copilot\n"
                f"Token cache: {result.token_dir}\n"
                f"API endpoint: {result.api_endpoint}",
                title="Authentication Complete",
                border_style="green",
            )
        )
        return True
    except TimeoutError:
        console.print("[red]GitHub Copilot authentication timed out[/red]")
        return False
    except Exception as exc:
        console.print(f"[red]GitHub Copilot authentication failed:[/red] {exc}")
        return False


async def _read_manual_authorization_code(timeout: float) -> str:
    """Own the async prompt until its input reader and background tasks stop."""
    from prompt_toolkit import PromptSession

    prompt = PromptSession(
        enable_open_in_editor=False,
        enable_system_prompt=False,
        enable_suspend=False,
    )
    task = asyncio.create_task(prompt.prompt_async("Paste the authorization code: "))
    async with asyncio.timeout(timeout):
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Cancel the prompt once, then shield/join its cleanup. Keeping this
            # inside timeout also preserves a later caller cancellation instead
            # of converting it into an ordinary timeout result.
            task.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await await_owned_task(task)
            raise


async def _handle_manual_code_flow(auth_url: str, timeout: float) -> CallbackResult:
    import webbrowser

    console.print("Opening browser for authentication...")
    console.print(f"\n[dim]URL: {auth_url}[/dim]\n")
    webbrowser.open(auth_url)
    console.print(
        "[yellow]After authorizing, you'll see a code on the page.[/yellow]\n"
        "[yellow]Copy the entire code and paste it below.[/yellow]\n"
    )
    try:
        code = await _read_manual_authorization_code(timeout)
        if not code or not code.strip():
            return CallbackResult(
                success=False,
                error="empty_code",
                error_description="No authorization code provided",
            )
        return CallbackResult(success=True, code=code.strip())
    except asyncio.TimeoutError:
        return CallbackResult(
            success=False,
            error="timeout",
            error_description=f"No code entered within {timeout} seconds",
        )
    except EOFError:
        return CallbackResult(
            success=False,
            error="input_closed",
            error_description="Input closed before an authorization code was provided",
        )


async def _tokens_for_display(
    provider_id: str, tokens: OAuthTokens, storage: TokenStorage
) -> tuple[OAuthTokens | None, str]:
    """Refresh a display snapshot without publishing over logout or a newer login."""
    if tokens.is_expired():
        try:
            tokens = await async_refresh_token(provider_id, tokens, storage=storage)
        except Exception as exc:
            logger.debug(
                "Display token refresh failed for %s (%s)", provider_id, type(exc).__name__
            )
            tokens = None
        if tokens is None:
            return None, "unavailable"

    if tokens.models and tokens.is_models_cache_valid():
        return tokens, "cached"

    try:
        models, status = await get_provider(provider_id).list_models(tokens.access_token)
        cached = replace(tokens, models=models, models_fetched_at=int(time.time() * 1000))
        if await run_sync_owned(storage.save_if_current, tokens, cached):
            return cached, "api" if status.get("source") == "api" else "cached"
    except Exception as exc:
        logger.debug("Display model refresh failed for %s (%s)", provider_id, type(exc).__name__)
    # A lost CAS or request error cannot justify returning or re-saving an old
    # account snapshot. Keep a newer cache, or show unavailability after logout.
    try:
        current = await run_sync_owned(storage.load, provider_id)
    except Exception:
        current = None
    return current, "cached" if current is not None and current.models else "unavailable"


def _print_credentials_unavailable(provider_id: str) -> None:
    console.print(
        Panel(
            "Credentials unavailable: refresh failed or authentication changed.",
            title=f"[bold]{provider_id}[/bold]",
            border_style="yellow",
        )
    )


async def handle_list() -> None:
    storage = await run_sync_owned(get_token_storage)
    all_tokens = await run_sync_owned(storage.get_all_tokens)
    if not all_tokens:
        console.print(
            "\n[yellow]No OAuth providers configured.[/yellow]\n"
            "Use 'koder auth login <provider>' to authenticate.\n"
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        )
        return

    for provider_id, tokens in all_tokens.items():
        tokens, source = await _tokens_for_display(provider_id, tokens, storage)
        if tokens is None:
            _print_credentials_unavailable(provider_id)
            continue
        description = PROVIDER_DESCRIPTIONS.get(provider_id, "")
        account = tokens.email if tokens.email else None
        info = (
            f"[bold]Account:[/bold] {account}\n[bold]Type:[/bold] {description}\n"
            if account
            else f"[bold]Type:[/bold] {description}\n"
        )
        models = tokens.models
        if models:
            source_label = "[green]API[/green]" if source == "api" else "[cyan]cached[/cyan]"
            info += f"\n[bold]Models ({len(models)}):[/bold] {source_label}\n"
            for model in sorted(models):
                info += f"  • {model}\n"
        else:
            info += "\n[dim]No models available[/dim]\n"

        console.print(
            Panel(info.strip(), title=f"[bold cyan]{provider_id}[/bold cyan]", border_style="blue")
        )
        console.print()


async def handle_revoke(provider_id: str) -> bool:
    provider_id = _normalize_provider_id(provider_id)
    if provider_id == GITHUB_COPILOT_PROVIDER_ID:
        console.print(
            "[yellow]GitHub Copilot tokens are managed by LiteLLM. "
            "Run `koder auth login github_copilot` to refresh the login.[/yellow]"
        )
        return False

    storage = await run_sync_owned(get_token_storage)
    tokens = await run_sync_owned(storage.load, provider_id)
    if not tokens:
        console.print(f"[yellow]No tokens found for provider '{provider_id}'[/yellow]")
        return False

    try:
        provider = get_provider(provider_id)
        await provider.revoke_token(tokens.refresh_token)
    except Exception:
        logger.debug("Failed to revoke token for provider %s", provider_id, exc_info=True)

    try:
        deleted = await run_sync_owned(storage.delete, provider_id)
    except Exception:
        deleted = False
    if not deleted:
        console.print(f"[red]Unable to remove local tokens for {provider_id}[/red]")
        return False
    console.print(f"[green]Local tokens removed for {provider_id}[/green]")
    return True


async def handle_status(provider_id: Optional[str] = None) -> None:
    if provider_id:
        provider_id = _normalize_provider_id(provider_id)
    if provider_id == GITHUB_COPILOT_PROVIDER_ID:
        await _print_github_copilot_status()
        return

    storage = await run_sync_owned(get_token_storage)
    if provider_id:
        tokens = await run_sync_owned(storage.load, provider_id)
        if not tokens:
            console.print(f"[yellow]No tokens found for provider '{provider_id}'[/yellow]")
            return
        await _print_token_details(provider_id, tokens, storage)
        return

    all_tokens = await run_sync_owned(storage.get_all_tokens)
    if not all_tokens:
        console.print("[yellow]No OAuth providers configured.[/yellow]")
        return

    for pid, tokens in all_tokens.items():
        await _print_token_details(pid, tokens, storage)
        console.print()


def _token_status_label(tokens) -> str:
    if tokens.is_expired(0):
        return "expired"
    if tokens.is_expired(TOKEN_EXPIRY_BUFFER_MS):
        return "expiring_soon"
    return "valid"


def _build_token_status_dict(provider_id: str, tokens) -> dict:
    """Build a JSON-serializable, redacted status dict for one provider."""
    now_ms = int(datetime.now().timestamp() * 1000)
    time_left_ms = tokens.expires_at - now_ms
    return {
        "provider": provider_id,
        "status": _token_status_label(tokens),
        "account": tokens.email,
        "expires_at": tokens.expires_at,
        "time_left_minutes": max(0, time_left_ms // 60000),
        "has_access_token": bool(tokens.access_token),
        "has_refresh_token": bool(tokens.refresh_token),
        "models": sorted(tokens.models) if tokens.models else [],
    }


async def handle_status_json(provider_id: Optional[str] = None) -> None:
    """Emit auth status as a serializable JSON dict (no Rich panels)."""
    if provider_id:
        provider_id = _normalize_provider_id(provider_id)

    payload: dict[str, object]

    if provider_id == GITHUB_COPILOT_PROVIDER_ID:
        payload = {"providers": [_github_copilot_status_dict()]}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    storage = await run_sync_owned(get_token_storage)
    if provider_id:
        tokens = await run_sync_owned(storage.load, provider_id)
        if not tokens:
            payload = {"providers": [{"provider": provider_id, "status": "not_configured"}]}
        else:
            payload = {"providers": [_build_token_status_dict(provider_id, tokens)]}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    all_tokens = await run_sync_owned(storage.get_all_tokens)
    providers = [_build_token_status_dict(pid, tok) for pid, tok in all_tokens.items()]
    print(json.dumps({"providers": providers}, ensure_ascii=False, indent=2))


async def _print_token_details(provider_id: str, tokens, storage) -> None:
    tokens, source = await _tokens_for_display(provider_id, tokens, storage)
    if tokens is None:
        _print_credentials_unavailable(provider_id)
        return

    if tokens.is_expired(0):
        status = "[red]EXPIRED[/red]"
    elif tokens.is_expired(TOKEN_EXPIRY_BUFFER_MS):
        status = "[yellow]EXPIRING SOON[/yellow]"
    else:
        status = "[green]VALID[/green]"

    now_ms = int(datetime.now().timestamp() * 1000)
    time_left_ms = tokens.expires_at - now_ms
    time_left_mins = max(0, time_left_ms // 60000)
    expires = datetime.fromtimestamp(tokens.expires_at / 1000)

    account_line = f"Account: {tokens.email}\n" if tokens.email else ""
    info = (
        f"Status: {status}\n"
        f"{account_line}"
        f"Expires: {expires.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Time left: {time_left_mins} minutes\n"
        f"Access token: {'present' if tokens.access_token else 'missing'}\n"
        f"Refresh token: {'present' if tokens.refresh_token else 'missing'}"
    )

    models = tokens.models
    if models:
        source_label = "[green]API[/green]" if source == "api" else "[cyan]cached[/cyan]"
        info += f"\n\n[bold]Models ({len(models)}):[/bold] {source_label}\n"
        for model in sorted(models):
            info += f"  • {model}\n"

    console.print(Panel(info.strip(), title=f"[bold]{provider_id}[/bold]", border_style="blue"))


def _github_copilot_status_dict() -> dict:
    """Build a JSON-serializable status dict for GitHub Copilot."""
    try:
        from litellm.llms.github_copilot.authenticator import Authenticator

        authenticator = Authenticator()
        access_token_exists = bool(
            authenticator.access_token_file and os.path.exists(authenticator.access_token_file)
        )
        api_key_info = None
        try:
            with open(authenticator.api_key_file, encoding="utf-8") as file:
                api_key_info = json.load(file)
        except Exception:
            logger.debug("Failed to read GitHub Copilot API key file", exc_info=True)

        raw_expires_at = api_key_info.get("expires_at") if isinstance(api_key_info, dict) else None
        expires_at = raw_expires_at if isinstance(raw_expires_at, numbers.Real) else None
        if expires_at:
            status = "valid" if expires_at > time.time() else "expired"
        else:
            status = "needs_login"
        return {
            "provider": GITHUB_COPILOT_PROVIDER_ID,
            "status": status,
            "has_access_token": access_token_exists,
            "expires_at": float(expires_at) if expires_at else None,
            "token_cache": str(authenticator.token_dir),
        }
    except Exception as exc:
        return {
            "provider": GITHUB_COPILOT_PROVIDER_ID,
            "status": "unavailable",
            "error": str(exc),
        }


async def _print_github_copilot_status() -> None:
    try:
        from litellm.llms.github_copilot.authenticator import Authenticator

        authenticator = Authenticator()
        access_token_exists = bool(
            authenticator.access_token_file and os.path.exists(authenticator.access_token_file)
        )
        api_key_info = None
        try:
            with open(authenticator.api_key_file, encoding="utf-8") as file:
                api_key_info = json.load(file)
        except Exception:
            logger.debug("Failed to read GitHub Copilot API key file", exc_info=True)

        raw_expires_at = api_key_info.get("expires_at") if isinstance(api_key_info, dict) else None
        expires_at = raw_expires_at if isinstance(raw_expires_at, numbers.Real) else None
        status = "[yellow]NEEDS LOGIN[/yellow]"
        expires_line = "Expires: unavailable"
        if expires_at:
            expires = datetime.fromtimestamp(expires_at)
            expires_line = f"Expires: {expires.strftime('%Y-%m-%d %H:%M:%S')}"
            status = "[green]VALID[/green]" if expires_at > time.time() else "[red]EXPIRED[/red]"

        info = (
            f"Status: {status}\n"
            f"Access token: {'present' if access_token_exists else 'missing'}\n"
            f"{expires_line}\n"
            f"Token cache: {authenticator.token_dir}\n\n"
            "Refresh login: koder auth login github_copilot"
        )
        console.print(Panel(info, title="[bold]github_copilot[/bold]", border_style="blue"))
    except Exception as exc:
        console.print(f"[red]Unable to inspect GitHub Copilot status:[/red] {exc}")


def show_auth_help() -> None:
    help_text = """[bold]OAuth Authentication Commands[/bold]

Commands:
  login <provider>    Authenticate with a provider
  list                List configured OAuth providers and models
  revoke <provider>   Revoke OAuth tokens
  status [provider]   Show OAuth token status

Providers:
  google, claude, chatgpt, antigravity, github_copilot
"""
    console.print(Panel(help_text, title="koder auth", border_style="blue"))


async def handle_auth_subcommand(args) -> int:
    if args.auth_command == "login":
        token_arg = getattr(args, "token", None)
        if token_arg is not None or os.environ.get(AUTH_TOKEN_ENV):
            token = _resolve_ingested_token(token_arg)
            if not token:
                console.print(
                    "[red]No token provided via --token, stdin, or KODER_AUTH_TOKEN.[/red]"
                )
                return 1
            success = await run_sync_owned(handle_token_login, args.provider, token)
            return 0 if success else 1
        success = await handle_login(args.provider, timeout=args.timeout)
        return 0 if success else 1
    if args.auth_command == "list":
        await handle_list()
        return 0
    if args.auth_command == "revoke":
        success = await handle_revoke(args.provider)
        return 0 if success else 1
    if args.auth_command == "status":
        if getattr(args, "json_output", False):
            await handle_status_json(getattr(args, "provider", None))
            return 0
        await handle_status(getattr(args, "provider", None))
        return 0
    show_auth_help()
    return 0
