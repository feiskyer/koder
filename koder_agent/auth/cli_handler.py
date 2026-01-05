"""CLI handler for OAuth authentication commands.

Provides CLI commands for:
- koder auth login <provider>
- koder auth list
- koder auth revoke <provider>
- koder auth status
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Optional

from rich.console import Console
from rich.panel import Panel

from koder_agent.auth.callback_server import CallbackResult, run_oauth_flow
from koder_agent.auth.constants import SUPPORTED_PROVIDERS, TOKEN_EXPIRY_BUFFER_MS
from koder_agent.auth.providers import get_provider
from koder_agent.auth.token_storage import get_token_storage

console = Console()

# Provider descriptions (static metadata)
PROVIDER_DESCRIPTIONS: Dict[str, str] = {
    "google": "Gemini CLI (free with Google account)",
    "claude": "Claude Max subscription",
    "chatgpt": "ChatGPT Plus/Pro subscription",
    "antigravity": "Antigravity (Gemini 3 + Claude models)",
}


async def handle_login(provider_id: str, timeout: float = 300) -> bool:
    """Handle OAuth login for a provider.

    Args:
        provider_id: Provider identifier (google, claude, chatgpt, antigravity)
        timeout: Timeout in seconds for OAuth flow

    Returns:
        True if login succeeded
    """
    if provider_id not in SUPPORTED_PROVIDERS:
        console.print(
            f"[red]Error:[/red] Unknown provider '{provider_id}'. "
            f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )
        return False

    console.print(f"\n[bold]Authenticating with {provider_id}...[/bold]\n")

    try:
        # Get provider
        provider = get_provider(provider_id)

        # Generate authorization URL
        auth_url, verifier = provider.get_authorization_url()

        # Anthropic uses a different flow - manual code entry
        if provider_id == "claude":
            result = await _handle_manual_code_flow(auth_url, timeout)
        else:
            # Use localhost callback server for other providers
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

        # Exchange code for tokens
        console.print("Exchanging authorization code for tokens...")
        exchange_result = await provider.exchange_code(result.code, verifier)

        if not exchange_result.success:
            console.print(f"[red]Token exchange failed:[/red] {exchange_result.error}")
            return False

        tokens = exchange_result.tokens

        # Fetch available models from provider API
        console.print("Fetching available models...")
        models, _ = await provider.list_models(tokens.access_token)
        tokens.models = models
        tokens.models_fetched_at = int(time.time() * 1000)

        # Save tokens with cached models
        storage = get_token_storage()
        storage.save(tokens)

        # Display success with models from API
        email = tokens.email or "Unknown"
        description = PROVIDER_DESCRIPTIONS.get(provider_id, "")

        # Build success message
        success_msg = (
            f"[green]Successfully authenticated![/green]\n\n"
            f"Provider: {provider_id} ({description})\n"
            f"Account: {email}\n"
        )

        if models:
            success_msg += f"\n[bold]Available Models ({len(models)}):[/bold]\n"
            # Show first 5 models
            display_models = models[:5]
            success_msg += ", ".join(display_models)
            if len(models) > 5:
                success_msg += f"\n  +{len(models) - 5} more (see 'koder auth list')"

            # Show usage example with first model
            success_msg += f'\n\n[bold]Usage:[/bold]\nKODER_MODEL="{models[0]}" koder "your prompt"'
        else:
            success_msg += "\n[dim]No models found (API may not support model listing)[/dim]"

        console.print(
            Panel(
                success_msg,
                title="Authentication Complete",
                border_style="green",
            )
        )

        return True

    except Exception as e:
        console.print(f"[red]Error during authentication:[/red] {e}")
        return False


async def _handle_manual_code_flow(auth_url: str, timeout: float) -> "CallbackResult":
    """Handle OAuth flow where user must manually paste the code.

    Used for providers like Anthropic that redirect to their own site
    instead of localhost.

    Args:
        auth_url: Authorization URL to open
        timeout: Timeout in seconds

    Returns:
        CallbackResult with code or error
    """
    import webbrowser

    console.print("Opening browser for authentication...")
    console.print(f"\n[dim]URL: {auth_url}[/dim]\n")
    webbrowser.open(auth_url)

    console.print(
        "[yellow]After authorizing, you'll see a code on the page.[/yellow]\n"
        "[yellow]Copy the entire code (including any # and text after it).[/yellow]\n"
    )

    try:
        code = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(
                None, lambda: input("Paste the authorization code: ")
            ),
            timeout=timeout,
        )

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


async def handle_list() -> None:
    """List all configured OAuth providers with their models."""
    storage = get_token_storage()
    all_tokens = storage.get_all_tokens()

    if not all_tokens:
        console.print(
            "\n[yellow]No OAuth providers configured.[/yellow]\n"
            "Use 'koder auth login <provider>' to authenticate.\n"
            f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
        )
        return

    for provider_id, tokens in all_tokens.items():
        description = PROVIDER_DESCRIPTIONS.get(provider_id, "")
        # Claude doesn't provide user info via OAuth
        account = tokens.email if tokens.email else None
        if account:
            info = f"[bold]Account:[/bold] {account}\n[bold]Type:[/bold] {description}\n"
        else:
            info = f"[bold]Type:[/bold] {description}\n"

        # Auto-refresh token if expired
        access_token = tokens.access_token
        if tokens.is_expired():
            console.print(f"[dim]Refreshing {provider_id} token...[/dim]")
            try:
                provider = get_provider(provider_id)
                result = await provider.refresh_tokens(tokens.refresh_token)
                if result.success and result.tokens:
                    storage.save(result.tokens)
                    tokens = result.tokens
                    access_token = tokens.access_token
            except Exception as e:
                info += f"\n[red]Token refresh failed: {e}[/red]\n"

        # Fetch models from API if cache is stale or empty
        models = tokens.models
        source = "cached"

        if not tokens.is_models_cache_valid() or not models:
            try:
                provider = get_provider(provider_id)
                models, status = await provider.list_models(access_token)
                source = status.get("source", "api")

                # Update cache
                tokens.models = models
                tokens.models_fetched_at = int(time.time() * 1000)
                storage.save(tokens)
            except Exception:
                source = "cached" if models else "unavailable"

        # Display models
        if models:
            source_label = "[green]API[/green]" if source == "api" else "[cyan]cached[/cyan]"
            info += f"\n[bold]Models ({len(models)}):[/bold] {source_label}\n"
            for model in sorted(models):
                info += f"  • {model}\n"
        else:
            info += "\n[dim]No models available[/dim]\n"

        console.print(
            Panel(
                info.strip(),
                title=f"[bold cyan]{provider_id}[/bold cyan]",
                border_style="blue",
            )
        )
        console.print()


async def handle_revoke(provider_id: str) -> bool:
    """Revoke OAuth tokens for a provider.

    Args:
        provider_id: Provider identifier

    Returns:
        True if revocation succeeded
    """
    storage = get_token_storage()
    tokens = storage.load(provider_id)

    if not tokens:
        console.print(f"[yellow]No tokens found for provider '{provider_id}'[/yellow]")
        return False

    console.print(f"Revoking tokens for {provider_id}...")

    try:
        # Try to revoke with provider
        provider = get_provider(provider_id)
        await provider.revoke_token(tokens.refresh_token)
    except Exception:
        # Continue even if revocation fails
        pass

    # Delete local tokens
    storage.delete(provider_id)

    console.print(f"[green]Tokens revoked for {provider_id}[/green]")
    return True


async def handle_status(provider_id: Optional[str] = None) -> None:
    """Show OAuth token status.

    Args:
        provider_id: Optional specific provider to show
    """
    storage = get_token_storage()

    if provider_id:
        # Show single provider
        tokens = storage.load(provider_id)
        if not tokens:
            console.print(f"[yellow]No tokens found for provider '{provider_id}'[/yellow]")
            return

        await _print_token_details(provider_id, tokens, storage)
    else:
        # Show all providers
        all_tokens = storage.get_all_tokens()
        if not all_tokens:
            console.print("[yellow]No OAuth providers configured.[/yellow]")
            return

        for pid, tokens in all_tokens.items():
            await _print_token_details(pid, tokens, storage)
            console.print()


async def _print_token_details(provider_id: str, tokens, storage) -> None:
    """Print detailed token information with models."""
    # Auto-refresh token if expired
    access_token = tokens.access_token
    if tokens.is_expired():
        console.print(f"[dim]Refreshing {provider_id} token...[/dim]")
        try:
            provider = get_provider(provider_id)
            result = await provider.refresh_tokens(tokens.refresh_token)
            if result.success and result.tokens:
                storage.save(result.tokens)
                tokens = result.tokens
                access_token = tokens.access_token
        except Exception:
            pass

    # Determine status
    if tokens.is_expired(0):
        status = "[red]EXPIRED[/red]"
    elif tokens.is_expired(TOKEN_EXPIRY_BUFFER_MS):
        status = "[yellow]EXPIRING SOON[/yellow]"
    else:
        status = "[green]VALID[/green]"

    # Calculate time until expiry
    now_ms = int(datetime.now().timestamp() * 1000)
    time_left_ms = tokens.expires_at - now_ms
    time_left_mins = max(0, time_left_ms // 60000)

    expires = datetime.fromtimestamp(tokens.expires_at / 1000)

    # Build detailed info
    account_line = f"Account: {tokens.email}\n" if tokens.email else ""
    info = (
        f"Status: {status}\n"
        f"{account_line}"
        f"Expires: {expires.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Time left: {time_left_mins} minutes\n"
        f"Access token: {tokens.access_token[:20]}...\n"
        f"Refresh token: {tokens.refresh_token[:20]}..."
    )

    # Fetch models from API if cache is stale or empty
    models = tokens.models
    source = "cached"

    if not tokens.is_models_cache_valid() or not models:
        try:
            provider = get_provider(provider_id)
            models, status_info = await provider.list_models(access_token)
            source = "api" if status_info.get("source") == "api" else "cached"

            # Update cache
            tokens.models = models
            tokens.models_fetched_at = int(time.time() * 1000)
            storage.save(tokens)
        except Exception:
            pass

    # Add models info
    if models:
        source_label = "[green]API[/green]" if source == "api" else "[cyan]cached[/cyan]"
        info += f"\n\n[bold]Models ({len(models)}):[/bold] {source_label}\n"
        for model in sorted(models):
            info += f"  • {model}\n"

    console.print(
        Panel(
            info.strip(),
            title=f"[bold]{provider_id}[/bold]",
            border_style="blue",
        )
    )


def show_auth_help() -> None:
    """Display auth command help."""
    help_text = """[bold]OAuth Authentication Commands[/bold]

Manage OAuth authentication for subscription-based model access.

[bold]Commands:[/bold]
  login <provider>    Authenticate with a provider
  list                List configured OAuth providers and models
  revoke <provider>   Revoke OAuth tokens
  status [provider]   Show OAuth token status

[bold]Providers:[/bold]
  google       Gemini CLI (free with Google account)
  claude       Claude Max subscription
  chatgpt      ChatGPT Plus/Pro subscription
  antigravity  Antigravity (Gemini 3 + Claude models)

[bold]Examples:[/bold]
  koder auth login google      # Start OAuth login with Google
  koder auth list              # List all providers and available models
  koder auth status claude     # Show Claude token status
  koder auth revoke chatgpt    # Revoke ChatGPT tokens
"""
    console.print(Panel(help_text, title="koder auth", border_style="blue"))


async def run_auth_command(args) -> int:
    """Run auth subcommand.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success)
    """
    if args.auth_command == "login":
        success = await handle_login(args.provider, timeout=args.timeout)
        return 0 if success else 1

    elif args.auth_command == "list":
        await handle_list()
        return 0

    elif args.auth_command == "revoke":
        success = await handle_revoke(args.provider)
        return 0 if success else 1

    elif args.auth_command == "status":
        await handle_status(getattr(args, "provider", None))
        return 0

    else:
        # Show help when no subcommand provided
        show_auth_help()
        return 0
