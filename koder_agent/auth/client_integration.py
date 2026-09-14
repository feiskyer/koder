"""OAuth integration with the client module.

Provides functions to integrate OAuth tokens with the existing
client setup for seamless authentication.
"""

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from filelock import Timeout as FileLockTimeout

from koder_agent.auth.base import OAuthTokens
from koder_agent.auth.constants import TOKEN_EXPIRY_BUFFER_MS
from koder_agent.auth.token_storage import TokenStorage, get_token_storage
from koder_agent.utils.async_tasks import await_owned_task, run_sync_owned

logger = logging.getLogger(__name__)
OAUTH_REFRESH_TIMEOUT_SECONDS = 30.0
_REFRESH_LOCK_POLL_SECONDS = 0.05


def _normalize_provider_for_log(provider: str) -> str:
    """Return a stable provider identifier suitable for refresh logs."""
    return provider.strip().lower()


def get_oauth_token(provider: str) -> Optional[OAuthTokens]:
    """Get OAuth tokens for a provider if available.

    Args:
        provider: Provider identifier

    Returns:
        OAuthTokens if available and valid, None otherwise
    """
    storage = get_token_storage()
    tokens = storage.load(provider)

    if tokens is None:
        return None

    # Check if token needs refresh
    if tokens.is_expired(TOKEN_EXPIRY_BUFFER_MS):
        # Try to refresh
        refreshed = _sync_refresh_token(provider, tokens, storage=storage)
        if refreshed:
            return refreshed
        # If refresh failed, token is still expired
        return None

    return tokens


async def async_get_oauth_token(provider: str) -> Optional[OAuthTokens]:
    """Acquire credentials without blocking the event loop on local storage."""
    provider = _normalize_provider_for_log(provider)
    storage = await run_sync_owned(get_token_storage)
    tokens = await run_sync_owned(storage.load, provider)
    if tokens is None:
        return None
    if tokens.is_expired(TOKEN_EXPIRY_BUFFER_MS):
        return await async_refresh_token(provider, tokens, storage=storage)
    return tokens


def _commit_refresh(
    provider: str,
    previous: OAuthTokens,
    refreshed: OAuthTokens,
    *,
    storage: TokenStorage | None = None,
) -> Optional[OAuthTokens]:
    """Commit only against the refresh input, or reuse a concurrent valid winner."""
    if previous.provider != provider.strip().lower() or refreshed.provider != previous.provider:
        raise ValueError("Refresh token provider mismatch")
    if storage is None:
        storage = get_token_storage()
    if storage.save_if_current(previous, refreshed):
        logger.info("Refreshed OAuth tokens for %s", previous.provider)
        return refreshed
    current = storage.load(previous.provider)
    if current is not None and not current.is_expired(TOKEN_EXPIRY_BUFFER_MS):
        return current
    return None


def _sync_refresh_token(
    provider: str, tokens: OAuthTokens, *, storage: TokenStorage | None = None
) -> Optional[OAuthTokens]:
    """Synchronously refresh OAuth tokens.

    Args:
        provider: Provider identifier
        tokens: Current tokens

    Returns:
        Refreshed tokens or None if refresh failed
    """

    def refresh():
        return asyncio.run(async_refresh_token(provider, tokens, storage=storage))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return refresh()

    # Compatibility for explicitly synchronous callers. This necessarily blocks
    # its caller: normal async code must use async_get_oauth_token instead.
    # The shared refresh operation owns its deadline and publication; an outer
    # Future timeout must not abandon a potentially rotated credential.
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(refresh).result()


@dataclass
class _RefreshState:
    deadline: float
    cancellation_requested: bool = False


async def async_refresh_token(
    provider: str, tokens: OAuthTokens, *, storage: TokenStorage | None = None
) -> Optional[OAuthTokens]:
    """Asynchronously refresh OAuth tokens.

    Args:
        provider: Provider identifier
        tokens: Current tokens

    Returns:
        Refreshed tokens or None if refresh failed
    """
    provider = _normalize_provider_for_log(provider)
    state = _RefreshState(asyncio.get_running_loop().time() + OAUTH_REFRESH_TIMEOUT_SECONDS)
    work = asyncio.create_task(_refresh_job(provider, tokens, storage, state))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError as original:
        state.cancellation_requested = True
        # An already-started refresh may rotate the credential. Finish its
        # bounded operation and conditional publication rather than losing the
        # result. Waiters that have not sent a request stop in the lease loop.
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await await_owned_task(work)
        raise original


async def _refresh_job(
    provider: str, tokens: OAuthTokens, storage: TokenStorage | None, state: _RefreshState
) -> Optional[OAuthTokens]:
    try:
        if tokens.provider != provider:
            raise ValueError("Refresh token provider mismatch")
        if state.cancellation_requested:
            return None
        if storage is None:
            storage = await run_sync_owned(get_token_storage)
        return await _refresh_under_lease(provider, tokens, storage, state)
    except asyncio.TimeoutError:
        logger.warning("OAuth token refresh failed provider=%s category=timeout", provider)
        return None
    except Exception as error:
        logger.error(
            "OAuth token refresh failed provider=%s category=exception exception_type=%s",
            provider,
            type(error).__name__,
        )
        return None


async def _refresh_under_lease(
    provider: str, tokens: OAuthTokens, storage: TokenStorage, state: _RefreshState
) -> Optional[OAuthTokens]:
    lock = await run_sync_owned(storage.refresh_lock, provider)
    acquired = False
    loop = asyncio.get_running_loop()
    try:
        while not acquired:
            if state.cancellation_requested:
                return None
            remaining = state.deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            try:
                # Each attempt is nonblocking. Do not leave a cancelled thread
                # waiting to acquire a lease after its caller has returned.
                await run_sync_owned(lock.acquire, timeout=0)
                acquired = True
            except FileLockTimeout:
                await asyncio.sleep(min(_REFRESH_LOCK_POLL_SECONDS, remaining))

        if state.cancellation_requested:
            return None
        current = await run_sync_owned(storage.load, provider)
        if current is None:
            return None
        if current != tokens:
            return current if not current.is_expired(TOKEN_EXPIRY_BUFFER_MS) else None
        if state.cancellation_requested:
            return None
        remaining = state.deadline - loop.time()
        if remaining <= 0:
            raise asyncio.TimeoutError

        from koder_agent.auth.providers import get_provider

        result = await asyncio.wait_for(
            get_provider(provider).refresh_tokens(tokens.refresh_token),
            timeout=remaining,
        )
        if result.success and result.tokens:
            return await run_sync_owned(
                _commit_refresh, provider, tokens, result.tokens, storage=storage
            )
        logger.warning("OAuth token refresh failed provider=%s category=refresh_rejected", provider)
        return None
    finally:
        if acquired:
            await run_sync_owned(lock.release)


def get_oauth_api_key(provider: str) -> Optional[str]:
    """Get access token as API key for a provider.

    This is used by providers that accept Bearer tokens in place of API keys.

    Args:
        provider: Provider identifier

    Returns:
        Access token string or None
    """
    tokens = get_oauth_token(provider)
    if tokens:
        return tokens.access_token
    return None


def get_oauth_headers(provider: str) -> Dict[str, str]:
    """Get OAuth authorization headers for a provider.

    Args:
        provider: Provider identifier

    Returns:
        Headers dict for API requests
    """
    tokens = get_oauth_token(provider)
    if not tokens:
        return {}

    try:
        from koder_agent.auth.providers import get_provider

        oauth_provider = get_provider(provider)
        return oauth_provider.get_auth_headers(tokens.access_token)
    except Exception:
        # Fallback to basic Bearer auth
        return {"Authorization": f"Bearer {tokens.access_token}"}


def has_oauth_token(provider: str) -> bool:
    """Check if a provider has valid OAuth tokens.

    Args:
        provider: Provider identifier

    Returns:
        True if valid OAuth tokens exist
    """
    storage = get_token_storage()
    return storage.has_valid_token(provider)


def has_oauth_credentials(provider: str) -> bool:
    """Check if a provider has stored OAuth credentials (valid or expired).

    This is used to decide routing to OAuth handlers even if a token
    needs refresh.

    Args:
        provider: Provider identifier

    Returns:
        True if a token file exists for the provider
    """
    storage = get_token_storage()
    return storage.load(provider) is not None


def get_provider_auth_info(provider: str) -> Tuple[Optional[str], Optional[Dict[str, str]], bool]:
    """Get authentication info for a provider.

    Returns API key and extra headers for the provider, preferring OAuth
    tokens over environment variables.

    Args:
        provider: Provider identifier

    Returns:
        Tuple of (api_key, extra_headers, is_oauth)
    """
    # Check for OAuth tokens first
    tokens = get_oauth_token(provider)
    if tokens:
        try:
            from koder_agent.auth.providers import get_provider

            oauth_provider = get_provider(provider)
            headers = oauth_provider.get_auth_headers(tokens.access_token)
            return tokens.access_token, headers, True
        except Exception:
            return tokens.access_token, {"Authorization": f"Bearer {tokens.access_token}"}, True

    return None, None, False


def map_provider_to_oauth(model_provider: str) -> Optional[str]:
    """Map a model provider name to OAuth provider name.

    Only OAuth providers (google, claude, chatgpt, antigravity) return themselves.
    API-based providers (anthropic, openai, gemini, azure, etc.) return None.

    Args:
        model_provider: Provider name from model config

    Returns:
        OAuth provider name or None if not an OAuth provider
    """
    if not model_provider:
        return None
    oauth_providers = {"google", "claude", "chatgpt", "antigravity"}
    provider = model_provider.strip().lower()
    return provider if provider in oauth_providers else None


def list_oauth_provider_ids() -> list[str]:
    """Return the supported OAuth provider IDs from the current auth runtime."""
    from koder_agent.auth.providers import list_providers

    return list_providers()
