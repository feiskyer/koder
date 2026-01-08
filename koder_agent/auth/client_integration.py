"""OAuth integration with the client module.

Provides functions to integrate OAuth tokens with the existing
client setup for seamless authentication.
"""

import asyncio
import logging
from typing import Dict, Optional, Tuple

from koder_agent.auth.base import OAuthTokens
from koder_agent.auth.constants import TOKEN_EXPIRY_BUFFER_MS
from koder_agent.auth.token_storage import get_token_storage

logger = logging.getLogger(__name__)


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
        refreshed = _sync_refresh_token(provider, tokens)
        if refreshed:
            return refreshed
        # If refresh failed, token is still expired
        return None

    return tokens


def _sync_refresh_token(provider: str, tokens: OAuthTokens) -> Optional[OAuthTokens]:
    """Synchronously refresh OAuth tokens.

    Args:
        provider: Provider identifier
        tokens: Current tokens

    Returns:
        Refreshed tokens or None if refresh failed
    """
    try:
        from koder_agent.auth.providers import get_provider

        oauth_provider = get_provider(provider)

        # Run refresh in event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new event loop for synchronous context
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, oauth_provider.refresh_tokens(tokens.refresh_token)
                )
                result = future.result(timeout=30)
        else:
            result = loop.run_until_complete(oauth_provider.refresh_tokens(tokens.refresh_token))

        if result.success and result.tokens:
            # Save refreshed tokens
            storage = get_token_storage()
            storage.save(result.tokens)
            logger.info(f"Refreshed OAuth tokens for {provider}")
            return result.tokens

        logger.warning(f"Failed to refresh OAuth tokens for {provider}: {result.error}")
        return None

    except Exception as e:
        logger.error(f"Error refreshing OAuth tokens for {provider}: {e}")
        return None


async def async_refresh_token(provider: str, tokens: OAuthTokens) -> Optional[OAuthTokens]:
    """Asynchronously refresh OAuth tokens.

    Args:
        provider: Provider identifier
        tokens: Current tokens

    Returns:
        Refreshed tokens or None if refresh failed
    """
    try:
        from koder_agent.auth.providers import get_provider

        oauth_provider = get_provider(provider)
        result = await oauth_provider.refresh_tokens(tokens.refresh_token)

        if result.success and result.tokens:
            # Save refreshed tokens
            storage = get_token_storage()
            storage.save(result.tokens)
            logger.info(f"Refreshed OAuth tokens for {provider}")
            return result.tokens

        logger.warning(f"Failed to refresh OAuth tokens for {provider}: {result.error}")
        return None

    except Exception as e:
        logger.error(f"Error refreshing OAuth tokens for {provider}: {e}")
        return None


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
