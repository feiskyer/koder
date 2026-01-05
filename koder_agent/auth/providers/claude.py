"""Claude OAuth provider implementation.

Provides OAuth authentication for Claude Max subscription using
PKCE authorization code flow. Supports both Claude Pro/Max
subscription access and API key creation.

Note: This OAuth provider is named 'claude' to avoid conflict with
the 'anthropic' API key provider.
"""

import time
from typing import Any, Dict, Optional

import aiohttp

from koder_agent.auth.base import OAuthProvider, OAuthResult, OAuthTokens
from koder_agent.auth.constants import (
    ANTHROPIC_AUTH_URL_CONSOLE,
    ANTHROPIC_AUTH_URL_MAX,
    ANTHROPIC_BETA_HEADERS,
    ANTHROPIC_CLIENT_ID,
    ANTHROPIC_CREATE_API_KEY_URL,
    ANTHROPIC_REDIRECT_URI,
    ANTHROPIC_SCOPES,
    ANTHROPIC_TOKEN_URL,
)


class ClaudeOAuthProvider(OAuthProvider):
    """Claude OAuth provider for Claude Max subscription access.

    Uses Anthropic's OAuth 2.0 with PKCE to authenticate users
    and obtain access tokens for Claude API via subscription.

    Supports two modes:
    - "max": Claude Pro/Max subscription (uses claude.ai OAuth)
    - "console": Create API key (uses console.anthropic.com OAuth)
    """

    provider_id = "claude"
    token_url = ANTHROPIC_TOKEN_URL
    redirect_uri = ANTHROPIC_REDIRECT_URI
    client_id = ANTHROPIC_CLIENT_ID
    scopes = ANTHROPIC_SCOPES

    def __init__(self, mode: str = "max"):
        """Initialize Claude OAuth provider.

        Args:
            mode: OAuth mode - "max" for Claude Pro/Max, "console" for API key
        """
        super().__init__()
        self.mode = mode
        self.auth_url = ANTHROPIC_AUTH_URL_MAX if mode == "max" else ANTHROPIC_AUTH_URL_CONSOLE

    def _build_auth_params(self) -> Dict[str, str]:
        """Build Anthropic-specific authorization parameters."""
        params = {
            "code": "true",  # Anthropic-specific parameter
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
            "code_challenge": self._pkce.challenge,
            "code_challenge_method": "S256",
            "state": self._pkce.verifier,  # Anthropic uses verifier as state
        }
        return params

    def _build_token_request(self, code: str, verifier: str) -> Dict[str, str]:
        """Build Anthropic token exchange request.

        Anthropic expects JSON body instead of form-encoded.
        """
        # Handle code format: code#state or just code
        if "#" in code:
            auth_code, state = code.split("#", 1)
        else:
            auth_code = code
            state = verifier

        return {
            "code": auth_code,
            "state": state,
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "code_verifier": verifier,
        }

    async def exchange_code(self, code: str, verifier: str) -> OAuthResult:
        """Exchange authorization code for tokens.

        Anthropic uses JSON request body instead of form-encoded.
        """
        try:
            data = self._build_token_request(code, verifier)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.token_url,
                    json=data,  # JSON body for Anthropic
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        return OAuthResult(
                            success=False,
                            error=f"Token exchange failed ({response.status}): {error_text}",
                        )

                    token_data = await response.json()
                    return await self._process_token_response(token_data)

        except Exception as e:
            return OAuthResult(success=False, error=str(e))

    async def _process_token_response(self, token_data: Dict[str, Any]) -> OAuthResult:
        """Process Anthropic token response.

        Args:
            token_data: Token response from Anthropic

        Returns:
            OAuthResult with tokens or error
        """
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 3600)

        if not access_token:
            return OAuthResult(success=False, error="Missing access_token in response")

        if not refresh_token:
            return OAuthResult(success=False, error="Missing refresh_token in response")

        # Calculate expiry timestamp
        expires_at = int(time.time() * 1000) + (expires_in * 1000)

        tokens = OAuthTokens(
            provider=self.provider_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            extra={"mode": self.mode},
        )

        return OAuthResult(success=True, tokens=tokens)

    async def refresh_tokens(self, refresh_token: str) -> OAuthResult:
        """Refresh Anthropic access token.

        Anthropic uses JSON request body for token refresh.
        """
        try:
            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.token_url,
                    json=data,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        return OAuthResult(
                            success=False,
                            error=f"Token refresh failed ({response.status}): {error_text}",
                        )

                    token_data = await response.json()

                    # Use existing refresh token if not returned
                    if "refresh_token" not in token_data:
                        token_data["refresh_token"] = refresh_token

                    return await self._process_token_response(token_data)

        except Exception as e:
            return OAuthResult(success=False, error=str(e))

    async def create_api_key(self, access_token: str) -> Optional[str]:
        """Create an API key using OAuth access token.

        Only works in "console" mode.

        Args:
            access_token: Valid OAuth access token

        Returns:
            API key string or None if failed
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    ANTHROPIC_CREATE_API_KEY_URL,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {access_token}",
                    },
                ) as response:
                    if response.ok:
                        data = await response.json()
                        return data.get("raw_key")
        except Exception:
            pass
        return None

    def get_auth_headers(self, access_token: str) -> Dict[str, str]:
        """Get Anthropic authorization headers.

        Includes OAuth beta headers required for API access.
        """
        return {
            "Authorization": f"Bearer {access_token}",
            "anthropic-beta": ",".join(ANTHROPIC_BETA_HEADERS),
        }

    async def list_models(self, access_token: str, verbose: bool = False) -> tuple[list[str], dict]:
        """List available Claude models.

        Note: Anthropic doesn't provide a public models listing API.
        Returns commonly available models for Claude Max subscription.

        Args:
            access_token: Valid Claude OAuth access token
            verbose: If True, return detailed status info

        Returns:
            Tuple of (model list, status dict with 'source' info)
        """
        # Anthropic doesn't have a models listing API
        # Return commonly available models for Claude Max subscription
        # Model names must match actual Anthropic API model identifiers
        status = {
            "source": "hardcoded",
            "error": None,
            "reason": "Anthropic has no public models API",
        }
        models = [
            # Claude 4.5 models (latest versions)
            f"{self.provider_id}/claude-sonnet-4-5-20250929",
            f"{self.provider_id}/claude-opus-4-5-20251101",
            # Aliases without date suffix
            f"{self.provider_id}/claude-sonnet-4-5",
            f"{self.provider_id}/claude-opus-4-5",
            # Claude 4.1 models
            f"{self.provider_id}/claude-opus-4-1-20250805",
            f"{self.provider_id}/claude-opus-4-1",
            # Claude 4 models
            f"{self.provider_id}/claude-sonnet-4-20250514",
            f"{self.provider_id}/claude-opus-4-20250514",
            # Legacy Claude 3.5/3 models
            f"{self.provider_id}/claude-3-5-sonnet-20241022",
            f"{self.provider_id}/claude-3-opus-20240229",
        ]
        return models, status
