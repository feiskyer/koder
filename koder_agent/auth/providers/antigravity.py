"""Antigravity OAuth provider implementation.

Provides OAuth authentication for Antigravity which gives access to
Gemini 3 and Claude models via Google OAuth credentials.
Uses dual quota system: Antigravity quota + Gemini CLI quota.
"""

import time
from typing import Any, Dict, List, Optional

import aiohttp

from koder_agent.auth.base import OAuthProvider, OAuthResult, OAuthTokens
from koder_agent.auth.constants import (
    ANTIGRAVITY_API_BASE,
    ANTIGRAVITY_AUTH_URL,
    ANTIGRAVITY_CALLBACK_PATH,
    ANTIGRAVITY_CALLBACK_PORT,
    ANTIGRAVITY_CLIENT_ID,
    ANTIGRAVITY_CLIENT_SECRET,
    ANTIGRAVITY_REDIRECT_URI,
    ANTIGRAVITY_SCOPES,
    ANTIGRAVITY_TOKEN_URL,
    GOOGLE_USERINFO_URL,
)


class AntigravityOAuthProvider(OAuthProvider):
    """Antigravity OAuth provider for Gemini 3 + Claude access.

    Uses Google OAuth to authenticate and access models through
    Antigravity's quota system. Supports multi-account rotation
    for higher combined quotas.

    Available models via Antigravity:
    - gemini-3-flash, gemini-3-pro-low, gemini-3-pro-high
    - claude-sonnet-4-5, claude-opus-4-5 (with thinking variants)
    """

    provider_id = "antigravity"
    auth_url = ANTIGRAVITY_AUTH_URL
    token_url = ANTIGRAVITY_TOKEN_URL
    redirect_uri = ANTIGRAVITY_REDIRECT_URI
    client_id = ANTIGRAVITY_CLIENT_ID
    client_secret = ANTIGRAVITY_CLIENT_SECRET
    scopes = ANTIGRAVITY_SCOPES
    callback_port = ANTIGRAVITY_CALLBACK_PORT
    callback_path = ANTIGRAVITY_CALLBACK_PATH

    def __init__(self):
        """Initialize Antigravity OAuth provider."""
        super().__init__()
        self._accounts: List[OAuthTokens] = []

    def _build_auth_params(self) -> Dict[str, str]:
        """Build Antigravity-specific authorization parameters."""
        params = super()._build_auth_params()

        # Request refresh token and force consent
        params["access_type"] = "offline"
        params["prompt"] = "consent"

        return params

    async def _process_token_response(self, token_data: Dict[str, Any]) -> OAuthResult:
        """Process Antigravity token response.

        Args:
            token_data: Token response from Google

        Returns:
            OAuthResult with tokens or error
        """
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 3600)

        if not access_token:
            return OAuthResult(success=False, error="Missing access_token in response")

        if not refresh_token:
            return OAuthResult(
                success=False,
                error="Missing refresh_token. Try revoking access and re-authenticating.",
            )

        # Calculate expiry timestamp
        expires_at = int(time.time() * 1000) + (expires_in * 1000)

        # Fetch user info
        email = None
        user_info = await self.get_user_info(access_token)
        if user_info:
            email = user_info.get("email")

        tokens = OAuthTokens(
            provider=self.provider_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            email=email,
            extra={
                "quota_type": "antigravity",  # Primary quota
                "fallback_quota": "gemini_cli",  # Fallback quota
            },
        )

        return OAuthResult(success=True, tokens=tokens)

    async def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Fetch user info from Google.

        Args:
            access_token: Valid access token

        Returns:
            User info dict with email and profile info
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{GOOGLE_USERINFO_URL}?alt=json",
                    headers=self.get_auth_headers(access_token),
                ) as response:
                    if response.ok:
                        return await response.json()
        except Exception:
            pass
        return None

    async def revoke_token(self, token: str) -> bool:
        """Revoke an Antigravity OAuth token.

        Args:
            token: Access or refresh token to revoke

        Returns:
            True if revocation succeeded
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://oauth2.googleapis.com/revoke",
                    data={"token": token},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                ) as response:
                    return response.ok
        except Exception:
            return False

    def get_model_mapping(self) -> Dict[str, str]:
        """Get Antigravity model name to API model mapping.

        Returns:
            Dict mapping user model names to API model names
        """
        return {
            # Gemini 3 models (Antigravity quota)
            "antigravity-gemini-3-flash": "gemini-3-flash",
            "antigravity-gemini-3-pro-low": "gemini-3-pro-low",
            "antigravity-gemini-3-pro-high": "gemini-3-pro-high",
            # Claude models (Antigravity quota)
            "antigravity-claude-sonnet-4-5": "claude-sonnet-4-5",
            "antigravity-claude-sonnet-4-5-thinking-low": "claude-sonnet-4-5-thinking-low",
            "antigravity-claude-sonnet-4-5-thinking-medium": "claude-sonnet-4-5-thinking-medium",
            "antigravity-claude-sonnet-4-5-thinking-high": "claude-sonnet-4-5-thinking-high",
            "antigravity-claude-opus-4-5-thinking-low": "claude-opus-4-5-thinking-low",
            "antigravity-claude-opus-4-5-thinking-medium": "claude-opus-4-5-thinking-medium",
            "antigravity-claude-opus-4-5-thinking-high": "claude-opus-4-5-thinking-high",
            # Gemini CLI quota fallback models
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-pro": "gemini-2.5-pro",
            "gemini-3-flash-preview": "gemini-3-flash-preview",
            "gemini-3-pro-preview": "gemini-3-pro-preview",
        }

    def get_api_base_url(self, model: str) -> str:
        """Get API base URL for a model.

        Args:
            model: Model name

        Returns:
            API base URL
        """
        return ANTIGRAVITY_API_BASE

    async def list_models(self, access_token: str) -> list[str]:
        """List available Antigravity models.

        Note: Antigravity provides access to Gemini 3 + Claude models
        through Google OAuth. Model availability depends on quota.

        Args:
            access_token: Valid Antigravity OAuth access token

        Returns:
            List of model names with antigravity/ prefix
        """
        # Antigravity models based on the reference implementation
        return [
            # Gemini 3 models (Antigravity quota)
            f"{self.provider_id}/antigravity-gemini-3-flash",
            f"{self.provider_id}/antigravity-gemini-3-pro-low",
            f"{self.provider_id}/antigravity-gemini-3-pro-high",
            # Claude models (Antigravity quota)
            f"{self.provider_id}/antigravity-claude-sonnet-4-5",
            f"{self.provider_id}/antigravity-claude-sonnet-4-5-thinking-low",
            f"{self.provider_id}/antigravity-claude-sonnet-4-5-thinking-medium",
            f"{self.provider_id}/antigravity-claude-sonnet-4-5-thinking-high",
            f"{self.provider_id}/antigravity-claude-opus-4-5-thinking-low",
            f"{self.provider_id}/antigravity-claude-opus-4-5-thinking-medium",
            f"{self.provider_id}/antigravity-claude-opus-4-5-thinking-high",
            # Gemini CLI quota fallback models
            f"{self.provider_id}/gemini-2.5-flash",
            f"{self.provider_id}/gemini-2.5-pro",
            f"{self.provider_id}/gemini-3-flash-preview",
            f"{self.provider_id}/gemini-3-pro-preview",
        ]
