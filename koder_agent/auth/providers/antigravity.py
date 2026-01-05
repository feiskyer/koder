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

# Antigravity internal API endpoints
CLOUD_CODE_BASE_URL = "https://cloudcode-pa.googleapis.com"
LOAD_CODE_ASSIST_URL = f"{CLOUD_CODE_BASE_URL}/v1internal:loadCodeAssist"
FETCH_MODELS_URL = f"{CLOUD_CODE_BASE_URL}/v1internal:fetchAvailableModels"
ANTIGRAVITY_USER_AGENT = "antigravity/1.11.3 Darwin/arm64"
DEFAULT_PROJECT_ID = "bamboo-precept-lgxtn"


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

    async def _fetch_project_id(self, access_token: str) -> tuple[Optional[str], Optional[str]]:
        """Fetch project ID and subscription tier from loadCodeAssist API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    LOAD_CODE_ASSIST_URL,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                        "User-Agent": ANTIGRAVITY_USER_AGENT,
                    },
                    json={"metadata": {"ideType": "ANTIGRAVITY"}},
                ) as response:
                    if response.ok:
                        data = await response.json()
                        project_id = data.get("cloudaicompanionProject")
                        paid_tier = data.get("paidTier", {})
                        current_tier = data.get("currentTier", {})
                        tier_id = paid_tier.get("id") or current_tier.get("id")
                        return project_id, tier_id
        except Exception:
            pass
        return None, None

    async def _fetch_available_models(
        self, access_token: str, project_id: str
    ) -> tuple[list[str], dict]:
        """Fetch available models from fetchAvailableModels API."""
        models = []
        status = {"source": "api", "error": None, "project_id": project_id}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    FETCH_MODELS_URL,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                        "User-Agent": ANTIGRAVITY_USER_AGENT,
                    },
                    json={"project": project_id},
                ) as response:
                    if response.ok:
                        data = await response.json()
                        models_data = data.get("models", {})

                        for model_name, model_info in models_data.items():
                            # Skip internal models
                            if model_info.get("isInternal"):
                                continue
                            # Only include models with quota info
                            if model_info.get("quotaInfo"):
                                models.append(f"{self.provider_id}/{model_name}")

                        status["model_count"] = len(models_data)
                        status["filtered_count"] = len(models)
                    else:
                        error_text = await response.text()
                        status["error"] = f"API returned {response.status}: {error_text[:200]}"
                        status["source"] = "fallback"
        except Exception as e:
            status["error"] = f"API request failed: {str(e)}"
            status["source"] = "fallback"

        return models, status

    async def list_models(self, access_token: str, verbose: bool = False) -> tuple[list[str], dict]:
        """List available Antigravity models from live API.

        Args:
            access_token: Valid Antigravity OAuth access token
            verbose: If True, return detailed status info

        Returns:
            Tuple of (model list, status dict with 'source' info)
        """
        # Fetch project ID first
        project_id, tier_id = await self._fetch_project_id(access_token)
        project_id = project_id or DEFAULT_PROJECT_ID

        # Fetch models from API
        models, status = await self._fetch_available_models(access_token, project_id)

        if tier_id:
            status["subscription_tier"] = tier_id

        # Fallback to hardcoded list if API fails
        if not models:
            status["source"] = "fallback"
            if not status.get("error"):
                status["error"] = "API returned no models"
            models = [
                f"{self.provider_id}/gemini-3-flash",
                f"{self.provider_id}/gemini-3-pro-low",
                f"{self.provider_id}/gemini-3-pro-high",
                f"{self.provider_id}/claude-sonnet-4-5",
                f"{self.provider_id}/claude-sonnet-4-5-thinking",
                f"{self.provider_id}/claude-opus-4-5-thinking",
                f"{self.provider_id}/gemini-2.5-flash",
                f"{self.provider_id}/gemini-2.5-pro",
            ]

        return models, status
