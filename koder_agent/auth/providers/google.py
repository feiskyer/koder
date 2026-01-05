"""Google OAuth provider implementation.

Provides OAuth authentication for Gemini CLI (free with Google account)
using PKCE authorization code flow.

Note: This OAuth provider is named 'google' to avoid conflict with
the 'gemini' API key provider.
"""

import time
from typing import Any, Dict, Optional

import aiohttp

from koder_agent.auth.base import OAuthProvider, OAuthResult, OAuthTokens
from koder_agent.auth.constants import (
    GOOGLE_AUTH_URL,
    GOOGLE_CALLBACK_PATH,
    GOOGLE_CALLBACK_PORT,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_MODELS_URL,
    GOOGLE_REDIRECT_URI,
    GOOGLE_SCOPES,
    GOOGLE_TOKEN_URL,
    GOOGLE_USERINFO_URL,
)


class GoogleOAuthProvider(OAuthProvider):
    """Google OAuth provider for Gemini API access.

    Uses Google's OAuth 2.0 with PKCE to authenticate users
    and obtain access tokens for Gemini API.
    """

    provider_id = "google"
    auth_url = GOOGLE_AUTH_URL
    token_url = GOOGLE_TOKEN_URL
    redirect_uri = GOOGLE_REDIRECT_URI
    client_id = GOOGLE_CLIENT_ID
    client_secret = GOOGLE_CLIENT_SECRET
    scopes = GOOGLE_SCOPES
    callback_port = GOOGLE_CALLBACK_PORT
    callback_path = GOOGLE_CALLBACK_PATH

    def _build_auth_params(self) -> Dict[str, str]:
        """Build Google-specific authorization parameters."""
        params = super()._build_auth_params()

        # Google-specific parameters
        params["access_type"] = "offline"  # Request refresh token
        params["prompt"] = "consent"  # Force consent to get refresh token

        return params

    async def _process_token_response(self, token_data: Dict[str, Any]) -> OAuthResult:
        """Process Google token response.

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
                error="Missing refresh_token in response. Try revoking access and re-authenticating.",
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
        """Revoke a Google OAuth token.

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

    async def list_models(self, access_token: str, verbose: bool = False) -> tuple[list[str], dict]:
        """List available Gemini models.

        Note: The OAuth scope may not include model listing permission.
        Falls back to known models if API call fails.

        Args:
            access_token: Valid Google OAuth access token
            verbose: If True, return detailed status info

        Returns:
            Tuple of (model list, status dict with 'source' and optional 'error')
        """
        models = []
        status = {"source": "api", "error": None}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    GOOGLE_MODELS_URL,
                    headers=self.get_auth_headers(access_token),
                ) as response:
                    if response.ok:
                        data = await response.json()
                        for model in data.get("models", []):
                            model_name = model.get("name", "")
                            # Extract model ID from "models/gemini-xxx" format
                            if model_name.startswith("models/"):
                                model_id = model_name[7:]  # Remove "models/" prefix
                                # Filter for generation models (exclude embedding, etc.)
                                if "generateContent" in model.get("supportedGenerationMethods", []):
                                    models.append(f"{self.provider_id}/{model_id}")
                    else:
                        error_text = await response.text()
                        status["error"] = f"API returned {response.status}: {error_text[:200]}"
        except Exception as e:
            status["error"] = f"API request failed: {str(e)}"

        # Fallback to known models if API call fails or returns empty
        if not models:
            status["source"] = "fallback"
            models = [
                # Gemini 3 models (preview)
                f"{self.provider_id}/gemini-3-pro-preview",
                f"{self.provider_id}/gemini-3-flash-preview",
                # Gemini 2.5 models
                f"{self.provider_id}/gemini-2.5-pro",
                f"{self.provider_id}/gemini-2.5-flash",
                # Gemini 2.0 models
                f"{self.provider_id}/gemini-2.0-flash",
                # Legacy models
                f"{self.provider_id}/gemini-1.5-pro",
                f"{self.provider_id}/gemini-1.5-flash",
            ]

        return models, status
