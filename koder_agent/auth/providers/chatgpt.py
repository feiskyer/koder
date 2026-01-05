"""ChatGPT OAuth provider implementation.

Provides OAuth authentication for ChatGPT Plus/Pro subscription using
PKCE authorization code flow. Uses stateless mode (store:false)
as required by ChatGPT backend.

Note: This OAuth provider is named 'chatgpt' to avoid conflict with
the 'openai' API key provider.
"""

import base64
import json
import time
from typing import Any, Dict, Optional

import aiohttp

from koder_agent.auth.base import OAuthProvider, OAuthResult, OAuthTokens
from koder_agent.auth.constants import (
    OPENAI_AUTH_URL,
    OPENAI_CALLBACK_PATH,
    OPENAI_CALLBACK_PORT,
    OPENAI_CLIENT_ID,
    OPENAI_MODELS_URL,
    OPENAI_REDIRECT_URI,
    OPENAI_SCOPES,
    OPENAI_TOKEN_URL,
)


class ChatGPTOAuthProvider(OAuthProvider):
    """ChatGPT OAuth provider for ChatGPT Plus/Pro subscription access.

    Uses OpenAI's OAuth 2.0 with PKCE to authenticate users
    and obtain access tokens for ChatGPT API via subscription.

    Note: ChatGPT backend requires store:false (stateless mode)
    which means full message history must be sent in every request.
    """

    provider_id = "chatgpt"
    auth_url = OPENAI_AUTH_URL
    token_url = OPENAI_TOKEN_URL
    redirect_uri = OPENAI_REDIRECT_URI
    client_id = OPENAI_CLIENT_ID
    scopes = OPENAI_SCOPES
    callback_port = OPENAI_CALLBACK_PORT
    callback_path = OPENAI_CALLBACK_PATH

    def _build_auth_params(self) -> Dict[str, str]:
        """Build OpenAI-specific authorization parameters."""
        params = super()._build_auth_params()

        # OpenAI-specific parameters for Codex CLI compatibility
        params["id_token_add_organizations"] = "true"
        params["codex_cli_simplified_flow"] = "true"
        params["originator"] = "codex_cli_rs"

        return params

    async def _process_token_response(self, token_data: Dict[str, Any]) -> OAuthResult:
        """Process OpenAI token response.

        Args:
            token_data: Token response from OpenAI

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

        # Extract email from JWT if present
        email = self._extract_email_from_jwt(access_token)

        tokens = OAuthTokens(
            provider=self.provider_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            email=email,
        )

        return OAuthResult(success=True, tokens=tokens)

    def _extract_email_from_jwt(self, token: str) -> Optional[str]:
        """Extract email from JWT access token.

        Args:
            token: JWT access token

        Returns:
            Email string or None if not found
        """
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            # Decode payload (second part)
            payload = parts[1]
            # Add padding if needed
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += "=" * padding

            decoded = base64.urlsafe_b64decode(payload)
            data = json.loads(decoded)
            return data.get("email") or data.get("sub")
        except Exception:
            return None

    def get_auth_headers(self, access_token: str) -> Dict[str, str]:
        """Get OpenAI authorization headers.

        Returns headers suitable for ChatGPT API requests.
        """
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration for OpenAI requests.

        Returns config needed for stateless ChatGPT API mode.
        """
        return {
            "store": False,  # Required for ChatGPT backend
            "stream": True,
        }

    async def list_models(self, access_token: str, verbose: bool = False) -> tuple[list[str], dict]:
        """List available OpenAI models.

        Note: Falls back to known models if API call fails.

        Args:
            access_token: Valid ChatGPT OAuth access token
            verbose: If True, return detailed status info

        Returns:
            Tuple of (model list, status dict with 'source' and optional 'error')
        """
        models = []
        status = {"source": "api", "error": None}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    OPENAI_MODELS_URL,
                    headers=self.get_auth_headers(access_token),
                ) as response:
                    if response.ok:
                        data = await response.json()
                        for model in data.get("data", []):
                            model_id = model.get("id", "")
                            # Filter for chat completion models
                            if model_id and any(
                                prefix in model_id for prefix in ["gpt-", "o1", "o3", "chatgpt"]
                            ):
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
                # GPT-5.2 Codex models
                f"{self.provider_id}/gpt-5.2-codex",
                # GPT-5.2 general purpose
                f"{self.provider_id}/gpt-5.2",
                # GPT-5.1 Codex Max models
                f"{self.provider_id}/gpt-5.1-codex-max",
                # GPT-5.1 Codex models
                f"{self.provider_id}/gpt-5.1-codex",
                f"{self.provider_id}/gpt-5.1-codex-mini",
                # GPT-5.1 general purpose
                f"{self.provider_id}/gpt-5.1",
                # Legacy GPT-4 models (for compatibility)
                f"{self.provider_id}/gpt-4o",
                f"{self.provider_id}/gpt-4o-mini",
            ]

        return models, status
