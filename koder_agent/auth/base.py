"""Base OAuth provider class for authentication.

Provides abstract base class for OAuth providers with PKCE support,
token exchange, refresh, and revocation functionality.
"""

import base64
import hashlib
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlencode

import aiohttp


@dataclass
class PKCEPair:
    """PKCE code verifier and challenge pair."""

    verifier: str
    challenge: str


# Model cache TTL: 1 day in milliseconds
MODEL_CACHE_TTL_MS = 24 * 60 * 60 * 1000


@dataclass
class OAuthTokens:
    """OAuth token storage structure."""

    provider: str
    access_token: str
    refresh_token: str
    expires_at: int  # Unix timestamp in milliseconds
    email: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    # Cached models with TTL
    models: list[str] = field(default_factory=list)
    models_fetched_at: Optional[int] = None  # Unix timestamp in milliseconds

    def is_expired(self, buffer_ms: int = 60000) -> bool:
        """Check if access token is expired or will expire within buffer period."""
        return time.time() * 1000 >= self.expires_at - buffer_ms

    def is_models_cache_valid(self) -> bool:
        """Check if models cache is still valid (within TTL)."""
        if not self.models_fetched_at:
            return False
        return time.time() * 1000 < self.models_fetched_at + MODEL_CACHE_TTL_MS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "provider": self.provider,
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "email": self.email,
            "extra": self.extra,
            "models": self.models,
            "models_fetched_at": self.models_fetched_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OAuthTokens":
        """Create from dictionary."""
        return cls(
            provider=data["provider"],
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data["expires_at"],
            email=data.get("email"),
            extra=data.get("extra", {}),
            models=data.get("models", []),
            models_fetched_at=data.get("models_fetched_at"),
        )


@dataclass
class OAuthResult:
    """Result of an OAuth operation."""

    success: bool
    tokens: Optional[OAuthTokens] = None
    error: Optional[str] = None
    api_key: Optional[str] = None  # For providers that create API keys


class OAuthProvider(ABC):
    """Abstract base class for OAuth providers.

    Implements PKCE flow for secure authorization in CLI applications.
    Subclasses must implement provider-specific URLs and token handling.
    """

    # Provider identifier
    provider_id: str = ""

    # OAuth endpoints (to be set by subclasses)
    auth_url: str = ""
    token_url: str = ""
    redirect_uri: str = ""
    client_id: str = ""
    client_secret: Optional[str] = None
    scopes: list = []

    # Callback server configuration (provider-specific)
    callback_port: int = 1455
    callback_path: str = "/auth/callback"

    def __init__(self):
        """Initialize the OAuth provider."""
        self._pkce: Optional[PKCEPair] = None
        self._state: Optional[str] = None

    @staticmethod
    def generate_pkce() -> PKCEPair:
        """Generate PKCE code verifier and challenge.

        Returns:
            PKCEPair with verifier and S256 challenge
        """
        # Generate random verifier (43-128 characters)
        verifier = secrets.token_urlsafe(32)

        # Create S256 challenge
        digest = hashlib.sha256(verifier.encode("utf-8")).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("utf-8")

        return PKCEPair(verifier=verifier, challenge=challenge)

    @staticmethod
    def generate_state() -> str:
        """Generate random state parameter for CSRF protection."""
        return secrets.token_hex(16)

    def get_authorization_url(self) -> Tuple[str, str]:
        """Generate the OAuth authorization URL with PKCE.

        Returns:
            Tuple of (authorization_url, code_verifier)
        """
        self._pkce = self.generate_pkce()
        self._state = self.generate_state()

        params = self._build_auth_params()

        url = f"{self.auth_url}?{urlencode(params)}"
        return url, self._pkce.verifier

    def _build_auth_params(self) -> Dict[str, str]:
        """Build authorization URL parameters.

        Override in subclasses for provider-specific parameters.
        """
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "code_challenge": self._pkce.challenge,
            "code_challenge_method": "S256",
            "state": self._state,
        }

        if self.scopes:
            params["scope"] = (
                " ".join(self.scopes) if isinstance(self.scopes, list) else self.scopes
            )

        return params

    async def exchange_code(self, code: str, verifier: str) -> OAuthResult:
        """Exchange authorization code for tokens.

        Args:
            code: Authorization code from callback
            verifier: PKCE code verifier

        Returns:
            OAuthResult with tokens or error
        """
        try:
            data = self._build_token_request(code, verifier)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.token_url,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
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

    def _build_token_request(self, code: str, verifier: str) -> Dict[str, str]:
        """Build token exchange request parameters.

        Override in subclasses for provider-specific parameters.
        """
        data = {
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": self.redirect_uri,
        }

        if self.client_secret:
            data["client_secret"] = self.client_secret

        return data

    @abstractmethod
    async def _process_token_response(self, token_data: Dict[str, Any]) -> OAuthResult:
        """Process token response from provider.

        Must be implemented by subclasses to handle provider-specific
        token response format and fetch user info if needed.
        """
        pass

    async def refresh_tokens(self, refresh_token: str) -> OAuthResult:
        """Refresh access token using refresh token.

        Args:
            refresh_token: The refresh token

        Returns:
            OAuthResult with new tokens or error
        """
        try:
            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
            }

            if self.client_secret:
                data["client_secret"] = self.client_secret

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.token_url,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
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

    async def revoke_token(self, token: str) -> bool:
        """Revoke an access or refresh token.

        Default implementation does nothing. Override for providers
        that support token revocation.

        Args:
            token: The token to revoke

        Returns:
            True if revocation succeeded
        """
        return True

    async def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Fetch user information using access token.

        Override in subclasses for provider-specific user info endpoint.

        Args:
            access_token: Valid access token

        Returns:
            User info dict or None if not available
        """
        return None

    def get_auth_headers(self, access_token: str) -> Dict[str, str]:
        """Get authorization headers for API requests.

        Override in subclasses for provider-specific headers.

        Args:
            access_token: Valid access token

        Returns:
            Headers dict for API requests
        """
        return {"Authorization": f"Bearer {access_token}"}

    async def list_models(self, access_token: str, verbose: bool = False) -> tuple[list[str], dict]:
        """List available models for this provider.

        Override in subclasses to call provider's models API.

        Args:
            access_token: Valid access token
            verbose: If True, return detailed status info

        Returns:
            Tuple of (model list, status dict with 'source' and optional 'error')
            - source: "api", "fallback", or "hardcoded"
            - error: Error message if API call failed (None otherwise)
            - reason: Optional explanation (e.g., "No public API available")
        """
        return [], {"source": "hardcoded", "error": None}
