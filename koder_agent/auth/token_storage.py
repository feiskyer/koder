"""Token storage for OAuth credentials.

Stores OAuth tokens in plain JSON files under ~/.koder/tokens/
with file permissions set to 0600 for basic security.
"""

import json
import os
import stat
from pathlib import Path
from typing import Dict, List, Optional

from koder_agent.auth.base import OAuthTokens
from koder_agent.auth.constants import SUPPORTED_PROVIDERS


class TokenStorage:
    """Manages OAuth token storage in the filesystem.

    Tokens are stored as JSON files in ~/.koder/tokens/<provider>.json
    with restricted file permissions (0600).
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize token storage.

        Args:
            base_dir: Base directory for token storage.
                     Defaults to ~/.koder/tokens/
        """
        if base_dir is None:
            base_dir = Path.home() / ".koder" / "tokens"
        self.base_dir = Path(base_dir)
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure the tokens directory exists with proper permissions."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # Set directory permissions to 0700
        os.chmod(self.base_dir, stat.S_IRWXU)

    def _get_token_path(self, provider: str) -> Path:
        """Get the path for a provider's token file."""
        return self.base_dir / f"{provider}.json"

    def save(self, tokens: OAuthTokens) -> None:
        """Save tokens for a provider.

        Args:
            tokens: OAuth tokens to save
        """
        token_path = self._get_token_path(tokens.provider)

        # Write tokens to file
        with open(token_path, "w") as f:
            json.dump(tokens.to_dict(), f, indent=2)

        # Set file permissions to 0600 (owner read/write only)
        os.chmod(token_path, stat.S_IRUSR | stat.S_IWUSR)

    def load(self, provider: str) -> Optional[OAuthTokens]:
        """Load tokens for a provider.

        Args:
            provider: Provider identifier

        Returns:
            OAuthTokens if found, None otherwise
        """
        token_path = self._get_token_path(provider)

        if not token_path.exists():
            return None

        try:
            with open(token_path, "r") as f:
                data = json.load(f)
            return OAuthTokens.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def delete(self, provider: str) -> bool:
        """Delete tokens for a provider.

        Args:
            provider: Provider identifier

        Returns:
            True if tokens were deleted, False if not found
        """
        token_path = self._get_token_path(provider)

        if token_path.exists():
            token_path.unlink()
            return True
        return False

    def list_providers(self) -> List[str]:
        """List all providers with stored tokens.

        Returns:
            List of provider identifiers
        """
        providers = []
        for token_file in self.base_dir.glob("*.json"):
            provider = token_file.stem
            if provider in SUPPORTED_PROVIDERS:
                providers.append(provider)
        return providers

    def get_all_tokens(self) -> Dict[str, OAuthTokens]:
        """Load all stored tokens.

        Returns:
            Dict mapping provider to tokens
        """
        tokens = {}
        for provider in self.list_providers():
            token = self.load(provider)
            if token:
                tokens[provider] = token
        return tokens

    def has_valid_token(self, provider: str, buffer_ms: int = 60000) -> bool:
        """Check if provider has a valid (non-expired) access token.

        Args:
            provider: Provider identifier
            buffer_ms: Buffer time before expiry to consider expired

        Returns:
            True if valid token exists
        """
        tokens = self.load(provider)
        if tokens is None:
            return False
        return not tokens.is_expired(buffer_ms)

    def update_access_token(
        self,
        provider: str,
        access_token: str,
        expires_at: int,
        refresh_token: Optional[str] = None,
    ) -> bool:
        """Update access token for a provider.

        Args:
            provider: Provider identifier
            access_token: New access token
            expires_at: Expiry timestamp in milliseconds
            refresh_token: Optional new refresh token

        Returns:
            True if updated, False if provider not found
        """
        tokens = self.load(provider)
        if tokens is None:
            return False

        tokens.access_token = access_token
        tokens.expires_at = expires_at
        if refresh_token:
            tokens.refresh_token = refresh_token

        self.save(tokens)
        return True


# Global token storage instance
_token_storage: Optional[TokenStorage] = None


def get_token_storage() -> TokenStorage:
    """Get the global token storage instance."""
    global _token_storage
    if _token_storage is None:
        _token_storage = TokenStorage()
    return _token_storage
