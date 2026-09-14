"""Token storage for OAuth credentials.

Stores OAuth tokens in plain JSON files under ~/.koder/tokens/
with file permissions set to 0600 for basic security.
On macOS, attempts to use Keychain when available.
"""

import json
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

from filelock import FileLock

from koder_agent.auth.base import OAuthTokens
from koder_agent.auth.constants import SUPPORTED_PROVIDERS
from koder_agent.auth.secure_storage import KeychainMutationUncertainError, SecureStorage
from koder_agent.utils.atomic_file import write_text_atomic

TOKEN_LOCK_TIMEOUT_SECONDS = 5


def _decode_keychain_record(value: str) -> dict:
    """Accept OAuth JSON and legacy hexadecimal OAuth record encodings.

    This belongs at the JSON-record layer: a generic hexadecimal-looking
    password must not be reinterpreted by SecureStorage.
    """
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        if len(value) % 2 or not re.fullmatch(r"[0-9a-fA-F]+", value):
            raise
        data = json.loads(bytes.fromhex(value).decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("OAuth Keychain record must be a JSON object")
    return data


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
        # Try to use secure storage (macOS Keychain) when available
        self._secure_storage = SecureStorage()
        self._use_keychain = self._secure_storage.is_available()

    def _ensure_directory(self) -> None:
        """Ensure the tokens directory exists with proper permissions."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # Set directory permissions to 0700
        os.chmod(self.base_dir, stat.S_IRWXU)

    def _get_token_path(self, provider: str) -> Path:
        """Get the path for a provider's token file."""
        if not isinstance(provider, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", provider):
            raise ValueError("Invalid token provider identifier")
        path = self.base_dir / f"{provider}.json"
        if path.is_symlink():
            raise ValueError("Refusing symlink token file")
        return path

    def save(self, tokens: OAuthTokens) -> None:
        """Save tokens for a provider.

        Args:
            tokens: OAuth tokens to save
        """
        with self._provider_lock(tokens.provider):
            self._save(tokens)

    @contextmanager
    def _provider_lock(self, provider: str):
        """Coordinate local stores, never network work; fail on bounded contention."""
        self._get_token_path(provider)
        lock_path = self.base_dir / f".{provider}.lock"
        if lock_path.is_symlink():
            raise ValueError("Refusing symlink token lock")
        # Credential replacement/deletion never removes an active lock.
        # The lock implementation owns its file lifecycle on release.
        with FileLock(str(lock_path), timeout=TOKEN_LOCK_TIMEOUT_SECONDS, mode=0o600):
            yield

    def refresh_lock(self, provider: str) -> FileLock:
        """Return a separate lease for cooperating network refresh operations.

        Storage mutations keep their short provider lock, so logout need not
        wait for network work. Async callers may acquire/release this lease on
        different worker threads; all lock ownership stays with this object.
        """
        self._get_token_path(provider)
        lock_path = self.base_dir / f".{provider}.refresh.lock"
        if lock_path.is_symlink():
            raise ValueError("Refusing symlink refresh lock")
        return FileLock(str(lock_path), timeout=0, thread_local=False, mode=0o600)

    def _save(self, tokens: OAuthTokens) -> None:
        """Publish while the caller holds the provider lock."""
        token_path = self._get_token_path(tokens.provider)
        # Serialize before touching either backing store.
        # Keep stable single-line ASCII JSON for legacy consumers. The native
        # pipe now preserves bytes; old pretty/hex records remain readable.
        token_json = json.dumps(tokens.to_dict(), ensure_ascii=True, separators=(",", ":"))
        stored_in_keychain = False
        if self._use_keychain:
            try:
                stored_in_keychain = self._secure_storage.store_checked(
                    "koder-oauth", tokens.provider, token_json
                )
            except KeychainMutationUncertainError:
                # A timed-out helper may already have written Keychain.
                # Do not publish a fallback or overwrite the previous file.
                raise
            except Exception:
                pass  # Fall back to file storage
        if stored_in_keychain:
            # A successful migration must not leave an older plaintext fallback.
            token_path.unlink(missing_ok=True)
            return

        if token_path.exists():
            os.chmod(token_path, stat.S_IRUSR | stat.S_IWUSR)
        file_data = tokens.to_dict()
        # Distinguish a new authoritative fallback from legacy plaintext copies
        # that older releases left behind after successfully saving to Keychain.
        file_data["_storage"] = "file_fallback"
        write_text_atomic(token_path, json.dumps(file_data, indent=2))

    def load(self, provider: str) -> Optional[OAuthTokens]:
        """Load tokens for a provider.

        Args:
            provider: Provider identifier

        Returns:
            OAuthTokens if found, None otherwise
        """
        with self._provider_lock(provider):
            return self._load(provider)

    def _load(self, provider: str, *, strict: bool = False) -> Optional[OAuthTokens]:
        """Read under the provider lock; CAS must not mistake a failed read for absence."""
        token_path = self._get_token_path(provider)
        file_tokens = None
        if token_path.exists():
            try:
                data = json.loads(token_path.read_text(encoding="utf-8"))
                file_tokens = OAuthTokens.from_dict(data)
                if file_tokens.provider != provider:
                    return None
                if data.get("_storage") == "file_fallback":
                    return file_tokens
            except (ValueError, KeyError, TypeError):
                # Includes invalid JSON/UTF-8 and malformed credential fields.
                # Filesystem errors remain distinct from a rejected record.
                return None

        if self._use_keychain:
            try:
                retrieve = (
                    self._secure_storage.retrieve_checked
                    if strict
                    else self._secure_storage.retrieve
                )
                token_json = retrieve("koder-oauth", provider)
                if token_json is not None:
                    data = _decode_keychain_record(token_json)
                    tokens = OAuthTokens.from_dict(data)
                    return tokens if tokens.provider == provider else None
            except Exception:
                if strict:
                    raise
                pass  # Fall back to file storage

        return file_tokens

    def save_if_current(self, expected: OAuthTokens, refreshed: OAuthTokens) -> bool:
        """Publish a refresh only if its complete input snapshot is still current.

        Missing or changed credentials reject the write. Storage errors propagate,
        not a claim of revocation. All cooperating writers must use this store;
        ``save`` remains the unconditional explicit-login API.
        """
        if refreshed.provider != expected.provider:
            raise ValueError("Refresh token provider mismatch")
        with self._provider_lock(expected.provider):
            if self._load(expected.provider, strict=True) != expected:
                return False
            self._save(refreshed)
            return True

    def delete(self, provider: str) -> bool:
        """Delete tokens for a provider.

        Args:
            provider: Provider identifier

        Returns:
            True if tokens were deleted, False if not found
        """
        with self._provider_lock(provider):
            return self._delete(provider)

    def _delete(self, provider: str) -> bool:
        """Remove both stores without inferring absence from a failed read."""
        token_path = self._get_token_path(provider)
        deleted = False
        if self._use_keychain:
            deleted = self._secure_storage.delete_checked("koder-oauth", provider)
        if token_path.exists():
            token_path.unlink()
            deleted = True
        return deleted

    def list_providers(self) -> List[str]:
        """List all providers with stored tokens.

        Returns:
            List of provider identifiers
        """
        providers = set()
        for token_file in self.base_dir.glob("*.json"):
            provider = token_file.stem
            if provider in SUPPORTED_PROVIDERS:
                providers.add(provider)
        if self._use_keychain:
            for provider in SUPPORTED_PROVIDERS:
                if provider not in providers and self.load(provider) is not None:
                    providers.add(provider)
        return sorted(providers)

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
        with self._provider_lock(provider):
            tokens = self._load(provider, strict=True)
            if tokens is None:
                return False

            tokens.access_token = access_token
            tokens.expires_at = expires_at
            if refresh_token:
                tokens.refresh_token = refresh_token
            self._save(tokens)
            return True


# Global token storage instance
_token_storage: Optional[TokenStorage] = None


def get_token_storage() -> TokenStorage:
    """Get the global token storage instance."""
    global _token_storage
    if _token_storage is None:
        _token_storage = TokenStorage()
    return _token_storage
