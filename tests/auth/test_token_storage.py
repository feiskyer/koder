"""Unit tests for OAuth token storage."""

import os
import stat
import tempfile
import time
from pathlib import Path

import pytest

from koder_agent.auth.base import OAuthTokens
from koder_agent.auth.token_storage import TokenStorage


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for token storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def storage(temp_storage_dir):
    """Create a TokenStorage instance with temp directory."""
    return TokenStorage(base_dir=temp_storage_dir)


@pytest.fixture
def sample_tokens():
    """Create sample OAuth tokens."""
    return OAuthTokens(
        provider="google",
        access_token="test_access_token_12345",
        refresh_token="test_refresh_token_67890",
        expires_at=int(time.time() * 1000) + 3600000,  # 1 hour from now
        email="test@example.com",
    )


class TestTokenStorage:
    """Tests for TokenStorage class."""

    def test_save_and_load(self, storage, sample_tokens):
        """Test saving and loading tokens."""
        storage.save(sample_tokens)

        loaded = storage.load("google")
        assert loaded is not None
        assert loaded.provider == sample_tokens.provider
        assert loaded.access_token == sample_tokens.access_token
        assert loaded.refresh_token == sample_tokens.refresh_token
        assert loaded.expires_at == sample_tokens.expires_at
        assert loaded.email == sample_tokens.email

    def test_load_nonexistent(self, storage):
        """Test loading tokens for nonexistent provider."""
        loaded = storage.load("nonexistent")
        assert loaded is None

    def test_delete(self, storage, sample_tokens):
        """Test deleting tokens."""
        storage.save(sample_tokens)
        assert storage.load("google") is not None

        result = storage.delete("google")
        assert result is True
        assert storage.load("google") is None

    def test_delete_nonexistent(self, storage):
        """Test deleting nonexistent tokens."""
        result = storage.delete("nonexistent")
        assert result is False

    def test_list_providers(self, storage, sample_tokens):
        """Test listing providers with stored tokens."""
        assert storage.list_providers() == []

        storage.save(sample_tokens)
        providers = storage.list_providers()
        assert "google" in providers

    def test_get_all_tokens(self, storage, sample_tokens):
        """Test getting all stored tokens."""
        assert storage.get_all_tokens() == {}

        storage.save(sample_tokens)
        all_tokens = storage.get_all_tokens()
        assert "google" in all_tokens
        assert all_tokens["google"].email == sample_tokens.email

    def test_has_valid_token_expired(self, storage):
        """Test has_valid_token with expired token."""
        expired_tokens = OAuthTokens(
            provider="google",
            access_token="expired_token",
            refresh_token="refresh",
            expires_at=int(time.time() * 1000) - 1000,  # Expired
            email="test@example.com",
        )
        storage.save(expired_tokens)

        assert storage.has_valid_token("google") is False

    def test_has_valid_token_valid(self, storage, sample_tokens):
        """Test has_valid_token with valid token."""
        storage.save(sample_tokens)
        assert storage.has_valid_token("google") is True

    def test_update_access_token(self, storage, sample_tokens):
        """Test updating access token."""
        storage.save(sample_tokens)

        new_expires = int(time.time() * 1000) + 7200000
        result = storage.update_access_token("google", "new_access_token", new_expires)
        assert result is True

        loaded = storage.load("google")
        assert loaded.access_token == "new_access_token"
        assert loaded.expires_at == new_expires

    def test_update_access_token_nonexistent(self, storage):
        """Test updating token for nonexistent provider."""
        result = storage.update_access_token("nonexistent", "token", int(time.time() * 1000))
        assert result is False

    def test_file_permissions(self, storage, sample_tokens, temp_storage_dir):
        """Test that token files have restricted permissions."""
        storage.save(sample_tokens)

        token_path = temp_storage_dir / "google.json"
        assert token_path.exists()

        # Check file permissions (0600 = owner read/write only)
        mode = os.stat(token_path).st_mode
        assert mode & stat.S_IRWXU == stat.S_IRUSR | stat.S_IWUSR
        assert mode & stat.S_IRWXG == 0
        assert mode & stat.S_IRWXO == 0

    def test_malformed_json(self, storage, temp_storage_dir):
        """Test handling of malformed JSON in token file."""
        token_path = temp_storage_dir / "google.json"
        token_path.write_text("not valid json {")

        loaded = storage.load("google")
        assert loaded is None


class TestOAuthTokens:
    """Tests for OAuthTokens dataclass."""

    def test_is_expired_false(self, sample_tokens):
        """Test is_expired returns False for valid token."""
        assert sample_tokens.is_expired() is False

    def test_is_expired_true(self):
        """Test is_expired returns True for expired token."""
        expired = OAuthTokens(
            provider="test",
            access_token="token",
            refresh_token="refresh",
            expires_at=int(time.time() * 1000) - 1000,
        )
        assert expired.is_expired() is True

    def test_is_expired_with_buffer(self):
        """Test is_expired considers buffer time."""
        # Token expires in 30 seconds
        almost_expired = OAuthTokens(
            provider="test",
            access_token="token",
            refresh_token="refresh",
            expires_at=int(time.time() * 1000) + 30000,
        )
        # With 60 second buffer, should be considered expired
        assert almost_expired.is_expired(buffer_ms=60000) is True
        # With 10 second buffer, should be valid
        assert almost_expired.is_expired(buffer_ms=10000) is False

    def test_to_dict(self, sample_tokens):
        """Test converting tokens to dictionary."""
        d = sample_tokens.to_dict()
        assert d["provider"] == "google"
        assert d["access_token"] == sample_tokens.access_token
        assert d["refresh_token"] == sample_tokens.refresh_token
        assert d["email"] == sample_tokens.email

    def test_from_dict(self):
        """Test creating tokens from dictionary."""
        data = {
            "provider": "claude",
            "access_token": "access",
            "refresh_token": "refresh",
            "expires_at": 1234567890000,
            "email": "user@example.com",
            "extra": {"mode": "max"},
        }
        tokens = OAuthTokens.from_dict(data)
        assert tokens.provider == "claude"
        assert tokens.access_token == "access"
        assert tokens.email == "user@example.com"
        assert tokens.extra["mode"] == "max"
