"""Unit tests for OAuth token refresh logic."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from koder_agent.auth.base import OAuthResult, OAuthTokens
from koder_agent.auth.client_integration import (
    async_refresh_token,
    get_oauth_api_key,
    get_oauth_token,
    get_provider_auth_info,
    has_oauth_token,
    map_provider_to_oauth,
)
from koder_agent.auth.constants import TOKEN_EXPIRY_BUFFER_MS


@pytest.fixture
def valid_tokens():
    """Create valid (non-expired) tokens."""
    return OAuthTokens(
        provider="google",
        access_token="valid_access_token",
        refresh_token="valid_refresh_token",
        expires_at=int(time.time() * 1000) + 3600000,  # 1 hour from now
        email="user@example.com",
    )


@pytest.fixture
def expired_tokens():
    """Create expired tokens."""
    return OAuthTokens(
        provider="google",
        access_token="expired_access_token",
        refresh_token="valid_refresh_token",
        expires_at=int(time.time() * 1000) - 1000,  # Already expired
        email="user@example.com",
    )


@pytest.fixture
def almost_expired_tokens():
    """Create tokens that will expire within buffer time."""
    return OAuthTokens(
        provider="google",
        access_token="almost_expired_access_token",
        refresh_token="valid_refresh_token",
        expires_at=int(time.time() * 1000) + 30000,  # 30 seconds from now
        email="user@example.com",
    )


class TestTokenExpiry:
    """Tests for token expiry checking."""

    def test_valid_token_not_expired(self, valid_tokens):
        """Test that valid tokens are not expired."""
        assert valid_tokens.is_expired() is False

    def test_expired_token_is_expired(self, expired_tokens):
        """Test that expired tokens are marked as expired."""
        assert expired_tokens.is_expired() is True

    def test_almost_expired_with_buffer(self, almost_expired_tokens):
        """Test expiry check with buffer time."""
        # With default buffer (60s), should be considered expired
        assert almost_expired_tokens.is_expired(buffer_ms=TOKEN_EXPIRY_BUFFER_MS) is True

        # Without buffer, should not be expired
        assert almost_expired_tokens.is_expired(buffer_ms=0) is False

    def test_expiry_buffer_boundary(self):
        """Test expiry at exact buffer boundary."""
        boundary_tokens = OAuthTokens(
            provider="test",
            access_token="token",
            refresh_token="refresh",
            expires_at=int(time.time() * 1000) + TOKEN_EXPIRY_BUFFER_MS,
        )
        # At exact boundary, should be considered expired (using <=)
        assert boundary_tokens.is_expired(buffer_ms=TOKEN_EXPIRY_BUFFER_MS) is True


class TestGetOAuthToken:
    """Tests for get_oauth_token function."""

    def test_returns_valid_token(self, valid_tokens):
        """Test that valid tokens are returned directly."""
        with patch("koder_agent.auth.client_integration.get_token_storage") as mock_storage:
            mock_storage.return_value.load.return_value = valid_tokens

            result = get_oauth_token("google")

            assert result is not None
            assert result.access_token == valid_tokens.access_token

    def test_returns_none_for_nonexistent(self):
        """Test that None is returned for nonexistent provider."""
        with patch("koder_agent.auth.client_integration.get_token_storage") as mock_storage:
            mock_storage.return_value.load.return_value = None

            result = get_oauth_token("nonexistent")

            assert result is None

    def test_refreshes_expired_token(self, expired_tokens):
        """Test that expired tokens trigger refresh."""
        refreshed = OAuthTokens(
            provider="google",
            access_token="new_access_token",
            refresh_token="new_refresh_token",
            expires_at=int(time.time() * 1000) + 3600000,
            email="user@example.com",
        )

        with patch("koder_agent.auth.client_integration.get_token_storage") as mock_storage:
            mock_storage.return_value.load.return_value = expired_tokens

            with patch("koder_agent.auth.client_integration._sync_refresh_token") as mock_refresh:
                mock_refresh.return_value = refreshed

                result = get_oauth_token("google")

                mock_refresh.assert_called_once_with("google", expired_tokens)
                assert result.access_token == "new_access_token"

    def test_returns_none_on_refresh_failure(self, expired_tokens):
        """Test that None is returned if refresh fails."""
        with patch("koder_agent.auth.client_integration.get_token_storage") as mock_storage:
            mock_storage.return_value.load.return_value = expired_tokens

            with patch("koder_agent.auth.client_integration._sync_refresh_token") as mock_refresh:
                mock_refresh.return_value = None

                result = get_oauth_token("google")

                assert result is None


class TestAsyncRefreshToken:
    """Tests for async_refresh_token function."""

    @pytest.mark.asyncio
    async def test_successful_refresh(self, expired_tokens):
        """Test successful token refresh."""
        refreshed = OAuthTokens(
            provider="google",
            access_token="refreshed_token",
            refresh_token="new_refresh",
            expires_at=int(time.time() * 1000) + 3600000,
        )
        refresh_result = OAuthResult(success=True, tokens=refreshed)

        with patch("koder_agent.auth.providers.get_provider") as mock_get:
            mock_provider = AsyncMock()
            mock_provider.refresh_tokens.return_value = refresh_result
            mock_get.return_value = mock_provider

            with patch("koder_agent.auth.client_integration.get_token_storage") as mock_storage:
                mock_storage.return_value.save = MagicMock()

                result = await async_refresh_token("google", expired_tokens)

                assert result is not None
                assert result.access_token == "refreshed_token"
                mock_storage.return_value.save.assert_called_once_with(refreshed)

    @pytest.mark.asyncio
    async def test_failed_refresh(self, expired_tokens):
        """Test failed token refresh."""
        refresh_result = OAuthResult(success=False, error="Invalid refresh token")

        with patch("koder_agent.auth.providers.get_provider") as mock_get:
            mock_provider = AsyncMock()
            mock_provider.refresh_tokens.return_value = refresh_result
            mock_get.return_value = mock_provider

            result = await async_refresh_token("google", expired_tokens)

            assert result is None

    @pytest.mark.asyncio
    async def test_exception_during_refresh(self, expired_tokens):
        """Test exception handling during refresh."""
        with patch("koder_agent.auth.providers.get_provider") as mock_get:
            mock_get.side_effect = Exception("Network error")

            result = await async_refresh_token("google", expired_tokens)

            assert result is None


class TestGetOAuthApiKey:
    """Tests for get_oauth_api_key function."""

    def test_returns_access_token(self, valid_tokens):
        """Test that access token is returned as API key."""
        with patch("koder_agent.auth.client_integration.get_oauth_token") as mock_get:
            mock_get.return_value = valid_tokens

            result = get_oauth_api_key("google")

            assert result == valid_tokens.access_token

    def test_returns_none_for_no_token(self):
        """Test that None is returned when no token exists."""
        with patch("koder_agent.auth.client_integration.get_oauth_token") as mock_get:
            mock_get.return_value = None

            result = get_oauth_api_key("google")

            assert result is None


class TestHasOAuthToken:
    """Tests for has_oauth_token function."""

    def test_returns_true_for_valid_token(self):
        """Test returns True when valid token exists."""
        with patch("koder_agent.auth.client_integration.get_token_storage") as mock_storage:
            mock_storage.return_value.has_valid_token.return_value = True

            result = has_oauth_token("google")

            assert result is True

    def test_returns_false_for_no_token(self):
        """Test returns False when no token exists."""
        with patch("koder_agent.auth.client_integration.get_token_storage") as mock_storage:
            mock_storage.return_value.has_valid_token.return_value = False

            result = has_oauth_token("nonexistent")

            assert result is False


class TestGetProviderAuthInfo:
    """Tests for get_provider_auth_info function."""

    def test_returns_oauth_info_when_available(self, valid_tokens):
        """Test returns OAuth info when tokens exist."""
        with patch("koder_agent.auth.client_integration.get_oauth_token") as mock_get:
            mock_get.return_value = valid_tokens

            with patch("koder_agent.auth.providers.get_provider") as mock_provider:
                mock_provider.return_value.get_auth_headers.return_value = {
                    "Authorization": f"Bearer {valid_tokens.access_token}"
                }

                api_key, headers, is_oauth = get_provider_auth_info("google")

                assert api_key == valid_tokens.access_token
                assert headers is not None
                assert is_oauth is True

    def test_returns_none_when_no_oauth(self):
        """Test returns None when no OAuth tokens exist."""
        with patch("koder_agent.auth.client_integration.get_oauth_token") as mock_get:
            mock_get.return_value = None

            api_key, headers, is_oauth = get_provider_auth_info("google")

            assert api_key is None
            assert headers is None
            assert is_oauth is False


class TestMapProviderToOAuth:
    """Tests for map_provider_to_oauth function."""

    def test_oauth_providers_map_to_themselves(self):
        """OAuth providers should map to themselves."""
        assert map_provider_to_oauth("google") == "google"
        assert map_provider_to_oauth("claude") == "claude"
        assert map_provider_to_oauth("chatgpt") == "chatgpt"
        assert map_provider_to_oauth("antigravity") == "antigravity"

    def test_api_providers_return_none(self):
        """API-based providers should NOT map to OAuth."""
        assert map_provider_to_oauth("anthropic") is None
        assert map_provider_to_oauth("openai") is None
        assert map_provider_to_oauth("gemini") is None
        assert map_provider_to_oauth("azure") is None
        assert map_provider_to_oauth("unknown") is None

    def test_case_insensitive(self):
        """Provider names should be case insensitive."""
        assert map_provider_to_oauth("GOOGLE") == "google"
        assert map_provider_to_oauth("Claude") == "claude"
        assert map_provider_to_oauth("ChatGPT") == "chatgpt"
        assert map_provider_to_oauth("ANTHROPIC") is None
        assert map_provider_to_oauth("OpenAI") is None

    def test_whitespace_handling(self):
        """Leading/trailing whitespace should be stripped."""
        assert map_provider_to_oauth("  google  ") == "google"
        assert map_provider_to_oauth(" claude") == "claude"
        assert map_provider_to_oauth("chatgpt ") == "chatgpt"
        assert map_provider_to_oauth("  anthropic  ") is None

    def test_invalid_inputs(self):
        """None and empty string should return None."""
        assert map_provider_to_oauth(None) is None
        assert map_provider_to_oauth("") is None
        assert map_provider_to_oauth("   ") is None
