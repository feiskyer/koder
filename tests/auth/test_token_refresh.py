"""Unit tests for OAuth token refresh logic."""

import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from koder_agent.auth.base import OAuthResult, OAuthTokens
from koder_agent.auth.client_integration import (
    _sync_refresh_token,
    async_refresh_token,
    get_oauth_api_key,
    get_oauth_token,
    get_provider_auth_info,
    has_oauth_token,
    map_provider_to_oauth,
)
from koder_agent.auth.constants import TOKEN_EXPIRY_BUFFER_MS

LOGGER_NAME = "koder_agent.auth.client_integration"
SYNTHETIC_SECRET_CANARY = "synthetic-secret-canary"
SYNTHETIC_ERROR_CANARY = "synthetic-error-canary"
SYNTHETIC_PATH_CANARY = "/synthetic/credential/path-canary.json"
SYNTHETIC_BODY_CANARY = '{"error":"synthetic-body-canary"}'
SYNTHETIC_TOKEN_CANARY = "synthetic-token-canary"
SENSITIVE_CANARIES = (
    SYNTHETIC_SECRET_CANARY,
    SYNTHETIC_ERROR_CANARY,
    SYNTHETIC_PATH_CANARY,
    SYNTHETIC_BODY_CANARY,
    SYNTHETIC_TOKEN_CANARY,
)


def _synthetic_failure_detail() -> str:
    return " ".join(SENSITIVE_CANARIES)


def _assert_sanitized_refresh_log(caplog, expected_message: str) -> None:
    records = [record for record in caplog.records if record.name == LOGGER_NAME]
    assert len(records) == 1
    assert records[0].getMessage() == expected_message
    assert records[0].exc_info is None
    for canary in SENSITIVE_CANARIES:
        assert canary not in caplog.text
        assert canary not in records[0].getMessage()


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


@pytest.fixture
def sensitive_expired_tokens():
    """Create expired tokens containing synthetic secret canaries."""
    return OAuthTokens(
        provider="google",
        access_token=SYNTHETIC_TOKEN_CANARY,
        refresh_token=SYNTHETIC_SECRET_CANARY,
        expires_at=int(time.time() * 1000) - 1000,
        email="synthetic@example.com",
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

    def test_valid_token_does_not_refresh(self, valid_tokens):
        """Test that a current token bypasses synchronous refresh."""
        with patch("koder_agent.auth.client_integration.get_token_storage") as mock_storage:
            mock_storage.return_value.load.return_value = valid_tokens

            with patch("koder_agent.auth.client_integration._sync_refresh_token") as mock_refresh:
                result = get_oauth_token("google")

                assert result is valid_tokens
                mock_refresh.assert_not_called()

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

                mock_refresh.assert_called_once_with(
                    "google", expired_tokens, storage=mock_storage.return_value
                )
                assert result.access_token == "new_access_token"

    def test_returns_none_on_refresh_failure(self, expired_tokens):
        """Test that None is returned if refresh fails."""
        with patch("koder_agent.auth.client_integration.get_token_storage") as mock_storage:
            mock_storage.return_value.load.return_value = expired_tokens

            with patch("koder_agent.auth.client_integration._sync_refresh_token") as mock_refresh:
                mock_refresh.return_value = None

                result = get_oauth_token("google")

                assert result is None


class TestSyncRefreshToken:
    """Tests for _sync_refresh_token function."""

    def test_unsuccessful_result_logs_only_normalized_provider_and_category(
        self, sensitive_expired_tokens, caplog
    ):
        """Test unsuccessful sync refresh logs no provider-supplied detail."""
        refresh_result = OAuthResult(success=False, error=_synthetic_failure_detail())
        mock_provider = MagicMock()
        mock_provider.refresh_tokens = AsyncMock(return_value=refresh_result)
        storage = MagicMock()
        storage.load.return_value = sensitive_expired_tokens

        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            with patch("koder_agent.auth.providers.get_provider", return_value=mock_provider):
                result = _sync_refresh_token(
                    "  GoOgLe  ", sensitive_expired_tokens, storage=storage
                )

        assert result is None
        mock_provider.refresh_tokens.assert_awaited_once_with(SYNTHETIC_SECRET_CANARY)
        storage.save_if_current.assert_not_called()
        _assert_sanitized_refresh_log(
            caplog,
            "OAuth token refresh failed provider=google category=refresh_rejected",
        )

    def test_exception_logs_only_normalized_provider_category_and_class(
        self, sensitive_expired_tokens, caplog
    ):
        """Test sync refresh exceptions omit exception text and traceback data."""
        mock_provider = MagicMock()
        mock_provider.refresh_tokens = AsyncMock(
            side_effect=RuntimeError(_synthetic_failure_detail())
        )
        storage = MagicMock()
        storage.load.return_value = sensitive_expired_tokens

        with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
            with patch("koder_agent.auth.providers.get_provider", return_value=mock_provider):
                result = _sync_refresh_token(
                    "  GoOgLe  ", sensitive_expired_tokens, storage=storage
                )

        assert result is None
        mock_provider.refresh_tokens.assert_awaited_once_with(SYNTHETIC_SECRET_CANARY)
        storage.save_if_current.assert_not_called()
        _assert_sanitized_refresh_log(
            caplog,
            "OAuth token refresh failed provider=google category=exception "
            "exception_type=RuntimeError",
        )


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
                mock_storage.return_value.load.return_value = expired_tokens

                result = await async_refresh_token("google", expired_tokens)

                assert result is not None
                assert result.access_token == "refreshed_token"
                mock_storage.return_value.save_if_current.assert_called_once_with(
                    expired_tokens, refreshed
                )
                mock_storage.return_value.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_unsuccessful_result_logs_only_normalized_provider_and_category(
        self, sensitive_expired_tokens, caplog
    ):
        """Test unsuccessful async refresh logs no provider-supplied detail."""
        refresh_result = OAuthResult(success=False, error=_synthetic_failure_detail())

        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            with patch("koder_agent.auth.providers.get_provider") as mock_get:
                mock_provider = AsyncMock()
                mock_provider.refresh_tokens.return_value = refresh_result
                mock_get.return_value = mock_provider

                with patch("koder_agent.auth.client_integration.get_token_storage") as mock_storage:
                    mock_storage.return_value.load.return_value = sensitive_expired_tokens
                    result = await async_refresh_token("  GoOgLe  ", sensitive_expired_tokens)

        assert result is None
        mock_provider.refresh_tokens.assert_awaited_once_with(SYNTHETIC_SECRET_CANARY)
        mock_storage.return_value.save_if_current.assert_not_called()
        _assert_sanitized_refresh_log(
            caplog,
            "OAuth token refresh failed provider=google category=refresh_rejected",
        )

    @pytest.mark.asyncio
    async def test_exception_logs_only_normalized_provider_category_and_class(
        self, sensitive_expired_tokens, caplog
    ):
        """Test async refresh exceptions omit exception text and traceback data."""
        with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
            with patch("koder_agent.auth.providers.get_provider") as mock_get:
                mock_provider = AsyncMock()
                mock_provider.refresh_tokens.side_effect = RuntimeError(_synthetic_failure_detail())
                mock_get.return_value = mock_provider

                with patch("koder_agent.auth.client_integration.get_token_storage") as mock_storage:
                    mock_storage.return_value.load.return_value = sensitive_expired_tokens
                    result = await async_refresh_token("  GoOgLe  ", sensitive_expired_tokens)

        assert result is None
        mock_provider.refresh_tokens.assert_awaited_once_with(SYNTHETIC_SECRET_CANARY)
        mock_storage.return_value.save_if_current.assert_not_called()
        _assert_sanitized_refresh_log(
            caplog,
            "OAuth token refresh failed provider=google category=exception "
            "exception_type=RuntimeError",
        )


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
