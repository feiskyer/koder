"""Unit tests for OAuth providers.

OAuth providers use different names than API key providers to avoid conflicts:
- google (OAuth) → Gemini CLI subscription
- claude (OAuth) → Claude Max subscription
- chatgpt (OAuth) → ChatGPT Plus/Pro subscription
- antigravity (OAuth) → Antigravity (Gemini 3 + Claude)
"""

import base64
import hashlib
from unittest.mock import patch

import pytest

from koder_agent.auth import constants
from koder_agent.auth.base import PKCEPair
from koder_agent.auth.providers.antigravity import AntigravityOAuthProvider
from koder_agent.auth.providers.chatgpt import ChatGPTOAuthProvider
from koder_agent.auth.providers.claude import ClaudeOAuthProvider
from koder_agent.auth.providers.google import GoogleOAuthProvider


class MockResponse:
    """Mock aiohttp response."""

    def __init__(self, status, json_data=None, text_data=None):
        self.status = status
        self.ok = status < 400
        self._json_data = json_data
        self._text_data = text_data or ""

    async def json(self):
        return self._json_data

    async def text(self):
        return self._text_data


class MockSession:
    """Mock aiohttp ClientSession with async context manager support."""

    def __init__(self, post_response=None, get_response=None):
        self._post_response = post_response
        self._get_response = get_response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def post(self, *args, **kwargs):
        return AsyncContextManager(self._post_response)

    def get(self, *args, **kwargs):
        return AsyncContextManager(self._get_response)


class AsyncContextManager:
    """Async context manager wrapper for mock responses."""

    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        pass


class TestPKCEGeneration:
    """Tests for PKCE code verifier and challenge generation."""

    def test_generate_pkce_verifier_length(self):
        """Test that PKCE verifier has correct length."""
        provider = GoogleOAuthProvider()
        pkce = provider.generate_pkce()

        # Verifier should be 43 characters (base64url of 32 bytes)
        assert len(pkce.verifier) >= 43
        assert len(pkce.verifier) <= 128

    def test_generate_pkce_verifier_charset(self):
        """Test that PKCE verifier uses valid characters."""
        provider = GoogleOAuthProvider()
        pkce = provider.generate_pkce()

        # Valid base64url characters (no padding)
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")
        assert all(c in valid_chars for c in pkce.verifier)

    def test_generate_pkce_challenge_derivation(self):
        """Test that challenge is correctly derived from verifier."""
        provider = GoogleOAuthProvider()
        pkce = provider.generate_pkce()

        # Manually compute expected challenge
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(pkce.verifier.encode("utf-8")).digest())
            .decode("utf-8")
            .rstrip("=")
        )

        assert pkce.challenge == expected

    def test_generate_pkce_uniqueness(self):
        """Test that each PKCE generation is unique."""
        provider = GoogleOAuthProvider()
        pairs = [provider.generate_pkce() for _ in range(10)]
        verifiers = [p.verifier for p in pairs]
        challenges = [p.challenge for p in pairs]

        assert len(set(verifiers)) == 10
        assert len(set(challenges)) == 10

    def test_pkce_pair_dataclass(self):
        """Test PKCEPair dataclass attributes."""
        pkce = PKCEPair(verifier="test_verifier", challenge="test_challenge")
        assert pkce.verifier == "test_verifier"
        assert pkce.challenge == "test_challenge"


class TestGoogleOAuthProvider:
    """Tests for Google OAuth provider."""

    def test_provider_id(self):
        """Test provider ID is correct."""
        provider = GoogleOAuthProvider()
        assert provider.provider_id == "google"

    def test_get_authorization_url(self):
        """Test authorization URL generation."""
        provider = GoogleOAuthProvider()
        url, verifier = provider.get_authorization_url()

        assert constants.GOOGLE_AUTH_URL in url
        assert f"client_id={constants.GOOGLE_CLIENT_ID}" in url
        assert "redirect_uri=" in url
        assert "response_type=code" in url
        assert "code_challenge=" in url
        assert "code_challenge_method=S256" in url
        assert "state=" in url
        # Check scopes are present
        assert "scope=" in url
        # Verifier should be returned
        assert len(verifier) >= 43

    def test_get_auth_headers(self):
        """Test auth headers generation."""
        provider = GoogleOAuthProvider()
        headers = provider.get_auth_headers("test_token")

        assert headers["Authorization"] == "Bearer test_token"

    @pytest.mark.asyncio
    async def test_exchange_code_success(self):
        """Test successful code exchange."""
        provider = GoogleOAuthProvider()

        mock_token_response = {
            "access_token": "google_access_token",
            "refresh_token": "google_refresh_token",
            "expires_in": 3600,
            "token_type": "Bearer",
        }

        mock_userinfo_response = {
            "email": "user@gmail.com",
        }

        token_resp = MockResponse(200, json_data=mock_token_response)
        userinfo_resp = MockResponse(200, json_data=mock_userinfo_response)
        mock_session = MockSession(post_response=token_resp, get_response=userinfo_resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await provider.exchange_code("auth_code", "verifier")

            assert result.success is True
            assert result.tokens is not None
            assert result.tokens.access_token == "google_access_token"
            assert result.tokens.refresh_token == "google_refresh_token"
            assert result.tokens.email == "user@gmail.com"

    @pytest.mark.asyncio
    async def test_exchange_code_failure(self):
        """Test code exchange failure."""
        provider = GoogleOAuthProvider()

        error_resp = MockResponse(400, text_data="invalid_grant")
        mock_session = MockSession(post_response=error_resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await provider.exchange_code("bad_code", "verifier")

            assert result.success is False
            assert result.error is not None


class TestClaudeOAuthProvider:
    """Tests for Claude OAuth provider."""

    def test_provider_id(self):
        """Test provider ID is correct."""
        provider = ClaudeOAuthProvider()
        assert provider.provider_id == "claude"

    def test_get_authorization_url_default_mode(self):
        """Test authorization URL with default (max) mode."""
        provider = ClaudeOAuthProvider()
        url, verifier = provider.get_authorization_url()

        assert constants.ANTHROPIC_AUTH_URL_MAX in url
        assert f"client_id={constants.ANTHROPIC_CLIENT_ID}" in url
        # Verifier should be returned
        assert len(verifier) >= 43

    def test_get_authorization_url_console_mode(self):
        """Test authorization URL with console mode."""
        provider = ClaudeOAuthProvider(mode="console")
        url, verifier = provider.get_authorization_url()

        # URL should be generated
        assert "https://" in url

    def test_get_auth_headers(self):
        """Test auth headers include proper authorization."""
        provider = ClaudeOAuthProvider()
        headers = provider.get_auth_headers("test_token")

        # Should have Authorization header
        assert "Authorization" in headers or "x-api-key" in headers

    @pytest.mark.asyncio
    async def test_exchange_code_success(self):
        """Test successful Claude code exchange."""
        provider = ClaudeOAuthProvider()

        mock_response = {
            "access_token": "claude_access_token",
            "refresh_token": "claude_refresh_token",
            "expires_in": 86400,
        }

        token_resp = MockResponse(200, json_data=mock_response)
        mock_session = MockSession(post_response=token_resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await provider.exchange_code("auth_code", "verifier")

            assert result.success is True
            assert result.tokens is not None
            assert result.tokens.access_token == "claude_access_token"


class TestChatGPTOAuthProvider:
    """Tests for ChatGPT OAuth provider."""

    def test_provider_id(self):
        """Test provider ID is correct."""
        provider = ChatGPTOAuthProvider()
        assert provider.provider_id == "chatgpt"

    def test_get_authorization_url(self):
        """Test authorization URL generation."""
        provider = ChatGPTOAuthProvider()
        url, verifier = provider.get_authorization_url()

        assert constants.OPENAI_AUTH_URL in url
        assert f"client_id={constants.OPENAI_CLIENT_ID}" in url
        # Verifier should be returned
        assert len(verifier) >= 43

    def test_get_auth_headers(self):
        """Test auth headers generation."""
        provider = ChatGPTOAuthProvider()
        headers = provider.get_auth_headers("test_token")

        assert headers["Authorization"] == "Bearer test_token"

    @pytest.mark.asyncio
    async def test_exchange_code_success(self):
        """Test successful ChatGPT code exchange."""
        provider = ChatGPTOAuthProvider()

        # OpenAI returns JWT access token
        mock_response = {
            "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InVzZXJAZXhhbXBsZS5jb20ifQ.signature",
            "refresh_token": "chatgpt_refresh_token",
            "expires_in": 3600,
        }

        token_resp = MockResponse(200, json_data=mock_response)
        mock_session = MockSession(post_response=token_resp)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await provider.exchange_code("auth_code", "verifier")

            assert result.success is True
            assert result.tokens is not None
            assert result.tokens.provider == "chatgpt"


class TestAntigravityOAuthProvider:
    """Tests for Antigravity OAuth provider."""

    def test_provider_id(self):
        """Test provider ID is correct."""
        provider = AntigravityOAuthProvider()
        assert provider.provider_id == "antigravity"

    def test_get_authorization_url(self):
        """Test authorization URL uses Google OAuth."""
        provider = AntigravityOAuthProvider()
        url, verifier = provider.get_authorization_url()

        # Antigravity uses Google OAuth
        assert "accounts.google.com" in url or "googleapis.com" in url
        assert len(verifier) >= 43

    def test_get_auth_headers(self):
        """Test auth headers generation."""
        provider = AntigravityOAuthProvider()
        headers = provider.get_auth_headers("test_token")

        assert "Authorization" in headers


class TestProviderFactory:
    """Tests for provider factory function."""

    def test_get_provider_google(self):
        """Test getting Google provider."""
        from koder_agent.auth.providers import get_provider

        provider = get_provider("google")
        assert isinstance(provider, GoogleOAuthProvider)

    def test_get_provider_claude(self):
        """Test getting Claude provider."""
        from koder_agent.auth.providers import get_provider

        provider = get_provider("claude")
        assert isinstance(provider, ClaudeOAuthProvider)

    def test_get_provider_chatgpt(self):
        """Test getting ChatGPT provider."""
        from koder_agent.auth.providers import get_provider

        provider = get_provider("chatgpt")
        assert isinstance(provider, ChatGPTOAuthProvider)

    def test_get_provider_antigravity(self):
        """Test getting Antigravity provider."""
        from koder_agent.auth.providers import get_provider

        provider = get_provider("antigravity")
        assert isinstance(provider, AntigravityOAuthProvider)

    def test_get_provider_unknown(self):
        """Test getting unknown provider raises error."""
        from koder_agent.auth.providers import get_provider

        with pytest.raises(ValueError, match="Unsupported provider"):
            get_provider("unknown_provider")

    def test_list_providers(self):
        """Test listing available providers."""
        from koder_agent.auth.providers import list_providers

        providers = list_providers()
        assert "google" in providers
        assert "claude" in providers
        assert "chatgpt" in providers
        assert "antigravity" in providers
