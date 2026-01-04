"""Manual integration test for Google OAuth.

This test requires user interaction to complete the OAuth flow.
Run with: pytest tests/integration/test_google_oauth.py -m manual --manual -s
"""

import pytest

from koder_agent.auth.callback_server import run_oauth_flow
from koder_agent.auth.providers.google import GoogleOAuthProvider
from koder_agent.auth.token_storage import TokenStorage


@pytest.mark.manual
@pytest.mark.asyncio
async def test_google_oauth_flow():
    """Test full Google OAuth flow with user interaction.

    This test will:
    1. Generate authorization URL
    2. Open browser for user to login
    3. Wait for OAuth callback
    4. Exchange code for tokens
    5. Verify tokens are valid
    """
    provider = GoogleOAuthProvider()

    print("\n" + "=" * 60)
    print("Google OAuth Integration Test")
    print("=" * 60)
    print("This test requires you to complete login in the browser.")
    print("The browser should open automatically.\n")

    result = await run_oauth_flow(provider, timeout=300)

    assert result.success, f"OAuth flow failed: {result.error}"
    assert result.tokens is not None, "No tokens returned"
    assert result.tokens.access_token, "No access token"
    assert result.tokens.refresh_token, "No refresh token"
    assert result.tokens.email, "No email in tokens"

    print(f"\n✓ Successfully authenticated as: {result.tokens.email}")
    print(f"✓ Access token obtained (length: {len(result.tokens.access_token)})")
    print(f"✓ Refresh token obtained (length: {len(result.tokens.refresh_token)})")


@pytest.mark.manual
@pytest.mark.asyncio
async def test_google_token_refresh():
    """Test token refresh for Google OAuth.

    Requires existing tokens from a previous login.
    """
    storage = TokenStorage()
    tokens = storage.load("google")

    if tokens is None:
        pytest.skip("No existing Google tokens - run login test first")

    provider = GoogleOAuthProvider()
    result = await provider.refresh_tokens(tokens.refresh_token)

    assert result.success, f"Token refresh failed: {result.error}"
    assert result.tokens is not None
    assert result.tokens.access_token != tokens.access_token, "Access token unchanged"

    print("\n✓ Token refresh successful")
    print("✓ New access token obtained")


@pytest.mark.manual
def test_google_stored_tokens():
    """Verify Google tokens are properly stored."""
    storage = TokenStorage()
    tokens = storage.load("google")

    if tokens is None:
        pytest.skip("No Google tokens stored")

    assert tokens.provider == "google"
    assert tokens.access_token
    assert tokens.refresh_token

    print(f"\n✓ Google tokens found for: {tokens.email}")
    print(f"✓ Token expires at: {tokens.expires_at}")
    print(f"✓ Token expired: {tokens.is_expired()}")
