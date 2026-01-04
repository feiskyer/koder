"""Manual integration test for Antigravity OAuth.

Antigravity uses Google OAuth to access Gemini 3 + Claude models
through Antigravity's quota system.

This test requires user interaction to complete the OAuth flow.
Run with: pytest tests/integration/test_antigravity_oauth.py -m manual --manual -s
"""

import pytest

from koder_agent.auth.callback_server import run_oauth_flow
from koder_agent.auth.providers.antigravity import AntigravityOAuthProvider
from koder_agent.auth.token_storage import TokenStorage


@pytest.mark.manual
@pytest.mark.asyncio
async def test_antigravity_oauth_flow():
    """Test full Antigravity OAuth flow with user interaction.

    This test will:
    1. Generate Google OAuth authorization URL
    2. Open browser for user to login with Google
    3. Wait for OAuth callback
    4. Exchange code for tokens
    5. Verify tokens are valid for Antigravity API

    Note: Antigravity uses Google OAuth to authenticate, but the
    tokens are used to access Gemini 3 and Claude models through
    Antigravity's quota system.
    """
    provider = AntigravityOAuthProvider()

    print("\n" + "=" * 60)
    print("Antigravity OAuth Integration Test")
    print("=" * 60)
    print("This test requires you to complete Google login in the browser.")
    print("Antigravity provides access to Gemini 3 + Claude via Google OAuth.")
    print("The browser should open automatically.\n")

    result = await run_oauth_flow(provider, timeout=300)

    assert result.success, f"OAuth flow failed: {result.error}"
    assert result.tokens is not None, "No tokens returned"
    assert result.tokens.access_token, "No access token"
    assert result.tokens.refresh_token, "No refresh token"

    print("\n✓ Successfully authenticated")
    if result.tokens.email:
        print(f"✓ Email: {result.tokens.email}")
    print(f"✓ Access token obtained (length: {len(result.tokens.access_token)})")
    print("✓ Refresh token obtained")
    print("✓ Ready to use Antigravity API")


@pytest.mark.manual
@pytest.mark.asyncio
async def test_antigravity_token_refresh():
    """Test token refresh for Antigravity OAuth."""
    storage = TokenStorage()
    tokens = storage.load("antigravity")

    if tokens is None:
        pytest.skip("No existing Antigravity tokens - run login test first")

    provider = AntigravityOAuthProvider()
    result = await provider.refresh_tokens(tokens.refresh_token)

    assert result.success, f"Token refresh failed: {result.error}"
    assert result.tokens is not None
    assert result.tokens.access_token != tokens.access_token

    print("\n✓ Token refresh successful")
    print("✓ New access token obtained")


@pytest.mark.manual
def test_antigravity_stored_tokens():
    """Verify Antigravity tokens are properly stored."""
    storage = TokenStorage()
    tokens = storage.load("antigravity")

    if tokens is None:
        pytest.skip("No Antigravity tokens stored")

    assert tokens.provider == "antigravity"
    assert tokens.access_token
    assert tokens.refresh_token

    print("\n✓ Antigravity tokens found")
    if tokens.email:
        print(f"✓ Email: {tokens.email}")
    print(f"✓ Token expires at: {tokens.expires_at}")
    print(f"✓ Token expired: {tokens.is_expired()}")
