"""Manual integration test for Claude OAuth.

This test requires user interaction to complete the OAuth flow.
Run with: pytest tests/integration/test_claude_oauth.py -m manual --manual -s
"""

import pytest

from koder_agent.auth.callback_server import run_oauth_flow
from koder_agent.auth.providers.claude import ClaudeOAuthProvider
from koder_agent.auth.token_storage import TokenStorage


@pytest.mark.manual
@pytest.mark.asyncio
async def test_claude_oauth_flow_max_mode():
    """Test full Claude OAuth flow with Max mode.

    This test will:
    1. Generate authorization URL for Claude Max
    2. Open browser for user to login
    3. Wait for OAuth callback
    4. Exchange code for tokens
    5. Verify tokens are valid
    """
    provider = ClaudeOAuthProvider(mode="max")

    print("\n" + "=" * 60)
    print("Claude OAuth Integration Test (Max Mode)")
    print("=" * 60)
    print("This test requires you to complete login in the browser.")
    print("You need a Claude Max subscription for this mode.")
    print("The browser should open automatically.\n")

    result = await run_oauth_flow(provider, timeout=300)

    assert result.success, f"OAuth flow failed: {result.error}"
    assert result.tokens is not None, "No tokens returned"
    assert result.tokens.access_token, "No access token"

    print("\n✓ Successfully authenticated")
    print(f"✓ Access token obtained (length: {len(result.tokens.access_token)})")
    if result.tokens.refresh_token:
        print("✓ Refresh token obtained")


@pytest.mark.manual
@pytest.mark.asyncio
async def test_claude_oauth_flow_console_mode():
    """Test Claude OAuth flow with Console mode (API key creation).

    This test will authenticate and create an API key via OAuth.
    """
    provider = ClaudeOAuthProvider(mode="console")

    print("\n" + "=" * 60)
    print("Claude OAuth Integration Test (Console Mode)")
    print("=" * 60)
    print("This test requires you to complete login in the browser.")
    print("This mode creates an API key via OAuth.\n")

    result = await run_oauth_flow(provider, timeout=300)

    assert result.success, f"OAuth flow failed: {result.error}"

    if result.api_key:
        print("\n✓ API key created successfully")
    else:
        print("\n✓ OAuth flow completed")


@pytest.mark.manual
@pytest.mark.asyncio
async def test_claude_token_refresh():
    """Test token refresh for Claude OAuth."""
    storage = TokenStorage()
    tokens = storage.load("claude")

    if tokens is None:
        pytest.skip("No existing Claude tokens - run login test first")

    if not tokens.refresh_token:
        pytest.skip("No refresh token available")

    provider = ClaudeOAuthProvider()
    result = await provider.refresh_tokens(tokens.refresh_token)

    assert result.success, f"Token refresh failed: {result.error}"
    assert result.tokens is not None

    print("\n✓ Token refresh successful")


@pytest.mark.manual
def test_claude_stored_tokens():
    """Verify Claude tokens are properly stored."""
    storage = TokenStorage()
    tokens = storage.load("claude")

    if tokens is None:
        pytest.skip("No Claude tokens stored")

    assert tokens.provider == "claude"
    assert tokens.access_token

    print("\n✓ Claude tokens found")
    print(f"✓ Token expires at: {tokens.expires_at}")
    print(f"✓ Token expired: {tokens.is_expired()}")
