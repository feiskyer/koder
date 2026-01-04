"""Manual integration test for ChatGPT OAuth.

This test requires user interaction to complete the OAuth flow.
Run with: pytest tests/integration/test_chatgpt_oauth.py -m manual --manual -s
"""

import pytest

from koder_agent.auth.callback_server import run_oauth_flow
from koder_agent.auth.providers.chatgpt import ChatGPTOAuthProvider
from koder_agent.auth.token_storage import TokenStorage


@pytest.mark.manual
@pytest.mark.asyncio
async def test_chatgpt_oauth_flow():
    """Test full ChatGPT OAuth flow with user interaction.

    This test will:
    1. Generate authorization URL
    2. Open browser for user to login
    3. Wait for OAuth callback
    4. Exchange code for tokens
    5. Verify tokens are valid
    """
    provider = ChatGPTOAuthProvider()

    print("\n" + "=" * 60)
    print("ChatGPT OAuth Integration Test")
    print("=" * 60)
    print("This test requires you to complete login in the browser.")
    print("You need a ChatGPT Plus/Pro subscription.")
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


@pytest.mark.manual
@pytest.mark.asyncio
async def test_chatgpt_token_refresh():
    """Test token refresh for ChatGPT OAuth."""
    storage = TokenStorage()
    tokens = storage.load("chatgpt")

    if tokens is None:
        pytest.skip("No existing ChatGPT tokens - run login test first")

    provider = ChatGPTOAuthProvider()
    result = await provider.refresh_tokens(tokens.refresh_token)

    assert result.success, f"Token refresh failed: {result.error}"
    assert result.tokens is not None
    assert result.tokens.access_token

    print("\n✓ Token refresh successful")
    print("✓ New access token obtained")


@pytest.mark.manual
def test_chatgpt_stored_tokens():
    """Verify ChatGPT tokens are properly stored."""
    storage = TokenStorage()
    tokens = storage.load("chatgpt")

    if tokens is None:
        pytest.skip("No ChatGPT tokens stored")

    assert tokens.provider == "chatgpt"
    assert tokens.access_token
    assert tokens.refresh_token

    print("\n✓ ChatGPT tokens found")
    if tokens.email:
        print(f"✓ Email: {tokens.email}")
    print(f"✓ Token expires at: {tokens.expires_at}")
    print(f"✓ Token expired: {tokens.is_expired()}")
