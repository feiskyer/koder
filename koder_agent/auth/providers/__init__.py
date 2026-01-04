"""OAuth provider implementations.

OAuth providers use different names than API key providers to avoid conflicts:
- google (OAuth) → Gemini CLI subscription
- claude (OAuth) → Claude Max subscription
- chatgpt (OAuth) → ChatGPT Plus/Pro subscription
- antigravity (OAuth) → Antigravity (Gemini 3 + Claude)
"""

from koder_agent.auth.providers.antigravity import AntigravityOAuthProvider
from koder_agent.auth.providers.chatgpt import ChatGPTOAuthProvider
from koder_agent.auth.providers.claude import ClaudeOAuthProvider
from koder_agent.auth.providers.google import GoogleOAuthProvider

__all__ = [
    "GoogleOAuthProvider",
    "ClaudeOAuthProvider",
    "ChatGPTOAuthProvider",
    "AntigravityOAuthProvider",
    "get_provider",
    "list_providers",
]


def list_providers():
    """List all available OAuth provider IDs.

    Returns:
        List of provider ID strings
    """
    return ["google", "claude", "chatgpt", "antigravity"]


def get_provider(provider_id: str):
    """Get OAuth provider instance by ID.

    Args:
        provider_id: Provider identifier (google, claude, chatgpt, antigravity)

    Returns:
        OAuth provider instance

    Raises:
        ValueError: If provider is not supported
    """
    providers = {
        "google": GoogleOAuthProvider,
        "claude": ClaudeOAuthProvider,
        "chatgpt": ChatGPTOAuthProvider,
        "antigravity": AntigravityOAuthProvider,
    }

    if provider_id not in providers:
        raise ValueError(
            f"Unsupported provider: {provider_id}. Supported: {', '.join(providers.keys())}"
        )

    return providers[provider_id]()
