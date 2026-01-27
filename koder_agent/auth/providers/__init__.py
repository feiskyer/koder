"""OAuth provider implementations.

OAuth providers use different names than API key providers to avoid conflicts:
- google (OAuth) → Gemini CLI subscription
- claude (OAuth) → Claude Max subscription
- chatgpt (OAuth) → ChatGPT Plus/Pro subscription
- antigravity (OAuth) → Antigravity (Gemini 3 + Claude)
"""

import logging
from typing import Optional

import litellm

from koder_agent.auth.base import OAuthProvider
from koder_agent.auth.providers.antigravity import (
    AntigravityOAuthLLM,
    AntigravityOAuthProvider,
)
from koder_agent.auth.providers.chatgpt import ChatGPTOAuthLLM, ChatGPTOAuthProvider
from koder_agent.auth.providers.claude import ClaudeOAuthLLM, ClaudeOAuthProvider
from koder_agent.auth.providers.google import GoogleOAuthLLM, GoogleOAuthProvider

logger = logging.getLogger(__name__)

# OAuth provider identifiers - single source of truth
OAUTH_PROVIDER_IDS = ("google", "claude", "chatgpt", "antigravity")

__all__ = [
    # OAuth Providers
    "GoogleOAuthProvider",
    "ClaudeOAuthProvider",
    "ChatGPTOAuthProvider",
    "AntigravityOAuthProvider",
    # LLM Handlers
    "GoogleOAuthLLM",
    "ClaudeOAuthLLM",
    "ChatGPTOAuthLLM",
    "AntigravityOAuthLLM",
    # Utility functions
    "get_provider",
    "list_providers",
    "register_oauth_providers",
    "get_oauth_model_prefix",
    "is_oauth_provider",
    # Constants
    "OAUTH_PROVIDER_IDS",
]


# Singleton instances for LiteLLM custom handlers
_google_oauth_llm = GoogleOAuthLLM()
_claude_oauth_llm = ClaudeOAuthLLM()
_chatgpt_oauth_llm = ChatGPTOAuthLLM()
_antigravity_oauth_llm = AntigravityOAuthLLM()


def list_providers() -> list[str]:
    """List all available OAuth provider IDs.

    Returns:
        List of provider ID strings
    """
    return list(OAUTH_PROVIDER_IDS)


def get_provider(provider_id: str) -> OAuthProvider:
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


def register_oauth_providers() -> None:
    """Register all OAuth providers with LiteLLM.

    Call this function at startup to enable OAuth-based model access
    with prefixes like 'google/', 'claude/', 'chatgpt/', 'antigravity/'.

    Registration strategy:
    1. Add to provider_list - allows get_llm_provider() to recognize the prefix
    2. Add to _custom_providers - enables routing to custom handlers
    3. Remove ChatGPT models from open_ai_chat_completion_models - prevents
       the model name check from routing to OpenAI before custom handler
    """
    # Register custom handlers
    litellm.custom_provider_map = [
        {"provider": "google", "custom_handler": _google_oauth_llm},
        {"provider": "claude", "custom_handler": _claude_oauth_llm},
        {"provider": "chatgpt", "custom_handler": _chatgpt_oauth_llm},
        {"provider": "antigravity", "custom_handler": _antigravity_oauth_llm},
    ]

    # Add to provider_list for get_llm_provider() to recognize the prefix
    for provider in OAUTH_PROVIDER_IDS:
        if provider not in litellm.provider_list:
            litellm.provider_list.append(provider)

    # Add to _custom_providers for custom handler routing in acompletion()
    existing_custom = list(litellm._custom_providers) if litellm._custom_providers else []
    for provider in OAUTH_PROVIDER_IDS:
        if provider not in existing_custom:
            existing_custom.append(provider)
    litellm._custom_providers = existing_custom

    # Remove ChatGPT/Codex models from open_ai_chat_completion_models
    # This prevents the model name check from catching these models
    # and routing to OpenAI before our custom handler is checked
    chatgpt_models = [
        m for m in litellm.open_ai_chat_completion_models if "gpt-5" in m or "codex" in m.lower()
    ]
    for model in chatgpt_models:
        if model in litellm.open_ai_chat_completion_models:
            litellm.open_ai_chat_completion_models.remove(model)

    logger.info(
        "Registered OAuth providers with LiteLLM: %s. "
        "Removed %d ChatGPT models from OpenAI routing.",
        ", ".join(OAUTH_PROVIDER_IDS),
        len(chatgpt_models),
    )


def get_oauth_model_prefix(provider: str) -> Optional[str]:
    """Get the LiteLLM model prefix for an OAuth provider.

    Args:
        provider: OAuth provider name (google, claude, chatgpt, antigravity)

    Returns:
        LiteLLM model prefix for OAuth access (same as provider name)
    """
    provider_lower = provider.lower()
    return provider_lower if provider_lower in OAUTH_PROVIDER_IDS else None


def is_oauth_provider(provider: str) -> bool:
    """Check if a provider name is an OAuth provider.

    Args:
        provider: Provider name to check

    Returns:
        True if this is an OAuth provider
    """
    return provider.lower() in OAUTH_PROVIDER_IDS
