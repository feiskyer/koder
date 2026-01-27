"""Custom LiteLLM provider for OAuth-based authentication.

DEPRECATED: Import from koder_agent.auth.providers instead.

This module is maintained for backwards compatibility. All OAuth LLM handler
classes and registration functions have been moved to their respective
provider modules in koder_agent.auth.providers/.
"""

from koder_agent.auth.providers import (
    AntigravityOAuthLLM,
    ChatGPTOAuthLLM,
    ClaudeOAuthLLM,
    GoogleOAuthLLM,
    get_oauth_model_prefix,
    is_oauth_provider,
    register_oauth_providers,
)

__all__ = [
    "register_oauth_providers",
    "get_oauth_model_prefix",
    "is_oauth_provider",
    "GoogleOAuthLLM",
    "ClaudeOAuthLLM",
    "ChatGPTOAuthLLM",
    "AntigravityOAuthLLM",
]
