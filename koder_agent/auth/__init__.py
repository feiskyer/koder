"""OAuth authentication module for Koder.

This module provides OAuth authentication support for multiple providers:
- Google (Gemini API)
- Anthropic (Claude API)
- OpenAI (ChatGPT/Codex API)
- Antigravity (Google-based access to Gemini 3 + Claude)
"""

from koder_agent.auth.base import OAuthProvider, OAuthResult, OAuthTokens
from koder_agent.auth.constants import (
    ANTHROPIC_CLIENT_ID,
    GOOGLE_CLIENT_ID,
    OPENAI_CLIENT_ID,
    SUPPORTED_PROVIDERS,
)
from koder_agent.auth.token_storage import TokenStorage

__all__ = [
    "OAuthProvider",
    "OAuthTokens",
    "OAuthResult",
    "TokenStorage",
    "SUPPORTED_PROVIDERS",
    "GOOGLE_CLIENT_ID",
    "ANTHROPIC_CLIENT_ID",
    "OPENAI_CLIENT_ID",
]
