"""OAuth constants for authentication providers.

Contains client IDs, endpoints, scopes, and other OAuth configuration
for Google, Anthropic, OpenAI, and Antigravity providers.
"""

from typing import Dict, List

# Supported OAuth providers
SUPPORTED_PROVIDERS = ["google", "claude", "chatgpt", "antigravity"]

# Token expiry buffer (refresh tokens 60s before they expire)
TOKEN_EXPIRY_BUFFER_MS = 60 * 1000

# ============================================================================
# Google/Gemini OAuth Configuration
# ============================================================================
# Reference: /tmp/oauth-providers/google-auth/src/constants.ts
# NOTE: Each provider has its own registered redirect URI - cannot be changed

GOOGLE_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GOOGLE_CLIENT_ID = "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v1/userinfo"
GOOGLE_REDIRECT_URI = "http://localhost:8085/oauth2callback"
GOOGLE_CALLBACK_PORT = 8085
GOOGLE_CALLBACK_PATH = "/oauth2callback"
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

# ============================================================================
# Anthropic/Claude OAuth Configuration
# ============================================================================
# Reference: /tmp/oauth-providers/claude-auth/index.mjs

ANTHROPIC_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
ANTHROPIC_AUTH_URL_MAX = "https://claude.ai/oauth/authorize"
ANTHROPIC_AUTH_URL_CONSOLE = "https://console.anthropic.com/oauth/authorize"
ANTHROPIC_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
ANTHROPIC_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
ANTHROPIC_CREATE_API_KEY_URL = "https://api.anthropic.com/api/oauth/claude_cli/create_api_key"
ANTHROPIC_SCOPES = [
    "org:create_api_key",
    "user:profile",
    "user:inference",
]

# Anthropic beta headers for OAuth
ANTHROPIC_BETA_HEADERS = [
    "oauth-2025-04-20",
    "claude-code-20250219",
    "interleaved-thinking-2025-05-14",
    "fine-grained-tool-streaming-2025-05-14",
]

# ============================================================================
# OpenAI/ChatGPT OAuth Configuration
# ============================================================================
# Reference: /tmp/oauth-providers/chatgpt-auth/lib/auth/auth.ts

OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_AUTH_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_REDIRECT_URI = "http://localhost:1455/auth/callback"
OPENAI_CALLBACK_PORT = 1455
OPENAI_CALLBACK_PATH = "/auth/callback"
OPENAI_SCOPES = "openid profile email offline_access"

# ============================================================================
# Antigravity OAuth Configuration
# ============================================================================
# Reference: /tmp/oauth-providers/antigravity-auth/src/constants.ts
# Antigravity uses Google OAuth to access Gemini 3 + Claude models

ANTIGRAVITY_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
ANTIGRAVITY_CLIENT_SECRET = "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"
ANTIGRAVITY_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
ANTIGRAVITY_TOKEN_URL = "https://oauth2.googleapis.com/token"
ANTIGRAVITY_REDIRECT_URI = "http://localhost:51121/oauth-callback"
ANTIGRAVITY_CALLBACK_PORT = 51121
ANTIGRAVITY_CALLBACK_PATH = "/oauth-callback"
ANTIGRAVITY_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]

# Antigravity API endpoints (in fallback order: daily → autopush → prod)
ANTIGRAVITY_ENDPOINT_DAILY = "https://daily-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_ENDPOINT_AUTOPUSH = "https://autopush-cloudcode-pa.sandbox.googleapis.com"
ANTIGRAVITY_ENDPOINT_PROD = "https://cloudcode-pa.googleapis.com"
ANTIGRAVITY_API_BASE = ANTIGRAVITY_ENDPOINT_DAILY

# ============================================================================
# Provider-specific model mappings
# ============================================================================

PROVIDER_MODEL_PREFIXES: Dict[str, List[str]] = {
    "google": ["gemini-", "google/"],
    "anthropic": ["claude-", "anthropic/"],
    "openai": ["gpt-", "o1-", "o3-", "chatgpt-", "codex-", "openai/"],
    "antigravity": ["antigravity-", "antigravity/"],
}
