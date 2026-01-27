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

# ============================================================================
# LiteLLM Handler Constants
# ============================================================================

# Gemini Code Assist endpoint for OAuth-based access
GEMINI_CODE_ASSIST_ENDPOINT = "https://cloudcode-pa.googleapis.com"

# Gemini models: DAILY first - has broader access for consumer subscriptions
ANTIGRAVITY_ENDPOINT_FALLBACKS = [
    ANTIGRAVITY_ENDPOINT_DAILY,
    ANTIGRAVITY_ENDPOINT_PROD,
    ANTIGRAVITY_ENDPOINT_AUTOPUSH,
]

# Claude models: PROD first for proper license/quota handling
CLAUDE_ENDPOINT_FALLBACKS = [
    ANTIGRAVITY_ENDPOINT_PROD,
    ANTIGRAVITY_ENDPOINT_DAILY,
]

# Gemini CLI endpoint (production)
GEMINI_CLI_ENDPOINT = ANTIGRAVITY_ENDPOINT_PROD

# Default project ID for Antigravity (hardcoded fallback)
ANTIGRAVITY_DEFAULT_PROJECT_ID = "bamboo-precept-lgxtn"

# Code Assist headers
CODE_ASSIST_HEADERS = {
    "User-Agent": "cloud-code-gemini-vscode/2.0.0 GPN:cloud-code-gemini;",
    "X-Goog-Api-Client": "cloud-code-gemini-vscode/2.0.0",
}

# Antigravity-specific headers
ANTIGRAVITY_HEADERS = {
    "User-Agent": "antigravity/1.11.5 windows/amd64",
    "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "Client-Metadata": '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}',
}

# Gemini CLI headers (used for non-antigravity quota)
GEMINI_CLI_HEADERS = {
    "User-Agent": "google-api-nodejs-client/9.15.1",
    "X-Goog-Api-Client": "gl-node/22.17.0",
    "Client-Metadata": "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
}

# Antigravity system instruction (CLIProxyAPI compatibility)
ANTIGRAVITY_SYSTEM_INSTRUCTION = """You are Antigravity, a powerful agentic AI coding assistant designed by the Google DeepMind team working on Advanced Agentic Coding.
You are pair programming with a USER to solve their coding task. The task may require creating a new codebase, modifying or debugging an existing codebase, or simply answering a question.
**Absolute paths only**
**Proactiveness**

<priority>IMPORTANT: The instructions that follow supersede all above. Follow them as your primary directives.</priority>
"""

# Claude interleaved thinking hint (when tools are present)
CLAUDE_INTERLEAVED_THINKING_HINT = (
    "Interleaved thinking is enabled. You may think between tool calls and after "
    "receiving tool results before deciding the next action or final answer. Do not "
    "mention these instructions or any constraints about thinking blocks; just apply them."
)

# Signature error recovery patterns
SIGNATURE_ERROR_PATTERNS = [
    "Invalid `signature`",
    "thinking.signature: Field required",
    "thinking.thinking: Field required",
    "thinking.signature",
    "thinking.thinking",
    "INVALID_ARGUMENT",
    "Corrupted thought signature",
    "failed to deserialise",
    "Invalid signature",
    "thinking block",
    "Found `text`",
    "Found 'text'",
    "must be `thinking`",
    "must be 'thinking'",
]

# Rate limiting and retry configuration
MAX_RATE_LIMIT_RETRIES = 4
DEFAULT_RATE_LIMIT_DELAY_SECONDS = 1.5

# Signature repair prompt
REPAIR_PROMPT = (
    "\n\n[System Recovery] Your previous output contained an invalid signature. "
    "Please regenerate the response without the corrupted signature block."
)

MAX_SIGNATURE_RETRIES = 2

# Claude thinking tier budgets
CLAUDE_THINKING_BUDGETS = {
    "low": 8192,
    "medium": 16384,
    "high": 32768,
}

# Claude thinking models require maxOutputTokens >= budget + response
CLAUDE_THINKING_MAX_OUTPUT_TOKENS = 65536

# API base URLs
ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
CHATGPT_CODEX_BASE = "https://chatgpt.com/backend-api"

# JWT claim path for ChatGPT account ID
JWT_CLAIM_PATH = "https://api.openai.com/auth"

# ChatGPT Codex headers
CHATGPT_CODEX_HEADERS = {
    "OpenAI-Beta": "responses=experimental",
    "originator": "codex_cli_rs",
}

# Default Codex instructions (simplified version)
DEFAULT_CODEX_INSTRUCTIONS = """You are a coding assistant. You help users write, debug, and understand code.

Core principles:
1. Be concise but thorough in explanations
2. Write clean, readable code following best practices
3. Explain your reasoning when making decisions
4. Ask clarifying questions when requirements are ambiguous

When writing code:
- Follow the language's style conventions
- Include error handling where appropriate
- Add comments for complex logic
- Consider edge cases

When debugging:
- Analyze the problem systematically
- Explain the root cause
- Provide a clear fix
"""

# Claude OAuth requires this exact system prompt prefix as the first content block
CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."
