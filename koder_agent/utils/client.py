"""OpenAI client setup with configuration support."""

import logging
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Optional

from agents import (
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ..config import get_config, get_config_manager
from ..core.turn_cancellation import await_with_turn_cancellation
from ..harness.memory.budget import (
    ContextPreflightError,
    estimate_messages_tokens,
    estimate_model_request_preflight,
    truncate_messages_to_token_budget,
)
from ..litellm_cost_map import get_litellm
from .async_tasks import run_sync_owned
from .model_deprecation import check_model_deprecation
from .model_info import get_context_window_size, get_maximum_output_tokens, resolve_model_alias

litellm = get_litellm()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompletionTruncationMetadata:
    """Truthful description of an explicitly truncated auxiliary request."""

    model: str
    context_window: int
    response_reserve: int
    original_input_tokens: int
    sent_input_tokens: int


@dataclass(frozen=True)
class LLMCompletionResult:
    """Auxiliary completion text plus request-integrity metadata."""

    text: str
    truncation: CompletionTruncationMetadata | None = None


# Suppress debug info from litellm
litellm.suppress_debug_info = True
litellm.drop_params = True

# LiteLLM changed exception namespace in some versions; unify access.
_LITELLM_EXC = getattr(litellm, "exceptions", litellm)
LITELLM_RETRYABLE_ERRORS = (
    getattr(_LITELLM_EXC, "ServiceUnavailableError", Exception),
    getattr(_LITELLM_EXC, "RateLimitError", Exception),
    getattr(_LITELLM_EXC, "APIConnectionError", Exception),
    getattr(_LITELLM_EXC, "Timeout", Exception),
    getattr(_LITELLM_EXC, "InternalServerError", Exception),
)

# Configure global retry settings for LiteLLM
# num_retries will be applied to all litellm API calls unless overridden
litellm.num_retries = 3  # Default, will be updated with config values
litellm.num_retries_per_request = 3

# GitHub Copilot spoof headers (single source of truth; also imported by
# ``koder_agent.agentic.agent``). These emulate the official VSCode Copilot
# client. Excludes x-request-id, which is generated per-use and merged in at
# the call site so it can rotate. NOTE: routing a GitHub Copilot subscription
# through an unofficial client may violate GitHub's terms of service — users
# opt in to this behavior knowingly.
GITHUB_COPILOT_HEADERS: dict[str, str] = {
    "copilot-integration-id": "vscode-chat",
    "editor-version": "vscode/1.98.1",
    "editor-plugin-version": "copilot-chat/0.26.7",
    "user-agent": "GitHubCopilotChat/0.26.7",
    "openai-intent": "conversation-panel",
    "x-github-api-version": "2025-04-01",
    "x-vscode-user-agent-library-version": "electron-fetch",
}

# Well-known environment variable mappings for common providers
# For providers not listed here, the api_key from config will be set
# to the provider's expected env var (e.g., {PROVIDER}_API_KEY)
# OAuth provider to LiteLLM provider mapping
# OAuth providers use different names than LiteLLM expects
OAUTH_TO_LITELLM_PROVIDER = {
    "google": "gemini",  # google OAuth → gemini/ for LiteLLM
    "claude": "anthropic",  # claude OAuth → anthropic/ for LiteLLM
    "chatgpt": "openai",  # chatgpt OAuth → openai/ for LiteLLM
    # antigravity uses custom API, not LiteLLM routing
}

PROVIDER_ENV_VARS = {
    "openai": {"api_key": "OPENAI_API_KEY", "base_url": "OPENAI_BASE_URL"},
    "anthropic": {"api_key": "ANTHROPIC_API_KEY", "base_url": "ANTHROPIC_BASE_URL"},
    "google": {"api_key": "GOOGLE_API_KEY"},
    "gemini": {"api_key": "GEMINI_API_KEY"},
    "azure": {
        "api_key": "AZURE_API_KEY",
        "base_url": "AZURE_API_BASE",
        "api_version": "AZURE_API_VERSION",
    },
    "vertex_ai": {
        "credentials_path": "GOOGLE_APPLICATION_CREDENTIALS",
        "location": "VERTEXAI_LOCATION",
    },
    "bedrock": {"api_key": "AWS_ACCESS_KEY_ID"},
    "cohere": {"api_key": "COHERE_API_KEY"},
    "replicate": {"api_key": "REPLICATE_API_TOKEN"},
    "huggingface": {"api_key": "HUGGINGFACE_API_KEY"},
    "together_ai": {"api_key": "TOGETHERAI_API_KEY"},
    "openrouter": {"api_key": "OPENROUTER_API_KEY"},
    "deepinfra": {"api_key": "DEEPINFRA_API_KEY"},
    "groq": {"api_key": "GROQ_API_KEY"},
    "mistral": {"api_key": "MISTRAL_API_KEY"},
    "perplexity": {"api_key": "PERPLEXITYAI_API_KEY"},
    "fireworks_ai": {"api_key": "FIREWORKS_AI_API_KEY"},
    "cloudflare": {"api_key": "CLOUDFLARE_API_KEY"},
    "ollama": {"base_url": "OLLAMA_BASE_URL"},
    "custom": {"api_key": "OPENAI_API_KEY", "base_url": "OPENAI_BASE_URL"},
}


@dataclass(frozen=True)
class AuthResolutionStatus:
    provider: str
    configured: bool
    oauth_provider: Optional[str] = None
    credential_mode: Literal["api_key", "provider_managed"] = "api_key"

    def __post_init__(self) -> None:
        if self.credential_mode == "provider_managed" and not self.configured:
            raise ValueError("provider_managed authentication must be configured")


_PROVIDER_MANAGED_AUTH_PROVIDERS = frozenset({"ollama", "vertex_ai", "bedrock"})


def _ensure_oauth_providers_registered() -> None:
    """Reconcile actual SDK state; a past success cannot certify a reset registry."""
    try:
        from ..auth.litellm_oauth import register_oauth_providers

        register_oauth_providers()
    except ImportError:
        pass  # Auth module not available


def _get_provider_env_var_name(provider: str) -> str:
    """Get the expected API key environment variable name for a provider."""
    provider_lower = provider.lower()
    if provider_lower in PROVIDER_ENV_VARS:
        return PROVIDER_ENV_VARS[provider_lower].get("api_key", f"{provider.upper()}_API_KEY")
    # Default pattern for unknown providers
    return f"{provider.upper()}_API_KEY"


def get_provider_api_env_var(provider: str) -> str:
    """Public helper so other modules can discover the provider's API key env var."""
    return _get_provider_env_var_name(provider)


def _read_environment_value(
    name: str,
    env: Mapping[str, str | None] | None = None,
) -> Optional[str]:
    """Read one environment value without mutating the selected environment."""
    source = os.environ if env is None else env
    return source.get(name)


def _split_model_identifier(model: str) -> tuple[Optional[str], str, bool]:
    """Split a model identifier into provider/model parts.

    Returns:
        (provider, model_name, had_litellm_prefix)
    """
    if not model:
        return None, "", False

    remainder = model
    had_prefix = False
    if remainder.startswith("litellm/"):
        had_prefix = True
        remainder = remainder[len("litellm/") :]

    if "/" not in remainder:
        return None, remainder, had_prefix

    provider_part, model_part = remainder.split("/", 1)
    return provider_part.lower(), model_part, had_prefix


def _is_openai_native_model(raw_model: str) -> bool:
    """
    Detect whether a model string refers to an OpenAI-native model.

    Handles plain names ("gpt-4o") and provider-prefixed strings
    ("openai/gpt-4o", "litellm/openai/gpt-4o").

    Also checks for "dall-e-" prefix to cover image models.
    """
    if not raw_model:
        return False
    _, model_part, _ = _split_model_identifier(raw_model)
    ml = model_part.lower()
    return ml.startswith(("gpt-", "o1-", "o3-", "o4-", "chatgpt-", "dall-e-"))


def _strip_matching_provider(raw_model: str, provider: str) -> str:
    """Strip provider prefix if it matches the resolved provider."""
    explicit_provider, model_part, _ = _split_model_identifier(raw_model)
    if explicit_provider and explicit_provider == provider.lower():
        return model_part
    return raw_model


def _resolve_model_settings(env: Mapping[str, str | None] | None = None):
    """Resolve the effective config, provider, and raw model string.

    When model comes from KODER_MODEL env var, it may use 'provider/model' format
    (e.g., 'openrouter/x-ai/grok-4.1-fast:free'), so we extract the provider from it.

    When model comes from config file, the provider is explicitly specified in
    config.model.provider, so we use that directly without parsing the model name.
    Model names can contain '/' (e.g., 'x-ai/grok-4.1-fast:free') which should not
    be interpreted as a provider prefix.
    """
    config = get_config()
    config_manager = get_config_manager()

    env_model = _read_environment_value("KODER_MODEL", env)
    model_from_env = env_model is not None
    raw_model = env_model if model_from_env else config.model.name

    # Resolve model aliases (e.g., 'sonnet' → 'claude-sonnet-4-6')
    raw_model = resolve_model_alias(raw_model)

    provider = config.model.provider.lower()

    # Only extract provider from model string if it came from environment variable
    # Environment variable uses 'provider/model' format (e.g., 'openrouter/deepseek-r1')
    # Config file has separate 'provider' field, so model name may contain '/' as part of name
    if model_from_env:
        explicit_provider, _, _ = _split_model_identifier(raw_model)
        if explicit_provider:
            provider = explicit_provider

    return config, config_manager, provider, raw_model, model_from_env


def _get_provider_api_key(
    config,
    provider: str,
    env: Mapping[str, str | None] | None = None,
) -> Optional[str]:
    """Get API key with priority: KODER_API_KEY > OAuth > ENV > Config."""
    provider = provider.lower()
    if provider == "github_copilot":
        return None

    koder_api_key = _read_environment_value("KODER_API_KEY", env)
    if koder_api_key:
        return koder_api_key

    try:
        from ..auth.client_integration import get_oauth_api_key, map_provider_to_oauth

        oauth_provider = map_provider_to_oauth(provider)
        if oauth_provider:
            oauth_key = get_oauth_api_key(oauth_provider)
            if oauth_key:
                return oauth_key
    except ImportError:
        pass

    env_value = _read_environment_value(_get_provider_env_var_name(provider), env)
    if env_value is not None:
        return env_value

    config_value = config.model.api_key if config.model.provider.lower() == provider else None
    return config_value


def _setup_provider_env_vars(config, provider: str):
    """Set up environment variables for the provider (used by LiteLLM)."""
    config_provider = config.model.provider.lower()
    if config_provider != provider:
        return

    # Set provider-specific env vars from config if not already set
    if config_provider == "azure":
        if config.model.azure_api_version and not os.environ.get("AZURE_API_VERSION"):
            os.environ["AZURE_API_VERSION"] = config.model.azure_api_version
        if config.model.base_url and not os.environ.get("AZURE_API_BASE"):
            os.environ["AZURE_API_BASE"] = config.model.base_url

    elif config_provider == "vertex_ai":
        if config.model.vertex_ai_credentials_path and not os.environ.get(
            "GOOGLE_APPLICATION_CREDENTIALS"
        ):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.model.vertex_ai_credentials_path
        if config.model.vertex_ai_location and not os.environ.get("VERTEXAI_LOCATION"):
            os.environ["VERTEXAI_LOCATION"] = config.model.vertex_ai_location

    # SECURITY (P1): Deliberately do NOT write the provider API key into os.environ.
    #
    # Subprocesses spawned by run_shell inherit the parent environment, so any
    # secret placed here would be readable by the model via `env`/`printenv`.
    # The output filter only masks a narrow set of patterns, so most provider
    # keys would leak. Keys must therefore flow ONLY through explicit call kwargs:
    #   - llm_completion()  -> litellm.acompletion/aresponses(api_key=...)
    #   - setup_openai_client() native path -> AsyncOpenAI(api_key=...)
    #   - get_litellm_model_kwargs() -> LitellmModel(api_key=...)
    # All of these resolve the key via _get_provider_api_key() and pass it
    # explicitly, so the previous os.environ[...] = api_key write was redundant.
    #
    # Only non-secret routing vars (Azure API version/base, Vertex location, and
    # GOOGLE_APPLICATION_CREDENTIALS which is a file path, not a secret) are set
    # above, because some provider SDK / LiteLLM routing paths read them from env.


def _map_oauth_to_litellm_provider(provider: str) -> str:
    """Map OAuth provider names to LiteLLM-compatible provider names.

    OAuth providers use different names than LiteLLM expects:
    - google (OAuth) → gemini (LiteLLM)
    - claude (OAuth) → anthropic (LiteLLM)
    - chatgpt (OAuth) → openai (LiteLLM)

    Args:
        provider: Provider name (e.g., 'google', 'claude', 'chatgpt')

    Returns:
        LiteLLM-compatible provider name
    """
    return OAUTH_TO_LITELLM_PROVIDER.get(provider.lower(), provider.lower())


def _should_use_oauth_provider(provider: str) -> bool:
    """Check if we should use OAuth custom handler for a provider.

    Returns True if:
    1. The provider is an OAuth provider (google, claude, chatgpt, antigravity)
    2. A valid OAuth token exists for the provider

    Args:
        provider: Provider name to check

    Returns:
        True if OAuth custom handler should be used
    """
    try:
        from ..auth.client_integration import has_oauth_credentials, map_provider_to_oauth
        from ..auth.litellm_oauth import is_oauth_provider

        if not is_oauth_provider(provider):
            return False

        oauth_provider = map_provider_to_oauth(provider)
        if oauth_provider and has_oauth_credentials(oauth_provider):
            return True
    except ImportError:
        pass
    return False


def _get_oauth_model_prefix(provider: str) -> Optional[str]:
    """Get the LiteLLM custom provider prefix for OAuth access.

    Args:
        provider: OAuth provider name (google, claude, chatgpt, antigravity)

    Returns:
        Custom provider prefix (e.g., 'google_oauth') or None
    """
    try:
        from ..auth.litellm_oauth import get_oauth_model_prefix

        return get_oauth_model_prefix(provider)
    except ImportError:
        return None


def _normalize_model_name(provider: str, raw_model: str, model_from_env: bool = False) -> str:
    """Return a LiteLLM-compatible identifier, always using litellm/<provider>/<model>.

    When OAuth tokens are available for the provider, uses custom OAuth handler
    (e.g., 'google_oauth/gemini-pro') instead of standard LiteLLM provider.

    Args:
        provider: The resolved provider name
        raw_model: The raw model string
        model_from_env: If True, the model string came from KODER_MODEL env var and may
                       contain an explicit provider prefix (e.g., 'openrouter/model-name').
                       If False, the model name should be used as-is since provider comes
                       from config file.
    """
    if not raw_model:
        return raw_model
    if raw_model.startswith("litellm/"):
        return raw_model

    # Only parse provider from model string if it came from environment variable
    if model_from_env:
        explicit_provider, remainder, _ = _split_model_identifier(raw_model)
        if explicit_provider:
            # Check if OAuth should be used for this provider
            if _should_use_oauth_provider(explicit_provider):
                oauth_prefix = _get_oauth_model_prefix(explicit_provider)
                if oauth_prefix:
                    return f"litellm/{oauth_prefix}/{remainder}"

            # Fall back to standard LiteLLM provider mapping
            litellm_provider = _map_oauth_to_litellm_provider(explicit_provider)
            return f"litellm/{litellm_provider}/{remainder}"

    # Check if OAuth should be used for this provider
    if _should_use_oauth_provider(provider):
        oauth_prefix = _get_oauth_model_prefix(provider)
        if oauth_prefix:
            return f"litellm/{oauth_prefix}/{raw_model}"

    # Fall back to standard LiteLLM provider mapping
    litellm_provider = _map_oauth_to_litellm_provider(provider)
    return f"litellm/{litellm_provider}/{raw_model}"


_native_vs_litellm_logged = False


def _compute_effective_model(
    config, provider, raw_model, model_from_env=False, *, resolve_credentials=True
):
    """Determine the model name and whether to use native OpenAI integration."""
    global _native_vs_litellm_logged
    # Non-native model identity does not depend on a credential. Status rendering
    # and override-name lookup must not trigger a synchronous OAuth refresh.
    api_key = (
        _get_provider_api_key(config, provider)
        if resolve_credentials or provider in ("openai", "custom")
        else None
    )

    # Determine whether to use native OpenAI client:
    # 1. Primary: known OpenAI model prefix (gpt-, o1-, o3-, o4-, chatgpt-, dall-e-)
    # 2. Secondary: provider is "openai" with a key, model name is a simple name
    #    (no sub-path like "x-ai/grok-...") and no custom base URL override.
    #    This catches new OpenAI model families whose names don't match known prefixes.
    use_native = False
    if provider in ("openai", "custom") and api_key:
        if _is_openai_native_model(raw_model):
            use_native = True
        elif provider == "openai":
            # Secondary detection: treat as native if model looks like a direct
            # OpenAI model (no sub-routing path) and no custom base URL is set.
            # Check both: the model_part after splitting AND the raw_model itself
            # (since raw_model like "x-ai/grok-..." is a compound routing name).
            stripped = _strip_matching_provider(raw_model, provider)
            has_routing_path = "/" in stripped
            has_custom_base = os.environ.get("KODER_BASE_URL") is not None
            if not has_routing_path and not has_custom_base:
                use_native = True

    if not _native_vs_litellm_logged:
        _native_vs_litellm_logged = True
        logger = logging.getLogger(__name__)
        if logger.isEnabledFor(logging.INFO):
            path = "native-openai" if use_native else "litellm"
            logger.info("Model routing: provider=%s model=%s path=%s", provider, raw_model, path)

    if use_native:
        # If the raw model string included a provider prefix, strip it for native calls
        normalized_raw = _strip_matching_provider(raw_model, provider)
        return normalized_raw, True, api_key

    return _normalize_model_name(provider, raw_model, model_from_env), False, api_key


def _resolve_completion_settings(
    model_override: Optional[str] = None,
    env: Mapping[str, str | None] | None = None,
):
    """Resolve provider/model settings for a completion call, honoring overrides."""
    config, config_manager, provider, raw_model, model_from_env = _resolve_model_settings(env)

    if model_override is not None:
        # Resolve aliases in overrides (e.g., 'sonnet' → 'claude-sonnet-4-6')
        raw_model = resolve_model_alias(model_override)
        override_provider, _, _ = _split_model_identifier(raw_model)
        if override_provider is not None:
            provider = override_provider
            model_from_env = True
        else:
            model_from_env = False

    return config, config_manager, provider, raw_model, model_from_env


def get_model_name(model_override: Optional[str] = None):
    """Get the appropriate model name with priority: ENV > Config > Default."""
    config, _, provider, raw_model, model_from_env = _resolve_completion_settings(model_override)
    model, _, _ = _compute_effective_model(
        config, provider, raw_model, model_from_env, resolve_credentials=False
    )
    # Check for model deprecation and warn if applicable
    try:
        warning = check_model_deprecation(raw_model)
        if warning:
            logging.warning(warning)
    except Exception:
        pass  # Don't fail if deprecation check fails
    return model


def resolve_model_override_name(raw_model: str) -> str:
    """Resolve a model override into the actual call model name."""
    config, _, provider, raw_model, model_from_env = _resolve_completion_settings(raw_model)
    model, _, _ = _compute_effective_model(
        config, provider, raw_model, model_from_env, resolve_credentials=False
    )
    return model


def get_api_key(
    model_override: Optional[str] = None,
    *,
    env: Mapping[str, str | None] | None = None,
) -> Optional[str]:
    """Get API key with priority: KODER_API_KEY > OAuth > ENV > Config."""
    config, _, provider, _, _ = _resolve_completion_settings(model_override, env)
    return _get_provider_api_key(config, provider, env)


def resolve_auth_status(
    model_override: Optional[str] = None,
    *,
    env: Mapping[str, str | None] | None = None,
) -> AuthResolutionStatus:
    """Resolve non-secret authentication status for the selected provider."""
    config, _, provider, _, _ = _resolve_completion_settings(model_override, env)
    if provider == "github_copilot":
        return AuthResolutionStatus(provider=provider, configured=False)
    if provider in _PROVIDER_MANAGED_AUTH_PROVIDERS:
        return AuthResolutionStatus(
            provider=provider,
            configured=True,
            credential_mode="provider_managed",
        )
    configured = bool(_get_provider_api_key(config, provider, env))
    oauth_provider = None
    try:
        from ..auth.client_integration import map_provider_to_oauth

        oauth_provider = map_provider_to_oauth(provider)
    except ImportError:
        pass
    return AuthResolutionStatus(
        provider=provider,
        configured=configured,
        oauth_provider=oauth_provider,
    )


def _resolve_base_url(config, config_manager, provider: str) -> Optional[str]:
    """Resolve base URL with priority: KODER_BASE_URL > ENV > Config.

    Args:
        config: Configuration object
        config_manager: Configuration manager
        provider: Provider name

    Returns:
        Base URL or None
    """
    koder_base_url = os.environ.get("KODER_BASE_URL")
    if koder_base_url:
        return koder_base_url

    base_url_env_var = PROVIDER_ENV_VARS.get(provider, {}).get(
        "base_url", f"{provider.upper()}_BASE_URL"
    )
    base_url_config = config.model.base_url if config.model.provider.lower() == provider else None
    return config_manager.get_effective_value(base_url_config, base_url_env_var)


def get_base_url(model_override: Optional[str] = None):
    """Get base URL with priority: KODER_BASE_URL > ENV > Config."""
    config, config_manager, provider, _, _ = _resolve_completion_settings(model_override)
    return _resolve_base_url(config, config_manager, provider)


def _get_oauth_extra_headers(provider: str) -> Optional[dict]:
    """Get OAuth-specific headers for a provider.

    Args:
        provider: Provider identifier

    Returns:
        Dict of extra headers or None
    """
    try:
        from ..auth.client_integration import (
            get_oauth_headers,
            has_oauth_token,
            map_provider_to_oauth,
        )

        oauth_provider = map_provider_to_oauth(provider)
        if oauth_provider and has_oauth_token(oauth_provider):
            headers = get_oauth_headers(oauth_provider)
            # Remove Authorization header as it's handled via api_key
            headers.pop("Authorization", None)
            return headers if headers else None
    except ImportError:
        pass
    return None


def get_litellm_model_kwargs(model_override: Optional[str] = None) -> dict:
    """Get kwargs for creating a LitellmModel instance.

    Returns a dict with 'model', 'api_key', 'base_url', and retry configuration
    that can be passed directly to LitellmModel constructor.
    """
    config, config_manager, provider, raw_model, model_from_env = _resolve_completion_settings(
        model_override
    )

    # Get normalized model name for LiteLLM
    model = _normalize_model_name(provider, raw_model, model_from_env)
    # Strip the 'litellm/' prefix since LitellmModel adds it internally
    if model.startswith("litellm/"):
        model = model[len("litellm/") :]

    api_key = _get_provider_api_key(config, provider)
    base_url = _resolve_base_url(config, config_manager, provider)

    kwargs = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "max_retries": 3,
    }

    # Add OAuth-specific headers if available
    oauth_headers = _get_oauth_extra_headers(provider)
    if oauth_headers:
        kwargs["extra_headers"] = oauth_headers

    return kwargs


def get_model_client_snapshot(model_override: Optional[str] = None) -> dict:
    """Return a stable snapshot of the currently resolved client settings."""
    config = get_config()
    config_manager = get_config_manager()
    model_name = get_model_name(model_override)
    context_window = get_configured_context_window(
        model_name,
        use_main_override=model_override in (None, "", "inherit"),
    )
    return {
        "model_name": model_name,
        "api_key": get_api_key(model_override),
        "base_url": get_base_url(model_override),
        "litellm_kwargs": get_litellm_model_kwargs(model_override),
        "native_openai": is_native_openai_provider(model_override),
        "context_window": context_window,
        "max_output_tokens": get_maximum_output_tokens(
            model_name,
            max_context_size=context_window,
        ),
        "reasoning_effort": config_manager.get_effective_value(
            config.model.reasoning_effort,
            "KODER_REASONING_EFFORT",
        ),
    }


def is_native_openai_provider(model_override: Optional[str] = None) -> bool:
    """Check if the current provider should use native OpenAI client."""
    config, _, provider, raw_model, _ = _resolve_completion_settings(model_override)
    api_key = _get_provider_api_key(config, provider)

    return (
        provider in ("openai", "custom")
        and api_key is not None
        and _is_openai_native_model(raw_model)
    )


def get_small_model_name() -> Optional[str]:
    """Resolve the small/cheap model for auxiliary LLM calls.

    Priority: KODER_SMALL_MODEL env > config.model.small_model > None.
    When None is returned, callers should use the main model.
    """
    env_val = os.environ.get("KODER_SMALL_MODEL")
    if env_val:
        return env_val
    try:
        config = get_config()
        if config.model.small_model:
            return config.model.small_model
    except Exception:
        pass
    return None


def _positive_int(value: object, *, setting: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{setting} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{setting} must be a positive integer")
    return parsed


def get_configured_context_window(
    model: str,
    *,
    use_small: bool = False,
    use_main_override: bool = True,
) -> int:
    """Resolve an authoritative registry or explicit configured model window."""
    config = get_config()
    if use_small:
        override = os.environ.get("KODER_SMALL_MODEL_CONTEXT_WINDOW")
        if override is None:
            override = config.model.small_model_context_window
        setting = "model.small_model_context_window"
    elif use_main_override:
        override = os.environ.get("KODER_CONTEXT_WINDOW")
        if override is None:
            override = config.model.context_window
        setting = "model.context_window"
    else:
        override = None
        setting = "model.context_window"
    parsed_override = _positive_int(override, setting=setting)
    if parsed_override is None:
        return get_context_window_size(model)
    return get_context_window_size(model, max_context_size=parsed_override)


@retry(
    retry=retry_if_exception_type(LITELLM_RETRYABLE_ERRORS),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1),
    before_sleep=before_sleep_log(logger, logging.INFO),
    reraise=True,
)
async def llm_completion(
    messages: list,
    model: Optional[str] = None,
    use_small: bool = False,
    *,
    response_reserve: int | None = None,
    overflow_policy: Literal["error", "truncate"] = "error",
    return_metadata: bool = False,
) -> str | LLMCompletionResult:
    """
    Make an LLM completion call using the configured provider settings.

    This function reuses the same configuration as the main agent, ensuring
    consistent API key and model settings. Includes automatic retry for 429 errors.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: Optional model override. If None, uses configured model.
        use_small: If True and a small model is configured (via KODER_SMALL_MODEL
                   env or config.model.small_model), override the model for this call.
                   When no small model is configured, this flag is ignored.
        response_reserve: Tokens reserved for the auxiliary response. Defaults
                          to the model's normal maximum output allowance.
        overflow_policy: Reject by default. Truncation is an explicit opt-in,
                         never triggers compaction, and requires metadata.
        return_metadata: Return :class:`LLMCompletionResult`. Required when
                         ``overflow_policy="truncate"`` so callers cannot hide
                         partial-input provenance.

    Returns:
        The completion response content as string
    """
    # If use_small requested and no explicit model override, try small model
    selected_small_model = False
    explicit_model_override = model is not None
    if use_small and model is None:
        small = get_small_model_name()
        if small:
            model = small
            selected_small_model = True

    if overflow_policy == "truncate" and not return_metadata:
        raise ValueError(
            'overflow_policy="truncate" requires return_metadata=True so truncation is surfaced'
        )

    config, config_manager, provider, raw_model, model_from_env = await run_sync_owned(
        _resolve_completion_settings, model
    )

    # Ensure provider env vars are set (for litellm to pick up)
    _setup_provider_env_vars(config, provider)

    # Get model name and API key
    model, use_native, api_key = await run_sync_owned(
        _compute_effective_model, config, provider, raw_model, model_from_env
    )

    # When the model is OpenAI-native but llm_completion goes through litellm,
    # ensure the model has a provider prefix so litellm can route it correctly.
    # Without this, newer OpenAI models (e.g. gpt-5.4) that litellm doesn't
    # recognize by name will fail with "LLM Provider NOT provided".
    if use_native and not model.startswith(("openai/", "azure/")):
        model = f"openai/{model}"

    # Strip the 'litellm/' prefix since litellm.acompletion doesn't expect it
    # (_compute_effective_model returns litellm-prefixed names for non-native models)
    if model.startswith("litellm/"):
        model = model[len("litellm/") :]

    context_window = get_configured_context_window(
        model,
        use_small=selected_small_model,
        use_main_override=not explicit_model_override and not selected_small_model,
    )
    reserve = (
        get_maximum_output_tokens(model, max_context_size=context_window)
        if response_reserve is None
        else max(0, response_reserve)
    )
    fixed_messages = []
    dynamic_messages = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") in {"system", "developer"}:
            fixed_messages.append(message)
        else:
            dynamic_messages.append(message)
    estimate = estimate_model_request_preflight(
        context_window=context_window,
        response_reserve=reserve,
        instructions=fixed_messages,
        input_items=dynamic_messages,
        model=model,
    )
    truncation: CompletionTruncationMetadata | None = None
    if not estimate.fits:
        fitted_messages = None
        if overflow_policy == "truncate":
            fitted_messages = truncate_messages_to_token_budget(
                messages,
                estimate.available_input_tokens,
            )
        if fitted_messages is None:
            raise ContextPreflightError(estimate, subject="Auxiliary request")
        logger.warning(
            "Truncated auxiliary LLM input for context preflight: model=%s required=%s window=%s",
            model,
            estimate.required_tokens,
            estimate.context_window,
        )
        truncation = CompletionTruncationMetadata(
            model=model,
            context_window=context_window,
            response_reserve=reserve,
            original_input_tokens=estimate_messages_tokens(messages, model=model),
            sent_input_tokens=estimate_messages_tokens(fitted_messages, model=model),
        )
        messages = fitted_messages

    # Use the same base URL priority as the main agent path:
    # KODER_BASE_URL > provider-specific environment > config.
    base_url = _resolve_base_url(config, config_manager, provider)

    # Build kwargs for litellm with retry configuration
    kwargs = {
        "model": model,
        "messages": messages,
        "metadata": {"source": "koder"},
        "max_tokens": reserve,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    model_lower = str(model).lower()
    is_copilot = model_lower.startswith("github_copilot/")
    extra_headers = None
    if is_copilot:
        # Reuse the shared static header set and add a fresh per-request id.
        extra_headers = {
            **GITHUB_COPILOT_HEADERS,
            "x-request-id": str(uuid.uuid4()),
        }

    if is_copilot and "codex" in model_lower:
        # Import lazily: agentic's package initializer imports this client module.
        from ..agentic.responses_compat import terminal_response_error

        if not hasattr(litellm, "aresponses"):
            raise RuntimeError(
                "GitHub Copilot Codex models require LiteLLM Responses API support. "
                "Please upgrade litellm to a version that provides `aresponses`."
            )
        responses_kwargs = {
            "model": model,
            "input": messages,
            "metadata": {"source": "koder"},
            "stream": False,
            "max_output_tokens": reserve,
        }
        if api_key:
            responses_kwargs["api_key"] = api_key
        if base_url:
            responses_kwargs["base_url"] = base_url
        if extra_headers:
            responses_kwargs["extra_headers"] = extra_headers
        response = await await_with_turn_cancellation(litellm.aresponses(**responses_kwargs))
        if failure := terminal_response_error(response):
            raise failure
        text = _extract_responses_text(response)
        return LLMCompletionResult(text=text, truncation=truncation) if return_metadata else text

    if extra_headers:
        kwargs["extra_headers"] = extra_headers
    from ..auth.oauth_routing import acompletion

    response = await await_with_turn_cancellation(acompletion(**kwargs))
    text = response.choices[0].message.content
    return LLMCompletionResult(text=text, truncation=truncation) if return_metadata else text


def _extract_responses_text(response: object) -> str:
    """
    Best-effort extraction of assistant text from a LiteLLM Responses API response.
    Handles dict-like and pydantic-like objects.
    """
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")

    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            item_dict = item
            if hasattr(item, "model_dump"):
                try:
                    item_dict = item.model_dump()
                except Exception:
                    item_dict = item
            if not isinstance(item_dict, dict):
                continue
            if item_dict.get("type") == "message":
                for content in item_dict.get("content", []) or []:
                    if isinstance(content, dict) and content.get("type") in (
                        "output_text",
                        "text",
                    ):
                        text_val = content.get("text") or content.get("content")
                        if text_val:
                            parts.append(str(text_val))
            elif item_dict.get("type") in ("output_text", "text"):
                text_val = item_dict.get("text")
                if text_val:
                    parts.append(str(text_val))
        if parts:
            return "".join(parts).strip()

    if isinstance(response, dict):
        try:
            first = response.get("output", [])[0]
            if isinstance(first, dict):
                content = first.get("content", [])
                if content and isinstance(content[0], dict):
                    return str(content[0].get("text", "")).strip()
        except Exception:
            pass

    return ""


def setup_openai_client():
    """Set up the OpenAI client with priority: ENV > Config > Default.

    Also configures global LiteLLM retry settings for all providers.
    OAuth calls select their Koder-owned handlers at the request boundary.
    """
    set_tracing_disabled(True)

    config, config_manager, provider, raw_model, model_from_env = _resolve_model_settings()

    # Setup provider environment variables for LiteLLM
    _setup_provider_env_vars(config, provider)

    model, use_native, api_key = _compute_effective_model(
        config, provider, raw_model, model_from_env
    )
    base_url = _resolve_base_url(config, config_manager, provider)

    if use_native:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=3,
        )
        set_default_openai_client(client)
        return client

    # Fall back to LiteLLM integration for other providers
    return None
