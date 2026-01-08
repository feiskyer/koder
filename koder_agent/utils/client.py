"""OpenAI client setup with configuration support."""

import os
import uuid
from typing import Optional

import backoff
import litellm
from agents import (
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

from ..config import get_config, get_config_manager

# Suppress debug info from litellm
litellm.suppress_debug_info = True
litellm.drop_params = True

# Register OAuth providers with LiteLLM
# This must happen early so custom providers are available for model routing
_oauth_providers_registered = False

# LiteLLM changed exception namespace in some versions; unify access.
_LITELLM_EXC = getattr(litellm, "exceptions", litellm)
_LITELLM_ERRORS = (
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
    "github_copilot": {"api_key": "GITHUB_TOKEN"},
    "ollama": {"base_url": "OLLAMA_BASE_URL"},
    "custom": {"api_key": "OPENAI_API_KEY", "base_url": "OPENAI_BASE_URL"},
}


def _ensure_oauth_providers_registered() -> None:
    """Ensure OAuth providers are registered with LiteLLM.

    This function is idempotent and will only register providers once.
    """
    global _oauth_providers_registered
    if _oauth_providers_registered:
        return

    try:
        from ..auth.litellm_oauth import register_oauth_providers

        register_oauth_providers()
        _oauth_providers_registered = True
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
    """
    if not raw_model:
        return False
    _, model_part, _ = _split_model_identifier(raw_model)
    ml = model_part.lower()
    return ml.startswith(("gpt-", "o1-", "o3-", "o4-", "chatgpt-"))


def _strip_matching_provider(raw_model: str, provider: str) -> str:
    """Strip provider prefix if it matches the resolved provider."""
    explicit_provider, model_part, _ = _split_model_identifier(raw_model)
    if explicit_provider and explicit_provider == provider.lower():
        return model_part
    return raw_model


def _resolve_model_settings():
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

    # Check if KODER_MODEL env var is set
    env_model = os.environ.get("KODER_MODEL")
    model_from_env = env_model is not None

    raw_model = config_manager.get_effective_value(config.model.name, "KODER_MODEL")
    provider = config.model.provider.lower()

    # Only extract provider from model string if it came from environment variable
    # Environment variable uses 'provider/model' format (e.g., 'openrouter/deepseek-r1')
    # Config file has separate 'provider' field, so model name may contain '/' as part of name
    if model_from_env:
        explicit_provider, _, _ = _split_model_identifier(raw_model)
        if explicit_provider:
            provider = explicit_provider

    return config, config_manager, provider, raw_model, model_from_env


def _get_provider_api_key(config, config_manager, provider: str):
    """Get API key with priority: KODER_API_KEY > OAuth > ENV > Config."""
    koder_api_key = os.environ.get("KODER_API_KEY")
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

    env_var_name = _get_provider_env_var_name(provider)
    config_value = config.model.api_key if config.model.provider.lower() == provider else None
    return config_manager.get_effective_value(config_value, env_var_name)


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

    # Set API key if configured and not in env
    api_key = config.model.api_key
    if api_key:
        env_var_name = _get_provider_env_var_name(config_provider)
        if not os.environ.get(env_var_name):
            os.environ[env_var_name] = api_key


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

    # Ensure OAuth providers are registered before checking
    _ensure_oauth_providers_registered()

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


def _compute_effective_model(config, config_manager, provider, raw_model, model_from_env=False):
    """Determine the model name and whether to use native OpenAI integration."""
    api_key = _get_provider_api_key(config, config_manager, provider)
    use_native = provider in ("openai", "custom") and api_key and _is_openai_native_model(raw_model)

    if use_native:
        # If the raw model string included a provider prefix, strip it for native calls
        normalized_raw = _strip_matching_provider(raw_model, provider)
        return normalized_raw, True, api_key

    return _normalize_model_name(provider, raw_model, model_from_env), False, api_key


def get_model_name():
    """Get the appropriate model name with priority: ENV > Config > Default."""
    config, config_manager, provider, raw_model, model_from_env = _resolve_model_settings()
    model, _, _ = _compute_effective_model(
        config, config_manager, provider, raw_model, model_from_env
    )
    return model


def get_api_key():
    """Get API key with priority: KODER_API_KEY > OAuth > ENV > Config."""
    config, config_manager, provider, _, _ = _resolve_model_settings()
    return _get_provider_api_key(config, config_manager, provider)


def get_base_url():
    """Get base URL with priority: KODER_BASE_URL > ENV > Config."""
    koder_base_url = os.environ.get("KODER_BASE_URL")
    if koder_base_url:
        return koder_base_url

    config, config_manager, provider, _, _ = _resolve_model_settings()
    base_url_env_var = PROVIDER_ENV_VARS.get(provider, {}).get(
        "base_url", f"{provider.upper()}_BASE_URL"
    )
    base_url_config = config.model.base_url if config.model.provider.lower() == provider else None
    return config_manager.get_effective_value(base_url_config, base_url_env_var)


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


def get_litellm_model_kwargs() -> dict:
    """Get kwargs for creating a LitellmModel instance.

    Returns a dict with 'model', 'api_key', 'base_url', and retry configuration
    that can be passed directly to LitellmModel constructor.
    """
    config, config_manager, provider, raw_model, model_from_env = _resolve_model_settings()

    # Get normalized model name for LiteLLM
    model = _normalize_model_name(provider, raw_model, model_from_env)
    # Strip the 'litellm/' prefix since LitellmModel adds it internally
    if model.startswith("litellm/"):
        model = model[len("litellm/") :]

    api_key = _get_provider_api_key(config, config_manager, provider)

    koder_base_url = os.environ.get("KODER_BASE_URL")
    if koder_base_url:
        base_url = koder_base_url
    else:
        base_url_env_var = PROVIDER_ENV_VARS.get(provider, {}).get(
            "base_url", f"{provider.upper()}_BASE_URL"
        )
        base_url_config = (
            config.model.base_url if config.model.provider.lower() == provider else None
        )
        base_url = config_manager.get_effective_value(base_url_config, base_url_env_var)

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


def is_native_openai_provider() -> bool:
    """Check if the current provider should use native OpenAI client."""
    config, config_manager, provider, raw_model, _ = _resolve_model_settings()
    api_key = _get_provider_api_key(config, config_manager, provider)

    return (
        provider in ("openai", "custom")
        and api_key is not None
        and _is_openai_native_model(raw_model)
    )


@backoff.on_exception(
    backoff.expo,
    _LITELLM_ERRORS,
    max_tries=3,
    jitter=backoff.full_jitter,
)
async def llm_completion(messages: list, model: Optional[str] = None) -> str:
    """
    Make an LLM completion call using the configured provider settings.

    This function reuses the same configuration as the main agent, ensuring
    consistent API key and model settings. Includes automatic retry for 429 errors.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: Optional model override. If None, uses configured model.

    Returns:
        The completion response content as string
    """

    config, config_manager, provider, raw_model, model_from_env = _resolve_model_settings()

    # Ensure provider env vars are set (for litellm to pick up)
    _setup_provider_env_vars(config, provider)

    # Get model name and API key
    if model is None:
        model, _, api_key = _compute_effective_model(
            config, config_manager, provider, raw_model, model_from_env
        )
    else:
        api_key = _get_provider_api_key(config, config_manager, provider)

    # Get base URL if configured
    base_url_env_var = PROVIDER_ENV_VARS.get(provider, {}).get("base_url", "OPENAI_BASE_URL")
    base_url_config = config.model.base_url if config.model.provider.lower() == provider else None
    base_url = config_manager.get_effective_value(base_url_config, base_url_env_var)

    # Build kwargs for litellm with retry configuration
    kwargs = {
        "model": model,
        "messages": messages,
        "metadata": {"source": "koder"},
    }
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    model_lower = str(model).lower()
    is_copilot = "github_copilot/" in model_lower
    extra_headers = None
    if is_copilot:
        extra_headers = {
            "copilot-integration-id": "vscode-chat",
            "editor-version": "vscode/1.98.1",
            "editor-plugin-version": "copilot-chat/0.26.7",
            "user-agent": "GitHubCopilotChat/0.26.7",
            "openai-intent": "conversation-panel",
            "x-github-api-version": "2025-04-01",
            "x-request-id": str(uuid.uuid4()),
            "x-vscode-user-agent-library-version": "electron-fetch",
        }

    if is_copilot and "codex" in model_lower:
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
        }
        if api_key:
            responses_kwargs["api_key"] = api_key
        if base_url:
            responses_kwargs["base_url"] = base_url
        if extra_headers:
            responses_kwargs["extra_headers"] = extra_headers
        response = await litellm.aresponses(**responses_kwargs)
        return _extract_responses_text(response)

    if extra_headers:
        kwargs["extra_headers"] = extra_headers
    response = await litellm.acompletion(**kwargs)
    return response.choices[0].message.content


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
    Registers OAuth custom providers with LiteLLM if auth module is available.
    """
    set_tracing_disabled(True)

    # Register OAuth providers with LiteLLM
    _ensure_oauth_providers_registered()

    config, config_manager, provider, raw_model, model_from_env = _resolve_model_settings()

    # Setup provider environment variables for LiteLLM
    _setup_provider_env_vars(config, provider)

    model, use_native, api_key = _compute_effective_model(
        config, config_manager, provider, raw_model, model_from_env
    )

    koder_base_url = os.environ.get("KODER_BASE_URL")
    if koder_base_url:
        base_url = koder_base_url
    else:
        base_url_env_var = PROVIDER_ENV_VARS.get(provider, {}).get("base_url", "OPENAI_BASE_URL")
        base_url_config = (
            config.model.base_url if config.model.provider.lower() == provider else None
        )
        base_url = config_manager.get_effective_value(base_url_config, base_url_env_var)

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
