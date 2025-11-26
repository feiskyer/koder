"""OpenAI client setup with configuration support."""

import os
from typing import Optional

import litellm
from agents import (
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

from ..config import get_config, get_config_manager

# Suppress debug info from litellm
litellm.suppress_debug_info = True

# Well-known environment variable mappings for common providers
# For providers not listed here, the api_key from config will be set
# to the provider's expected env var (e.g., {PROVIDER}_API_KEY)
PROVIDER_ENV_VARS = {
    "openai": {"api_key": "OPENAI_API_KEY", "base_url": "OPENAI_BASE_URL"},
    "anthropic": {"api_key": "ANTHROPIC_API_KEY"},
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


def _resolve_model_settings():
    """Resolve the effective config, provider, and raw model string."""
    config = get_config()
    config_manager = get_config_manager()
    raw_model = config_manager.get_effective_value(config.model.name, "KODER_MODEL")
    provider = config.model.provider.lower()

    explicit_provider, _, _ = _split_model_identifier(raw_model)
    if explicit_provider:
        provider = explicit_provider

    return config, config_manager, provider, raw_model


def _get_provider_api_key(config, config_manager, provider: str):
    """Get the API key for the effective provider."""
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


def _normalize_model_name(provider: str, raw_model: str) -> str:
    """Return a LiteLLM-compatible identifier, always using litellm/<provider>/<model>."""
    if not raw_model:
        return raw_model
    if raw_model.startswith("litellm/"):
        return raw_model

    explicit_provider, remainder, _ = _split_model_identifier(raw_model)
    if explicit_provider:
        return f"litellm/{explicit_provider}/{remainder}"

    provider = provider.lower()
    return f"litellm/{provider}/{raw_model}"


def _compute_effective_model(config, config_manager, provider, raw_model):
    """Determine the model name and whether to use native OpenAI integration."""
    api_key = _get_provider_api_key(config, config_manager, provider)
    use_native = provider in ("openai", "custom") and api_key

    if use_native:
        return raw_model, True, api_key

    return _normalize_model_name(provider, raw_model), False, api_key


def get_model_name():
    """Get the appropriate model name with priority: ENV > Config > Default."""
    config, config_manager, provider, raw_model = _resolve_model_settings()
    model, _, _ = _compute_effective_model(config, config_manager, provider, raw_model)
    return model


def get_api_key():
    """Get the API key for the current provider with priority: ENV > Config."""
    config, config_manager, provider, _ = _resolve_model_settings()
    return _get_provider_api_key(config, config_manager, provider)


async def llm_completion(messages: list, model: Optional[str] = None) -> str:
    """
    Make an LLM completion call using the configured provider settings.

    This function reuses the same configuration as the main agent, ensuring
    consistent API key and model settings.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: Optional model override. If None, uses configured model.

    Returns:
        The completion response content as string
    """

    config, config_manager, provider, raw_model = _resolve_model_settings()

    # Ensure provider env vars are set (for litellm to pick up)
    _setup_provider_env_vars(config, provider)

    # Get model name and API key
    if model is None:
        model, _, api_key = _compute_effective_model(config, config_manager, provider, raw_model)
    else:
        api_key = _get_provider_api_key(config, config_manager, provider)

    # Get base URL if configured
    base_url_env_var = PROVIDER_ENV_VARS.get(provider, {}).get("base_url", "OPENAI_BASE_URL")
    base_url_config = config.model.base_url if config.model.provider.lower() == provider else None
    base_url = config_manager.get_effective_value(base_url_config, base_url_env_var)

    # Build kwargs for litellm
    kwargs = {
        "model": model,
        "messages": messages,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    response = await litellm.acompletion(**kwargs)
    return response.choices[0].message.content


def setup_openai_client():
    """Set up the OpenAI client with priority: ENV > Config > Default."""
    set_tracing_disabled(True)
    config, config_manager, provider, raw_model = _resolve_model_settings()

    # Setup provider environment variables for LiteLLM
    _setup_provider_env_vars(config, provider)

    model, use_native, api_key = _compute_effective_model(
        config, config_manager, provider, raw_model
    )

    # Get base URL with priority: ENV > Config
    base_url_env_var = PROVIDER_ENV_VARS.get(provider, {}).get("base_url", "OPENAI_BASE_URL")
    base_url_config = config.model.base_url if config.model.provider.lower() == provider else None
    base_url = config_manager.get_effective_value(base_url_config, base_url_env_var)

    # Use OpenAI native integration for openai/custom providers with API key
    if use_native:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        set_default_openai_client(client)
        return client

    # Fall back to LiteLLM integration for other providers
    return None
