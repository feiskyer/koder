from pathlib import Path

import pytest
import yaml

from koder_agent.config import reset_config_manager
from koder_agent.config.manager import ConfigManager
from koder_agent.utils.client import (
    get_api_key,
    get_base_url,
    get_litellm_model_kwargs,
    get_model_name,
    is_native_openai_provider,
)


def _write_config(tmp_path, data: dict) -> None:
    """Write a config file under the temp HOME used for tests."""
    config_dir = tmp_path / ".koder"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch, tmp_path):
    """
    Isolate HOME and clear relevant env vars between tests.

    The client code reads ~/.koder/config.yaml via Path.home(), so we
    point HOME to a temp directory and reset the config manager cache.
    """
    # Redirect config location to temp HOME
    config_path = Path(tmp_path) / ".koder" / "config.yaml"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(ConfigManager, "DEFAULT_CONFIG_PATH", config_path)

    for var in [
        "KODER_MODEL",
        "KODER_API_KEY",
        "KODER_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "AZURE_API_KEY",
        "AZURE_API_BASE",
        "AZURE_API_VERSION",
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)
    reset_config_manager()
    yield
    reset_config_manager()


def test_env_model_provider_overrides_config(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-4.1", "provider": "openai"}},
    )
    monkeypatch.setenv("KODER_MODEL", "openrouter/x-ai/grok-4.1-fast:free")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    # Env-supplied provider should be used and normalized for LiteLLM
    assert get_model_name() == "litellm/openrouter/x-ai/grok-4.1-fast:free"
    kwargs = get_litellm_model_kwargs()
    assert kwargs["model"] == "openrouter/x-ai/grok-4.1-fast:free"


def test_openai_native_model_uses_raw(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-4.1", "provider": "openai"}},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    assert get_model_name() == "gpt-4.1"
    assert is_native_openai_provider()


def test_openai_provider_non_openai_model_uses_litellm(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        {"model": {"name": "x-ai/grok-4.1-fast:free", "provider": "openai"}},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    model = get_model_name()
    assert model == "litellm/openai/x-ai/grok-4.1-fast:free"
    assert not is_native_openai_provider()
    kwargs = get_litellm_model_kwargs()
    assert kwargs["model"] == "openai/x-ai/grok-4.1-fast:free"


def test_azure_provider_uses_litellm_and_base_url(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-4o-mini", "provider": "azure"}},
    )
    monkeypatch.setenv("AZURE_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_API_BASE", "https://example.azure.com")
    monkeypatch.setenv("AZURE_API_VERSION", "2025-04-01-preview")

    # Azure should always go through LiteLLM path
    assert get_model_name() == "litellm/azure/gpt-4o-mini"
    kwargs = get_litellm_model_kwargs()
    assert kwargs["model"] == "azure/gpt-4o-mini"
    assert kwargs["base_url"] == "https://example.azure.com"
    assert kwargs["api_key"] == "azure-key"


def test_openrouter_config_path(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        {"model": {"name": "anthropic/claude-3-opus", "provider": "openrouter"}},
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter")

    assert get_model_name() == "litellm/openrouter/anthropic/claude-3-opus"
    kwargs = get_litellm_model_kwargs()
    assert kwargs["model"] == "openrouter/anthropic/claude-3-opus"
    assert kwargs["api_key"] == "sk-openrouter"


def test_env_openai_model_overrides_non_openai_config(monkeypatch, tmp_path):
    # Config says azure, but KODER_MODEL supplies an OpenAI-native model with provider prefix
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-4o-mini", "provider": "azure"}},
    )
    monkeypatch.setenv("KODER_MODEL", "openai/gpt-4o")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    assert get_model_name() == "gpt-4o"
    assert is_native_openai_provider()


# =============================================================================
# KODER_BASE_URL Tests
# =============================================================================


def test_koder_base_url_overrides_config_base_url(monkeypatch, tmp_path):
    """KODER_BASE_URL env var should override base_url from config file."""
    _write_config(
        tmp_path,
        {
            "model": {
                "name": "gpt-4o",
                "provider": "openai",
                "base_url": "http://config-file.local/v1",
            }
        },
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("KODER_BASE_URL", "http://koder-override.local/v1")

    assert get_base_url() == "http://koder-override.local/v1"
    kwargs = get_litellm_model_kwargs()
    assert kwargs["base_url"] == "http://koder-override.local/v1"


def test_koder_base_url_overrides_provider_env_var(monkeypatch, tmp_path):
    """KODER_BASE_URL should take priority over provider-specific env vars like OPENAI_BASE_URL."""
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-4o", "provider": "openai"}},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai-specific.local/v1")
    monkeypatch.setenv("KODER_BASE_URL", "http://koder-override.local/v1")

    assert get_base_url() == "http://koder-override.local/v1"
    kwargs = get_litellm_model_kwargs()
    assert kwargs["base_url"] == "http://koder-override.local/v1"


def test_provider_env_base_url_works_without_koder_base_url(monkeypatch, tmp_path):
    """Provider-specific env var (OPENAI_BASE_URL) should work when KODER_BASE_URL is not set."""
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-4o", "provider": "openai"}},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai-specific.local/v1")

    assert get_base_url() == "http://openai-specific.local/v1"
    kwargs = get_litellm_model_kwargs()
    assert kwargs["base_url"] == "http://openai-specific.local/v1"


def test_config_base_url_works_without_env_vars(monkeypatch, tmp_path):
    """Config file base_url should work when no env vars are set."""
    _write_config(
        tmp_path,
        {
            "model": {
                "name": "gpt-4o",
                "provider": "openai",
                "base_url": "http://config-file.local/v1",
            }
        },
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    assert get_base_url() == "http://config-file.local/v1"
    kwargs = get_litellm_model_kwargs()
    assert kwargs["base_url"] == "http://config-file.local/v1"


def test_provider_env_base_url_overrides_config(monkeypatch, tmp_path):
    """Provider env var should override config file base_url (existing behavior)."""
    _write_config(
        tmp_path,
        {
            "model": {
                "name": "gpt-4o",
                "provider": "openai",
                "base_url": "http://config-file.local/v1",
            }
        },
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://openai-specific.local/v1")

    assert get_base_url() == "http://openai-specific.local/v1"
    kwargs = get_litellm_model_kwargs()
    assert kwargs["base_url"] == "http://openai-specific.local/v1"


def test_azure_base_url_with_koder_override(monkeypatch, tmp_path):
    """KODER_BASE_URL should override Azure provider base_url."""
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-4o-mini", "provider": "azure"}},
    )
    monkeypatch.setenv("AZURE_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_API_BASE", "https://azure.example.com")
    monkeypatch.setenv("AZURE_API_VERSION", "2025-04-01-preview")
    monkeypatch.setenv("KODER_BASE_URL", "http://koder-override.local/v1")

    assert get_base_url() == "http://koder-override.local/v1"
    kwargs = get_litellm_model_kwargs()
    assert kwargs["base_url"] == "http://koder-override.local/v1"


def test_no_base_url_returns_none(monkeypatch, tmp_path):
    """When no base_url is configured anywhere, get_base_url should return None."""
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-4o", "provider": "openai"}},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    assert get_base_url() is None
    kwargs = get_litellm_model_kwargs()
    assert kwargs["base_url"] is None


# =============================================================================
# KODER_API_KEY Tests
# =============================================================================


def test_koder_api_key_overrides_config_api_key(monkeypatch, tmp_path):
    """KODER_API_KEY env var should override api_key from config file."""
    _write_config(
        tmp_path,
        {
            "model": {
                "name": "gpt-4o",
                "provider": "openai",
                "api_key": "sk-from-config",
            }
        },
    )
    monkeypatch.setenv("KODER_API_KEY", "sk-koder-override")

    assert get_api_key() == "sk-koder-override"
    kwargs = get_litellm_model_kwargs()
    assert kwargs["api_key"] == "sk-koder-override"


def test_koder_api_key_overrides_provider_env_var(monkeypatch, tmp_path):
    """KODER_API_KEY should take priority over provider-specific env vars like OPENAI_API_KEY."""
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-4o", "provider": "openai"}},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-specific")
    monkeypatch.setenv("KODER_API_KEY", "sk-koder-override")

    assert get_api_key() == "sk-koder-override"
    kwargs = get_litellm_model_kwargs()
    assert kwargs["api_key"] == "sk-koder-override"


def test_koder_api_key_works_across_providers(monkeypatch, tmp_path):
    """KODER_API_KEY should work regardless of provider (OpenAI, Anthropic, etc.)."""
    # Test with Anthropic provider
    _write_config(
        tmp_path,
        {"model": {"name": "claude-3-opus", "provider": "anthropic"}},
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic-specific")
    monkeypatch.setenv("KODER_API_KEY", "sk-koder-universal")

    assert get_api_key() == "sk-koder-universal"
    kwargs = get_litellm_model_kwargs()
    assert kwargs["api_key"] == "sk-koder-universal"


def test_provider_env_api_key_works_without_koder_api_key(monkeypatch, tmp_path):
    """Provider-specific env var (OPENAI_API_KEY) should work when KODER_API_KEY is not set."""
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-4o", "provider": "openai"}},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-specific")

    assert get_api_key() == "sk-openai-specific"
    kwargs = get_litellm_model_kwargs()
    assert kwargs["api_key"] == "sk-openai-specific"


def test_config_api_key_works_without_env_vars(monkeypatch, tmp_path):
    """Config file api_key should work when no env vars are set."""
    _write_config(
        tmp_path,
        {
            "model": {
                "name": "gpt-4o",
                "provider": "openai",
                "api_key": "sk-from-config",
            }
        },
    )

    assert get_api_key() == "sk-from-config"
    kwargs = get_litellm_model_kwargs()
    assert kwargs["api_key"] == "sk-from-config"


def test_provider_env_api_key_overrides_config(monkeypatch, tmp_path):
    """Provider env var should override config file api_key (existing behavior)."""
    _write_config(
        tmp_path,
        {
            "model": {
                "name": "gpt-4o",
                "provider": "openai",
                "api_key": "sk-from-config",
            }
        },
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-specific")

    assert get_api_key() == "sk-openai-specific"
    kwargs = get_litellm_model_kwargs()
    assert kwargs["api_key"] == "sk-openai-specific"


def test_azure_api_key_with_koder_override(monkeypatch, tmp_path):
    """KODER_API_KEY should override Azure provider api_key."""
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-4o-mini", "provider": "azure"}},
    )
    monkeypatch.setenv("AZURE_API_KEY", "azure-key")
    monkeypatch.setenv("AZURE_API_BASE", "https://azure.example.com")
    monkeypatch.setenv("AZURE_API_VERSION", "2025-04-01-preview")
    monkeypatch.setenv("KODER_API_KEY", "sk-koder-override")

    assert get_api_key() == "sk-koder-override"
    kwargs = get_litellm_model_kwargs()
    assert kwargs["api_key"] == "sk-koder-override"


# =============================================================================
# OAuth vs API Provider Separation Tests
# =============================================================================


def test_api_providers_do_not_map_to_oauth():
    """API-based providers should NOT map to OAuth providers."""
    from koder_agent.auth.client_integration import map_provider_to_oauth

    assert map_provider_to_oauth("anthropic") is None
    assert map_provider_to_oauth("openai") is None
    assert map_provider_to_oauth("gemini") is None
    assert map_provider_to_oauth("azure") is None


def test_oauth_providers_map_to_themselves():
    """OAuth providers should map to themselves."""
    from koder_agent.auth.client_integration import map_provider_to_oauth

    assert map_provider_to_oauth("claude") == "claude"
    assert map_provider_to_oauth("chatgpt") == "chatgpt"
    assert map_provider_to_oauth("google") == "google"
    assert map_provider_to_oauth("antigravity") == "antigravity"
