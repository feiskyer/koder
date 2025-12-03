from pathlib import Path

import pytest
import yaml

from koder_agent.config import reset_config_manager
from koder_agent.config.manager import ConfigManager
from koder_agent.utils.client import (
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
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "AZURE_API_KEY",
        "AZURE_API_BASE",
        "AZURE_API_VERSION",
        "OPENROUTER_API_KEY",
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
