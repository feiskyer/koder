from pathlib import Path

import pytest
import yaml

from koder_agent.config import reset_config_manager
from koder_agent.config.manager import ConfigManager
from koder_agent.providers.compat import ProviderCompat


def _write_config(tmp_path, data: dict) -> None:
    config_dir = tmp_path / ".koder"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture(autouse=True)
def isolate_provider_compat(monkeypatch, tmp_path):
    config_path = Path(tmp_path) / ".koder" / "config.yaml"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(ConfigManager, "DEFAULT_CONFIG_PATH", config_path)
    for var in [
        "KODER_MODEL",
        "KODER_API_KEY",
        "KODER_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)
    reset_config_manager()
    yield
    reset_config_manager()


def test_provider_compat_exposes_existing_oauth_providers():
    compat = ProviderCompat.from_current_runtime()
    assert {"google", "claude", "chatgpt", "antigravity"} <= set(compat.oauth_providers)


def test_provider_compat_can_resolve_model_client_without_harness_runtime(monkeypatch, tmp_path):
    _write_config(tmp_path, {"model": {"name": "gpt-4.1", "provider": "openai"}})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    compat = ProviderCompat.from_current_runtime()
    resolved = compat.resolve_model_client()
    assert resolved.model_name == "gpt-4.1"
    assert resolved.api_key == "sk-test"
    # LiteLLM kwargs retain the canonical provider-qualified identity; the
    # separate native model_name field remains unqualified.
    assert resolved.litellm_kwargs["model"] == "openai/gpt-4.1"


@pytest.mark.parametrize(
    ("provider", "model", "override"),
    [
        ("openai", "gpt-4.1", "anthropic/claude-opus-4-1"),
        ("anthropic", "claude-opus-4-1", "openai/gpt-4.1"),
        ("anthropic", "claude-opus-4-1", "anthropic/claude-sonnet-4-6"),
    ],
)
def test_override_uses_one_canonical_provider_snapshot(
    monkeypatch, tmp_path, provider, model, override
):
    from koder_agent.utils.client import get_model_client_snapshot

    _write_config(tmp_path, {"model": {"name": model, "provider": provider}})
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-anthropic-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example.invalid")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example.invalid")
    monkeypatch.setattr("koder_agent.auth.client_integration.get_oauth_api_key", lambda _: None)
    monkeypatch.setattr("koder_agent.auth.client_integration.has_oauth_token", lambda _: False)

    expected = get_model_client_snapshot(override)
    actual = ProviderCompat().resolve_model_client(override)
    for field in ("model_name", "api_key", "base_url", "native_openai", "litellm_kwargs"):
        assert getattr(actual, field) == expected[field], field
