"""Reading a model label must not refresh unrelated provider credentials."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from koder_agent.core.usage_tracker import UsageTracker
from koder_agent.utils import client


@pytest.mark.parametrize("provider", ["anthropic", "google", "chatgpt", "github_copilot"])
@pytest.mark.parametrize("consumer", ["status", "override"])
def test_nonnative_identity_lookup_does_not_acquire_credentials(monkeypatch, provider, consumer):
    config = object()
    monkeypatch.setattr(
        client,
        "_resolve_completion_settings",
        lambda *_args: (config, None, provider, "synthetic-model", False),
    )
    credential_lookup = Mock(side_effect=AssertionError("model label attempted credential refresh"))
    monkeypatch.setattr(client, "_get_provider_api_key", credential_lookup)
    monkeypatch.setattr(
        client,
        "_normalize_model_name",
        lambda selected, model, _from_env: f"litellm/{selected}/{model}",
    )

    actual = (
        UsageTracker().model
        if consumer == "status"
        else client.resolve_model_override_name("synthetic-model")
    )
    assert actual == f"litellm/{provider}/synthetic-model"
    credential_lookup.assert_not_called()


def test_completion_routing_still_acquires_nonnative_credentials(monkeypatch):
    config = object()
    credential_lookup = Mock(return_value="synthetic-credential")
    monkeypatch.setattr(client, "_get_provider_api_key", credential_lookup)
    monkeypatch.setattr(
        client, "_normalize_model_name", lambda *_args: "litellm/anthropic/synthetic-model"
    )
    model, native, key = client._compute_effective_model(config, "anthropic", "synthetic-model")
    assert model == "litellm/anthropic/synthetic-model"
    assert native is False
    assert key == "synthetic-credential"
    credential_lookup.assert_called_once_with(config, "anthropic")


@pytest.mark.parametrize("provider", ["openai", "custom"])
@pytest.mark.parametrize("key", [None, "", "synthetic-valid-key"])
def test_native_client_setup_requires_a_nonempty_credential(monkeypatch, provider, key):
    config = SimpleNamespace(model=SimpleNamespace(provider=provider, api_key=None))
    monkeypatch.setenv("KODER_API_KEY", "")
    if key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", key)
    monkeypatch.setattr(
        "koder_agent.auth.client_integration.get_oauth_api_key", lambda _provider: None
    )
    monkeypatch.setattr(client, "_should_use_oauth_provider", lambda _provider: False)
    monkeypatch.setattr(
        client, "_resolve_model_settings", lambda: (config, None, provider, "gpt-4.1", False)
    )
    monkeypatch.setattr(client, "_resolve_base_url", lambda *_args: "http://127.0.0.1:1/v1")
    constructed = Mock()
    registered = Mock()
    monkeypatch.setattr(client, "AsyncOpenAI", constructed)
    monkeypatch.setattr(client, "set_default_openai_client", registered)
    monkeypatch.setattr(client, "set_tracing_disabled", Mock())

    result = client.setup_openai_client()

    if key:
        constructed.assert_called_once_with(
            api_key=key, base_url="http://127.0.0.1:1/v1", max_retries=3
        )
        assert result is constructed.return_value
        registered.assert_called_once_with(result)
    else:
        constructed.assert_not_called()
        registered.assert_not_called()
        assert result is None
