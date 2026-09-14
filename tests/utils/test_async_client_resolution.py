"""Auxiliary model resolution must not block the running event loop."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from koder_agent.auth import oauth_routing
from koder_agent.utils import client


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["settings", "credentials"])
async def test_auxiliary_credentials_are_resolved_off_loop(
    monkeypatch, event_loop_progress_probe, phase
):
    wrap, observed = event_loop_progress_probe

    def settings(_model):
        return None, None, "openai", "fixture", False

    def credentials(*_args):
        return "openai/fixture", False, "synthetic-api-key"

    monkeypatch.setattr(
        client, "_resolve_completion_settings", wrap(settings) if phase == "settings" else settings
    )
    monkeypatch.setattr(client, "_setup_provider_env_vars", lambda *_args: None)
    monkeypatch.setattr(
        client,
        "_compute_effective_model",
        wrap(credentials) if phase == "credentials" else credentials,
    )
    monkeypatch.setattr(client, "get_configured_context_window", lambda *_a, **_kw: 4096)
    monkeypatch.setattr(client, "get_maximum_output_tokens", lambda *_a, **_kw: 256)
    monkeypatch.setattr(
        client, "estimate_model_request_preflight", lambda **_kwargs: SimpleNamespace(fits=True)
    )
    monkeypatch.setattr(client, "_resolve_base_url", lambda *_args: None)
    request = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="fixture response"))]
        )
    )
    monkeypatch.setattr(oauth_routing, "acompletion", request)

    result = await client.llm_completion(
        [{"role": "user", "content": "fixture"}], model="openai/fixture"
    )

    assert result == "fixture response"
    assert observed == [True], "auxiliary credential resolution blocked the event loop"
    assert request.await_args.kwargs["api_key"] == "synthetic-api-key"
