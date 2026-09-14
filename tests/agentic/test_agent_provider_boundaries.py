"""Controlled provider I/O at the real Koder adapter; no live model requests."""

import asyncio
from types import SimpleNamespace

import litellm.exceptions
import pytest
from agents import ModelSettings
from agents.models.interface import ModelTracing

from koder_agent.agentic.agent import RetryingLitellmModel, create_dev_agent
from koder_agent.harness.config.schema import RuntimeConfig


def _model():
    return RetryingLitellmModel(
        model="github_copilot/gpt-5.1-codex",
        context_window=100_000,
    )


def _call(model, *, streaming):
    method = model.stream_response if streaming else model.get_response
    return method(
        "system",
        "hello",
        ModelSettings(max_tokens=20),
        [],
        None,
        [],
        ModelTracing.DISABLED,
    )


def _transient_error():
    return litellm.exceptions.InternalServerError(
        message="synthetic transient failure",
        model="test-model",
        llm_provider="test-provider",
    )


async def _no_sleep(_seconds):
    return None


@pytest.mark.parametrize(("streaming", "expected_attempts"), [(False, 3), (True, 5)])
def test_responses_retry_limit_before_provider_output(monkeypatch, streaming, expected_attempts):
    attempts = 0

    async def fail(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise _transient_error()

    monkeypatch.setattr("koder_agent.agentic.agent.litellm.aresponses", fail)
    monkeypatch.setattr("koder_agent.agentic.agent.asyncio.sleep", _no_sleep)

    async def run():
        with pytest.raises(litellm.exceptions.InternalServerError):
            if streaming:
                _ = [event async for event in _call(_model(), streaming=True)]
            else:
                await _call(_model(), streaming=False)

    asyncio.run(run())
    assert attempts == expected_attempts


def test_cancelling_responses_retry_wait_prevents_next_attempt(monkeypatch):
    attempts = 0
    waiting = asyncio.Event()

    async def fail(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise _transient_error()

    async def retry_wait(_seconds):
        waiting.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("koder_agent.agentic.agent.litellm.aresponses", fail)
    monkeypatch.setattr("koder_agent.agentic.agent.asyncio.sleep", retry_wait)

    async def consume():
        return [event async for event in _call(_model(), streaming=True)]

    async def run():
        task = asyncio.create_task(consume())
        await asyncio.wait_for(waiting.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert attempts == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_cancellation_during_responses_fetch_is_not_retried(monkeypatch, streaming):
    attempts = 0
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    async def fetch(**_kwargs):
        nonlocal attempts
        attempts += 1
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    monkeypatch.setattr("koder_agent.agentic.agent.litellm.aresponses", fetch)

    async def consume():
        if streaming:
            return [event async for event in _call(_model(), streaming=True)]
        return await _call(_model(), streaming=False)

    async def run():
        task = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set()

    asyncio.run(run())
    assert attempts == 1


class _Stream:
    def __init__(self, events, error=None):
        self.events = iter(events)
        self.error = error
        self.closed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.events)
        except StopIteration:
            if self.error is not None:
                raise self.error
            raise StopAsyncIteration from None

    async def aclose(self):
        self.closed += 1


def test_responses_failure_after_delta_never_replays_and_closes_transport(monkeypatch):
    attempts = 0
    provider = _Stream(
        [{"type": "response.output_text.delta", "delta": "partial"}],
        error=_transient_error(),
    )

    async def fetch(**_kwargs):
        nonlocal attempts
        attempts += 1
        return provider

    monkeypatch.setattr("koder_agent.agentic.agent.litellm.aresponses", fetch)
    monkeypatch.setattr("koder_agent.agentic.agent.asyncio.sleep", _no_sleep)

    async def run():
        events = []
        with pytest.raises(litellm.exceptions.InternalServerError):
            async for event in _call(_model(), streaming=True):
                events.append(event)
        assert len(events) == 1
        assert events[0].delta == "partial"
        assert provider.closed == 1

    asyncio.run(run())
    assert attempts == 1


def test_completed_stream_preserves_response_id_request_id_and_usage(monkeypatch):
    provider = _Stream(
        [
            {
                "type": "response.completed",
                "response": {
                    "id": "resp-synthetic",
                    "_request_id": "request-synthetic",
                    "status": "completed",
                    "output": [],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "total_tokens": 14,
                        "input_tokens_details": {"cached_tokens": 6, "cache_write_tokens": 2},
                        "output_tokens_details": {"reasoning_tokens": 3},
                    },
                },
            }
        ]
    )

    async def fetch(**_kwargs):
        return provider

    monkeypatch.setattr("koder_agent.agentic.agent.litellm.aresponses", fetch)

    async def run():
        stream = _call(_model(), streaming=True)
        event = await anext(stream)
        await stream.aclose()
        assert provider.closed == 1
        return event.response

    response = asyncio.run(run())
    assert response.id == "resp-synthetic"
    assert response._request_id == "request-synthetic"
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 4
    assert response.usage.total_tokens == 14
    assert response.usage.input_tokens_details.cached_tokens == 6
    assert response.usage.input_tokens_details.cache_write_tokens == 2
    assert response.usage.output_tokens_details.reasoning_tokens == 3


@pytest.mark.parametrize("override", [None, "inherit", "openai/gpt-4.1"])
def test_native_model_override_uses_its_resolved_client(
    monkeypatch, override, event_loop_progress_probe
):
    default_client = SimpleNamespace(
        api_key="synthetic-default-key",
        base_url="https://default.example.invalid/v1",
    )
    resolved_key = (
        "synthetic-override-key" if override == "openai/gpt-4.1" else default_client.api_key
    )
    resolved_url = (
        "https://override.example.invalid/v1"
        if override == "openai/gpt-4.1"
        else default_client.base_url
    )
    constructed = []

    def make_client(**kwargs):
        constructed.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setenv("KODER_SIMPLE", "1")
    monkeypatch.setattr("koder_agent.agentic.agent.get_config", RuntimeConfig)
    monkeypatch.setattr(
        "koder_agent.agentic.agent.get_default_openai_client", lambda: default_client
    )
    monkeypatch.setattr("koder_agent.agentic.agent.AsyncOpenAI", make_client)

    def snapshot(_override):
        return {
            "model_name": "gpt-4.1",
            "native_openai": True,
            "api_key": resolved_key,
            "base_url": resolved_url,
            "context_window": 100_000,
            "max_output_tokens": 20,
        }

    monkeypatch.setattr("koder_agent.agentic.agent._get_skills_metadata", lambda _config: "")
    monkeypatch.setattr("koder_agent.agentic.agent._get_agents_metadata", lambda: "")
    monkeypatch.setattr("koder_agent.agentic.agent._get_environment_info", lambda _model: "")
    monkeypatch.setattr("koder_agent.agentic.agent.should_use_reasoning_param", lambda: False)

    wrap, observed = event_loop_progress_probe

    async def build():
        monkeypatch.setattr("koder_agent.agentic.agent.get_model_client_snapshot", wrap(snapshot))
        return await create_dev_agent([], model_override=override, instructions_override="hello")

    agent = asyncio.run(build())

    assert agent.model._client.api_key == resolved_key
    assert agent.model._client.base_url == resolved_url
    assert len(constructed) == (1 if override == "openai/gpt-4.1" else 0)
    assert default_client.api_key == "synthetic-default-key"
    assert observed == [True], "credential snapshot acquisition blocked agent execution"
