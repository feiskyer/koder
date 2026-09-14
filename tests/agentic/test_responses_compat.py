"""Regressions at Koder's LiteLLM Responses provider boundary.

These tests run the actual Koder adapter, replacing only provider I/O. They do
not read configuration, use credentials, or claim to exercise a live model.
"""

import asyncio
from types import SimpleNamespace

import pytest
from agents import Agent, ModelBehaviorError, ModelSettings, handoff
from agents.extensions.models.litellm_model import LitellmModel
from agents.models.interface import ModelTracing
from openai import NotGiven, Omit

from koder_agent.agentic.agent import RetryingLitellmModel
from koder_agent.agentic.responses_compat import (
    close_provider_stream,
    present_request_value,
    responses_usage,
)
from koder_agent.harness.memory.budget import ContextPreflightError


def _model(*, context_window=100_000):
    return RetryingLitellmModel(
        model="github_copilot/gpt-5.1-codex",
        context_window=context_window,
    )


def _response(*, status="completed", usage=None):
    return {
        "id": "resp-test",
        "object": "response",
        "created_at": 0,
        "model": "test-model",
        "status": status,
        "output": [],
        "usage": usage,
        "parallel_tool_calls": False,
        "tools": [],
        "tool_choice": "auto",
        "error": None,
    }


def _patch_response(monkeypatch, payload):
    calls = []

    async def respond(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(**payload)

    monkeypatch.setattr("koder_agent.agentic.agent.litellm.aresponses", respond)
    return calls


async def _get_response(model, settings=None, handoffs=None):
    return await model.get_response(
        "system",
        "hello",
        settings or ModelSettings(max_tokens=20),
        [],
        None,
        handoffs or [],
        ModelTracing.DISABLED,
    )


@pytest.mark.parametrize("status", ["failed", "incomplete"])
def test_terminal_failure_is_not_returned_as_success(monkeypatch, status):
    payload = _response(status=status)
    payload["error"] = {"message": "private-provider-payload"}
    _patch_response(monkeypatch, payload)

    with pytest.raises(ModelBehaviorError, match=status) as caught:
        asyncio.run(_get_response(_model()))

    assert "private-provider-payload" not in str(caught.value)


def test_completed_request_is_counted_when_usage_is_missing(monkeypatch):
    _patch_response(monkeypatch, _response())

    result = asyncio.run(_get_response(_model()))

    assert result.usage.requests == 1
    assert result.usage.total_tokens == 0


def test_usage_preserves_cache_and_reasoning_details(monkeypatch):
    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=4,
        total_tokens=14,
        input_tokens_details=SimpleNamespace(cached_tokens=6, cache_write_tokens=2),
        output_tokens_details=SimpleNamespace(reasoning_tokens=3),
    )
    _patch_response(monkeypatch, _response(usage=usage))

    result = asyncio.run(_get_response(_model()))

    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 4
    assert result.usage.input_tokens_details.cached_tokens == 6
    assert result.usage.input_tokens_details.cache_write_tokens == 2
    assert result.usage.output_tokens_details.reasoning_tokens == 3


def test_store_false_is_forwarded_to_provider(monkeypatch):
    calls = _patch_response(monkeypatch, _response())

    asyncio.run(_get_response(_model(), ModelSettings(max_tokens=20, store=False)))

    assert calls[0]["store"] is False


def test_parallel_tool_calls_are_omitted_without_tools(monkeypatch):
    calls = _patch_response(monkeypatch, _response())

    asyncio.run(_get_response(_model(), ModelSettings(max_tokens=20, parallel_tool_calls=False)))

    assert calls[0].get("parallel_tool_calls") is None


def test_parallel_tool_calls_include_handoff_tools(monkeypatch):
    calls = _patch_response(monkeypatch, _response())
    target = handoff(Agent(name="Specialist", model="test-model"))

    asyncio.run(
        _get_response(
            _model(),
            ModelSettings(max_tokens=20, parallel_tool_calls=True),
            [target],
        )
    )

    assert calls[0]["parallel_tool_calls"] is True
    assert calls[0]["tools"]


def test_extension_payload_cannot_bypass_context_preflight(monkeypatch):
    calls = _patch_response(monkeypatch, _response())
    settings = ModelSettings(max_tokens=20, extra_body={"large_context": "x" * 100_000})

    with pytest.raises(ContextPreflightError):
        asyncio.run(_get_response(_model(context_window=200), settings))

    assert calls == []


def test_extra_args_cannot_replace_primary_input(monkeypatch):
    calls = _patch_response(monkeypatch, _response())
    settings = ModelSettings(
        max_tokens=20,
        extra_args={"input": "private-replacement-input"},
    )

    with pytest.raises(TypeError, match="multiple values for: input") as caught:
        asyncio.run(_get_response(_model(), settings))

    assert calls == []
    assert "private-replacement-input" not in str(caught.value)


def test_extra_args_output_reserve_is_checked_when_unset_on_settings(monkeypatch):
    calls = _patch_response(monkeypatch, _response())
    settings = ModelSettings(extra_args={"max_output_tokens": 1000})

    with pytest.raises(ContextPreflightError):
        asyncio.run(_get_response(_model(context_window=200), settings))

    assert calls == []


def test_request_id_is_preserved(monkeypatch):
    _patch_response(monkeypatch, {**_response(), "_request_id": "request-test"})

    result = asyncio.run(_get_response(_model()))

    assert result.request_id == "request-test"


@pytest.mark.parametrize("sentinel", [Omit(), NotGiven()])
def test_all_omission_sentinels_are_absent(sentinel):
    assert present_request_value(sentinel) is None
    assert present_request_value(False) is False
    assert present_request_value(0) == 0


def test_usage_accepts_mapping_details_with_missing_fields():
    usage = responses_usage(
        {
            "input_tokens": 12,
            "output_tokens": None,
            "input_tokens_details": {"cached_tokens": 4},
            "output_tokens_details": None,
        }
    )
    assert usage.requests == 1
    assert usage.input_tokens == 12
    assert usage.input_tokens_details.cached_tokens == 4
    assert usage.input_tokens_details.cache_write_tokens == 0
    assert usage.output_tokens_details.reasoning_tokens == 0


class _ProviderStream:
    def __init__(self, events, *, close_error=None):
        self.events = iter(events)
        self.close_error = close_error
        self.closed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.events)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self):
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


def _patch_stream(monkeypatch, stream):
    async def respond(**kwargs):
        assert kwargs["stream"] is True
        return stream

    monkeypatch.setattr("koder_agent.agentic.agent.litellm.aresponses", respond)


def _stream(model):
    return model.stream_response(
        "system",
        "hello",
        ModelSettings(max_tokens=20),
        [],
        None,
        [],
        ModelTracing.DISABLED,
    )


def test_provider_stream_closes_after_completion(monkeypatch):
    provider = _ProviderStream([{"type": "response.completed", "response": _response()}])
    _patch_stream(monkeypatch, provider)

    async def run():
        events = [event async for event in _stream(_model())]
        assert provider.closed == 1
        return events

    assert asyncio.run(run())[-1].type == "response.completed"


def test_provider_stream_closes_before_generator_close_returns(monkeypatch):
    provider = _ProviderStream([{"type": "response.output_text.delta", "delta": "partial"}])
    _patch_stream(monkeypatch, provider)

    async def run():
        stream = _stream(_model())
        await anext(stream)
        await stream.aclose()
        assert provider.closed == 1

    asyncio.run(run())


def test_chat_wrapper_closes_sdk_generator_in_same_task(monkeypatch):
    closed = False

    async def sdk_stream(*_args, **_kwargs):
        nonlocal closed
        try:
            yield "partial"
        finally:
            closed = True

    monkeypatch.setattr(LitellmModel, "stream_response", sdk_stream)
    model = RetryingLitellmModel(model="test-model", context_window=100_000)

    async def run():
        stream = _stream(model)
        await anext(stream)
        await stream.aclose()
        assert closed

    asyncio.run(run())


def test_cleanup_error_does_not_replace_completed_response(monkeypatch, caplog):
    provider = _ProviderStream(
        [{"type": "response.completed", "response": _response()}],
        close_error=RuntimeError("private-cleanup-payload"),
    )
    _patch_stream(monkeypatch, provider)

    async def run():
        events = [event async for event in _stream(_model())]
        assert provider.closed == 1
        return events

    assert asyncio.run(run())[-1].type == "response.completed"
    assert "private-cleanup-payload" not in caplog.text


@pytest.mark.parametrize("status", ["failed", "incomplete"])
def test_stream_terminal_failure_is_reported_and_closed(monkeypatch, status):
    provider = _ProviderStream(
        [{"type": f"response.{status}", "response": _response(status=status)}]
    )
    _patch_stream(monkeypatch, provider)

    async def run():
        with pytest.raises(ModelBehaviorError, match=status):
            _ = [event async for event in _stream(_model())]
        assert provider.closed == 1

    asyncio.run(run())


@pytest.mark.parametrize("event_type", ["error", "response.error"])
def test_error_events_never_become_successful_empty_turns(monkeypatch, event_type):
    provider = _ProviderStream(
        [{"type": event_type, "message": "private-provider-error", "code": "server_error"}]
    )
    _patch_stream(monkeypatch, provider)

    async def run():
        with pytest.raises(ModelBehaviorError, match="terminal event") as caught:
            _ = [event async for event in _stream(_model())]
        assert provider.closed == 1
        assert "private-provider-error" not in str(caught.value)

    asyncio.run(run())


def test_stream_without_terminal_event_fails_closed(monkeypatch):
    provider = _ProviderStream([{"type": "response.output_text.delta", "delta": "partial"}])
    _patch_stream(monkeypatch, provider)

    async def run():
        with pytest.raises(ModelBehaviorError, match="before a terminal"):
            _ = [event async for event in _stream(_model())]
        assert provider.closed == 1

    asyncio.run(run())


def test_provider_error_is_preserved_if_cleanup_also_fails(monkeypatch):
    class BrokenStream(_ProviderStream):
        async def __anext__(self):
            raise ValueError("provider read failed")

    provider = BrokenStream([], close_error=RuntimeError("private-cleanup-error"))
    _patch_stream(monkeypatch, provider)

    async def run():
        with pytest.raises(ValueError, match="provider read failed"):
            _ = [event async for event in _stream(_model())]
        assert provider.closed == 1

    asyncio.run(run())


def test_cancelled_provider_stream_is_closed(monkeypatch):
    entered = asyncio.Event()

    class BlockingStream(_ProviderStream):
        async def __anext__(self):
            entered.set()
            await asyncio.Event().wait()

    provider = BlockingStream([])
    _patch_stream(monkeypatch, provider)

    async def consume():
        return [event async for event in _stream(_model())]

    async def run():
        task = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.closed == 1

    asyncio.run(run())


def test_second_cancel_does_not_abandon_inflight_transport_close(monkeypatch):
    reading = asyncio.Event()
    closing = asyncio.Event()
    release_close = asyncio.Event()
    closed = asyncio.Event()

    class SlowClosingStream(_ProviderStream):
        async def __anext__(self):
            reading.set()
            await asyncio.Event().wait()

        async def aclose(self):
            self.closed += 1
            closing.set()
            await release_close.wait()
            closed.set()

    provider = SlowClosingStream([])
    _patch_stream(monkeypatch, provider)

    async def consume():
        return [event async for event in _stream(_model())]

    async def run():
        task = asyncio.create_task(consume())
        await asyncio.wait_for(reading.wait(), timeout=1)
        task.cancel()
        await asyncio.wait_for(closing.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release_close.set()
        await asyncio.wait_for(closed.wait(), timeout=1)
        assert provider.closed == 1

    asyncio.run(run())


@pytest.mark.parametrize("async_close", [False, True])
def test_provider_cleanup_accepts_close_only_transports(async_close):
    calls = []

    def close():
        calls.append("sync")

    async def aclose():
        calls.append("async")

    stream = SimpleNamespace(close=aclose if async_close else close)
    asyncio.run(close_provider_stream(stream))
    assert calls == ["async" if async_close else "sync"]
