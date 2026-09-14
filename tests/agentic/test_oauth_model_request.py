"""Compare the OAuth adapter with the installed SDK without any provider I/O."""

import asyncio
import sys
from copy import deepcopy
from importlib import import_module
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents import AgentOutputSchema, FunctionTool, Handoff, ModelSettings
from agents.extensions.models import litellm_model as sdk
from agents.models._retry_runtime import provider_managed_retries_disabled
from agents.models.chatcmpl_helpers import HEADERS_OVERRIDE
from agents.models.interface import ModelTracing
from agents.tracing.span_data import GenerationSpanData
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import Response
from openai.types.shared import Reasoning
from pydantic import BaseModel

PUBLIC_MODELS = [
    "google/gemini-2.5-pro",
    "claude/claude-sonnet-4",
    "chatgpt/gpt-5",
    "antigravity/gemini-2.5-pro",
]


class StructuredAnswer(BaseModel):
    answer: str


class CloseableStream:
    """A transport-owned stream; the adapter must return it without wrapping."""

    def __init__(self):
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


@pytest.fixture
def routing(monkeypatch):
    # The main worker owns the real routing module. Never invoke it or inspect
    # credentials here, even when it becomes importable during integration.
    # Load model modules before injecting the stub so unrelated module-level
    # imports cannot accidentally retain a test routing function.
    import_module("koder_agent.agentic.oauth_model")
    module = ModuleType("koder_agent.auth.oauth_routing")
    module.is_oauth_model = lambda model: (
        model.removeprefix("litellm/").split("/", 1)[0]
        in {
            "google",
            "claude",
            "chatgpt",
            "antigravity",
        }
    )
    module.acompletion = AsyncMock()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def adapter(model="chatgpt/gpt-5", **kwargs):
    cls = import_module("koder_agent.agentic.oauth_model").OAuthAwareLitellmModel
    return cls(model=model, **kwargs)


def response():
    return sdk.litellm.ModelResponse(
        id="synthetic-response",
        model="transport-private-model",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "synthetic answer"},
            }
        ],
        usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    )


def span():
    return SimpleNamespace(span_data=GenerationSpanData(model="synthetic"))


def request(**overrides):
    values = dict(
        system_instructions="Be precise.",
        input="Hello",
        model_settings=ModelSettings(),
        tools=[],
        output_schema=None,
        handoffs=[],
        span=span(),
        tracing=ModelTracing.ENABLED,
        stream=False,
        prompt={"id": "unused-by-pinned-chat-adapter"},
    )
    values.update(overrides)
    return values


async def never_invoke(*args, **kwargs):
    raise AssertionError("Conversion must not invoke tools or handoffs")


def tools_and_handoffs():
    tool = FunctionTool(
        name="lookup",
        description="Look up a fixture",
        params_json_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        on_invoke_tool=never_invoke,
    )
    handoff = Handoff(
        tool_name="transfer_fixture",
        tool_description="Transfer to a fixture",
        input_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
        on_invoke_handoff=never_invoke,
        agent_name="fixture",
    )
    return [tool], [handoff]


def history(model):
    return [
        {"role": "user", "content": "Look up a fixture"},
        {
            "type": "reasoning",
            "id": "reasoning_fixture",
            "summary": [{"type": "summary_text", "text": "fixture reasoning"}],
            "content": [{"type": "reasoning_text", "text": "fixture thinking"}],
            "encrypted_content": "fixture-signature",
            "provider_data": {"model": model},
        },
        {
            "type": "function_call",
            "id": "fc_fixture",
            "call_id": "call_fixture",
            "name": "lookup",
            "arguments": '{"key":"fixture"}',
            "provider_data": {"thought_signature": "fixture-thought-signature"},
        },
        {
            "type": "function_call_output",
            "call_id": "call_fixture",
            "output": [
                {"type": "input_text", "text": "fixture result"},
                {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
            ],
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", PUBLIC_MODELS + ["litellm/" + name for name in PUBLIC_MODELS])
@pytest.mark.parametrize("stream", [False, True])
async def test_oauth_request_matches_sdk_kwargs(monkeypatch, routing, model, stream):
    settings = ModelSettings(
        temperature=0.2,
        top_p=0.8,
        frequency_penalty=0.3,
        presence_penalty=0.4,
        max_tokens=1234,
        parallel_tool_calls=True,
        tool_choice="lookup",
        include_usage=True,
        top_logprobs=3,
        reasoning=Reasoning(effort="high"),
        extra_headers={"X-Fixture": "header", "User-Agent": "fixture-agent"},
        extra_body={"reasoning_effort": "low", "fixture_body": {"nested": [1]}},
        extra_query={"fixture_query": "q"},
        metadata={"fixture_metadata": "m"},
        extra_args={"reasoning_effort": "medium", "seed": 9, "optional_fixture": None},
    )
    before_settings = deepcopy(settings)
    items = history(model)
    before_items = deepcopy(items)
    tools, handoffs = tools_and_handoffs()
    result = CloseableStream() if stream else response()
    raw_capture = AsyncMock(return_value=result)
    monkeypatch.setattr(sdk.litellm, "acompletion", raw_capture)
    routing.acompletion.return_value = result
    native_span, oauth_span = span(), span()
    args = request(
        input=items,
        model_settings=settings,
        tools=tools,
        handoffs=handoffs,
        output_schema=AgentOutputSchema(StructuredAnswer),
        stream=stream,
    )
    native = sdk.LitellmModel(model, base_url="https://fixture.invalid/v1", api_key="fixture-key")
    oauth = adapter(model, base_url=native.base_url, api_key=native.api_key)
    token = HEADERS_OVERRIDE.set({"X-Fixture": "overridden"})
    try:
        expected = await native._fetch_response(**{**args, "span": native_span})
        actual = await oauth._fetch_response(**{**args, "span": oauth_span})
    finally:
        HEADERS_OVERRIDE.reset(token)

    assert raw_capture.await_count == 1  # OAuth must never hit native dispatch.
    assert routing.acompletion.await_count == 1
    captured = routing.acompletion.await_args.kwargs
    assert captured == raw_capture.await_args.kwargs
    assert captured["model"] == oauth.model == model
    assert captured["reasoning_effort"] == "high"
    assert captured["extra_body"] == {"fixture_body": {"nested": [1]}}
    assert captured["extra_headers"]["X-Fixture"] == "overridden"
    assert len(captured["tools"]) == 2
    assert captured["messages"][0] == {"role": "system", "content": "Be precise."}
    call_message = next(message for message in captured["messages"] if message.get("tool_calls"))
    if "gemini" in model:
        assert call_message["tool_calls"][0]["provider_specific_fields"] == {
            "thought_signature": "fixture-thought-signature"
        }
    if "claude" in model:
        assert call_message["content"] == [
            {"type": "thinking", "thinking": "fixture thinking", "signature": "fixture-signature"}
        ]
    assert oauth_span.span_data.input == native_span.span_data.input == captured["messages"]
    assert settings == before_settings
    assert items == before_items
    if stream:
        stub, returned_stream = actual
        assert isinstance(stub, Response)
        assert returned_stream is result
        assert stub.model == model
        assert stub.model_dump(exclude={"created_at"}) == expected[0].model_dump(
            exclude={"created_at"}
        )
        await returned_stream.aclose()
        assert result.closed
    else:
        assert actual is expected is result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings",
    [
        ModelSettings(),
        ModelSettings(reasoning=Reasoning(effort="low")),
        ModelSettings(extra_body={"reasoning_effort": "none"}),
        ModelSettings(extra_args={"reasoning_effort": {"budget": 200}}),
        ModelSettings(
            extra_body={"reasoning_effort": "low"}, extra_args={"reasoning_effort": "high"}
        ),
        ModelSettings(extra_body={"reasoning_effort": None}),
        ModelSettings(extra_query={}, metadata={}, extra_args={"seed": None}),
        ModelSettings(extra_body={"fixture": "body"}, extra_args={"extra_body": {"override": 1}}),
        ModelSettings(parallel_tool_calls=False, include_usage=False, tool_choice="none"),
        ModelSettings(parallel_tool_calls=True, include_usage=True, tool_choice="required"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
async def test_default_and_escape_hatch_parity(monkeypatch, routing, settings, stream):
    raw = AsyncMock(return_value=response())
    monkeypatch.setattr(sdk.litellm, "acompletion", raw)
    routing.acompletion.return_value = response()
    before = deepcopy(settings)
    args = request(model_settings=settings, stream=stream, system_instructions=None)
    await sdk.LitellmModel("chatgpt/gpt-5")._fetch_response(**args)
    await adapter()._fetch_response(**args)
    assert routing.acompletion.await_args.kwargs == raw.await_args.kwargs
    assert settings == before


@pytest.mark.asyncio
@pytest.mark.parametrize("disable", [False, True])
async def test_runner_retry_flags_match_sdk(monkeypatch, routing, disable):
    raw = AsyncMock(return_value=response())
    monkeypatch.setattr(sdk.litellm, "acompletion", raw)
    routing.acompletion.return_value = response()
    settings = ModelSettings(extra_args={"num_retries": 4, "max_retries": 5})
    with provider_managed_retries_disabled(disable):
        await sdk.LitellmModel("chatgpt/gpt-5")._fetch_response(**request(model_settings=settings))
        await adapter()._fetch_response(**request(model_settings=settings))
    captured = routing.acompletion.await_args.kwargs
    assert captured == raw.await_args.kwargs
    assert captured["num_retries"] == (0 if disable else 4)
    assert captured["max_retries"] == (0 if disable else 5)
    assert settings.extra_args == {"num_retries": 4, "max_retries": 5}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "openai/gpt-5",
        "gpt-5",
        "anthropic/claude-sonnet-4",
        "gemini/gemini-2.5-pro",
        "github_copilot/gpt-5",
    ],
)
async def test_non_oauth_delegates_unchanged(monkeypatch, routing, model):
    sentinel = object()
    native = AsyncMock(return_value=sentinel)
    monkeypatch.setattr(sdk.LitellmModel, "_fetch_response", native)
    args = request(stream=True)
    instance = adapter(model)
    assert await instance._fetch_response(**args) is sentinel
    native.assert_awaited_once()
    call = native.await_args
    if call.args:
        assert call.args == tuple(args[name] for name in list(args)[:8])
        assert call.kwargs == {"stream": True, "prompt": args["prompt"]}
    else:
        assert call.kwargs == args
    routing.acompletion.assert_not_awaited()
    assert instance.model == model


@pytest.mark.asyncio
@pytest.mark.parametrize("tracing", list(ModelTracing))
async def test_trace_data_respects_mode(routing, tracing):
    routing.acompletion.return_value = response()
    target_span = span()
    await adapter()._fetch_response(**request(span=target_span, tracing=tracing))
    assert target_span.span_data.input == (
        routing.acompletion.await_args.kwargs["messages"] if tracing.include_data() else None
    )


@pytest.mark.asyncio
async def test_fetch_uses_same_conversion_helper_as_preflight(monkeypatch, routing):
    instance = adapter()
    messages, tools = [{"role": "user", "content": "preflight payload"}], [{"fixture": "tool"}]
    calls = []

    def converted(*args):
        calls.append(args)
        return messages, tools

    monkeypatch.setattr(instance, "_converted_chat_request", converted)
    routing.acompletion.return_value = response()
    args = request()
    await instance._fetch_response(**args)
    assert calls == [
        (
            args["system_instructions"],
            args["input"],
            args["model_settings"],
            args["tools"],
            args["handoffs"],
        )
    ]
    assert routing.acompletion.await_args.kwargs["messages"] is messages
    assert routing.acompletion.await_args.kwargs["tools"] is tools


@pytest.mark.asyncio
async def test_transport_error_propagates_without_rewriting_model(routing):
    error = RuntimeError("synthetic transport failure")
    routing.acompletion.side_effect = error
    instance = adapter()
    with pytest.raises(RuntimeError) as raised:
        await instance._fetch_response(**request())
    assert raised.value is error
    assert instance.model == "chatgpt/gpt-5"


@pytest.mark.asyncio
async def test_cancellation_propagates_without_transport_retry(routing):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def pending(**kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    routing.acompletion.side_effect = pending
    instance = adapter()
    task = asyncio.create_task(instance._fetch_response(**request()))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
        routing.acompletion.assert_awaited_once()
        assert instance.model == "chatgpt/gpt-5"
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", [None, "auto", "none", "required"])
async def test_stream_stub_defaults_match_sdk(monkeypatch, routing, choice):
    stream = CloseableStream()
    raw = AsyncMock(return_value=stream)
    monkeypatch.setattr(sdk.litellm, "acompletion", raw)
    routing.acompletion.return_value = stream
    args = request(stream=True, model_settings=ModelSettings(tool_choice=choice))
    expected, _ = await sdk.LitellmModel("chatgpt/gpt-5")._fetch_response(**args)
    actual, returned = await adapter()._fetch_response(**args)
    assert returned is stream
    assert actual.model_dump(exclude={"created_at"}) == expected.model_dump(exclude={"created_at"})
    assert actual.tool_choice == (choice or "auto")
    assert not actual.parallel_tool_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("parallel", [None, False, True])
async def test_handoff_only_parallel_setting_matches_sdk(monkeypatch, routing, parallel):
    _, handoffs = tools_and_handoffs()
    stream = CloseableStream()
    raw = AsyncMock(return_value=stream)
    monkeypatch.setattr(sdk.litellm, "acompletion", raw)
    routing.acompletion.return_value = stream
    args = request(
        handoffs=handoffs,
        stream=True,
        model_settings=ModelSettings(parallel_tool_calls=parallel),
    )
    expected, _ = await sdk.LitellmModel("chatgpt/gpt-5")._fetch_response(**args)
    actual, returned = await adapter()._fetch_response(**args)
    assert routing.acompletion.await_args.kwargs == raw.await_args.kwargs
    assert len(routing.acompletion.await_args.kwargs["tools"]) == 1
    assert actual.parallel_tool_calls == expected.parallel_tool_calls == bool(parallel)
    assert returned is stream


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["claude/claude-sonnet-4", "google/gemini-2.5-pro"])
async def test_out_of_order_tool_output_matches_sdk(monkeypatch, routing, model):
    items = history(model)
    items.insert(1, items.pop())
    before = deepcopy(items)
    raw = AsyncMock(return_value=response())
    monkeypatch.setattr(sdk.litellm, "acompletion", raw)
    routing.acompletion.return_value = response()
    args = request(input=items, model_settings=ModelSettings(reasoning=Reasoning(effort="high")))
    await sdk.LitellmModel(model)._fetch_response(**args)
    await adapter(model)._fetch_response(**args)
    messages = routing.acompletion.await_args.kwargs["messages"]
    assert messages == raw.await_args.kwargs["messages"]
    call_index = next(i for i, message in enumerate(messages) if message.get("tool_calls"))
    result_index = next(i for i, message in enumerate(messages) if message["role"] == "tool")
    assert call_index < result_index
    assert items == before


@pytest.mark.asyncio
async def test_reasoning_replay_callback_receives_public_model(routing):
    contexts = []

    def replay(context):
        contexts.append(context)
        return True

    model = "litellm/chatgpt/gpt-5"
    routing.acompletion.return_value = response()
    await adapter(model, should_replay_reasoning_content=replay)._fetch_response(
        **request(input=history(model))
    )
    assert len(contexts) == 1
    assert contexts[0].model == model
    messages = routing.acompletion.await_args.kwargs["messages"]
    assert (
        next(message for message in messages if message.get("tool_calls"))["reasoning_content"]
        == "fixture reasoning"
    )


@pytest.mark.asyncio
async def test_public_get_response_preserves_sdk_usage_and_provider_metadata(routing):
    routing.acompletion.return_value = sdk.litellm.ModelResponse(
        id="fixture-id",
        model="private-wire-alias",
        choices=[
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "fixture-call",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"key":"a"}'},
                            "provider_specific_fields": {"thought_signature": "fixture-signature"},
                        }
                    ],
                },
            }
        ],
        usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    )
    args = request(tracing=ModelTracing.DISABLED)
    args.pop("span")
    args.pop("stream")
    result = await adapter("google/gemini-2.5-pro").get_response(**args)
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 7
    assert result.usage.total_tokens == 18
    assert result.output[0].call_id == "fixture-call"
    assert result.output[0].provider_data["model"] == "google/gemini-2.5-pro"
    assert result.output[0].provider_data["response_id"] == "fixture-id"
    assert result.output[0].provider_data["thought_signature"] == "fixture-signature"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", PUBLIC_MODELS)
async def test_public_stream_response_keeps_public_model_metadata(routing, model):
    class ChunkStream(CloseableStream):
        def __init__(self):
            super().__init__()
            self.chunks = iter(
                [
                    ChatCompletionChunk(
                        id="fixture-stream-id",
                        created=1,
                        model="private-wire-alias",
                        object="chat.completion.chunk",
                        choices=[
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": "fixture answer"},
                                "finish_reason": None,
                            }
                        ],
                    ),
                    ChatCompletionChunk(
                        id="fixture-stream-id",
                        created=1,
                        model="private-wire-alias",
                        object="chat.completion.chunk",
                        choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                    ),
                ]
            )

        async def __anext__(self):
            try:
                return next(self.chunks)
            except StopIteration:
                raise StopAsyncIteration from None

    stream = ChunkStream()
    routing.acompletion.return_value = stream
    args = request(tracing=ModelTracing.DISABLED)
    args.pop("span")
    args.pop("stream")
    events = [event async for event in adapter(model).stream_response(**args)]
    created = next(event.response for event in events if event.type == "response.created")
    completed = next(event.response for event in events if event.type == "response.completed")
    assert created.model == completed.model == model
    assert completed.usage.input_tokens == 11
    assert completed.usage.output_tokens == 7
    assert completed.output[0].content[0].text == "fixture answer"
    assert completed.output[0].provider_data == {
        "model": model,
        "response_id": "fixture-stream-id",
    }
    assert stream.closed


class OwnedTransport(CloseableStream):
    """Controlled reads/close barriers without HTTP, credentials, or background work."""

    def __init__(self, *, blocked=False, failure=None, hold_close=False):
        super().__init__()
        self.blocked = blocked
        self.failure = failure
        self.read_started = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()
        if not hold_close:
            self.close_release.set()
        self.close_calls = 0
        self.read_cancelled = False
        self.sent_first = False

    async def __anext__(self):
        # The SDK emits response.created only after receiving its first chunk.
        if not self.sent_first:
            self.sent_first = True
            return ChatCompletionChunk(
                id="fixture-owned-stream",
                created=1,
                model="private-wire-alias",
                object="chat.completion.chunk",
                choices=[
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "fixture"},
                        "finish_reason": None,
                    }
                ],
            )
        self.read_started.set()
        if self.blocked:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.read_cancelled = True
                raise
        if self.failure is not None:
            raise self.failure
        raise StopAsyncIteration

    async def aclose(self):
        self.close_calls += 1
        self.close_started.set()
        await self.close_release.wait()
        self.closed = True


def stream_request(**overrides):
    args = request(tracing=ModelTracing.DISABLED)
    args.pop("span")
    args.pop("stream")
    args.update(overrides)
    return args


@pytest.fixture
def sdk_streams(monkeypatch):
    original = sdk.LitellmModel.stream_response
    calls = []

    class EventIterator:
        def __init__(self, iterator):
            self.iterator = iterator
            self.close_calls = 0
            self.close_failure = None

        def __aiter__(self):
            return self

        async def __anext__(self):
            return await anext(self.iterator)

        async def aclose(self):
            self.close_calls += 1
            await self.iterator.aclose()
            if self.close_failure is not None:
                raise self.close_failure

    def capture(model, *args, **kwargs):
        events = EventIterator(original(model, *args, **kwargs))
        calls.append(SimpleNamespace(model=model, args=args, kwargs=kwargs, events=events))
        return events

    monkeypatch.setattr(sdk.LitellmModel, "stream_response", capture)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("different_task", [False, True])
async def test_owned_stream_explicit_close_after_first_event(routing, sdk_streams, different_task):
    transport = OwnedTransport()
    routing.acompletion.return_value = transport
    model = adapter()
    model.fixture_marker = object()
    original_state = dict(vars(model))
    events = model.stream_response(**stream_request())
    try:
        assert (await anext(events)).type == "response.created"
        assert not transport.closed
        if different_task:
            await asyncio.create_task(events.aclose())
        else:
            await events.aclose()
        assert transport.closed and transport.close_calls == 1
        assert sdk_streams[0].events.close_calls == 1
        assert sdk_streams[0].model is not model
        assert sdk_streams[0].model.model == model.model
        assert sdk_streams[0].model.fixture_marker is model.fixture_marker
        assert vars(model) == original_state
    finally:
        await events.aclose()
        await transport.aclose()


@pytest.mark.asyncio
async def test_owned_stream_cancellation_during_blocked_read(routing, sdk_streams):
    transport = OwnedTransport(blocked=True, hold_close=True)
    routing.acompletion.return_value = transport
    model = adapter()
    original_state = dict(vars(model))
    events = model.stream_response(**stream_request())
    assert (await anext(events)).type == "response.created"

    async def consume_remaining():
        async for _ in events:
            pass

    reader = asyncio.create_task(consume_remaining())
    try:
        await asyncio.wait_for(transport.read_started.wait(), timeout=5)
        reader.cancel()
        await asyncio.wait_for(transport.close_started.wait(), timeout=5)
        reader.cancel()  # Repeated cancellation must not detach cleanup.
        await asyncio.sleep(0)
        assert not reader.done()
        transport.close_release.set()
        with pytest.raises(asyncio.CancelledError):
            await reader
        assert transport.read_cancelled
        assert transport.closed and transport.close_calls == 1
        assert sdk_streams[0].events.close_calls == 1
        assert vars(model) == original_state
    finally:
        transport.close_release.set()
        if not reader.done():
            reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
        await events.aclose()
        await transport.aclose()


@pytest.mark.asyncio
async def test_owned_stream_close_itself_is_cancellation_safe(routing, sdk_streams):
    transport = OwnedTransport(hold_close=True)
    routing.acompletion.return_value = transport
    events = adapter().stream_response(**stream_request())
    assert (await anext(events)).type == "response.created"
    closer = asyncio.create_task(events.aclose())
    try:
        await asyncio.wait_for(transport.close_started.wait(), timeout=5)
        closer.cancel()
        await asyncio.sleep(0)
        closer.cancel()
        await asyncio.sleep(0)
        assert not closer.done()
        transport.close_release.set()
        with pytest.raises(asyncio.CancelledError):
            await closer
        assert transport.closed and transport.close_calls == 1
        assert sdk_streams[0].events.close_calls == 1
    finally:
        transport.close_release.set()
        await asyncio.gather(closer, return_exceptions=True)
        await events.aclose()
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_owned_stream_closes_at_eof_or_transport_failure(routing, sdk_streams, failed):
    failure = RuntimeError("fixture read failure") if failed else None
    transport = OwnedTransport(failure=failure)
    routing.acompletion.return_value = transport
    events = adapter().stream_response(**stream_request())
    try:
        if failed:
            with pytest.raises(RuntimeError) as raised:
                async for _ in events:
                    pass
            assert raised.value is failure
        else:
            types = [event.type async for event in events]
            assert types[-1] == "response.completed"
        assert transport.closed and transport.close_calls == 1
        assert sdk_streams[0].events.close_calls == 1
    finally:
        await events.aclose()
        await transport.aclose()


@pytest.mark.asyncio
async def test_owned_streams_on_same_model_never_close_each_other(routing, sdk_streams):
    first_transport, second_transport = OwnedTransport(), OwnedTransport()
    transports = {"first": first_transport, "second": second_transport}

    async def route(**kwargs):
        return transports[kwargs["messages"][-1]["content"]]

    routing.acompletion.side_effect = route
    model = adapter()
    original_state = dict(vars(model))
    ready = {"first": asyncio.Event(), "second": asyncio.Event()}
    release = {"first": asyncio.Event(), "second": asyncio.Event()}

    async def consume(name, close_early):
        events = model.stream_response(**stream_request(input=name))
        try:
            await anext(events)
            ready[name].set()
            await release[name].wait()
            return [] if close_early else [event.type async for event in events]
        finally:
            await events.aclose()

    first = asyncio.create_task(consume("first", True))
    second = asyncio.create_task(consume("second", False))
    try:
        await asyncio.wait_for(
            asyncio.gather(ready["first"].wait(), ready["second"].wait()), timeout=5
        )
        first_call = next(call for call in sdk_streams if call.kwargs["input"] == "first")
        second_call = next(call for call in sdk_streams if call.kwargs["input"] == "second")
        release["first"].set()
        await asyncio.wait_for(first, timeout=5)
        assert first_transport.closed
        assert not second_transport.closed
        assert first_call.events.close_calls == 1
        assert second_call.events.close_calls == 0
        assert first_call.model is not second_call.model
        assert first_call.model is not model
        assert second_call.model is not model
        release["second"].set()
        remaining = await asyncio.wait_for(second, timeout=5)
        assert remaining[-1] == "response.completed"
        assert second_transport.closed
        assert first_transport.close_calls == second_transport.close_calls == 1
        assert vars(model) == original_state
    finally:
        release["first"].set()
        release["second"].set()
        for task in (first, second):
            if not task.done():
                task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)
        await first_transport.aclose()
        await second_transport.aclose()


@pytest.mark.asyncio
async def test_owned_transport_closes_even_when_sdk_iterator_close_fails(routing, sdk_streams):
    transport = OwnedTransport()
    routing.acompletion.return_value = transport
    events = adapter().stream_response(**stream_request())
    await anext(events)
    failure = RuntimeError("fixture event close failure")
    sdk_streams[0].events.close_failure = failure
    try:
        with pytest.raises(RuntimeError) as raised:
            await events.aclose()
        assert raised.value is failure
        assert transport.closed and transport.close_calls == 1
    finally:
        sdk_streams[0].events.close_failure = None
        await events.aclose()
        await transport.aclose()


@pytest.mark.asyncio
async def test_non_oauth_stream_delegates_iterator_without_clone(routing, sdk_streams, monkeypatch):
    transport = OwnedTransport()
    raw = AsyncMock(return_value=transport)
    monkeypatch.setattr(sdk.litellm, "acompletion", raw)
    model = adapter("openai/gpt-5")
    original_state = dict(vars(model))
    events = model.stream_response(
        **stream_request(
            previous_response_id="fixture-prev", conversation_id="fixture-conversation"
        )
    )
    try:
        assert events is sdk_streams[0].events
        assert sdk_streams[0].model is model
        await anext(events)
        await events.aclose()
        # The current SDK closes streams itself; Koder must not add another close
        # or a per-request clone to the inherited non-OAuth path.
        assert transport.closed and transport.close_calls == 1
        assert vars(model) == original_state
        routing.acompletion.assert_not_awaited()
    finally:
        await events.aclose()
        await transport.aclose()


@pytest.mark.asyncio
async def test_direct_fetch_keeps_stream_owned_by_caller(routing):
    transport = OwnedTransport()
    routing.acompletion.return_value = transport
    model = adapter()
    original_state = dict(vars(model))
    _, stream = await model._fetch_response(**request(stream=True, tracing=ModelTracing.DISABLED))
    assert stream is transport
    assert vars(model) == original_state
    assert not stream.closed and stream.close_calls == 0
    await stream.aclose()
