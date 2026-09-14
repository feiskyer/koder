"""Real SDK routing with synthetic handlers; native auth/network is intercepted."""

from __future__ import annotations

import asyncio
import atexit
import importlib
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace

import pytest
import pytest_asyncio
from agents import ModelSettings
from agents.models.interface import ModelTracing
from agents.tracing.span_data import GenerationSpanData
from litellm.llms.custom_llm import CustomLLM

from koder_agent.agentic.agent import RetryingLitellmModel
from koder_agent.auth import oauth_routing, providers
from koder_agent.harness.config.schema import RuntimeConfig
from koder_agent.utils import client


@pytest.fixture
def unexpected_native_calls():
    return []


@pytest_asyncio.fixture(autouse=True)
async def logging_owner(monkeypatch, unexpected_native_calls):
    """Keep this file's real SDK log tasks inside its live test loop."""
    module = importlib.import_module("litellm.litellm_core_utils.logging_worker")
    worker = module.LoggingWorker()
    monkeypatch.setattr(module, "GLOBAL_LOGGING_WORKER", worker)
    try:
        yield
    finally:
        try:
            await asyncio.wait_for(worker.flush(), timeout=3)
        finally:
            await worker.stop()
            atexit.unregister(worker._flush_on_exit)
        assert unexpected_native_calls == []


class SyntheticHandler(CustomLLM):
    def __init__(self, label="koder"):
        super().__init__()
        self.label = label
        self.calls = []
        self.closed = False

    async def acompletion(self, model, messages, **kwargs):
        self.calls.append((model, messages, kwargs))
        return providers.litellm.ModelResponse(
            model=model,
            choices=[{"index": 0, "message": {"role": "assistant", "content": self.label}}],
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        )

    async def astreaming(self, model, messages, **kwargs):
        self.calls.append((model, messages, kwargs))
        try:
            yield {
                "text": self.label,
                "is_finished": False,
                "finish_reason": "",
                "usage": None,
                "index": 0,
                "tool_use": None,
            }
            yield {
                "text": "",
                "is_finished": True,
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                "index": 0,
                "tool_use": None,
            }
        finally:
            self.closed = True


@pytest.fixture
def sdk_state(monkeypatch, unexpected_native_calls):
    sdk = providers.litellm
    for name in ("provider_list", "_custom_providers"):
        monkeypatch.setattr(sdk, name, list(getattr(sdk, name) or []))
    monkeypatch.setattr(sdk, "custom_provider_map", [])
    monkeypatch.setattr(sdk, "model_cost", dict(sdk.model_cost))
    native_models = set(sdk.open_ai_chat_completion_models)
    native_models.add("gpt-5.1")
    monkeypatch.setattr(sdk, "open_ai_chat_completion_models", native_models)
    monkeypatch.setattr(sdk, "_koder_oauth_handlers", {}, raising=False)
    monkeypatch.setattr(sdk, "_koder_private_oauth_handlers", {}, raising=False)

    def forbid_native(*_args, **_kwargs):
        unexpected_native_calls.append(
            [(frame.name, frame.lineno) for frame in traceback.extract_stack(limit=9)]
        )
        raise AssertionError("Native provider/auth must not run in OAuth route tests")

    monkeypatch.setattr(sdk.ChatGPTConfig, "_get_openai_compatible_provider_info", forbid_native)
    monkeypatch.setattr(
        importlib.import_module("litellm.main"), "_complete_custom_openai", forbid_native
    )
    return sdk


@pytest.mark.asyncio
async def test_agent_chatgpt_request_reaches_koder_handler_not_sdk_native_auth(
    sdk_state, monkeypatch
):
    handler = SyntheticHandler()
    monkeypatch.setattr(providers, "_chatgpt_oauth_llm", handler)
    native_calls = []

    def native_config(self, model, api_base, api_key, custom_llm_provider):
        native_calls.append("config")
        return api_base, api_key, custom_llm_provider

    def native_completion(_context):
        native_calls.append("completion")
        return sdk_state.ModelResponse(
            model="gpt-5.1",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "native"}}],
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        )

    monkeypatch.setattr(
        sdk_state.ChatGPTConfig,
        "_get_openai_compatible_provider_info",
        native_config,
    )
    main = importlib.import_module("litellm.main")
    monkeypatch.setattr(main, "_complete_custom_openai", native_completion)
    providers.register_oauth_providers()
    model = RetryingLitellmModel(model="chatgpt/gpt-5.1", context_window=100_000)
    result = await model.get_response(
        "system",
        "hello",
        ModelSettings(max_tokens=20),
        [],
        None,
        [],
        ModelTracing.DISABLED,
    )
    assert result.output[0].content[0].text == "koder"
    assert native_calls == []
    assert handler.calls[0][0] == "gpt-5.1"
    assert model.model == "chatgpt/gpt-5.1"


@pytest.mark.asyncio
async def test_sdk_supports_private_provider_and_non_native_model_alias(sdk_state):
    """Feasibility control: use the real SDK custom route without model APIs."""
    handler = SyntheticHandler("private-route")
    alias = "koder_fixture_oauth"
    sdk_state.custom_provider_map.append({"provider": alias, "custom_handler": handler})
    sdk_state.provider_list.append(alias)
    sdk_state._custom_providers.append(alias)
    response = await asyncio.wait_for(
        sdk_state.acompletion(
            model=f"{alias}/model-6770742d352e31",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=10,
        ),
        timeout=10,
    )
    assert response.choices[0].message.content == "private-route"
    assert handler.calls[0][0] == "model-6770742d352e31"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["google", "claude", "chatgpt", "antigravity"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_private_dispatch_preserves_public_identity_and_native_catalogs(
    sdk_state, monkeypatch, provider, streaming
):
    handler = SyntheticHandler(provider)
    monkeypatch.setattr(providers, f"_{provider}_oauth_llm", handler)
    before = set(sdk_state.open_ai_chat_completion_models)
    public_model = f"{provider}/gpt-5.1"
    result = await oauth_routing.acompletion(
        model=public_model,
        messages=[{"role": "user", "content": "hello"}],
        stream=streaming,
        max_tokens=20,
        temperature=0.4,
        metadata={"fixture": "routing"},
    )
    if streaming:
        chunks = [chunk async for chunk in result]
        assert any(chunk.choices and chunk.choices[0].delta.content == provider for chunk in chunks)
        assert all(chunk.model == public_model for chunk in chunks)
        assert handler.closed
    else:
        assert result.choices[0].message.content == provider
        assert result.model == public_model
    assert handler.calls[0][0] == "gpt-5.1"
    assert handler.calls[0][2]["optional_params"]["temperature"] == 0.4
    assert sdk_state.open_ai_chat_completion_models == before


def test_explicit_legacy_registration_does_not_remove_native_models(sdk_state):
    before = set(sdk_state.open_ai_chat_completion_models)
    providers.register_oauth_providers()
    assert sdk_state.open_ai_chat_completion_models == before


@pytest.mark.asyncio
async def test_oauth_caller_cannot_override_the_private_provider(sdk_state, monkeypatch):
    async def unexpected(**_kwargs):
        pytest.fail("routing override must fail before invoking the SDK")

    monkeypatch.setattr(sdk_state, "acompletion", unexpected)
    with pytest.raises(ValueError, match="routing"):
        await oauth_routing.acompletion(
            model="chatgpt/gpt-5.1",
            messages=[],
            custom_llm_provider="openai",
        )


@pytest.mark.asyncio
async def test_oauth_does_not_forward_other_provider_credentials_or_clients(sdk_state, monkeypatch):
    handler = SyntheticHandler()
    monkeypatch.setattr(providers, "_chatgpt_oauth_llm", handler)
    result = await oauth_routing.acompletion(
        model="chatgpt/gpt-5.1",
        messages=[{"role": "user", "content": "hello"}],
        api_key="synthetic-foreign-api-key",
        base_url="https://synthetic.invalid/v1",
        api_base="https://synthetic.invalid/v1",
        client=object(),
        max_tokens=10,
    )
    assert result.choices[0].message.content == "koder"
    received = handler.calls[0][2]
    assert received.get("api_key") == "koder-oauth-managed"
    assert received.get("api_base") is None
    assert received.get("client") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "public_model, raw_model",
    [
        ("chatgpt/gpt-5.1", "gpt-5.1"),
        ("chatgpt/github_copilot/gpt-codex-fixture", "github_copilot/gpt-codex-fixture"),
    ],
)
async def test_auxiliary_completion_uses_the_same_owned_route(
    sdk_state, monkeypatch, public_model, raw_model
):
    handler = SyntheticHandler("auxiliary")
    monkeypatch.setattr(providers, "_chatgpt_oauth_llm", handler)
    config = RuntimeConfig()
    monkeypatch.setattr(
        client,
        "_resolve_completion_settings",
        lambda _model: (config, object(), "chatgpt", "gpt-5.1", False),
    )
    monkeypatch.setattr(client, "_setup_provider_env_vars", lambda *_args: None)
    monkeypatch.setattr(
        client,
        "_compute_effective_model",
        lambda *_args: (f"litellm/{public_model}", False, "synthetic-unused-key"),
    )
    monkeypatch.setattr(client, "get_configured_context_window", lambda *_a, **_kw: 100_000)
    monkeypatch.setattr(client, "_resolve_base_url", lambda *_args: "https://synthetic.invalid")

    async def unexpected_responses(**_kwargs):
        pytest.fail("an OAuth model suffix must not select another provider")

    monkeypatch.setattr(sdk_state, "aresponses", unexpected_responses)
    result = await client.llm_completion(
        [{"role": "user", "content": "hello"}],
        model=public_model,
        response_reserve=20,
    )
    assert result == "auxiliary"
    assert handler.calls[0][0] == raw_model
    assert handler.calls[0][2]["api_key"] == "koder-oauth-managed"


@pytest.mark.asyncio
async def test_native_requests_keep_their_parameters_and_result_identity(sdk_state, monkeypatch):
    expected = object()
    seen = []

    async def capture(**kwargs):
        seen.append(kwargs)
        return expected

    monkeypatch.setattr(sdk_state, "acompletion", capture)
    metadata = {"fixture": True}
    kwargs = {
        "model": "openai/gpt-5.1",
        "messages": [],
        "api_key": "synthetic-api-key",
        "base_url": "https://synthetic.invalid/v1",
        "custom_llm_provider": "openai",
        "metadata": metadata,
    }
    assert await oauth_routing.acompletion(**kwargs) is expected
    assert seen == [kwargs]
    assert seen[0]["metadata"] is metadata
    assert not sdk_state.custom_provider_map


@pytest.mark.parametrize(
    "model_name, copilot, responses",
    [
        ("chatgpt/github_copilot/gpt-codex-fixture", False, False),
        ("openrouter/team/github_copilot-codex", False, False),
        ("github_copilot/gpt-5.1-codex", True, True),
        ("litellm/github_copilot/gpt-5.1-codex", True, True),
        ("github_copilot/gpt-4.1", True, False),
    ],
)
def test_provider_selection_uses_the_leading_provider_not_a_model_substring(
    model_name, copilot, responses
):
    model = RetryingLitellmModel(model=model_name, context_window=100_000)
    assert model._is_github_copilot() is copilot
    assert model._should_use_responses_api() is responses


@pytest.mark.asyncio
@pytest.mark.parametrize("collision", ["model", "provider"])
async def test_private_alias_collision_is_rejected_before_sdk_call(
    sdk_state, monkeypatch, collision
):
    alias = "koder_oauth_chatgpt"
    wire_model = "model-" + "gpt-5.1".encode().hex()
    if collision == "model":
        sdk_state.open_ai_chat_completion_models.add(wire_model)
    else:
        monkeypatch.setattr(
            sdk_state,
            "openai_compatible_providers",
            [*sdk_state.openai_compatible_providers, alias],
        )

    async def unexpected(**_kwargs):
        pytest.fail("a conflicting native route must not receive the request")

    monkeypatch.setattr(sdk_state, "acompletion", unexpected)
    with pytest.raises(ValueError, match="collision"):
        await oauth_routing.acompletion(model="chatgpt/gpt-5.1", messages=[])
    assert not sdk_state.custom_provider_map


@pytest.mark.asyncio
async def test_private_stream_preserves_tool_calls_usage_and_provider_metadata(
    sdk_state, monkeypatch
):
    class ToolHandler(SyntheticHandler):
        async def astreaming(self, model, messages, **kwargs):
            self.calls.append((model, messages, kwargs))
            try:
                yield {
                    "text": "",
                    "is_finished": False,
                    "finish_reason": "",
                    "usage": None,
                    "index": 0,
                    "tool_use": {
                        "id": "call-fixture",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"key":"value"}'},
                    },
                    "provider_specific_fields": {"fixture_provider_field": "retained"},
                }
                yield {
                    "text": "",
                    "is_finished": True,
                    "finish_reason": "tool_calls",
                    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                    "index": 0,
                    "tool_use": None,
                }
            finally:
                self.closed = True

    handler = ToolHandler()
    monkeypatch.setattr(providers, "_google_oauth_llm", handler)
    stream = await oauth_routing.acompletion(
        model="google/gemini-fixture",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        stream_options={"include_usage": True},
        num_retries=0,
    )
    chunks = [chunk async for chunk in stream]
    calls = [
        call
        for chunk in chunks
        for choice in chunk.choices
        for call in choice.delta.tool_calls or []
    ]
    assert calls[0].id == "call-fixture"
    assert calls[0].function.name == "lookup"
    assert calls[0].function.arguments == '{"key":"value"}'
    assert any(getattr(chunk, "fixture_provider_field", None) == "retained" for chunk in chunks)
    assert any(getattr(chunk, "usage", None) and chunk.usage.total_tokens == 8 for chunk in chunks)
    assert any(choice.finish_reason == "tool_calls" for chunk in chunks for choice in chunk.choices)
    assert all(chunk.model == "google/gemini-fixture" for chunk in chunks)
    assert handler.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["close", "cancel", "error"])
async def test_direct_private_stream_closes_its_owned_handler(sdk_state, monkeypatch, ending):
    entered = asyncio.Event()
    handler = SyntheticHandler()

    async def stream(model, messages, **kwargs):
        handler.calls.append((model, messages, kwargs))
        try:
            yield {
                "text": "first",
                "is_finished": False,
                "finish_reason": "",
                "usage": None,
                "index": 0,
                "tool_use": None,
            }
            entered.set()
            if ending == "error":
                raise RuntimeError("synthetic stream failure")
            await asyncio.Event().wait()
        finally:
            handler.closed = True

    monkeypatch.setattr(handler, "astreaming", stream)
    monkeypatch.setattr(providers, "_chatgpt_oauth_llm", handler)
    response = await oauth_routing.acompletion(
        model="chatgpt/gpt-5.1",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        num_retries=0,
    )
    try:

        async def read_first():
            async for chunk in response:
                if any(choice.delta.content == "first" for choice in chunk.choices):
                    return

        await asyncio.wait_for(read_first(), timeout=3)
        if ending == "close":
            await response.aclose()
        elif ending == "error":
            with pytest.raises(Exception, match="synthetic stream failure"):
                async for _chunk in response:
                    pass
        else:

            async def consume():
                async for _chunk in response:
                    pass

            task = asyncio.create_task(consume())
            await asyncio.wait_for(entered.wait(), timeout=3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert handler.closed
        assert len(handler.calls) == 1
    finally:
        await response.aclose()


@pytest.mark.asyncio
async def test_agent_retry_keeps_oauth_dispatch_and_public_identity(sdk_state, monkeypatch):
    attempts = 0
    handler = SyntheticHandler("recovered")
    complete = handler.acompletion

    async def flaky(model, messages, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sdk_state.InternalServerError(
                message="synthetic transient",
                model=model,
                llm_provider="chatgpt",
            )
        return await complete(model, messages, **kwargs)

    monkeypatch.setattr(handler, "acompletion", flaky)
    monkeypatch.setattr(providers, "_chatgpt_oauth_llm", handler)
    model = RetryingLitellmModel(model="chatgpt/gpt-5.1", context_window=100_000)
    result = await asyncio.wait_for(
        model.get_response(
            None,
            "hello",
            ModelSettings(max_tokens=20, extra_args={"num_retries": 0, "max_retries": 0}),
            [],
            None,
            [],
            ModelTracing.DISABLED,
        ),
        timeout=10,
    )
    assert result.output[0].content[0].text == "recovered"
    assert attempts == 2
    assert handler.calls[0][0] == "gpt-5.1"
    assert model.model == "chatgpt/gpt-5.1"


@pytest.mark.asyncio
async def test_agent_stream_failure_after_text_does_not_replay(sdk_state, monkeypatch):
    handler = SyntheticHandler()

    async def fail_after_text(model, messages, **kwargs):
        handler.calls.append((model, messages, kwargs))
        try:
            yield {
                "text": "delivered",
                "is_finished": False,
                "finish_reason": "",
                "usage": None,
                "index": 0,
                "tool_use": None,
            }
            raise RuntimeError("synthetic failure after text")
        finally:
            handler.closed = True

    monkeypatch.setattr(handler, "astreaming", fail_after_text)
    monkeypatch.setattr(providers, "_chatgpt_oauth_llm", handler)
    model = RetryingLitellmModel(model="chatgpt/gpt-5.1", context_window=100_000)
    events = []
    with pytest.raises(Exception, match="synthetic failure after text"):
        async for event in model.stream_response(
            None,
            "hello",
            ModelSettings(max_tokens=20, extra_args={"num_retries": 0, "max_retries": 0}),
            [],
            None,
            [],
            ModelTracing.DISABLED,
        ):
            events.append(event)
    assert any(
        event.type == "response.output_text.delta" and event.delta == "delivered"
        for event in events
    )
    assert len(handler.calls) == 1
    assert handler.closed


@pytest.mark.asyncio
async def test_agent_stream_closes_sdk_trace_in_the_consuming_context(sdk_state, monkeypatch):
    trace_scope = ContextVar("test_oauth_trace_scope", default=None)

    @contextmanager
    def local_span(**kwargs):
        token = trace_scope.set("active")
        try:
            yield SimpleNamespace(
                span_data=GenerationSpanData(model=kwargs["model"]),
                set_error=lambda _error: None,
            )
        finally:
            trace_scope.reset(token)

    sdk_model_module = importlib.import_module("agents.extensions.models.litellm_model")
    monkeypatch.setattr(sdk_model_module, "generation_span", local_span)
    handler = SyntheticHandler("traced")
    monkeypatch.setattr(providers, "_chatgpt_oauth_llm", handler)
    model = RetryingLitellmModel(model="chatgpt/gpt-5.1", context_window=100_000)
    stream = model.stream_response(
        None,
        "hello",
        ModelSettings(max_tokens=20),
        [],
        None,
        [],
        ModelTracing.ENABLED,
    )
    try:
        async for event in stream:
            if event.type == "response.output_text.delta":
                break
        assert trace_scope.get() == "active"
        await stream.aclose()
        assert trace_scope.get() is None
        assert handler.closed
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_native_sdk_chatgpt_route_is_unchanged_after_koder_oauth_call(sdk_state, monkeypatch):
    handler = SyntheticHandler("owned")
    monkeypatch.setattr(providers, "_chatgpt_oauth_llm", handler)
    before = set(sdk_state.open_ai_chat_completion_models)
    native_calls = []

    def native_config(self, model, api_base, api_key, custom_llm_provider):
        native_calls.append("config")
        return api_base, api_key, custom_llm_provider

    def native_completion(_context):
        native_calls.append("completion")
        return sdk_state.ModelResponse(
            model="gpt-5.1",
            choices=[{"index": 0, "message": {"role": "assistant", "content": "native"}}],
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        )

    monkeypatch.setattr(
        sdk_state.ChatGPTConfig, "_get_openai_compatible_provider_info", native_config
    )
    monkeypatch.setattr(
        importlib.import_module("litellm.main"), "_complete_custom_openai", native_completion
    )
    owned = await oauth_routing.acompletion(
        model="chatgpt/gpt-5.1", messages=[{"role": "user", "content": "hello"}]
    )
    assert owned.choices[0].message.content == "owned"
    assert not native_calls
    native = await sdk_state.acompletion(
        model="chatgpt/gpt-5.1",
        messages=[{"role": "user", "content": "hello"}],
        api_key="synthetic-native-key",
        num_retries=0,
    )
    assert native.choices[0].message.content == "native"
    assert "config" in native_calls and "completion" in native_calls
    assert sdk_state.open_ai_chat_completion_models == before


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_public_model_projection_never_mutates_sdk_owned_objects(
    sdk_state, monkeypatch, streaming
):
    original = sdk_state.ModelResponse(
        model="private-sdk-model",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "fixture"}}],
    )

    class OneResponse:
        def __init__(self):
            self.sent = False
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return original

        async def aclose(self):
            self.closed = True

    source = OneResponse()

    async def capture(**_kwargs):
        return source if streaming else original

    monkeypatch.setattr(sdk_state, "acompletion", capture)
    result = await oauth_routing.acompletion(model="chatgpt/gpt-5.1", messages=[], stream=streaming)
    if streaming:
        projected = await result.__anext__()
        await result.aclose()
        assert source.closed
    else:
        projected = result
    assert projected is not original
    assert projected.model == "chatgpt/gpt-5.1"
    assert original.model == "private-sdk-model"
