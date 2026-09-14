import asyncio
import copy

import litellm.exceptions
import pytest
from agents import ModelSettings
from agents.extensions.models.litellm_model import LitellmModel

from koder_agent.agentic.agent import (
    PreflightOpenAIResponsesModel,
    RetryingLitellmModel,
    create_dev_agent,
)
from koder_agent.harness.memory.budget import ContextPreflightError
from koder_agent.mcp.server_config import MCPServerConfig, MCPServerScope, MCPServerType

_MISSING = object()


def _retryable_error(message="boom"):
    return litellm.exceptions.InternalServerError(
        message=message, model="test-model", llm_provider="test-provider"
    )


class DummyTool:
    def __init__(self, name="tool", schema=_MISSING, strict_json_schema=_MISSING):
        self.name = name
        if schema is not _MISSING:
            self.params_json_schema = schema
        if strict_json_schema is not _MISSING:
            self.strict_json_schema = strict_json_schema


def _make_model(model_name: str) -> RetryingLitellmModel:
    model = RetryingLitellmModel.__new__(RetryingLitellmModel)
    model.model = model_name
    return model


def _schema_with_refs() -> dict:
    return {
        "type": "object",
        "$defs": {
            "Foo": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            }
        },
        "properties": {
            "foo": {"$ref": "#/$defs/Foo"},
        },
    }


def test_is_github_copilot_true():
    model = _make_model("github_copilot/anthropic/claude-3")
    assert model._is_github_copilot()

    model = _make_model("Litellm/GitHub_Copilot/claude-3")
    assert model._is_github_copilot()


def test_is_github_copilot_false():
    for name in ["gpt-4", "claude-3"]:
        model = _make_model(name)
        assert not model._is_github_copilot()


def test_clean_tools_no_copilot_returns_unchanged():
    model = _make_model("gpt-4")
    schema = _schema_with_refs()
    tool = DummyTool(name="tool", schema=schema, strict_json_schema=True)
    tools = [tool]

    result = model._clean_tools_for_github_copilot(tools)

    assert result is tools
    assert tool.params_json_schema is schema
    assert "$ref" in tool.params_json_schema["properties"]["foo"]
    assert tool.strict_json_schema is True


def test_clean_tools_for_copilot_cleans_schema_and_strict():
    model = _make_model("github_copilot/anthropic/claude-3")
    tool_with_schema = DummyTool(
        name="tool",
        schema=copy.deepcopy(_schema_with_refs()),
        strict_json_schema=True,
    )
    tool_without_schema = DummyTool(
        name="no_schema",
        schema=_MISSING,
        strict_json_schema=True,
    )
    tools = [tool_with_schema, tool_without_schema]

    result = model._clean_tools_for_github_copilot(tools)

    assert result is tools
    cleaned_schema = tool_with_schema.params_json_schema
    assert "$defs" not in cleaned_schema
    assert "$ref" not in cleaned_schema.get("properties", {}).get("foo", {})
    assert tool_with_schema.strict_json_schema is False

    assert not hasattr(tool_without_schema, "params_json_schema")
    assert tool_without_schema.strict_json_schema is True


def test_clean_tools_handles_empty_list():
    model = _make_model("github_copilot/anthropic/claude-3")
    tools = []

    result = model._clean_tools_for_github_copilot(tools)

    assert result is tools
    assert result == []


def test_stream_response_retries_before_first_chunk(monkeypatch):
    """A retryable error raised before any chunk is yielded triggers a retry
    and ultimately succeeds."""
    model = _make_model("gpt-4")  # not the responses API path
    calls = {"n": 0}

    async def fake_super_stream(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _retryable_error("transient")
        for chunk in ["a", "b", "c"]:
            yield chunk

    monkeypatch.setattr(LitellmModel, "stream_response", fake_super_stream)
    # Avoid real sleeps between retries.
    monkeypatch.setattr("koder_agent.agentic.agent.asyncio.sleep", _no_sleep)

    async def run():
        out = []
        async for chunk in model.stream_response(
            None, "in", _DummySettings(), [], None, [], _DummyTracing()
        ):
            out.append(chunk)
        return out

    result = asyncio.run(run())

    assert result == ["a", "b", "c"]
    assert calls["n"] == 2  # failed once, retried, succeeded


def test_stream_response_does_not_retry_after_first_chunk(monkeypatch):
    """A retryable error raised AFTER a chunk has been yielded propagates and is
    NOT retried (retrying would replay partial output)."""
    model = _make_model("gpt-4")
    calls = {"n": 0}

    async def fake_super_stream(*args, **kwargs):
        calls["n"] += 1
        yield "first"
        raise _retryable_error("mid-stream")

    monkeypatch.setattr(LitellmModel, "stream_response", fake_super_stream)
    monkeypatch.setattr("koder_agent.agentic.agent.asyncio.sleep", _no_sleep)

    async def run():
        out = []
        async for chunk in model.stream_response(
            None, "in", _DummySettings(), [], None, [], _DummyTracing()
        ):
            out.append(chunk)
        return out

    raised = None
    try:
        asyncio.run(run())
    except litellm.exceptions.InternalServerError as exc:
        raised = exc

    assert raised is not None  # error propagated
    assert calls["n"] == 1  # only one attempt, no retry


def test_get_response_retries(monkeypatch):
    """get_response retry still works (fails once then succeeds)."""
    model = _make_model("gpt-4")
    calls = {"n": 0}
    sentinel = object()

    async def fake_super_get(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _retryable_error("transient-get")
        return sentinel

    monkeypatch.setattr(LitellmModel, "get_response", fake_super_get)
    monkeypatch.setattr("koder_agent.agentic.agent.asyncio.sleep", _no_sleep)

    async def run():
        return await model.get_response(None, "in", _DummySettings(), [], None, [], _DummyTracing())

    result = asyncio.run(run())

    assert result is sentinel
    assert calls["n"] == 2  # failed once, retried, succeeded


class _DummySettings:
    max_tokens = 20
    reasoning = None

    def to_json_dict(self):
        return {}


class _DummyTracing:
    def is_disabled(self):
        return True

    def include_data(self):
        return False


async def _no_sleep(_seconds):
    return None


class _LargeOutputSchema:
    def is_plain_text(self):
        return False

    def is_strict_json_schema(self):
        return True

    def json_schema(self):
        return {
            "type": "object",
            "properties": {"payload": {"type": "string", "description": "s" * 20_000}},
            "required": ["payload"],
            "additionalProperties": False,
        }


class _NeverCalledResponses:
    def __init__(self):
        self.provider_calls = 0

    async def create(self, **_kwargs):
        self.provider_calls += 1
        raise AssertionError("provider must not be called")


class _FakeOpenAIClient:
    def __init__(self):
        self.responses = _NeverCalledResponses()


@pytest.mark.parametrize("streaming", [False, True])
def test_native_responses_preflights_exact_large_schema_before_provider_io(streaming):
    client = _FakeOpenAIClient()
    model = PreflightOpenAIResponsesModel(
        model="gpt-4.1",
        openai_client=client,
        context_window=200,
    )

    async def run():
        if streaming:
            async for _chunk in model.stream_response(
                "system",
                [{"role": "user", "content": "respond as json"}],
                ModelSettings(max_tokens=20),
                [],
                _LargeOutputSchema(),
                [],
                _DummyTracing(),
            ):
                pass
        else:
            await model.get_response(
                "system",
                [{"role": "user", "content": "respond as json"}],
                ModelSettings(max_tokens=20),
                [],
                _LargeOutputSchema(),
                [],
                _DummyTracing(),
            )

    with pytest.raises(ContextPreflightError) as exc_info:
        asyncio.run(run())

    assert exc_info.value.estimate.schema_tokens > 200
    assert client.responses.provider_calls == 0


@pytest.mark.parametrize("streaming", [False, True])
def test_model_wrapper_preflights_each_tool_loop_request(monkeypatch, streaming):
    model = RetryingLitellmModel(
        model="github_copilot/gpt-5.1-codex",
        context_window=128,
    )
    provider_calls = 0
    tool_loop_input = [
        {"role": "user", "content": "read the file"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "read_file",
            "arguments": '{"path":"large.txt"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "x" * 100_000,
        },
    ]

    async def fake_aresponses(**_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("provider must not be called")

    monkeypatch.setattr(
        "koder_agent.agentic.agent.litellm.aresponses",
        fake_aresponses,
        raising=False,
    )

    async def run():
        if streaming:
            async for _chunk in model.stream_response(
                "system",
                tool_loop_input,
                ModelSettings(max_tokens=20),
                [],
                None,
                [],
                _DummyTracing(),
            ):
                pass
        else:
            await model.get_response(
                "system",
                tool_loop_input,
                ModelSettings(max_tokens=20),
                [],
                None,
                [],
                _DummyTracing(),
            )

    with pytest.raises(ContextPreflightError):
        asyncio.run(run())

    assert provider_calls == 0


def test_create_dev_agent_cleans_connected_mcp_after_tool_construction_failure(monkeypatch):
    class ConnectedServer:
        name = "connected"

        def __init__(self):
            self.cleanup_calls = 0

        async def cleanup(self):
            self.cleanup_calls += 1

    server = ConnectedServer()

    async def load_servers():
        return [server]

    async def fail_tool_construction(*_args, **_kwargs):
        raise RuntimeError("tool construction failed")

    monkeypatch.delenv("KODER_SIMPLE", raising=False)
    monkeypatch.setattr("koder_agent.agentic.agent.load_mcp_servers", load_servers)
    monkeypatch.setattr(
        "koder_agent.agentic.agent._build_prefixed_mcp_tools",
        fail_tool_construction,
    )

    with pytest.raises(RuntimeError, match="tool construction failed"):
        asyncio.run(create_dev_agent([]))

    assert server.cleanup_calls == 1


def test_create_dev_agent_cleans_connected_mcp_after_agent_construction_failure(
    monkeypatch,
):
    from koder_agent.harness.config.schema import RuntimeConfig

    class ConnectedServer:
        name = "connected"

        def __init__(self):
            self.cleanup_calls = 0

        async def cleanup(self):
            self.cleanup_calls += 1

    server = ConnectedServer()
    config = RuntimeConfig()
    config.model.name = "gpt-4.1"
    config.model.provider = "openai"
    config.model.reasoning_effort = None

    async def load_servers():
        return [server]

    async def no_mcp_tools(*_args, **_kwargs):
        return []

    def fail_agent_construction(*_args, **_kwargs):
        raise RuntimeError("Agent construction failed")

    monkeypatch.delenv("KODER_SIMPLE", raising=False)
    monkeypatch.setattr("koder_agent.agentic.agent.load_mcp_servers", load_servers)
    monkeypatch.setattr("koder_agent.agentic.agent.get_config", lambda: config)
    monkeypatch.setattr("koder_agent.agentic.agent._build_prefixed_mcp_tools", no_mcp_tools)
    monkeypatch.setattr(
        "koder_agent.agentic.agent.get_model_client_snapshot",
        lambda _override: {
            "model_name": "gpt-4.1",
            "native_openai": True,
            "litellm_kwargs": {},
        },
    )
    monkeypatch.setattr("koder_agent.agentic.agent._get_skills_metadata", lambda _config: "")
    monkeypatch.setattr("koder_agent.agentic.agent._get_agents_metadata", lambda: "")
    monkeypatch.setattr("koder_agent.agentic.agent._get_environment_info", lambda _model: "")
    monkeypatch.setattr("koder_agent.agentic.agent.Agent", fail_agent_construction)

    with pytest.raises(RuntimeError, match="Agent construction failed"):
        asyncio.run(create_dev_agent([]))

    assert server.cleanup_calls == 1


def test_agent_build_and_factory_cancellation_clean_each_mcp_server_once(monkeypatch):
    from koder_agent import mcp as mcp_module
    from koder_agent.agentic import agent as agent_module

    class Server:
        def __init__(self, name: str, *, connect_error: Exception | None = None):
            self.name = name
            self.connect_error = connect_error
            self.cleanup_calls = 0
            self.cleanup_started = asyncio.Event()
            self.cleanup_release = asyncio.Event()

        async def connect(self):
            if self.connect_error is not None:
                raise self.connect_error

        async def cleanup(self):
            self.cleanup_calls += 1
            self.cleanup_started.set()
            if self.name == "failed-extra":
                await self.cleanup_release.wait()

    base = Server("base")
    connected_extra = Server("connected-extra")
    failed_extra = Server("failed-extra", connect_error=RuntimeError("connect failed"))
    created = iter([connected_extra, failed_extra])

    async def create_server(*_args, **_kwargs):
        return next(created)

    async def load_servers(extra_configs=None):
        owner = mcp_module.MCPServerSet([base])
        try:
            extra_servers = await mcp_module.MCPServerFactory.create_servers_from_configs(
                list(extra_configs or [])
            )
            for server in extra_servers:
                owner.adopt_server(server, server_name=server.name)
            return owner
        except BaseException:
            await owner.aclose(propagate_cancellation=False)
            raise

    monkeypatch.delenv("KODER_SIMPLE", raising=False)
    monkeypatch.setattr(agent_module, "load_mcp_servers", load_servers)
    monkeypatch.setattr(mcp_module.MCPServerFactory, "create_server", create_server)

    configs = [
        MCPServerConfig(
            name=name,
            transport_type=MCPServerType.STDIO,
            command="python",
            args=[],
            env_vars={},
            scope=MCPServerScope.USER,
            source_path="/tmp/test-agent-mcp.json",
        )
        for name in ("connected-extra", "failed-extra")
    ]

    async def scenario():
        task = asyncio.create_task(create_dev_agent([], extra_mcp_server_configs=configs))
        await failed_extra.cleanup_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        failed_extra.cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert base.cleanup_calls == 1
    assert connected_extra.cleanup_calls == 1
    assert failed_extra.cleanup_calls == 1


def test_create_dev_agent_uses_model_client_snapshot_for_litellm(monkeypatch):
    seen = {}

    def fake_snapshot(model_override):
        seen["model_override"] = model_override
        return {
            "model_name": "litellm/claude/claude-sonnet-4-6",
            "api_key": "oauth-access-token",
            "base_url": None,
            "native_openai": False,
            "litellm_kwargs": {
                "model": "claude/claude-sonnet-4-6",
                "api_key": "oauth-access-token",
                "base_url": None,
                "extra_headers": {"x-oauth-provider": "claude"},
            },
        }

    monkeypatch.setenv("KODER_SIMPLE", "1")
    monkeypatch.setattr("koder_agent.agentic.agent.get_model_client_snapshot", fake_snapshot)

    agent = asyncio.run(create_dev_agent([], model_override="inherit"))

    assert seen["model_override"] is None
    assert isinstance(agent.model, RetryingLitellmModel)
    assert agent.model.model == "claude/claude-sonnet-4-6"
    assert agent.model.api_key == "oauth-access-token"
    assert agent.model_settings.extra_headers == {"x-oauth-provider": "claude"}


def test_create_dev_agent_requests_reasoning_summary_when_display_enabled(monkeypatch):
    from koder_agent.harness.config.schema import RuntimeConfig

    config = RuntimeConfig()
    config.model.name = "gpt-5"
    config.model.provider = "openai"
    config.harness.reasoning_display = "summary"

    def fake_snapshot(_model_override):
        return {
            "model_name": "gpt-5",
            "api_key": "sk-test",
            "base_url": None,
            "native_openai": True,
            "litellm_kwargs": {},
        }

    monkeypatch.setenv("KODER_SIMPLE", "1")
    monkeypatch.delenv("KODER_REASONING_DISPLAY", raising=False)
    monkeypatch.setattr("koder_agent.agentic.agent.get_config", lambda: config)
    monkeypatch.setattr("koder_agent.agentic.agent.get_model_client_snapshot", fake_snapshot)
    monkeypatch.setattr("koder_agent.agentic.agent.should_use_reasoning_param", lambda: True)

    agent = asyncio.run(create_dev_agent([]))

    assert isinstance(agent.model, PreflightOpenAIResponsesModel)
    assert agent.model.model == "gpt-5"
    assert agent.model_settings.reasoning.summary == "detailed"
    assert agent.model_settings.reasoning.effort == "medium"


def test_create_dev_agent_passes_max_reasoning_effort_without_conversion(monkeypatch):
    from koder_agent.harness.config.schema import RuntimeConfig

    config = RuntimeConfig(
        model={
            "name": "gpt-5.6",
            "provider": "openai",
            "reasoning_effort": "max",
        }
    )

    def fake_snapshot(_model_override):
        return {
            "model_name": "gpt-5.6",
            "api_key": "sk-test",
            "base_url": None,
            "native_openai": True,
            "litellm_kwargs": {},
        }

    monkeypatch.setenv("KODER_SIMPLE", "1")
    monkeypatch.setattr("koder_agent.agentic.agent.get_config", lambda: config)
    monkeypatch.setattr("koder_agent.agentic.agent.get_model_client_snapshot", fake_snapshot)
    monkeypatch.setattr("koder_agent.agentic.agent.should_use_reasoning_param", lambda: True)

    agent = asyncio.run(create_dev_agent([]))

    assert agent.model_settings.reasoning.effort == "max"
    assert agent.model_settings.to_json_dict()["reasoning"]["effort"] == "max"


def test_create_dev_agent_cleans_loaded_mcp_servers_when_construction_is_cancelled(monkeypatch):
    entered_tool_build = asyncio.Event()
    never = asyncio.Event()

    class Server:
        name = "construction-server"

        def __init__(self):
            self.cleaned = False

        async def cleanup(self):
            self.cleaned = True

    server = Server()

    async def fake_load_mcp_servers():
        return [server]

    async def blocking_tool_build(*args, **kwargs):
        entered_tool_build.set()
        await never.wait()

    monkeypatch.delenv("KODER_SIMPLE", raising=False)
    monkeypatch.setattr("koder_agent.agentic.agent.load_mcp_servers", fake_load_mcp_servers)
    monkeypatch.setattr(
        "koder_agent.agentic.agent._build_prefixed_mcp_tools",
        blocking_tool_build,
    )

    async def scenario():
        construction = asyncio.create_task(create_dev_agent([]))
        await asyncio.wait_for(entered_tool_build.wait(), timeout=1)
        construction.cancel()
        try:
            await construction
        except asyncio.CancelledError:
            pass
        else:  # pragma: no cover - cancellation must propagate
            raise AssertionError("agent construction was not cancelled")

    asyncio.run(scenario())

    assert server.cleaned is True


def test_create_dev_agent_preserves_initial_cancel_during_slow_cleanup(
    monkeypatch, cancellation_observer
):
    observe, cancellations = cancellation_observer
    entered_tool_build = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    never = asyncio.Event()
    cleanup_calls = 0
    cleanup_completed = 0

    class Server:
        name = "construction-server"

        async def cleanup(self):
            nonlocal cleanup_calls, cleanup_completed
            cleanup_calls += 1
            cleanup_started.set()
            await cleanup_release.wait()
            cleanup_completed += 1

    server = Server()

    async def fake_load_mcp_servers():
        return [server]

    async def blocking_tool_build(*args, **kwargs):
        entered_tool_build.set()
        await never.wait()

    monkeypatch.delenv("KODER_SIMPLE", raising=False)
    monkeypatch.setattr("koder_agent.agentic.agent.load_mcp_servers", fake_load_mcp_servers)
    monkeypatch.setattr(
        "koder_agent.agentic.agent._build_prefixed_mcp_tools",
        blocking_tool_build,
    )

    async def scenario():
        construction = asyncio.create_task(observe(create_dev_agent([])))
        await asyncio.wait_for(entered_tool_build.wait(), timeout=1)
        construction.cancel("initial-parent-cancel")
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        construction.cancel("repeat-parent-cancel")
        await asyncio.sleep(0)
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await construction

    asyncio.run(scenario())

    assert cleanup_calls == 1
    assert cleanup_completed == 1
    assert [error.args for error in cancellations] == [("initial-parent-cancel",)]
