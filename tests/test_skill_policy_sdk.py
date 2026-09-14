from __future__ import annotations

import asyncio
import copy
import json
import shlex
import sys
from types import SimpleNamespace

import pytest
from agents import Agent, AgentHooks, FunctionTool, RunConfig, RunHooks, Runner, handoff
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.run_config import ToolExecutionConfig
from agents.usage import Usage
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

from koder_agent.agentic.hook_guardrail import hook_pretool_input_guardrail
from koder_agent.agentic.skill_guardrail import skill_restriction_guardrail
from koder_agent.core.scheduler import AgentScheduler
from koder_agent.harness.agents import service as agent_service_module
from koder_agent.harness.agents.definitions import AgentDefinition
from koder_agent.harness.commands.interactive import HarnessInteractiveCommandHandler
from koder_agent.harness.hooks.runtime import _run_agent_hook, list_configured_hooks
from koder_agent.tools import skill as skill_module
from koder_agent.tools.skill import Skill
from koder_agent.tools.skill_context import (
    clear_restrictions,
    get_active_restrictions,
    get_skill_activation_block_message,
    skill_invocation_scope,
    skill_run_scope,
)


class _SequenceModel(Model):
    def __init__(self, steps: list[list | BaseException]):
        self.steps = list(steps)
        self.inputs = []

    async def get_response(self, *args, **kwargs) -> ModelResponse:
        self.inputs.append(kwargs.get("input", args[1] if len(args) > 1 else None))
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return ModelResponse(output=step, usage=Usage(), response_id=None)

    async def stream_response(self, *args, **kwargs):
        if False:
            yield None
        raise NotImplementedError


class _StreamingSequenceModel(_SequenceModel):
    async def get_response(self, *args, **kwargs) -> ModelResponse:
        raise NotImplementedError

    async def stream_response(self, *args, **kwargs):
        self.inputs.append(kwargs.get("input", args[1] if len(args) > 1 else None))
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        response = Response(
            id=f"response-{len(self.inputs)}",
            created_at=0.0,
            model="test",
            object="response",
            output=step,
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        yield ResponseCompletedEvent(
            response=response,
            sequence_number=0,
            type="response.completed",
        )


def _tool_call(name: str, call_id: str, arguments: str = "{}") -> list:
    return [
        ResponseFunctionToolCall(
            arguments=arguments,
            call_id=call_id,
            name=name,
            type="function_call",
        )
    ]


def _message(text: str, message_id: str) -> list:
    return [
        ResponseOutputMessage(
            id=message_id,
            content=[ResponseOutputText(annotations=[], text=text, type="output_text")],
            role="assistant",
            status="completed",
            type="message",
        )
    ]


def _mixed_skill_batch(order: str) -> list:
    calls = {
        "skill": _tool_call("get_skill", "skill-1", '{"skill_name":"read-only"}')[0],
        "write": _tool_call("write_file", "write-1")[0],
    }
    return [calls[name] for name in order.split("-")]


def _function_tool(
    name: str,
    calls: list[str],
    *,
    with_hooks: bool = False,
) -> FunctionTool:
    async def invoke(_context, _arguments: str) -> str:
        calls.append(name)
        return f"ran {name}"

    guardrails = [skill_restriction_guardrail]
    if with_hooks:
        guardrails.append(hook_pretool_input_guardrail)
    return FunctionTool(
        name=name,
        description=f"Test tool {name}",
        params_json_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        on_invoke_tool=invoke,
        tool_input_guardrails=guardrails,
    )


class _RunnerScheduler:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.session = SimpleNamespace(session_id="skill-policy-sdk")
        self.called = False

    async def handle(self, prompt: str, render_output: bool = True) -> str:
        self.called = True
        with skill_run_scope() as run_hooks:
            result = await Runner.run(self.agent, prompt, hooks=run_hooks)
        return str(result.final_output or "")


@pytest.fixture(autouse=True)
def _clean_skill_state(monkeypatch, tmp_path):
    clear_restrictions()
    monkeypatch.setenv("HOME", str(tmp_path))
    yield
    clear_restrictions()


def test_manual_skill_rejects_real_sdk_function_tool(monkeypatch):
    calls: list[str] = []
    write_tool = _function_tool("write_file", calls)
    model = _SequenceModel([_tool_call("write_file", "write-1"), _message("done", "m1")])
    scheduler = _RunnerScheduler(
        Agent(name="manual", instructions="test", tools=[write_tool], model=model)
    )
    skill = Skill(
        name="read-only",
        description="read only",
        content="Inspect files",
        allowed_tools=["read_file"],
    )
    handler = HarnessInteractiveCommandHandler()
    monkeypatch.setattr(handler, "_available_skills", lambda: {skill.name: skill})

    result = asyncio.run(handler.handle_slash_input("/read-only", scheduler=scheduler))

    assert result == "done"
    assert calls == []
    assert "not permitted" in str(model.inputs[-1])
    assert get_active_restrictions() is None


def test_get_skill_parallel_batch_fails_closed_before_disallowed_tool(monkeypatch):
    calls: list[str] = []
    write_tool = _function_tool("write_file", calls)
    skill = Skill(
        name="read-only",
        description="read only",
        content="Inspect files",
        allowed_tools=["read_file"],
    )
    monkeypatch.setattr(skill_module, "_get_merged_skills", lambda: {skill.name: skill})
    model = _SequenceModel(
        [
            [
                *_tool_call("get_skill", "skill-1", '{"skill_name":"read-only"}'),
                *_tool_call("write_file", "write-1"),
            ],
            _message("done", "m1"),
        ]
    )
    agent = Agent(
        name="parallel-activation",
        instructions="test",
        tools=[skill_module.get_skill, write_tool],
        model=model,
    )

    async def scenario():
        with skill_run_scope() as run_hooks:
            return await Runner.run(agent, "load and write", hooks=run_hooks)

    result = asyncio.run(scenario())

    assert result.final_output == "done"
    assert calls == []
    assert "call this tool again in the next model step" in str(model.inputs[-1])
    assert get_active_restrictions() is None


@pytest.mark.parametrize("reuse_call_id", [False, True])
def test_batch_generation_respects_sdk_tool_call_identity(monkeypatch, reuse_call_id):
    writes: list[str] = []
    write_tool = _function_tool("write_file", writes)
    skill = Skill(
        name="read-only",
        description="test generation",
        content="Allow the next write",
        allowed_tools=["write_file"],
    )
    monkeypatch.setattr(skill_module, "_get_merged_skills", lambda: {skill.name: skill})
    model = _SequenceModel(
        [
            _mixed_skill_batch("skill-write"),
            _tool_call("write_file", "write-1" if reuse_call_id else "write-2"),
            _message("done", "generation-message"),
        ]
    )
    agent = Agent(
        name="response-generation",
        instructions="test",
        tools=[skill_module.get_skill, write_tool],
        model=model,
    )

    async def scenario():
        with skill_run_scope() as run_hooks:
            return await Runner.run(agent, "load then write", hooks=run_hooks)

    result = asyncio.run(scenario())

    assert result.final_output == "done"
    # The current SDK deduplicates completed call IDs, including calls that
    # produced a guardrail result. A new invocation needs a fresh ID; do not
    # bypass the SDK's replay protection to test Koder's per-response policy.
    assert writes == ([] if reuse_call_id else ["write_file"])
    assert "call this tool again in the next model step" in str(model.inputs[1])


def test_policy_generation_itself_does_not_retain_prior_response_call_ids():
    async def scenario():
        with skill_run_scope() as run_hooks:
            await run_hooks.on_llm_end(
                None, None, SimpleNamespace(output=_mixed_skill_batch("skill-write"))
            )
            assert get_skill_activation_block_message("write_file", "write-1") is not None
            await run_hooks.on_llm_end(
                None, None, SimpleNamespace(output=_tool_call("write_file", "write-1"))
            )
            assert get_skill_activation_block_message("write_file", "write-1") is None

    asyncio.run(scenario())


@pytest.mark.parametrize("order", ["skill-write", "write-skill"])
@pytest.mark.parametrize("rebuild", ["model_dump", "json"])
def test_wrapped_on_llm_end_reconstructed_calls_fail_closed(monkeypatch, order, rebuild):
    writes: list[str] = []
    write_tool = _function_tool("write_file", writes)
    skill = Skill(
        name="read-only",
        description="read only",
        content="Inspect files",
        allowed_tools=["read_file"],
    )
    monkeypatch.setattr(skill_module, "_get_merged_skills", lambda: {skill.name: skill})

    class ReconstructingHook(RunHooks):
        async def on_llm_end(self, context, agent, response) -> None:
            if rebuild == "model_dump":
                response.output = [
                    (
                        ResponseFunctionToolCall.model_validate(item.model_dump())
                        if isinstance(item, ResponseFunctionToolCall)
                        else item
                    )
                    for item in response.output
                ]
            else:
                response.output = [
                    (
                        ResponseFunctionToolCall.model_validate(json.loads(item.model_dump_json()))
                        if isinstance(item, ResponseFunctionToolCall)
                        else item
                    )
                    for item in response.output
                ]

    model = _SequenceModel(
        [_mixed_skill_batch(order), _message("done", f"message-{rebuild}-{order}")]
    )
    agent = Agent(
        name=f"reconstructed-{rebuild}-{order}",
        instructions="test",
        tools=[skill_module.get_skill, write_tool],
        model=model,
    )

    async def scenario():
        with skill_run_scope(ReconstructingHook()) as run_hooks:
            return await Runner.run(agent, "load and write", hooks=run_hooks)

    result = asyncio.run(scenario())

    assert result.final_output == "done"
    assert writes == []
    assert "call this tool again in the next model step" in str(model.inputs[-1])


@pytest.mark.parametrize("order", ["skill-write", "write-skill"])
@pytest.mark.parametrize("rebuild", ["model_dump", "json"])
def test_late_reconstruction_without_private_metadata_fails_closed(
    monkeypatch,
    order,
    rebuild,
):
    writes: list[str] = []
    write_tool = _function_tool("write_file", writes)
    skill = Skill(
        name="read-only",
        description="read only",
        content="Inspect files",
        allowed_tools=["read_file"],
    )
    monkeypatch.setattr(skill_module, "_get_merged_skills", lambda: {skill.name: skill})
    wrapped_hook_started = asyncio.Event()

    class WrappedHook(RunHooks):
        async def on_llm_end(self, context, agent, response) -> None:
            if any(isinstance(item, ResponseFunctionToolCall) for item in response.output):
                wrapped_hook_started.set()

    class LateReconstructingAgentHook(AgentHooks):
        async def on_llm_end(self, context, agent, response) -> None:
            await wrapped_hook_started.wait()
            if rebuild == "model_dump":
                response.output = [
                    (
                        ResponseFunctionToolCall.model_validate(item.model_dump())
                        if isinstance(item, ResponseFunctionToolCall)
                        else item
                    )
                    for item in response.output
                ]
            else:
                response.output = [
                    (
                        ResponseFunctionToolCall.model_validate_json(item.model_dump_json())
                        if isinstance(item, ResponseFunctionToolCall)
                        else item
                    )
                    for item in response.output
                ]

    model = _SequenceModel([_mixed_skill_batch(order), _message("done", f"late-{rebuild}-{order}")])
    agent = Agent(
        name=f"late-reconstructed-{rebuild}-{order}",
        instructions="test",
        tools=[skill_module.get_skill, write_tool],
        hooks=LateReconstructingAgentHook(),
        model=model,
    )

    async def scenario():
        with skill_run_scope(WrappedHook()) as run_hooks:
            return await Runner.run(agent, "load and write", hooks=run_hooks)

    result = asyncio.run(scenario())

    assert result.final_output == "done"
    assert writes == []
    assert "call this tool again in the next model step" in str(model.inputs[-1])


@pytest.mark.parametrize("order", ["skill-write", "write-skill"])
@pytest.mark.parametrize(
    "transform",
    ["copy", "deepcopy", "model_copy", "model_copy_deep", "reorder"],
)
def test_wrapped_on_llm_end_copied_or_reordered_calls_fail_closed(
    monkeypatch,
    order,
    transform,
):
    writes: list[str] = []
    write_tool = _function_tool("write_file", writes)
    skill = Skill(
        name="read-only",
        description="read only",
        content="Inspect files",
        allowed_tools=["read_file"],
    )
    monkeypatch.setattr(skill_module, "_get_merged_skills", lambda: {skill.name: skill})

    class CopyingHook(RunHooks):
        async def on_llm_end(self, context, agent, response) -> None:
            if not any(isinstance(item, ResponseFunctionToolCall) for item in response.output):
                return
            if transform == "copy":
                response.output = [copy.copy(item) for item in response.output]
            elif transform == "deepcopy":
                response.output = copy.deepcopy(response.output)
            elif transform == "model_copy":
                response.output = [item.model_copy() for item in response.output]
            elif transform == "model_copy_deep":
                response.output = [item.model_copy(deep=True) for item in response.output]
            else:
                response.output = list(reversed(response.output))

    model = _SequenceModel(
        [_mixed_skill_batch(order), _message("done", f"message-{transform}-{order}")]
    )
    agent = Agent(
        name=f"copied-{transform}-{order}",
        instructions="test",
        tools=[skill_module.get_skill, write_tool],
        model=model,
    )

    async def scenario():
        with skill_run_scope(CopyingHook()) as run_hooks:
            return await Runner.run(agent, "load and write", hooks=run_hooks)

    result = asyncio.run(scenario())

    assert result.final_output == "done"
    assert writes == []
    assert "call this tool again in the next model step" in str(model.inputs[-1])


@pytest.mark.parametrize("order", ["skill-write", "write-skill"])
def test_concurrent_runners_cannot_clear_another_response_batch_barrier(
    monkeypatch,
    order,
):
    writes: list[str] = []
    write_tool = _function_tool("write_file", writes)
    skill = Skill(
        name="read-only",
        description="read only",
        content="Inspect files",
        allowed_tools=["read_file"],
    )
    monkeypatch.setattr(skill_module, "_get_merged_skills", lambda: {skill.name: skill})

    mixed_hook_entered = asyncio.Event()
    unrelated_hook_finished = asyncio.Event()

    class MixedResponseHook(RunHooks):
        async def on_llm_end(self, context, agent, response) -> None:
            if any(getattr(item, "name", None) == "get_skill" for item in response.output):
                mixed_hook_entered.set()
                await unrelated_hook_finished.wait()

    class UnrelatedResponseHook(RunHooks):
        async def on_llm_end(self, context, agent, response) -> None:
            unrelated_hook_finished.set()

    class UnrelatedModel(_SequenceModel):
        async def get_response(self, *args, **kwargs) -> ModelResponse:
            await mixed_hook_entered.wait()
            return await super().get_response(*args, **kwargs)

    mixed_model = _SequenceModel(
        [
            _mixed_skill_batch(order),
            _message("mixed done", "m1"),
        ]
    )
    unrelated_model = UnrelatedModel([_message("unrelated done", "m2")])
    mixed_agent = Agent(
        name="mixed-runner",
        instructions="test",
        tools=[skill_module.get_skill, write_tool],
        model=mixed_model,
    )
    unrelated_agent = Agent(
        name="unrelated-runner",
        instructions="test",
        tools=[],
        model=unrelated_model,
    )
    serial_tools = RunConfig(tool_execution=ToolExecutionConfig(max_function_tool_concurrency=1))

    async def scenario():
        async def run_mixed():
            with skill_run_scope(MixedResponseHook()) as run_hooks:
                return await Runner.run(
                    mixed_agent,
                    "load and write",
                    hooks=run_hooks,
                    run_config=serial_tools,
                )

        async def run_unrelated():
            with skill_run_scope(UnrelatedResponseHook()) as run_hooks:
                return await Runner.run(unrelated_agent, "finish", hooks=run_hooks)

        with skill_invocation_scope():
            return await asyncio.gather(run_mixed(), run_unrelated())

    mixed_result, unrelated_result = asyncio.run(scenario())

    assert mixed_result.final_output == "mixed done"
    assert unrelated_result.final_output == "unrelated done"
    assert writes == []
    assert "call this tool again in the next model step" in str(mixed_model.inputs[-1])
    assert get_active_restrictions() is None


@pytest.mark.parametrize("order", ["skill-write", "write-skill"])
def test_streaming_mixed_batch_fails_closed(monkeypatch, order):
    writes: list[str] = []
    write_tool = _function_tool("write_file", writes)
    skill = Skill(
        name="read-only",
        description="read only",
        content="Inspect files",
        allowed_tools=["read_file"],
    )
    monkeypatch.setattr(skill_module, "_get_merged_skills", lambda: {skill.name: skill})
    model = _StreamingSequenceModel(
        [_mixed_skill_batch(order), _message("done", f"stream-message-{order}")]
    )
    agent = Agent(
        name=f"streaming-{order}",
        instructions="test",
        tools=[skill_module.get_skill, write_tool],
        model=model,
    )

    async def scenario():
        with skill_run_scope() as run_hooks:
            result = Runner.run_streamed(agent, "load and write", hooks=run_hooks)
            async for _event in result.stream_events():
                pass
            return result

    result = asyncio.run(scenario())

    assert result.final_output == "done"
    assert writes == []
    assert "call this tool again in the next model step" in str(model.inputs[-1])


@pytest.mark.parametrize("order", ["skill-handoff", "handoff-skill"])
def test_handoff_inherits_activated_policy_for_both_sibling_orders(monkeypatch, order):
    writes: list[str] = []
    write_tool = _function_tool("write_file", writes)
    skill = Skill(
        name="read-only",
        description="read only",
        content="Inspect files",
        allowed_tools=["read_file"],
    )
    monkeypatch.setattr(skill_module, "_get_merged_skills", lambda: {skill.name: skill})

    target_model = _SequenceModel(
        [_tool_call("write_file", "target-write-1"), _message("done", "target-message")]
    )
    target = Agent(
        name="target",
        instructions="test",
        tools=[write_tool],
        model=target_model,
    )
    transfer = handoff(target)
    calls = {
        "skill": _tool_call("get_skill", "skill-1", '{"skill_name":"read-only"}')[0],
        "handoff": _tool_call(transfer.tool_name, "handoff-1")[0],
    }
    source_model = _SequenceModel([[calls[name] for name in order.split("-")]])
    source = Agent(
        name="source",
        instructions="test",
        tools=[skill_module.get_skill],
        handoffs=[transfer],
        model=source_model,
    )

    async def scenario():
        with skill_run_scope() as run_hooks:
            return await Runner.run(source, "load and hand off", hooks=run_hooks)

    result = asyncio.run(scenario())

    assert result.final_output == "done"
    assert writes == []
    assert "not permitted" in str(target_model.inputs[-1])


def test_get_skill_union_persists_across_real_scheduler_continuation(monkeypatch):
    allowed_calls: list[str] = []
    forbidden_calls: list[str] = []
    allowed_tool = _function_tool("allowed_tool", allowed_calls)
    forbidden_tool = _function_tool("forbidden_tool", forbidden_calls)
    skills = {
        "first": Skill(
            name="first",
            description="first",
            content="first",
            allowed_tools=["read_file"],
        ),
        "second": Skill(
            name="second",
            description="second",
            content="second",
            allowed_tools=["allowed_tool"],
        ),
    }
    monkeypatch.setattr(skill_module, "_get_merged_skills", lambda: skills)
    model = _SequenceModel(
        [
            _tool_call("get_skill", "skill-1", '{"skill_name":"first"}'),
            _tool_call("get_skill", "skill-2", '{"skill_name":"second"}'),
            _message("first turn", "m1"),
            _tool_call("allowed_tool", "allowed-1"),
            _tool_call("forbidden_tool", "forbidden-1"),
            _message("continued", "m2"),
        ]
    )
    agent = Agent(
        name="continuation",
        instructions="test",
        tools=[skill_module.get_skill, allowed_tool, forbidden_tool],
        model=model,
    )
    scheduler = AgentScheduler.__new__(AgentScheduler)
    scheduler._turn_lock = asyncio.Lock()
    scheduler.session = SimpleNamespace(session_id="continuation")
    continuations = iter(["continue the goal", None])

    async def next_continuation_prompt():
        return next(continuations)

    scheduler.goal_runtime = SimpleNamespace(next_continuation_prompt=next_continuation_prompt)
    scheduler._goal_cumulative_tokens = lambda: 0

    async def run_turn(prompt: str, **_kwargs) -> str:
        result = await Runner.run(agent, prompt)
        return str(result.final_output or "")

    scheduler._handle_unlocked = run_turn

    result = asyncio.run(scheduler.handle("start", render_output=False))

    assert result == "continued"
    assert allowed_calls == ["allowed_tool"]
    assert forbidden_calls == []
    assert "not permitted" in str(model.inputs[-1])
    assert get_active_restrictions() is None


def test_model_get_skill_installs_deduplicated_hooks_for_real_tool_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "hook-count.txt"
    script = (
        "from pathlib import Path; "
        f"p=Path({str(marker)!r}); "
        "p.open('a', encoding='utf-8').write('x'); "
        'print(\'{"decision":"block","reason":"skill hook blocked"}\')'
    )
    hook_command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    skill = Skill(
        name="hooked",
        description="hooked",
        content="hooked",
        allowed_tools=["write_file"],
        hooks={
            "PreToolUse": [
                {
                    "matcher": "write_file",
                    "hooks": [{"type": "command", "command": hook_command}],
                }
            ]
        },
        base_dir=tmp_path,
    )
    monkeypatch.setattr(skill_module, "_get_merged_skills", lambda: {skill.name: skill})
    calls: list[str] = []
    write_tool = _function_tool("write_file", calls, with_hooks=True)
    model = _SequenceModel(
        [
            _tool_call("get_skill", "skill-1", '{"skill_name":"hooked"}'),
            _tool_call("get_skill", "skill-2", '{"skill_name":"hooked"}'),
            _tool_call("write_file", "write-1"),
            _message("done", "m1"),
        ]
    )
    agent = Agent(
        name="model-hook",
        instructions="test",
        tools=[skill_module.get_skill, write_tool],
        model=model,
    )

    with skill_invocation_scope():
        result = asyncio.run(Runner.run(agent, "load the skill"))

    assert result.final_output == "done"
    assert calls == []
    assert marker.read_text(encoding="utf-8") == "x"
    assert "skill hook blocked" in str(model.inputs[-1])
    assert not [hook for hook in list_configured_hooks(tmp_path) if hook.source == "skills"]


def test_raw_thread_hook_agent_inherits_skill_policy(monkeypatch):
    calls: list[str] = []
    write_tool = _function_tool("write_file", calls)
    model = _SequenceModel([_tool_call("write_file", "write-1"), _message("done", "m1")])

    async def create_hook_agent(tools, **_kwargs):
        return Agent(name="hook-agent", instructions="test", tools=tools, model=model)

    monkeypatch.setattr("koder_agent.agentic.agent.create_dev_agent", create_hook_agent)
    monkeypatch.setattr("koder_agent.tools.get_all_tools", lambda: [write_tool])
    skill = Skill(
        name="fork-read-only",
        description="read only",
        content="fork",
        allowed_tools=["read_file"],
    )

    async def scenario() -> str:
        with skill_invocation_scope(skill):
            return _run_agent_hook(prompt_text="check", payload_text="{}", model=None)

    result = asyncio.run(scenario())

    assert result == "done"
    assert calls == []
    assert "not permitted" in str(model.inputs[-1])


def test_unseeded_agent_service_fork_activates_skill_across_sdk_steps(monkeypatch):
    calls: list[str] = []
    write_tool = _function_tool("write_file", calls)
    skill = Skill(
        name="read-only",
        description="read only",
        content="Inspect files",
        allowed_tools=["read_file"],
    )
    monkeypatch.setattr(skill_module, "_get_merged_skills", lambda: {skill.name: skill})
    model = _SequenceModel(
        [
            _tool_call("get_skill", "skill-1", '{"skill_name":"read-only"}'),
            _tool_call("write_file", "write-1"),
            _message("done", "m1"),
        ]
    )

    async def create_fork_agent(tools, **_kwargs):
        return Agent(name="fork", instructions="test", tools=tools, model=model)

    class _Session:
        def __init__(self, session_id):
            self.session_id = session_id

        def close(self):
            pass

        async def get_items(self):
            return []

        async def add_items(self, _items):
            return None

    monkeypatch.setattr(agent_service_module, "create_dev_agent", create_fork_agent)
    monkeypatch.setattr(
        agent_service_module,
        "get_all_tools",
        lambda: [skill_module.get_skill, write_tool],
    )
    monkeypatch.setattr(agent_service_module, "EnhancedSQLiteSession", _Session)
    monkeypatch.setattr(agent_service_module, "get_display_hooks", RunHooks)
    definition = AgentDefinition(
        agent_type="worker",
        when_to_use="test",
        system_prompt="test",
        source="built-in",
    )

    result = asyncio.run(
        agent_service_module._execute_agent_run(
            agent_definition=definition,
            prompt="load then write",
            session_id="fork-skill-policy",
            seed_items=None,
            cwd=None,
        )
    )

    assert result == "done"
    assert calls == []
    assert "not permitted" in str(model.inputs[-1])
    assert get_active_restrictions() is None


def test_unseeded_agent_hook_activates_skill_across_sdk_steps(monkeypatch):
    calls: list[str] = []
    write_tool = _function_tool("write_file", calls)
    skill = Skill(
        name="read-only",
        description="read only",
        content="Inspect files",
        allowed_tools=["read_file"],
    )
    monkeypatch.setattr(skill_module, "_get_merged_skills", lambda: {skill.name: skill})
    model = _SequenceModel(
        [
            _tool_call("get_skill", "skill-1", '{"skill_name":"read-only"}'),
            _tool_call("write_file", "write-1"),
            _message("done", "m1"),
        ]
    )

    async def create_hook_agent(tools, **_kwargs):
        return Agent(name="hook-agent", instructions="test", tools=tools, model=model)

    monkeypatch.setattr("koder_agent.agentic.agent.create_dev_agent", create_hook_agent)
    monkeypatch.setattr(
        "koder_agent.tools.get_all_tools",
        lambda: [skill_module.get_skill, write_tool],
    )

    result = _run_agent_hook(prompt_text="check", payload_text="{}", model=None)

    assert result == "done"
    assert calls == []
    assert "not permitted" in str(model.inputs[-1])
    assert get_active_restrictions() is None


def test_overridden_remember_uses_normal_runner_and_cannot_bypass_policy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []
    write_tool = _function_tool("write_file", calls)
    model = _SequenceModel([_tool_call("write_file", "write-1"), _message("done", "m1")])
    scheduler = _RunnerScheduler(
        Agent(name="remember-override", instructions="test", tools=[write_tool], model=model)
    )
    override = Skill(
        name="remember",
        description="project override",
        content="Use tools to remember $ARGUMENTS",
        allowed_tools=["read_file"],
        source="project",
        base_dir=tmp_path / ".koder" / "skills" / "remember",
    )
    handler = HarnessInteractiveCommandHandler()
    monkeypatch.setattr(handler, "_available_skills", lambda: {"remember": override})

    result = asyncio.run(handler.handle_slash_input("/remember secret", scheduler=scheduler))

    assert result == "done"
    assert scheduler.called is True
    assert calls == []
    assert "not permitted" in str(model.inputs[-1])
    assert not (tmp_path / ".koder" / "memory").exists()


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("boom"), asyncio.CancelledError()],
    ids=["error", "cancel"],
)
def test_scheduler_error_and_cancel_cleanup_skill_policy(tmp_path, monkeypatch, failure):
    skill = Skill(
        name="cleanup",
        description="cleanup",
        content="cleanup",
        allowed_tools=["read_file"],
        hooks={"Stop": [{"hooks": [{"type": "command", "command": "true"}]}]},
        base_dir=tmp_path,
    )
    monkeypatch.setattr(skill_module, "_get_merged_skills", lambda: {skill.name: skill})
    model = _SequenceModel(
        [_tool_call("get_skill", "skill-1", '{"skill_name":"cleanup"}'), failure]
    )
    agent = Agent(
        name="cleanup",
        instructions="test",
        tools=[skill_module.get_skill],
        model=model,
    )
    scheduler = AgentScheduler.__new__(AgentScheduler)
    scheduler._turn_lock = asyncio.Lock()
    scheduler.session = SimpleNamespace(session_id="cleanup")

    async def run_turn(prompt: str, **_kwargs) -> str:
        result = await Runner.run(agent, prompt)
        return str(result.final_output or "")

    scheduler._handle_unlocked = run_turn

    with pytest.raises(type(failure), match="boom" if isinstance(failure, RuntimeError) else None):
        asyncio.run(scheduler.handle("cleanup", render_output=False))

    assert get_active_restrictions() is None
    assert not [hook for hook in list_configured_hooks(tmp_path) if hook.source == "skills"]
