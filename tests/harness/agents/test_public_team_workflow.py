"""Exercise public FunctionTools with real local team state and synthetic work."""

import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from koder_agent.harness.agents import service as agent_module
from koder_agent.harness.agents.runtime_context import agent_service_scope, agent_session_scope
from koder_agent.harness.agents.teams.context import team_tool_context
from koder_agent.harness.agents.teams.service import TeamService
from koder_agent.tools.agent import agent_tool
from koder_agent.tools.plan_mode import _get_plan_service
from koder_agent.tools.send_message import send_message
from koder_agent.tools.team import team_create, team_delete


async def invoke(tool, **arguments):
    return json.loads(await tool.on_invoke_tool(None, json.dumps(arguments)))


async def eventually(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest_asyncio.fixture
async def runtime(tmp_path, monkeypatch):
    profile = tmp_path / "profile"
    profile.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: profile))
    monkeypatch.setattr(
        agent_module,
        "_redacted_model_config_snapshot",
        lambda definition: {"model_override": definition.model},
    )
    for module in ("service", "task_service"):
        monkeypatch.setattr(
            f"koder_agent.harness.agents.teams.{module}.dispatch_project_hook_event",
            lambda **_: SimpleNamespace(blocked=False),
        )
    observed = []

    async def execute(**kwargs):
        observed.append(
            {
                **kwargs,
                "observed_plan_mode": _get_plan_service().is_plan_mode(),
            }
        )
        return "synthetic teammate result"

    monkeypatch.setattr(agent_module, "_execute_agent_run", execute)
    service = agent_module.AgentService()
    with agent_service_scope(service):
        try:
            yield service, observed
        finally:
            await service.aclose()


async def launch(runtime, **overrides):
    service, _ = runtime
    arguments = {
        "description": "Public teammate",
        "prompt": "Initial team work",
        "name": "worker",
        "team_name": "public-workflow",
        "run_in_background": True,
        **overrides,
    }
    result = await invoke(agent_tool, **arguments)
    assert result["status"] != "error", result
    await asyncio.wait_for(service.wait(result["agent_id"]), timeout=3)
    return result


@pytest.mark.asyncio
async def test_public_create_delete_preserves_context_across_tool_tasks(runtime):
    created = await asyncio.create_task(invoke(team_create, team_name="public-workflow"))
    assert created["status"] == "created", created
    deleted = await asyncio.create_task(invoke(team_delete))
    assert deleted["status"] == "deleted", deleted
    assert deleted["team_id"] == created["team_id"]
    again = await invoke(team_delete)
    assert again["status"] == "error"
    assert "No team context" in again["error"]


@pytest.mark.asyncio
async def test_public_team_spawn_registers_a_real_member(runtime):
    _, observed = runtime
    created = await invoke(team_create, team_name="public-workflow")
    launched = await launch(runtime)
    members = TeamService().member_records(created["team_id"])
    assert [(member.agent_id, member.name) for member in members] == [
        (launched["agent_id"], "worker")
    ]
    assert observed[0]["team_context"].team_id == created["team_id"]
    assert observed[0]["team_context"].sender_agent_id == launched["agent_id"]


@pytest.mark.asyncio
async def test_public_leader_followup_is_consumed_by_the_idle_teammate(runtime):
    service, observed = runtime
    created = await invoke(team_create, team_name="public-workflow")
    launched = await launch(runtime)
    await asyncio.wait_for(service.team_tool_runtime.runner.wait(launched["agent_id"]), 3)
    sent = await asyncio.create_task(
        invoke(send_message, to="worker", message="Follow-up team work")
    )
    assert sent["status"] == "sent", sent
    assert sent["routing"] == "team_mailbox", sent
    assert sent["team"] == created["team_id"]
    await eventually(lambda: len(observed) == 2)
    assert observed[1]["prompt"] == "Follow-up team work"
    assert observed[1]["team_context"].source == "mailbox"
    assert observed[1]["session_id"] == observed[0]["session_id"]


@pytest.mark.asyncio
async def test_public_teammate_consumes_shared_task_and_blocks_early_delete(runtime):
    service, observed = runtime
    await invoke(team_create, team_name="public-workflow")
    launched = await launch(runtime)
    tasks = TeamService().task_service("public-workflow")
    task = tasks.create_task("Queued shared task")
    await eventually(lambda: tasks.get_task(task.id).status == "completed")
    assert observed[-1]["prompt"] == "Queued shared task"
    assert observed[-1]["team_context"].source == "task"
    blocked = await invoke(team_delete)
    assert blocked["status"] == "error" and "active team" in blocked["error"]
    await service.team_tool_runtime.runner.terminate(launched["agent_id"])
    deleted = await invoke(team_delete)
    assert deleted["status"] == "deleted", deleted


@pytest.mark.asyncio
async def test_public_team_resume_queues_work_without_bypassing_its_consumer(runtime):
    service, observed = runtime
    await invoke(team_create, team_name="public-workflow")
    launched = await launch(runtime)
    await asyncio.wait_for(service.team_tool_runtime.runner.wait(launched["agent_id"]), 3)
    resumed = await invoke(
        agent_tool, description="Continue teammate", prompt="Resume follow-up", resume="worker"
    )
    assert resumed["status"] == "async_resumed", resumed
    assert resumed["delivery"] == "queued"
    assert resumed["agent_id"] == launched["agent_id"]
    await eventually(lambda: len(observed) == 2)
    assert observed[-1]["team_context"].source == "mailbox"
    assert observed[-1]["prompt"] == "Resume follow-up"


@pytest.mark.asyncio
async def test_leader_selection_is_separate_for_sessions_sharing_an_agent_service(runtime):
    first = SimpleNamespace(session_id="first")
    second = SimpleNamespace(session_id="second")
    with agent_session_scope(first):
        assert (await invoke(team_create, team_name="first-team"))["status"] == "created"
    with agent_session_scope(second):
        missing = await invoke(team_delete)
        assert missing["status"] == "error"
        assert (await invoke(team_create, team_name="second-team"))["status"] == "created"
    with agent_session_scope(first):
        deleted = await asyncio.create_task(invoke(team_delete))
        assert deleted["team_id"] == "first-team", deleted
    with agent_session_scope(second):
        assert TeamService().get("second-team")
        deleted = await invoke(team_delete)
        assert deleted["team_id"] == "second-team", deleted


@pytest.mark.asyncio
async def test_another_runtime_does_not_inherit_the_leaders_selection(runtime, tmp_path):
    await invoke(team_create, team_name="public-workflow")
    other = agent_module.AgentService.for_test(tmp_path / "other")
    try:
        with agent_service_scope(other):
            result = await invoke(team_delete)
            assert result["status"] == "error"
            assert "No team context" in result["error"]
    finally:
        await other.aclose()
    assert TeamService().get("public-workflow")


@pytest.mark.asyncio
async def test_retired_leader_selection_cannot_delete_recreated_team(runtime):
    await invoke(team_create, team_name="public-workflow")
    outsider = TeamService()
    outsider.delete_team("public-workflow")
    outsider.create_team("public-workflow")
    replacement = outsider.get("public-workflow")
    result = await invoke(team_delete)
    assert result["status"] == "error", result
    assert outsider.get("public-workflow").generation == replacement.generation


@pytest.mark.asyncio
async def test_member_context_takes_precedence_and_cannot_delete_the_team(runtime):
    _, observed = runtime
    await invoke(team_create, team_name="public-workflow")
    await launch(runtime)
    with team_tool_context(observed[0]["team_context"]):
        sent = await invoke(send_message, to="team-lead", message="Member reply")
        assert sent["routing"] == "team_mailbox" and sent["sender"] == "worker"
        deleted = await invoke(team_delete)
        assert deleted["status"] == "error"
        assert "leader" in deleted["error"]
    assert any(
        entry.content == "Member reply" for entry in TeamService().read_mailbox("public-workflow")
    )


@pytest.mark.asyncio
async def test_public_team_spawn_preserves_fork_seed_and_plan_mode(runtime):
    service, observed = runtime
    seed = [{"role": "user", "content": "parent conversation"}]

    async def get_items():
        return seed

    parent = SimpleNamespace(session_id="parent", get_items=get_items)
    with agent_session_scope(parent):
        await invoke(team_create, team_name="public-workflow")
        launched = await launch(runtime, context="fork", mode="plan")
    assert observed[0]["seed_items"] == seed
    assert observed[0]["observed_plan_mode"] is True
    assert service.get(launched["agent_id"]).permission_mode == "plan"
    member = TeamService().member_records("public-workflow")[0]
    assert member.mode == "plan" and member.plan_mode_required


@pytest.mark.asyncio
async def test_foreground_team_call_returns_initial_result_and_keeps_member_alive(runtime):
    service, _ = runtime
    await invoke(team_create, team_name="public-workflow")
    launched = await launch(runtime, run_in_background=False)
    assert launched["status"] == "completed", launched
    assert launched["result"] == "synthetic teammate result"
    assert service.team_tool_runtime.has_live_members
    with pytest.raises(RuntimeError, match="aclose"):
        service.close()
    await service.aclose()
    assert not service.team_tool_runtime.has_live_members
    assert not TeamService().member_records("public-workflow")[0].is_active


@pytest.mark.asyncio
async def test_cancelled_foreground_team_call_joins_its_member(runtime, monkeypatch):
    service, _ = runtime
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def execute(**_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(agent_module, "_execute_agent_run", execute)
    await invoke(team_create, team_name="public-workflow")
    call = asyncio.create_task(
        invoke(
            agent_tool,
            description="Cancelled teammate",
            prompt="Wait",
            team_name="public-workflow",
            name="worker",
            run_in_background=False,
        )
    )
    await asyncio.wait_for(started.wait(), 3)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    assert stopped.is_set()
    assert not service.team_tool_runtime.has_live_members
    assert not TeamService().member_records("public-workflow")[0].is_active


@pytest.mark.parametrize("team_name", ["", "missing-team"])
@pytest.mark.asyncio
async def test_invalid_team_does_not_fall_back_to_a_generic_agent(runtime, team_name):
    service, observed = runtime
    result = await invoke(
        agent_tool, description="Invalid team", prompt="Do not run", team_name=team_name
    )
    assert result["status"] == "error", result
    assert service.list_records() == [] and observed == []


@pytest.mark.asyncio
async def test_background_disable_is_honored_for_team_members(runtime, monkeypatch):
    service, observed = runtime
    await invoke(team_create, team_name="public-workflow")
    monkeypatch.setenv("KODER_DISABLE_BACKGROUND_TASKS", "1")
    result = await invoke(
        agent_tool,
        description="Disabled teammate",
        prompt="Do not run",
        team_name="public-workflow",
    )
    assert result["status"] == "error", result
    assert service.list_records() == [] and observed == []


@pytest.mark.asyncio
async def test_public_resume_rejects_changed_definition_before_queueing(runtime):
    service, observed = runtime
    await invoke(team_create, team_name="public-workflow")
    launched = await launch(runtime)
    await asyncio.wait_for(service.team_tool_runtime.runner.wait(launched["agent_id"]), 3)
    before = TeamService().member_records("public-workflow")[0]
    resumed = await invoke(
        agent_tool,
        description="Update teammate",
        prompt="Use updated model",
        resume=launched["agent_id"],
        model="synthetic/replacement",
    )
    assert resumed["status"] == "error", resumed
    assert "provenance" in resumed["error"]
    assert len(observed) == 1
    after = TeamService().member_records("public-workflow")[0]
    assert after.model == before.model
    assert after.generation == before.generation
    assert service.team_tool_runtime.manages(launched["agent_id"])
    sent = await invoke(send_message, to="worker", message="Valid follow-up")
    assert sent["status"] == "sent"
    await eventually(lambda: len(observed) == 2)
    assert observed[-1]["prompt"] == "Valid follow-up"


@pytest.mark.asyncio
async def test_public_teammate_worktree_input_reaches_execution(runtime, tmp_path):
    _, observed = runtime
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.name", "Synthetic Test"],
        ["git", "config", "user.email", "synthetic@example.invalid"],
        ["git", "commit", "--allow-empty", "-qm", "initial"],
    ):
        subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    await invoke(team_create, team_name="public-workflow")
    launched = await launch(runtime, isolation="worktree")
    execution_cwd = Path(observed[0]["cwd"])
    assert execution_cwd != tmp_path
    assert execution_cwd.is_relative_to(tmp_path / ".koder" / "worktrees")
    assert observed[0]["agent_definition"].isolation == "worktree"
    member = TeamService().member_records("public-workflow")[0]
    assert member.agent_id == launched["agent_id"]
    assert member.worktree_path == str(execution_cwd)


@pytest.mark.asyncio
async def test_interactive_commands_and_public_tools_share_the_runner(runtime):
    from koder_agent.harness.commands.interactive import HarnessInteractiveCommandHandler

    service, _ = runtime
    handler = HarnessInteractiveCommandHandler(agent_service=service, emit_console=False)
    await invoke(team_create, team_name="public-workflow")
    launched = await launch(runtime)
    assert service.team_tool_runtime.runner is handler.in_process_teammate_runner
    assert service.team_tool_runtime.team_service is handler.team_service
    assert handler.in_process_teammate_runner.manages(launched["agent_id"])
    await handler.in_process_teammate_runner.terminate(launched["agent_id"])
    assert (await invoke(team_delete))["status"] == "deleted"


@pytest.mark.asyncio
async def test_failed_mailbox_start_is_recorded_and_consumer_exception_is_observed(
    runtime, monkeypatch, caplog
):
    service, _ = runtime
    await invoke(team_create, team_name="public-workflow")
    launched = await launch(runtime)
    await asyncio.wait_for(service.team_tool_runtime.runner.wait(launched["agent_id"]), 3)

    async def cannot_resume(**_kwargs):
        raise OSError("synthetic startup failure")

    monkeypatch.setattr(service, "resume_background", cannot_resume)
    sent = await invoke(send_message, to="worker", message="Failed mailbox work")
    assert sent["status"] == "sent"
    await eventually(lambda: not service.team_tool_runtime.manages(launched["agent_id"]))
    await asyncio.sleep(0)
    teams = TeamService()
    assert any(
        "Could not start teammate work (OSError)" in message.content
        for message in teams.read_mailbox("public-workflow")
    )
    history = Path(teams.get("public-workflow").config_path).with_name("history.jsonl")
    events = [json.loads(line) for line in history.read_text().splitlines()]
    assert any(
        entry.get("prompt") == "Failed mailbox work" and entry.get("state") == "failed"
        for entry in events
    )
    assert "consumer worker stopped (OSError)" in caplog.text
    assert "Task exception was never retrieved" not in caplog.text


@pytest.mark.parametrize("as_member", [False, True])
@pytest.mark.asyncio
async def test_closing_owner_rejects_new_team_mail(runtime, monkeypatch, as_member):
    service, _ = runtime
    started, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    member_context = []

    async def execute(**kwargs):
        member_context.append(kwargs["team_context"])
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()

    monkeypatch.setattr(agent_module, "_execute_agent_run", execute)
    await invoke(team_create, team_name="public-workflow")
    await invoke(
        agent_tool,
        description="Closing worker",
        prompt="Wait for shutdown",
        name="worker",
        team_name="public-workflow",
        run_in_background=True,
    )
    await asyncio.wait_for(started.wait(), 3)
    close_task = asyncio.create_task(service.aclose())
    try:
        await asyncio.wait_for(closing.wait(), 3)
        with team_tool_context(member_context[0] if as_member else None):
            result = await invoke(send_message, to="worker", message="Too late")
        assert result["status"] == "error", result
        assert not any(
            message.content == "Too late"
            for message in TeamService().read_mailbox("public-workflow", recipient="worker")
        )
    finally:
        release.set()
        await close_task


@pytest.mark.asyncio
async def test_explicit_model_is_preserved_on_spawn_and_compatible_resume(runtime):
    service, observed = runtime
    await invoke(team_create, team_name="public-workflow")
    launched = await launch(runtime, model="synthetic/original")
    await asyncio.wait_for(service.team_tool_runtime.runner.wait(launched["agent_id"]), 3)
    assert observed[0]["agent_definition"].model == "synthetic/original"
    resumed = await invoke(
        agent_tool,
        description="Compatible resume",
        prompt="Same definition",
        resume=launched["agent_id"],
        model="synthetic/original",
    )
    assert resumed["status"] == "async_resumed", resumed
    await eventually(lambda: len(observed) == 2)
    assert observed[-1]["agent_definition"].model == "synthetic/original"
