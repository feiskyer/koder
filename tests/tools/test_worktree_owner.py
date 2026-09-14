"""Caller-level worktree ownership checks using disposable real Git repositories."""

import asyncio
import gc
import json
import subprocess
import threading
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.tool_context import ToolContext

from koder_agent.core.scheduler import _GoalTurnLifecycle
from koder_agent.harness.agents.definitions import AgentDefinition
from koder_agent.harness.agents.service import AgentService, _execute_agent_run
from koder_agent.harness.execution_context import execution_directory, get_execution_cwd
from koder_agent.harness.plugins.context import get_plugin_root
from koder_agent.harness.session_flow import (
    _SchedulerBuilder,
    _SchedulerState,
    _switch_active_session,
)
from koder_agent.tools.todo import (
    TodoRuntimeIdentity,
    TodoStore,
    reset_todo_context,
    set_todo_context,
)
from koder_agent.tools.worktree import (
    WorktreeLifecycle,
    _get_worktree_owner,
    _get_worktree_session,
    _set_worktree_session,
    enter_worktree_tool,
    exit_worktree_tool,
)


def git(repo, *arguments):
    return subprocess.run(
        ["git", *arguments], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def make_repo(path):
    path.mkdir()
    git(path, "init")
    git(path, "config", "user.email", "isolated@test.invalid")
    git(path, "config", "user.name", "Isolated Test")
    (path / ".gitignore").write_text(".koder/\nignored.log\n")
    (path / "tracked.txt").write_text("initial\n")
    git(path, "add", ".gitignore", "tracked.txt")
    git(path, "commit", "-m", "initial")
    return path.resolve()


def store(session="parent", agent="main", run="run"):
    return TodoStore(TodoRuntimeIdentity(session, agent, run))


@contextmanager
def owner_scope(owner):
    token = set_todo_context(owner)
    try:
        yield
    finally:
        reset_todo_context(token)


async def invoke(tool, **arguments):
    encoded = json.dumps(arguments)
    context = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="isolated-owner-proof",
        tool_arguments=encoded,
    )
    return json.loads(await tool.on_invoke_tool(context, encoded))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    # Every Git mutation, including discard/replacement cases, is below tmp_path.
    path = make_repo(tmp_path / "repo")
    monkeypatch.chdir(path)
    _set_worktree_session(None)
    yield path
    _set_worktree_session(None)


def main_scheduler(owner):
    return SimpleNamespace(
        session=None,
        todo_store=owner,
        plugin_root=get_plugin_root(),
        permission_service=None,
        approver=None,
        goal_runtime=SimpleNamespace(on_turn_start=AsyncMock()),
        _goal_cumulative_tokens=lambda: 0,
        _finish_goal_turn=AsyncMock(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["keep", "remove"])
async def test_main_session_switch_cannot_consume_previous_owner(repo, action):
    first, second = main_scheduler(store("first")), main_scheduler(store("second"))
    async with _GoalTurnLifecycle(first):
        entered = await asyncio.create_task(invoke(enter_worktree_tool, name="first"))
        session = _get_worktree_session()
    async with _GoalTurnLifecycle(second):
        result = await asyncio.create_task(
            invoke(exit_worktree_tool, action=action, discard_changes=True)
        )
        assert "No active" in result.get("message", ""), result
    assert Path(entered["worktree_path"]).exists()
    async with _GoalTurnLifecycle(first):
        assert _get_worktree_session() is session
        result = await asyncio.create_task(invoke(exit_worktree_tool, action="remove"))
        assert result["branch_deleted"] is True


@pytest.mark.asyncio
async def test_sibling_tools_share_only_their_runtime_owner(repo):
    first, second = store("first"), store("second")
    with owner_scope(first):
        entered_first = await asyncio.create_task(invoke(enter_worktree_tool, name="first"))
    with owner_scope(second):
        entered_second = await asyncio.create_task(invoke(enter_worktree_tool, name="second"))
        assert entered_second.get("worktree_created") is True, entered_second
        assert (await invoke(exit_worktree_tool, action="keep"))["action"] == "keep"
    with owner_scope(first):
        removed = await asyncio.create_task(invoke(exit_worktree_tool, action="remove"))
        assert removed["worktree_path"] == entered_first["worktree_path"]
    assert Path(entered_second["worktree_path"]).exists()


@pytest.mark.asyncio
async def test_equal_labels_do_not_transfer_ownership_between_independent_runtimes(repo):
    first, independent = store(), store()
    assert first.identity == independent.identity
    with owner_scope(first):
        entered = await invoke(enter_worktree_tool, name="private-owner")
    with owner_scope(independent):
        result = await invoke(exit_worktree_tool, action="remove", discard_changes=True)
        assert "No active" in result.get("message", ""), result
    with owner_scope(first):
        assert _get_worktree_session().worktree_path == entered["worktree_path"]
        await invoke(exit_worktree_tool, action="remove")


@pytest.mark.asyncio
async def test_direct_calls_cannot_consume_scoped_worktree(repo):
    actor = store()
    with owner_scope(actor):
        entered = await invoke(enter_worktree_tool, name="scoped")
    result = await invoke(exit_worktree_tool, action="remove", discard_changes=True)
    assert "No active" in result.get("message", ""), result
    assert Path(entered["worktree_path"]).exists()
    with owner_scope(actor):
        await invoke(exit_worktree_tool, action="remove")


@pytest.mark.asyncio
async def test_creation_uses_canonical_execution_directory(repo, tmp_path):
    child = make_repo(tmp_path / "child")
    subdir = child / "subdir"
    subdir.mkdir()
    with owner_scope(store("child")), execution_directory(subdir):
        entered = await invoke(enter_worktree_tool, name="child")
        assert entered["owner_root"] == str(child)
        assert entered["original_cwd"] == str(subdir)
        assert get_execution_cwd() == subdir
        assert Path.cwd() == repo
        await invoke(exit_worktree_tool, action="remove")
    assert not (repo / ".koder").exists()


@pytest.fixture
def model_free_runner(monkeypatch):
    operations = {}

    async def run(_agent, prompt, **_kwargs):
        return SimpleNamespace(final_output=await operations[prompt]())

    monkeypatch.setattr(
        "koder_agent.harness.agents.service.EnhancedSQLiteSession",
        lambda session_id: SimpleNamespace(session_id=session_id, close=lambda: None),
    )
    monkeypatch.setattr(
        "koder_agent.harness.agents.service.create_dev_agent",
        AsyncMock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr(
        "koder_agent.harness.agents.service._cleanup_agent_mcp_servers", AsyncMock()
    )
    monkeypatch.setattr("koder_agent.harness.agents.service.Runner.run", run)
    return operations


def definition(*, isolation=None):
    return AgentDefinition(
        agent_type="owner-proof",
        when_to_use="Isolated ownership tests",
        system_prompt="No model calls",
        source="built-in",
        isolation=isolation,
    )


@pytest.mark.asyncio
async def test_real_subagent_runner_cannot_exit_parent_worktree(repo, model_free_runner):
    parent = store()
    model_free_runner["child"] = lambda: invoke(
        exit_worktree_tool, action="remove", discard_changes=True
    )
    with owner_scope(parent):
        entered = await invoke(enter_worktree_tool, name="parent")
        output = await _execute_agent_run(
            agent_definition=definition(),
            prompt="child",
            session_id="subagent-proof",
            seed_items=None,
            cwd=str(repo),
        )
        assert "No active" in output, output
        assert Path(entered["worktree_path"]).exists()
        assert _get_worktree_session().name == "parent"
        await invoke(exit_worktree_tool, action="remove")


@pytest.mark.asyncio
@pytest.mark.parametrize("isolation", [None, "worktree"])
async def test_background_resume_retains_its_own_interactive_worktree(
    repo, tmp_path, model_free_runner, isolation
):
    service = AgentService(output_root=tmp_path / "agents")
    entered = {}

    async def enter():
        entered.update(await invoke(enter_worktree_tool, name="background"))
        return "entered"

    async def remove():
        result = await invoke(exit_worktree_tool, action="remove")
        assert result["worktree_path"] == entered["worktree_path"]
        assert result["branch_deleted"] is True
        return "removed"

    model_free_runner.update(enter=enter, remove=remove)
    with owner_scope(store("parent")):
        parent = await invoke(enter_worktree_tool, name="parent")
        record = await service.launch_background(
            agent_definition=definition(isolation=isolation),
            prompt="enter",
            description="owner proof",
            cwd=str(repo),
        )
        completed = await service.wait(record.id)
        assert completed.state == "completed", completed.error
        assert entered.get("worktree_created") is True, entered
        if isolation:
            assert entered["owner_root"] == record.worktree_path
            assert completed.worktree_path == record.worktree_path
        await service.resume_background(
            agent_id=record.id,
            agent_definition=definition(isolation=isolation),
            prompt="remove",
        )
        completed = await service.wait(record.id)
        assert completed.state == "completed", completed.error
        assert Path(parent["worktree_path"]).exists()
        assert _get_worktree_session().name == "parent"
        await invoke(exit_worktree_tool, action="remove")


@pytest.mark.asyncio
async def test_branch_cleanup_retry_remains_with_owner(repo, monkeypatch):
    first, second = store("first"), store("second")
    real_delete = WorktreeLifecycle.delete_owned_branch
    with owner_scope(first):
        entered = await invoke(enter_worktree_tool, name="pending")
        monkeypatch.setattr(
            WorktreeLifecycle, "delete_owned_branch", lambda *_: OSError("pause cleanup")
        )
        result = await invoke(exit_worktree_tool, action="remove")
        assert result["session_state"] == "branch_cleanup_pending"
        session = _get_worktree_session()
    monkeypatch.setattr(WorktreeLifecycle, "delete_owned_branch", real_delete)
    with owner_scope(second):
        result = await invoke(exit_worktree_tool, action="remove", discard_changes=True)
        assert "No active" in result.get("message", ""), result
    with owner_scope(first):
        assert _get_worktree_session() is session
        result = await invoke(exit_worktree_tool, action="remove")
        assert result["branch_deleted"] is True
    assert not Path(entered["worktree_path"]).exists()


@pytest.mark.asyncio
async def test_owner_keep_preserves_unmerged_and_ignored_work(repo):
    with owner_scope(store()):
        entered = await invoke(enter_worktree_tool, name="valuable")
        path = Path(entered["worktree_path"])
        (path / "committed.txt").write_text("unmerged work")
        git(path, "add", "committed.txt")
        git(path, "commit", "-m", "unmerged")
        commit = git(path, "rev-parse", "HEAD")
        (path / "ignored.log").write_text("local work")
        blocked = await invoke(exit_worktree_tool, action="remove")
        assert blocked["error_code"] == "worktree_not_clean"
        await invoke(exit_worktree_tool, action="keep")
        assert _get_worktree_session() is None
        assert (path / "ignored.log").read_text() == "local work"
        assert git(path, "rev-parse", "HEAD") == commit


@pytest.mark.asyncio
async def test_cancelled_background_owner_can_resume_cleanup(repo, tmp_path, model_free_runner):
    service = AgentService(output_root=tmp_path / "cancel-agents")
    ready = asyncio.Event()
    entered = {}

    async def enter_and_wait():
        entered.update(await invoke(enter_worktree_tool, name="cancelled"))
        ready.set()
        await asyncio.Event().wait()

    async def cleanup():
        result = await invoke(exit_worktree_tool, action="remove")
        assert result["worktree_path"] == entered["worktree_path"]
        assert result["branch_deleted"] is True
        return "removed"

    model_free_runner.update(wait=enter_and_wait, cleanup=cleanup)
    record = await service.launch_background(
        agent_definition=definition(),
        prompt="wait",
        description="cancelled owner proof",
        cwd=repo,
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
    finally:
        await service.cancel_background(record.id)
    assert Path(entered["worktree_path"]).exists()
    with owner_scope(store("unrelated")):
        result = await invoke(exit_worktree_tool, action="remove", discard_changes=True)
        assert "No active" in result["message"]
    await service.resume_background(
        agent_id=record.id, agent_definition=definition(), prompt="cleanup"
    )
    completed = await service.wait(record.id)
    assert completed.state == "completed", completed.error


@pytest.mark.asyncio
async def test_owner_lifetime_releases_memory_without_deleting_work(repo):
    runtime = store()
    with owner_scope(runtime):
        entered = await invoke(enter_worktree_tool, name="retained-on-release")
        state = _get_worktree_owner()
        runtime_ref, owner_ref = weakref.ref(runtime), weakref.ref(state)
    del runtime, state
    gc.collect()
    assert runtime_ref() is None
    assert owner_ref() is None
    assert Path(entered["worktree_path"]).exists()
    assert git(repo, "rev-parse", entered["worktree_branch"])


@pytest.mark.asyncio
async def test_simultaneous_owners_use_separate_state(repo, model_free_runner):
    first_ready, second_ready = asyncio.Event(), asyncio.Event()

    async def operation(label, ready, peer):
        entered = await asyncio.create_task(invoke(enter_worktree_tool, name=label))
        assert entered.get("worktree_created") is True, entered
        ready.set()
        await asyncio.wait_for(peer.wait(), timeout=5)
        removed = await asyncio.create_task(invoke(exit_worktree_tool, action="remove"))
        assert removed["worktree_path"] == entered["worktree_path"]
        assert removed["branch_deleted"] is True
        return label

    model_free_runner["first"] = lambda: operation("first", first_ready, second_ready)
    model_free_runner["second"] = lambda: operation("second", second_ready, first_ready)

    async def run(label):
        return await _execute_agent_run(
            agent_definition=definition(),
            prompt=label,
            session_id=label,
            seed_items=None,
            cwd=str(repo),
        )

    assert await asyncio.gather(run("first"), run("second")) == ["first", "second"]
    assert Path.cwd() == repo


@pytest.mark.asyncio
async def test_same_owner_thread_calls_cannot_replace_unfinished_creation(repo):
    ready = threading.Barrier(2)

    def enter(label):
        ready.wait(timeout=5)
        return asyncio.run(invoke(enter_worktree_tool, name=label))

    with owner_scope(store()):
        results = await asyncio.gather(
            asyncio.to_thread(enter, "thread-first"),
            asyncio.to_thread(enter, "thread-second"),
        )
        created = [result for result in results if result.get("worktree_created")]
        assert len(created) == 1, results
        assert sum("Already in" in result["message"] for result in results) == 1
        assert _get_worktree_session().worktree_path == created[0]["worktree_path"]
        removed = await invoke(exit_worktree_tool, action="remove")
        assert removed["branch_deleted"] is True
    assert len(git(repo, "worktree", "list", "--porcelain").split("worktree ")) == 2


@pytest.mark.asyncio
async def test_actual_session_switch_round_trip_retains_worktree_owner(repo, monkeypatch):
    """Use real switch/builder/turn/tool entrypoints; only storage/model state is synthetic."""

    class Metadata:
        def __init__(self, session_id):
            self.session_id = session_id

        async def get_cwd(self):
            return str(repo)

        async def get_agent(self):
            return None

    def scheduler_type(session_id, todo_store=None, **_kwargs):
        scheduler = main_scheduler(todo_store or store(session_id))
        scheduler.session = Metadata(session_id)
        scheduler.cleanup = AsyncMock()
        return scheduler

    monkeypatch.setattr("koder_agent.core.session.EnhancedSQLiteSession", Metadata)
    monkeypatch.setattr(
        "koder_agent.harness.session_flow.get_agent_definitions",
        lambda **_kwargs: SimpleNamespace(active_agents=[]),
    )
    builder = _SchedulerBuilder(
        scheduler_type=scheduler_type,
        streaming=False,
        agent_definition=None,
        instructions_override=None,
        instructions_append=None,
        permission_service=None,
        approver=None,
        agent_definitions=None,
    )
    state = _SchedulerState.create(builder, "first")
    args = SimpleNamespace(session="first", agent=None)
    try:
        async with _GoalTurnLifecycle(state.scheduler):
            entered = await invoke(enter_worktree_tool, name="switch-round-trip")
            original = _get_worktree_session()
        await _switch_active_session(state, args, "second")
        async with _GoalTurnLifecycle(state.scheduler):
            result = await invoke(exit_worktree_tool, action="remove", discard_changes=True)
            assert "No active" in result["message"]
        await _switch_active_session(state, args, "first")
        async with _GoalTurnLifecycle(state.scheduler):
            assert _get_worktree_session() is original
            result = await invoke(exit_worktree_tool, action="remove")
            assert result["branch_deleted"] is True
        assert not Path(entered["worktree_path"]).exists()
    finally:
        await state.cleanup()
