"""Real tools must use each concurrent agent's directory, not process cwd."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from agents.tool_context import ToolContext

from koder_agent.core.session import EnhancedSQLiteSession
from koder_agent.harness.agents.definitions import AgentDefinition
from koder_agent.harness.agents.service import _execute_agent_run
from koder_agent.harness.permissions.modes import PermissionMode
from koder_agent.harness.permissions.service import PermissionService
from koder_agent.tools.file import append_file, edit_file, read_file, write_file
from koder_agent.tools.notebook_edit import notebook_edit
from koder_agent.tools.search import glob_search, grep_search
from koder_agent.tools.shell import BackgroundShellManager, run_shell


async def _invoke(tool, **arguments):
    encoded = json.dumps(arguments)
    context = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="synthetic-directory-proof",
        tool_arguments=encoded,
    )
    return await tool.on_invoke_tool(context, encoded)


@pytest.fixture
def agent_runner(monkeypatch, tmp_path):
    origin = tmp_path / "parent"
    origin.mkdir()
    monkeypatch.chdir(origin)
    operations = {}
    sessions = []

    def create_session(session_id):
        session = EnhancedSQLiteSession(session_id, db_path=str(tmp_path / "sessions.db"))
        sessions.append(session)
        return session

    async def create_agent(*_args, **_kwargs):
        return SimpleNamespace()

    async def cleanup(*_args, **_kwargs):
        pass

    async def run_agent(_agent, prompt, **_kwargs):
        assert Path.cwd() == origin, "an agent changed the process-wide cwd"
        output = await operations[prompt]()
        assert Path.cwd() == origin
        return SimpleNamespace(final_output=output)

    monkeypatch.setattr("koder_agent.harness.agents.service.EnhancedSQLiteSession", create_session)
    monkeypatch.setattr("koder_agent.harness.agents.service.create_dev_agent", create_agent)
    monkeypatch.setattr("koder_agent.harness.agents.service._cleanup_agent_mcp_servers", cleanup)
    monkeypatch.setattr("koder_agent.harness.agents.service.Runner.run", run_agent)
    monkeypatch.setattr(
        "koder_agent.harness.tools.shell_executor.resolve_sandbox_settings",
        lambda _cwd: SimpleNamespace(enabled=False),
    )

    async def run(cwd, operation, *, permissions=None):
        key = f"operation-{len(operations)}"
        operations[key] = operation
        return await _execute_agent_run(
            agent_definition=AgentDefinition(
                agent_type="synthetic",
                when_to_use="Directory regression",
                system_prompt="Synthetic worker",
                source="built-in",
                isolation="worktree",
            ),
            prompt=key,
            session_id=f"directory-proof-{cwd.name}",
            seed_items=None,
            cwd=str(cwd),
            permission_service=permissions,
        )

    try:
        yield origin, run
    finally:
        unclosed = [session for session in sessions if not session._closed]
        for session in sessions:
            session.close()
        assert not unclosed, "agent execution did not close its Session"


@pytest.mark.asyncio
async def test_concurrent_agent_writes_are_task_local(agent_runner, tmp_path):
    origin, run = agent_runner
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    left_ready, right_ready = asyncio.Event(), asyncio.Event()

    async def write(label, ready, peer_ready):
        ready.set()
        await asyncio.wait_for(peer_ready.wait(), timeout=2)
        return await _invoke(write_file, path="result.txt", content=label)

    await asyncio.gather(
        run(left, lambda: write("left", left_ready, right_ready)),
        run(right, lambda: write("right", right_ready, left_ready)),
    )
    assert not (origin / "result.txt").exists()
    assert (left / "result.txt").read_text() == "left"
    assert (right / "result.txt").read_text() == "right"
    assert Path.cwd() == origin


@pytest.mark.asyncio
async def test_concurrent_agents_cannot_borrow_read_evidence_in_the_same_directory(
    agent_runner, tmp_path
):
    _origin, run = agent_runner
    child = tmp_path / "shared"
    child.mkdir()
    path = child / "shared.txt"
    path.write_text("shared original", encoding="utf-8")
    read_ready, write_done = asyncio.Event(), asyncio.Event()

    async def reader():
        assert "shared original" in await _invoke(read_file, path="shared.txt")
        read_ready.set()
        await asyncio.wait_for(write_done.wait(), timeout=5)
        return await _invoke(read_file, path="shared.txt")

    async def writer():
        await asyncio.wait_for(read_ready.wait(), timeout=5)
        try:
            return await _invoke(write_file, path="shared.txt", content="borrowed read")
        finally:
            write_done.set()

    async with asyncio.TaskGroup() as tasks:
        reader_task = tasks.create_task(run(child, reader))
        writer_task = tasks.create_task(run(child, writer))
    assert "has not been read" in writer_task.result()
    assert "prior content still in context" in reader_task.result()
    assert path.read_text(encoding="utf-8") == "shared original"


@pytest.mark.asyncio
async def test_read_append_and_edit_share_execution_directory(agent_runner, tmp_path):
    origin, run = agent_runner
    child = tmp_path / "child"
    child.mkdir()
    (origin / "note.txt").write_text("parent canary\n")
    (child / "note.txt").write_text("child value\n")

    async def edit_child():
        output = await _invoke(read_file, path="note.txt")
        assert "child value" in output and "parent canary" not in output
        await _invoke(append_file, path="note.txt", content="more\n")
        await _invoke(read_file, path="note.txt")
        return await _invoke(
            edit_file, path="note.txt", old_string="child value", new_string="updated child"
        )

    await run(child, edit_child)
    assert (origin / "note.txt").read_text() == "parent canary\n"
    assert (child / "note.txt").read_text() == "updated child\nmore\n"


@pytest.mark.asyncio
async def test_notebook_edit_uses_same_directory_as_read_guard(agent_runner, tmp_path):
    origin, run = agent_runner
    child = tmp_path / "child"
    child.mkdir()
    document = {"cells": [{"cell_type": "markdown", "source": "before", "metadata": {}}]}
    (child / "test.ipynb").write_text(json.dumps(document))

    async def edit_child():
        await _invoke(read_file, path="test.ipynb")
        return await _invoke(
            notebook_edit,
            notebook_path="test.ipynb",
            cell_index=0,
            operation="replace",
            new_source="after",
        )

    output = await run(child, edit_child)
    assert "successfully" in output
    assert json.loads((child / "test.ipynb").read_text())["cells"][0]["source"] == "after"
    assert not (origin / "test.ipynb").exists()


@pytest.mark.asyncio
async def test_search_inside_hidden_worktree_root(agent_runner, tmp_path):
    _origin, run = agent_runner
    child = tmp_path / ".koder" / "worktrees" / "child"
    child.mkdir(parents=True)
    (child / "marker.txt").write_text("only in child")
    (child / ".hidden").mkdir()
    (child / ".hidden" / "hidden.txt").write_text("hidden")

    async def search_child():
        glob_output = await _invoke(glob_search, pattern="**/*.txt")
        assert "marker.txt" in glob_output
        assert "hidden.txt" not in glob_output
        return await _invoke(grep_search, pattern="only in child", output_mode="content")

    output = await run(child, search_child)
    assert "only in child" in output


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True])
async def test_shell_process_starts_in_execution_directory(agent_runner, tmp_path, background):
    _origin, run = agent_runner
    child = tmp_path / "child"
    child.mkdir()
    shell_id = None

    async def child_shell():
        nonlocal shell_id
        output = await _invoke(run_shell, command="pwd", run_in_background=background)
        if not background:
            return output
        shell_id = output.split("shell_id: ", 1)[1].splitlines()[0]
        shell = BackgroundShellManager.get(shell_id)
        await asyncio.wait_for(shell.process.wait(), timeout=5)
        monitor = BackgroundShellManager._monitor_tasks.get(shell_id)
        if monitor is not None:
            await asyncio.wait_for(asyncio.shield(monitor), timeout=5)
        return "\n".join(shell.output_lines)

    try:
        output = await run(child, child_shell)
        assert str(child.resolve()) in output
    finally:
        if shell_id is not None:
            await BackgroundShellManager.terminate(shell_id)


@pytest.mark.asyncio
async def test_permission_workspace_follows_child_without_mutating_parent(agent_runner, tmp_path):
    origin, run = agent_runner
    child = origin / ".koder" / "worktrees" / "child"
    child.mkdir(parents=True)
    permissions = PermissionService(mode=PermissionMode.ACCEPT_EDITS, workspace_root=origin)

    async def write_child():
        denied = await _invoke(write_file, path=str(origin / "forbidden.txt"), content="no")
        assert "Permission denied" in denied
        return await _invoke(write_file, path="allowed.txt", content="child")

    output = await run(child, write_child, permissions=permissions)
    assert "Permission denied" not in output
    assert (child / "allowed.txt").read_text() == "child"
    assert not (origin / "forbidden.txt").exists()
    assert permissions.workspace_root == origin


@pytest.mark.asyncio
async def test_absolute_deny_rule_matches_child_relative_path(agent_runner, tmp_path):
    origin, run = agent_runner
    child = tmp_path / "child"
    child.mkdir()
    permissions = PermissionService(
        mode=PermissionMode.ACCEPT_EDITS,
        workspace_root=origin,
        rules={"write_file": {"deny": [str(child / "private.txt")]}},
    )

    async def write_child():
        return await _invoke(write_file, path="private.txt", content="must not write")

    output = await run(child, write_child, permissions=permissions)
    assert "Permission denied" in output
    assert not (origin / "private.txt").exists()
    assert not (child / "private.txt").exists()
