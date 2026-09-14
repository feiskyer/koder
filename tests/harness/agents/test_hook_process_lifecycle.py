"""Isolated real subprocess coverage of the subagent hook boundary."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from koder_agent.harness.agents import hooks as hooks_mod
from koder_agent.harness.agents.definitions import AgentDefinition
from koder_agent.harness.hooks import runtime


def _python_argv(script, *args):
    return [
        "uv",
        "run",
        "--no-project",
        "--no-env-file",
        "--python",
        sys.executable,
        "python",
        "-c",
        script,
        *(str(arg) for arg in args),
    ]


def _command(script, *args):
    return shlex.join(_python_argv(script, *args))


def _marker_command(path):
    return _command("import pathlib,sys; pathlib.Path(sys.argv[1]).touch()", path)


def _tree_command(tmp_path):
    child = _python_argv(
        "import pathlib,sys,time; "
        "pathlib.Path(sys.argv[1]).touch(); "
        "time.sleep(1); pathlib.Path(sys.argv[2]).touch()",
        tmp_path / "ready",
        tmp_path / "late-child",
    )
    # The parent waits for a real child; no shell chaining or shell exec
    # optimization is needed to prove descendant ownership.
    return _command(
        "import json,pathlib,subprocess,sys; "
        "subprocess.run(json.loads(sys.argv[1]), check=True); "
        "pathlib.Path(sys.argv[2]).touch()",
        json.dumps(child),
        tmp_path / "late-parent",
    )


@pytest.fixture(autouse=True)
def isolated_scopes(monkeypatch):
    @contextmanager
    def scopes(_cwd):
        yield []

    monkeypatch.setattr(runtime, "load_hook_scopes", scopes)
    # No child resolves project dependencies or .env files.
    monkeypatch.delenv("UV_NO_SYNC", raising=False)


def _lifecycle(tmp_path, event, commands, monkeypatch):
    rules = [{"hooks": commands}]
    settings_event = event in {"SubagentStart", "SubagentStop"}
    if settings_event:

        @contextmanager
        def scopes(_cwd):
            yield [
                runtime.HookScope(
                    source="user_settings",
                    file_path=tmp_path / "settings.json",
                    hooks={event: rules},
                )
            ]

        monkeypatch.setattr(runtime, "load_hook_scopes", scopes)
    definition = AgentDefinition(
        agent_type="reviewer",
        when_to_use="Reviews code",
        system_prompt="Review",
        source="projectSettings",
        hooks={} if settings_event else {event: rules},
    )
    return hooks_mod.SubagentLifecycleHooks(agent_definition=definition, cwd=tmp_path)


async def _invoke(hooks, event):
    agent = SimpleNamespace(name="reviewer")
    tool = SimpleNamespace(name="read_file")
    if event == "SubagentStart":
        await hooks.on_agent_start(None, agent)
    elif event in {"SubagentStop", "Stop"}:
        await hooks.on_agent_end(None, agent, "done")
    elif event == "PreToolUse":
        await hooks.on_tool_start(None, agent, tool)
    else:
        await hooks.on_tool_end(None, agent, tool, "ok")


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup")
def test_sync_frontmatter_timeout_stops_running_subtree_and_later_hooks(tmp_path):
    rules = [
        {
            "hooks": [
                {"type": "command", "command": _tree_command(tmp_path), "timeout": 0.5},
                {"type": "command", "command": _marker_command(tmp_path / "later-hook")},
            ]
        }
    ]
    error = None
    try:
        hooks_mod._run_command_hooks(rules, {"event": "PreToolUse"}, tmp_path)
    except PermissionError as exc:
        error = exc
    time.sleep(1.2)
    assert (tmp_path / "ready").exists(), "child must start for this test to be meaningful"
    assert not (tmp_path / "late-child").exists(), "descendant wrote after hook timeout"
    assert not (tmp_path / "late-parent").exists(), "parent wrote after hook timeout"
    assert not (tmp_path / "later-hook").exists(), "timed-out dispatch ran the next hook"
    assert error is not None and "timed out" in str(error).lower()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup")
@pytest.mark.parametrize(
    "event", ["PreToolUse", "PostToolUse", "Stop", "SubagentStart", "SubagentStop"]
)
def test_cancellation_stops_running_subtree_and_later_hooks(tmp_path, monkeypatch, event):
    hooks = _lifecycle(
        tmp_path,
        event,
        [
            {"type": "command", "command": _tree_command(tmp_path), "timeout": 3},
            {"type": "command", "command": _marker_command(tmp_path / "later-hook")},
        ],
        monkeypatch,
    )

    async def exercise():
        task = asyncio.create_task(_invoke(hooks, event))
        try:
            async with asyncio.timeout(4):
                while not (tmp_path / "ready").exists() and not task.done():
                    await asyncio.sleep(0.01)
            task.cancel("cancel-subagent-hook")
            result = (await asyncio.gather(task, return_exceptions=True))[0]
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(1.2)
        return result

    result = asyncio.run(exercise())
    assert (tmp_path / "ready").exists()
    assert not (tmp_path / "late-child").exists(), "cancellation did not interrupt the hook"
    assert not (tmp_path / "late-parent").exists()
    assert not (tmp_path / "later-hook").exists()
    assert isinstance(result, asyncio.CancelledError)


@pytest.mark.parametrize(
    "event", ["PreToolUse", "PostToolUse", "Stop", "SubagentStart", "SubagentStop"]
)
@pytest.mark.parametrize("block", ["exit", "json"])
def test_blocking_hook_raises_and_does_not_run_later_hooks(tmp_path, monkeypatch, event, block):
    command = (
        _command("import sys; sys.stderr.write('synthetic denial'); sys.exit(2)")
        if block == "exit"
        else _command('print(\'{"decision":"block","reason":"synthetic denial"}\')')
    )
    hooks = _lifecycle(
        tmp_path,
        event,
        [
            {"type": "command", "command": command},
            {"type": "command", "command": _marker_command(tmp_path / "later-hook")},
        ],
        monkeypatch,
    )
    with pytest.raises(PermissionError, match="synthetic denial"):
        asyncio.run(_invoke(hooks, event))
    assert not (tmp_path / "later-hook").exists()


@pytest.mark.parametrize("requires_approval", [False, True])
def test_non_guarded_permission_denial_precedes_wrapped_and_frontmatter_hooks(
    tmp_path, monkeypatch, requires_approval
):
    hooks = _lifecycle(
        tmp_path,
        "PreToolUse",
        [{"type": "command", "command": _marker_command(tmp_path / "hook")}],
        monkeypatch,
    )
    calls = []

    class Service:
        def evaluate_tool_call(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                allowed=False, requires_approval=requires_approval, reason="synthetic policy"
            )

    class Wrapped:
        async def on_tool_start(self, *_args):
            calls.append("wrapped")

    hooks._permission_service = Service()
    hooks.wrapped_hooks = Wrapped()
    with pytest.raises(PermissionError, match="synthetic policy"):
        asyncio.run(
            hooks.on_tool_start(None, SimpleNamespace(), SimpleNamespace(name="custom_tool"))
        )
    assert calls == [{"tool_name": "custom_tool", "arguments": {}}]
    assert not (tmp_path / "hook").exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup")
@pytest.mark.parametrize(
    "event", ["PreToolUse", "PostToolUse", "Stop", "SubagentStart", "SubagentStop"]
)
def test_async_timeout_propagates_denial_after_subtree_cleanup(tmp_path, monkeypatch, event):
    hooks = _lifecycle(
        tmp_path,
        event,
        [
            {"type": "command", "command": _tree_command(tmp_path), "timeout": 0.5},
            {"type": "command", "command": _marker_command(tmp_path / "later-hook")},
        ],
        monkeypatch,
    )
    with pytest.raises(PermissionError, match="timed out"):
        asyncio.run(_invoke(hooks, event))
    time.sleep(1.2)
    assert (tmp_path / "ready").exists()
    assert not (tmp_path / "late-child").exists()
    assert not (tmp_path / "late-parent").exists()
    assert not (tmp_path / "later-hook").exists()


@pytest.mark.parametrize("tool_name", ["custom_tool", "run_shell"])
def test_allowed_tool_preserves_wrapped_then_frontmatter_order(tmp_path, monkeypatch, tool_name):
    events = []
    hooks = _lifecycle(
        tmp_path,
        "PreToolUse",
        [
            {"type": "command", "command": "synthetic-first"},
            {"type": "command", "command": "synthetic-second"},
        ],
        monkeypatch,
    )

    class Service:
        def evaluate_tool_call(self, **_kwargs):
            assert tool_name != "run_shell", "guarded tools need their argument-level check"
            events.append("permission")
            return SimpleNamespace(allowed=True, requires_approval=False)

    class Wrapped:
        async def on_tool_start(self, *_args):
            events.append("wrapped")

    def run(command, **kwargs):
        assert json.loads(kwargs["input"])["tool_name"] == tool_name
        events.append(command)
        # Exit 1 is non-blocking in the canonical hook contract; exit 2 blocks.
        return SimpleNamespace(returncode=1, stdout="", stderr="diagnostic")

    hooks._permission_service = Service()
    hooks.wrapped_hooks = Wrapped()
    monkeypatch.setattr(hooks_mod, "run_command", run)
    asyncio.run(hooks.on_tool_start(None, SimpleNamespace(), SimpleNamespace(name=tool_name)))
    expected = ["wrapped", "synthetic-first", "synthetic-second"]
    assert events == (["permission"] + expected if tool_name == "custom_tool" else expected)


def test_blocked_settings_stop_never_runs_frontmatter_stop(tmp_path, monkeypatch):
    hooks = _lifecycle(
        tmp_path,
        "SubagentStop",
        [{"type": "command", "command": _command("import sys; sys.exit(2)")}],
        monkeypatch,
    )
    hooks.frontmatter_hooks = {
        "Stop": [{"hooks": [{"type": "command", "command": _marker_command(tmp_path / "stop")}]}]
    }
    with pytest.raises(PermissionError):
        asyncio.run(_invoke(hooks, "SubagentStop"))
    assert not (tmp_path / "stop").exists()


def test_sync_project_dispatch_keeps_result_contract(tmp_path, monkeypatch):
    expected = runtime.HookDispatchResult(matched_hooks=1, blocked=True, block_reason="denied")
    received = {}

    def dispatch(**kwargs):
        received.update(kwargs)
        return expected

    monkeypatch.setattr(hooks_mod, "dispatch_command_hooks", dispatch)
    assert (
        hooks_mod.dispatch_project_hook_event(
            cwd=tmp_path, event_name="TaskCompleted", match_value="task", payload={"id": "t"}
        )
        is expected
    )
    assert received == {
        "cwd": tmp_path,
        "event_name": "TaskCompleted",
        "match_value": "task",
        "payload": {"id": "t"},
    }


def test_cancel_before_worker_launch_never_starts_command(tmp_path, monkeypatch):
    cancelled = threading.Event()
    cancelled.set()
    rules = [{"hooks": [{"type": "command", "command": _marker_command(tmp_path / "started")}]}]
    with pytest.raises(hooks_mod.HookCommandCancelledError):
        hooks_mod._run_command_hooks(rules, {}, tmp_path, cancel_event=cancelled)
    assert not (tmp_path / "started").exists()


def test_repeated_cancellation_waits_for_owned_worker_cleanup(tmp_path, monkeypatch):
    entered = threading.Event()
    cleaning_up = threading.Event()
    release_cleanup = threading.Event()
    finished = threading.Event()

    def run(command, **kwargs):
        entered.set()
        assert kwargs["cancel_event"].wait(3)
        cleaning_up.set()
        assert release_cleanup.wait(3)
        finished.set()
        raise hooks_mod.HookCommandCancelledError()

    monkeypatch.setattr(hooks_mod, "run_command", run)
    rules = [{"hooks": [{"type": "command", "command": "synthetic-hook"}]}]

    async def wait_for(flag):
        async with asyncio.timeout(3):
            while not flag.is_set():
                await asyncio.sleep(0.01)

    async def exercise():
        task = asyncio.create_task(hooks_mod._run_command_hooks_async(rules, {}, tmp_path))
        try:
            await wait_for(entered)
            task.cancel("original-cancel")
            await wait_for(cleaning_up)
            task.cancel("second-cancel")
            await asyncio.sleep(0)
            assert not task.done(), "cleanup was abandoned on repeated cancellation"
            release_cleanup.set()
            with pytest.raises(asyncio.CancelledError, match="original-cancel"):
                await task
            assert finished.is_set()
        finally:
            release_cleanup.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup")
def test_cancelling_one_subagent_does_not_cancel_another(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_hooks = _lifecycle(
        first, "PreToolUse", [{"type": "command", "command": _tree_command(first)}], monkeypatch
    )
    second_hooks = _lifecycle(
        second, "PreToolUse", [{"type": "command", "command": _tree_command(second)}], monkeypatch
    )

    async def exercise():
        first_task = asyncio.create_task(_invoke(first_hooks, "PreToolUse"))
        second_task = asyncio.create_task(_invoke(second_hooks, "PreToolUse"))
        try:
            async with asyncio.timeout(4):
                while not ((first / "ready").exists() and (second / "ready").exists()):
                    await asyncio.sleep(0.01)
                first_task.cancel("first-only")
                with pytest.raises(asyncio.CancelledError, match="first-only"):
                    await first_task
                await second_task
        finally:
            for task in (first_task, second_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(first_task, second_task, return_exceptions=True)

    asyncio.run(exercise())
    assert not (first / "late-child").exists()
    assert (second / "late-child").exists()
    assert (second / "late-parent").exists()
