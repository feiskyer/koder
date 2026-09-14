"""Regression coverage for unified scheduler turn finalization."""

import asyncio
import subprocess
import sys
import textwrap
import threading
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from koder_agent.tools.goal import get_goal_runtime
from koder_agent.tools.permission_context import get_tool_permission_context
from koder_agent.tools.skill import Skill
from koder_agent.tools.skill_context import get_active_restrictions, skill_invocation_scope
from koder_agent.tools.todo import TodoRuntimeIdentity, get_todo_store_or_none


def _make_scheduler(session_id: str = "turn-lifecycle"):
    from koder_agent.core.scheduler import AgentScheduler

    with (
        patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
        patch("koder_agent.core.scheduler.get_display_hooks"),
        patch("koder_agent.core.scheduler.ApprovalHooks"),
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as session_cls,
    ):
        session = AsyncMock()
        session.session_id = session_id
        session.db_path = ":memory:"
        session.get_items = AsyncMock(return_value=[])
        session.add_items = AsyncMock()
        session_cls.return_value = session
        scheduler = AgentScheduler(session_id=session_id)

    scheduler.goal_runtime = AsyncMock()
    scheduler.goal_runtime.next_continuation_prompt = AsyncMock(return_value=None)
    scheduler._append_interruption_marker_if_needed = AsyncMock()
    return scheduler


def _assert_turn_contexts_reset():
    assert get_goal_runtime() is None
    assert get_tool_permission_context() is None
    assert get_active_restrictions() is None
    assert get_todo_store_or_none() is None


@pytest.mark.asyncio
async def test_turn_lifecycle_preserves_outer_skill_scope_and_scheduler_todo_until_goal_finish():
    scheduler = _make_scheduler("outer-skill-and-todo")
    scheduler.permission_service = object()
    scheduler.goal_runtime.next_continuation_prompt = AsyncMock(side_effect=["continue", None])
    finalization_started = asyncio.Event()
    release_finalization = asyncio.Event()
    turn_inputs: list[str] = []
    finish_calls = 0

    async def fake_run_turn(user_input, **_kwargs):
        turn_inputs.append(user_input)

        async def assert_child_contexts():
            restrictions = get_active_restrictions()
            assert restrictions is not None
            assert restrictions.allowed_tools == {"read_file"}
            assert get_tool_permission_context() is not None
            assert get_goal_runtime() is scheduler.goal_runtime
            assert get_todo_store_or_none() is scheduler.todo_store

        await asyncio.create_task(assert_child_contexts())
        return user_input

    async def finish_turn(*_args, **_kwargs):
        nonlocal finish_calls
        finish_calls += 1
        assert get_todo_store_or_none() is scheduler.todo_store
        restrictions = get_active_restrictions()
        assert restrictions is not None
        assert restrictions.allowed_tools == {"read_file"}
        if finish_calls == 2:
            finalization_started.set()
            await release_finalization.wait()

    scheduler._run_turn_unlocked = fake_run_turn
    scheduler.goal_runtime.on_turn_end = AsyncMock(side_effect=finish_turn)
    manual_skill = Skill(
        name="read-only",
        description="read only",
        content="Inspect files",
        allowed_tools=["read_file"],
    )

    with skill_invocation_scope(manual_skill):
        task = asyncio.create_task(scheduler.handle("initial", render_output=False))
        await finalization_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release_finalization.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        restored = get_active_restrictions()
        assert restored is not None
        assert restored.allowed_tools == {"read_file"}
        assert get_todo_store_or_none() is None

    assert len(turn_inputs) == 2
    assert turn_inputs[0] == "initial"
    assert turn_inputs[1].startswith("[Goal continuation]")
    assert scheduler.goal_runtime.on_turn_start.await_count == 2
    assert scheduler.goal_runtime.on_turn_end.await_count == 2
    _assert_turn_contexts_reset()


async def _start_background_shell(shell_id: str, *, owner: TodoRuntimeIdentity):
    from koder_agent.tools.shell import BackgroundShell, BackgroundShellManager

    subprocess_kwargs = {} if sys.platform == "win32" else {"start_new_session": True}
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        **subprocess_kwargs,
    )
    BackgroundShellManager.add(
        BackgroundShell(
            shell_id=shell_id,
            command="sleep",
            process=process,
            start_time=time.time(),
            owner=owner,
        )
    )
    await BackgroundShellManager.start_monitor(shell_id)
    return process, BackgroundShellManager._monitor_tasks[shell_id]


@pytest.mark.asyncio
async def test_scheduler_cleanup_joins_workers_tasks_and_owned_resources():
    scheduler = _make_scheduler()
    scheduler.session.close = Mock()
    baseline_threads = {thread.ident for thread in threading.enumerate()}
    current = asyncio.current_task()
    baseline_tasks = {
        task for task in asyncio.all_tasks() if task is not current and not task.done()
    }
    await scheduler.goal_store.get_goal("turn-lifecycle")
    scheduler._title_generation_task = asyncio.create_task(asyncio.Event().wait())
    from koder_agent.tools.shell import BackgroundShell, BackgroundShellManager

    subprocess_kwargs = {} if sys.platform == "win32" else {"start_new_session": True}
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        **subprocess_kwargs,
    )
    shell_id = "scheduler-cleanup-proof"
    BackgroundShellManager.add(
        BackgroundShell(
            shell_id=shell_id,
            command="sleep",
            process=process,
            start_time=time.time(),
            owner=scheduler.todo_store.identity,
        )
    )
    await BackgroundShellManager.start_monitor(shell_id)

    try:
        await asyncio.wait_for(scheduler.cleanup(), timeout=5)
        await asyncio.wait_for(scheduler.cleanup(), timeout=5)
    finally:
        if shell_id in BackgroundShellManager.get_available_ids():
            await BackgroundShellManager.terminate(shell_id)

    leaked_threads = [
        thread
        for thread in threading.enumerate()
        if thread.ident not in baseline_threads and not thread.daemon
    ]
    leaked_tasks = [
        task
        for task in asyncio.all_tasks()
        if task is not current and task not in baseline_tasks and not task.done()
    ]
    loop = asyncio.get_running_loop()
    leaked_transports = [
        transport
        for transport in getattr(loop, "_transports", {}).values()
        if transport is not None and not transport.is_closing()
    ]

    assert leaked_threads == []
    assert leaked_tasks == []
    assert leaked_transports == []
    assert process.returncode is not None
    assert BackgroundShellManager.get_available_ids() == []
    scheduler.session.close.assert_called_once()


@pytest.mark.asyncio
async def test_aborted_replacement_cleanup_preserves_old_session_background_shell():
    from koder_agent.harness.session_flow import _SchedulerState
    from koder_agent.tools.shell import BackgroundShellManager

    old = _make_scheduler("old-session")
    replacement = _make_scheduler("aborted-session")

    class Builder:
        agent_definition = None

        def build(self, session_id):
            assert session_id == "aborted-session"
            return replacement

    state = _SchedulerState(Builder(), old)
    preparation_started = asyncio.Event()

    async def blocked_prepare_commit(_replacement, target):
        preparation_started.set()
        await asyncio.Event().wait()
        return target

    old_process, old_monitor = await _start_background_shell(
        "old-session-shell",
        owner=old.todo_store.identity,
    )
    try:
        switch_task = asyncio.create_task(
            state.switch(
                "aborted-session",
                prepare_commit=blocked_prepare_commit,
            )
        )
        await preparation_started.wait()
        switch_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await switch_task

        replacement_cleanup = state._cleanup_tasks[id(replacement)][1]
        await asyncio.wait_for(replacement_cleanup, timeout=5)
        await asyncio.sleep(0)

        assert state.scheduler is old
        assert old_process.returncode is None
        assert not old_monitor.done()
        assert "old-session-shell" in BackgroundShellManager.get_available_ids()

        await asyncio.wait_for(state.cleanup(), timeout=5)
        await asyncio.sleep(0)

        assert old_process.returncode is not None
        assert old_monitor.done()
        assert BackgroundShellManager.get_available_ids() == []
    finally:
        for shell_id in list(BackgroundShellManager.get_available_ids()):
            await BackgroundShellManager.terminate(shell_id)


@pytest.mark.asyncio
async def test_retired_scheduler_cleanup_preserves_new_session_background_shell():
    from koder_agent.harness.session_flow import _SchedulerState
    from koder_agent.tools.shell import BackgroundShellManager

    old = _make_scheduler("old-session")
    replacement = _make_scheduler("new-session")

    class Builder:
        agent_definition = None

        def build(self, session_id):
            assert session_id == "new-session"
            return replacement

    state = _SchedulerState(Builder(), old)
    retirement_started = asyncio.Event()
    allow_retirement = asyncio.Event()

    async def blocked_reset_agent():
        retirement_started.set()
        await allow_retirement.wait()

    old.reset_agent = blocked_reset_agent
    old_process, old_monitor = await _start_background_shell(
        "old-session-shell",
        owner=old.todo_store.identity,
    )
    foreign_process, foreign_monitor = await _start_background_shell(
        "background-agent-shell",
        owner=TodoRuntimeIdentity(
            session_id="background-agent-session",
            agent_id="worker",
            run_id="background-agent-run",
        ),
    )

    try:
        switched = await state.switch("new-session")
        assert switched is replacement
        await retirement_started.wait()

        new_process, new_monitor = await _start_background_shell(
            "new-session-shell",
            owner=replacement.todo_store.identity,
        )
        allow_retirement.set()
        retirement_task = state._cleanup_tasks[id(old)][1]
        await asyncio.wait_for(retirement_task, timeout=5)
        await asyncio.sleep(0)

        old_transport = getattr(old_process, "_transport", None)
        assert old_process.returncode is not None
        assert old_monitor.done()
        assert old_transport is None or old_transport.is_closing()
        assert "old-session-shell" not in BackgroundShellManager.get_available_ids()

        assert new_process.returncode is None
        assert not new_monitor.done()
        assert "new-session-shell" in BackgroundShellManager.get_available_ids()
        assert foreign_process.returncode is None
        assert not foreign_monitor.done()
        assert "background-agent-shell" in BackgroundShellManager.get_available_ids()

        await asyncio.wait_for(state.cleanup(), timeout=5)
        await asyncio.sleep(0)

        new_transport = getattr(new_process, "_transport", None)
        assert new_process.returncode is not None
        assert new_monitor.done()
        assert new_transport is None or new_transport.is_closing()
        assert foreign_process.returncode is None
        assert not foreign_monitor.done()
        assert BackgroundShellManager.get_available_ids() == ["background-agent-shell"]

        await BackgroundShellManager.terminate("background-agent-shell")
        assert foreign_process.returncode is not None
        assert foreign_monitor.done()
        assert BackgroundShellManager.get_available_ids() == []
    finally:
        allow_retirement.set()
        try:
            await asyncio.wait_for(state.cleanup(), timeout=5)
        except BaseException:
            pass
        for shell_id in list(BackgroundShellManager.get_available_ids()):
            await BackgroundShellManager.terminate(shell_id)


@pytest.mark.asyncio
async def test_scheduler_cleanup_finishes_once_through_repeated_cancellation():
    scheduler = _make_scheduler()
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    async def blocked_close():
        close_started.set()
        await allow_close.wait()

    scheduler.goal_store.close = AsyncMock(side_effect=blocked_close)
    cleanup_task = asyncio.create_task(scheduler.cleanup())
    await close_started.wait()
    cleanup_task.cancel()
    await asyncio.sleep(0)
    cleanup_task.cancel()
    allow_close.set()

    with pytest.raises(asyncio.CancelledError):
        await cleanup_task

    await scheduler.cleanup()
    scheduler.goal_store.close.assert_awaited_once()
    scheduler.session.close.assert_awaited_once()


def test_scheduler_cleanup_allows_process_to_terminate_with_bounded_timeout(
    tmp_path, python_child_environment
):
    script = textwrap.dedent("""
        import asyncio
        import sys
        import threading
        import time
        from unittest.mock import AsyncMock, Mock, patch

        async def main():
            sessions = []

            def make_session(session_id, *_args, **_kwargs):
                session = AsyncMock()
                session.session_id = session_id
                session.db_path = ":memory:"
                session.get_items = AsyncMock(return_value=[])
                session.close = Mock()
                sessions.append(session)
                return session

            with (
                patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
                patch("koder_agent.core.scheduler.get_display_hooks"),
                patch("koder_agent.core.scheduler.ApprovalHooks"),
                patch(
                    "koder_agent.core.scheduler.EnhancedSQLiteSession",
                    side_effect=make_session,
                ),
            ):
                from koder_agent.core.scheduler import AgentScheduler
                from koder_agent.tools.shell import BackgroundShell, BackgroundShellManager

                schedulers = [
                    AgentScheduler(session_id=f"process-cleanup-{index}")
                    for index in range(8)
                ]
                for scheduler in schedulers:
                    await scheduler.goal_store.get_goal(scheduler.session.session_id)
                    scheduler._title_generation_task = asyncio.create_task(
                        asyncio.Event().wait()
                    )

                workers_before_cleanup = [
                    thread.name
                    for thread in threading.enumerate()
                    if thread is not threading.main_thread() and not thread.daemon
                ]
                assert len(workers_before_cleanup) >= 8, workers_before_cleanup

                subprocess_kwargs = (
                    {} if sys.platform == "win32" else {"start_new_session": True}
                )
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-c",
                    "import time; time.sleep(60)",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    **subprocess_kwargs,
                )
                shell_id = "process-cleanup-proof"
                BackgroundShellManager.add(
                    BackgroundShell(
                        shell_id=shell_id,
                        command="sleep",
                        process=process,
                        start_time=time.time(),
                        owner=schedulers[0].todo_store.identity,
                    )
                )
                await BackgroundShellManager.start_monitor(shell_id)

                await asyncio.wait_for(schedulers[0].cleanup(), timeout=5)
                await asyncio.wait_for(
                    asyncio.gather(*(scheduler.cleanup() for scheduler in schedulers[1:])),
                    timeout=5,
                )

                assert process.returncode is not None
                assert BackgroundShellManager.get_available_ids() == []
                for session in sessions:
                    session.close.assert_called_once()

            current = asyncio.current_task()
            pending = [
                task
                for task in asyncio.all_tasks()
                if task is not current and not task.done()
            ]
            workers = [
                thread.name
                for thread in threading.enumerate()
                if thread is not threading.main_thread() and not thread.daemon
            ]
            loop = asyncio.get_running_loop()
            transports = [
                repr(transport)
                for transport in getattr(loop, "_transports", {}).values()
                if transport is not None and not transport.is_closing()
            ]
            assert pending == [], pending
            assert workers == [], workers
            assert transports == [], transports

        asyncio.run(main())
        print("PROCESS_CLEAN_EXIT", flush=True)
        """)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=python_child_environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "PROCESS_CLEAN_EXIT"


@pytest.mark.asyncio
async def test_interactive_preflight_cancellation_finishes_goal_once_as_cancelled():
    scheduler = _make_scheduler()
    started = asyncio.Event()

    async def blocked_initialization():
        started.set()
        await asyncio.Event().wait()

    scheduler._ensure_agent_initialized = blocked_initialization

    task = asyncio.create_task(scheduler.handle("hello", render_output=False))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    scheduler.goal_runtime.on_turn_start.assert_awaited_once()
    scheduler.goal_runtime.on_turn_end.assert_awaited_once()
    assert scheduler.goal_runtime.on_turn_end.await_args.kwargs == {
        "error": False,
        "cancelled": True,
    }
    _assert_turn_contexts_reset()


@pytest.mark.asyncio
async def test_interactive_preflight_error_finishes_goal_once_as_error():
    scheduler = _make_scheduler()

    async def failing_initialization():
        raise RuntimeError("agent initialization failed")

    scheduler._ensure_agent_initialized = failing_initialization

    with pytest.raises(RuntimeError, match="agent initialization failed"):
        await scheduler.handle("hello", render_output=False)

    scheduler.goal_runtime.on_turn_start.assert_awaited_once()
    scheduler.goal_runtime.on_turn_end.assert_awaited_once()
    assert scheduler.goal_runtime.on_turn_end.await_args.kwargs == {
        "error": True,
        "cancelled": False,
    }
    _assert_turn_contexts_reset()


@pytest.mark.asyncio
async def test_stream_json_timeout_finishes_goal_once_as_error(monkeypatch):
    scheduler = _make_scheduler()
    scheduler.permission_service = object()
    release = asyncio.Event()
    manual_skill = Skill(
        name="read-only",
        description="read only",
        content="Inspect files",
        allowed_tools=["read_file"],
    )

    async def blocked_turn(*_args, **_kwargs):
        assert get_tool_permission_context() is not None
        assert get_goal_runtime() is scheduler.goal_runtime
        assert get_todo_store_or_none() is scheduler.todo_store
        restrictions = get_active_restrictions()
        assert restrictions is not None
        assert restrictions.allowed_tools == {"read_file"}
        await release.wait()

    monkeypatch.setattr("koder_agent.core.scheduler.get_turn_timeout", lambda: 0.01)
    scheduler._handle_stream_json_unlocked = blocked_turn

    with skill_invocation_scope(manual_skill):
        with pytest.raises(asyncio.TimeoutError):
            await scheduler.handle_stream_json("hello", on_event=lambda _event: None)
        restored = get_active_restrictions()
        assert restored is not None
        assert restored.allowed_tools == {"read_file"}
        assert get_todo_store_or_none() is None

    scheduler.goal_runtime.on_turn_start.assert_awaited_once()
    scheduler.goal_runtime.on_turn_end.assert_awaited_once()
    assert scheduler.goal_runtime.on_turn_end.await_args.kwargs == {
        "error": True,
        "cancelled": False,
    }
    scheduler._append_interruption_marker_if_needed.assert_not_awaited()
    _assert_turn_contexts_reset()


@pytest.mark.asyncio
async def test_stream_json_exhausted_deadline_uses_asyncio_timeout_contract(monkeypatch):
    scheduler = _make_scheduler()
    scheduler._handle_stream_json_unlocked = AsyncMock()

    async def slow_goal_start(*_args, **_kwargs):
        await asyncio.sleep(0.02)

    scheduler.goal_runtime.on_turn_start.side_effect = slow_goal_start
    monkeypatch.setattr("koder_agent.core.scheduler.get_turn_timeout", lambda: 0.001)

    with pytest.raises(asyncio.TimeoutError):
        await scheduler.handle_stream_json("hello", on_event=lambda _event: None)

    scheduler._handle_stream_json_unlocked.assert_not_awaited()
    scheduler.goal_runtime.on_turn_end.assert_awaited_once()
    assert scheduler.goal_runtime.on_turn_end.await_args.kwargs == {
        "error": True,
        "cancelled": False,
    }
    _assert_turn_contexts_reset()


@pytest.mark.asyncio
async def test_stream_json_task_cancellation_finishes_goal_once_as_cancelled(monkeypatch):
    scheduler = _make_scheduler()
    started = asyncio.Event()
    release = asyncio.Event()
    finalization_started = asyncio.Event()
    release_finalization = asyncio.Event()

    async def blocked_turn(*_args, **_kwargs):
        started.set()
        await release.wait()

    monkeypatch.setattr("koder_agent.core.scheduler.get_turn_timeout", lambda: 0)
    scheduler._handle_stream_json_unlocked = blocked_turn

    async def finish_goal(*_args, **_kwargs):
        finalization_started.set()
        await release_finalization.wait()

    scheduler.goal_runtime.on_turn_end = AsyncMock(side_effect=finish_goal)

    task = asyncio.create_task(scheduler.handle_stream_json("hello", on_event=lambda _event: None))
    await started.wait()
    task.cancel()
    await finalization_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    release_finalization.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    scheduler.goal_runtime.on_turn_start.assert_awaited_once()
    scheduler.goal_runtime.on_turn_end.assert_awaited_once()
    assert scheduler.goal_runtime.on_turn_end.await_args.kwargs == {
        "error": False,
        "cancelled": True,
    }
    scheduler._append_interruption_marker_if_needed.assert_not_awaited()
    _assert_turn_contexts_reset()


@pytest.mark.asyncio
async def test_stream_json_model_error_finishes_goal_once_as_error(monkeypatch):
    scheduler = _make_scheduler()

    async def failing_turn(*_args, **_kwargs):
        raise RuntimeError("provider failed")

    monkeypatch.setattr("koder_agent.core.scheduler.get_turn_timeout", lambda: 0)
    scheduler._handle_stream_json_unlocked = failing_turn

    with pytest.raises(RuntimeError, match="provider failed"):
        await scheduler.handle_stream_json("hello", on_event=lambda _event: None)

    scheduler.goal_runtime.on_turn_start.assert_awaited_once()
    scheduler.goal_runtime.on_turn_end.assert_awaited_once()
    assert scheduler.goal_runtime.on_turn_end.await_args.kwargs == {
        "error": True,
        "cancelled": False,
    }
    scheduler._append_interruption_marker_if_needed.assert_not_awaited()
    _assert_turn_contexts_reset()


class _PartialStreamingResult:
    final_output = None

    def __init__(self, *, error=None, started=None):
        self.error = error
        self.started = started

    async def stream_events(self):
        if self.started is not None:
            self.started.set()
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        if False:
            yield None


def _prepare_streaming_scheduler(scheduler):
    scheduler.dev_agent = object()
    scheduler._agent_initialized = True
    scheduler._migration_done = True
    scheduler._capture_usage = AsyncMock()
    scheduler._refresh_magic_docs_after_turn = AsyncMock()


@pytest.mark.asyncio
async def test_stream_json_api_error_captures_partial_usage(monkeypatch):
    scheduler = _make_scheduler()
    _prepare_streaming_scheduler(scheduler)
    result = _PartialStreamingResult(error=RuntimeError("provider failed"))
    monkeypatch.setattr("koder_agent.core.scheduler.Runner.run_streamed", lambda *_a, **_k: result)
    monkeypatch.setattr("koder_agent.core.scheduler.get_turn_timeout", lambda: 0)

    with pytest.raises(RuntimeError, match="provider failed"):
        await scheduler.handle_stream_json("hello", on_event=lambda _event: None)

    scheduler._capture_usage.assert_awaited_once_with(result)


@pytest.mark.asyncio
async def test_stream_json_timeout_captures_partial_usage(monkeypatch):
    scheduler = _make_scheduler()
    _prepare_streaming_scheduler(scheduler)
    started = asyncio.Event()
    result = _PartialStreamingResult(started=started)
    monkeypatch.setattr("koder_agent.core.scheduler.Runner.run_streamed", lambda *_a, **_k: result)
    monkeypatch.setattr("koder_agent.core.scheduler.get_turn_timeout", lambda: 0.01)

    with pytest.raises(asyncio.TimeoutError):
        await scheduler.handle_stream_json("hello", on_event=lambda _event: None)

    scheduler._capture_usage.assert_awaited_once_with(result)


@pytest.mark.asyncio
async def test_stream_json_external_cancellation_captures_partial_usage(monkeypatch):
    scheduler = _make_scheduler()
    _prepare_streaming_scheduler(scheduler)
    started = asyncio.Event()
    result = _PartialStreamingResult(started=started)
    monkeypatch.setattr("koder_agent.core.scheduler.Runner.run_streamed", lambda *_a, **_k: result)
    monkeypatch.setattr("koder_agent.core.scheduler.get_turn_timeout", lambda: 0)

    task = asyncio.create_task(scheduler.handle_stream_json("hello", on_event=lambda _event: None))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    scheduler._capture_usage.assert_awaited_once_with(result)


@pytest.mark.asyncio
async def test_interactive_streaming_external_cancellation_captures_partial_usage(monkeypatch):
    scheduler = _make_scheduler()
    _prepare_streaming_scheduler(scheduler)
    started = asyncio.Event()
    result = _PartialStreamingResult(started=started)
    monkeypatch.setattr("koder_agent.core.scheduler.Runner.run_streamed", lambda *_a, **_k: result)

    class StreamingUI:
        def update_output(self, _value):
            return None

        def set_cancel_callback(self, _callback):
            return None

    task = asyncio.create_task(scheduler._handle_streaming("hello", streaming_ui=StreamingUI()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    scheduler._capture_usage.assert_awaited_once_with(result)


@pytest.mark.asyncio
async def test_interactive_streaming_api_error_captures_partial_usage(monkeypatch):
    scheduler = _make_scheduler()
    _prepare_streaming_scheduler(scheduler)
    result = _PartialStreamingResult(error=RuntimeError("stream failed"))
    monkeypatch.setattr("koder_agent.core.scheduler.Runner.run_streamed", lambda *_a, **_k: result)

    class StreamingUI:
        def update_output(self, _value):
            return None

        def set_cancel_callback(self, _callback):
            return None

        def set_final_content(self, _value):
            return None

    response = await scheduler._handle_streaming("hello", streaming_ui=StreamingUI())

    assert "stream failed" in response
    scheduler._capture_usage.assert_awaited_once_with(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("display_kind", ["rich", "text", "empty"])
async def test_fixed_bottom_cancel_notice_preserves_partial_output(monkeypatch, display_kind):
    import io

    from rich.console import Console
    from rich.text import Text

    scheduler = _make_scheduler()
    _prepare_streaming_scheduler(scheduler)
    partial_text = "" if display_kind == "empty" else "partial provider response"
    display = Mock()
    display.get_display_content.return_value = (
        Text(partial_text) if display_kind == "rich" else None
    )
    display.get_final_text.return_value = partial_text
    monkeypatch.setattr("koder_agent.core.scheduler.StreamingDisplayManager", lambda _: display)

    class StreamingUI:
        cancel_callback = None
        final_content = None

        def update_output(self, _value):
            pass

        def set_cancel_callback(self, callback):
            self.cancel_callback = callback

        def set_final_content(self, content):
            self.final_content = content

        def set_final_text(self, text):
            self.final_content = text

    ui = StreamingUI()

    class Result:
        final_output = None
        cancel = Mock()

        async def stream_events(self):
            ui.cancel_callback()
            if False:
                yield None

    result = Result()
    monkeypatch.setattr("koder_agent.core.scheduler.Runner.run_streamed", lambda *_a, **_k: result)
    response = await scheduler._handle_streaming("hello", streaming_ui=ui)
    output = io.StringIO()
    Console(file=output, color_system=None).print(ui.final_content)

    assert "Operation cancelled by user" in output.getvalue()
    if partial_text:
        assert partial_text in output.getvalue()
        assert response == partial_text
    assert scheduler._last_turn_cancelled
    assert ui.cancel_callback is None
    result.cancel.assert_called_once_with(mode="immediate")
