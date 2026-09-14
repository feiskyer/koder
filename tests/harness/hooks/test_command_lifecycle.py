"""Real local hook processes must not outlive timeout or foreground cancellation."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import threading
import time

import pytest
from anyio import fail_after

from koder_agent.harness.hooks import command_process
from koder_agent.harness.hooks.runtime import (
    HookDispatchResult,
    _run_async_command,
    _run_command_hook,
    dispatch_command_hooks_async,
)


def _python_command(script, *arguments):
    # All Python children use the already-selected test interpreter through uv.
    return shlex.join(
        [
            "uv",
            "run",
            "--quiet",
            "--no-project",
            "--no-env-file",
            "--python",
            sys.executable,
            "python",
            "-c",
            script,
            *(str(value) for value in arguments),
        ]
    )


def _command_env(tmp_path):
    return {
        "HOME": str(tmp_path),
        "PATH": os.environ.get("PATH", os.defpath),
        "UV_NO_ENV_FILE": "1",
        "UV_PYTHON_PREFERENCE": "only-system",
    }


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup contract")
def test_command_timeout_stops_shell_descendant_side_effects(tmp_path):
    marker = tmp_path / "late-effect"
    command = _python_command(
        "import pathlib,sys,time; time.sleep(0.8); pathlib.Path(sys.argv[1]).touch()",
        marker,
    )
    # Keep the shell wrapper instead of allowing a last-command exec optimization.
    code, _stdout, stderr = _run_command_hook(
        command=command + "; :",
        payload_text="{}",
        cwd=tmp_path,
        env=_command_env(tmp_path),
        timeout=0.3,
    )
    assert code == 2
    assert "timed out" in stderr.lower()
    time.sleep(1)
    assert not marker.exists(), "hook descendant survived timeout and wrote after denial"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup contract")
def test_background_hook_timeout_stops_descendants_before_completion_callback(tmp_path):
    marker = tmp_path / "late-background-effect"
    done = threading.Event()
    command = _python_command(
        "import pathlib,sys,time; time.sleep(0.8); pathlib.Path(sys.argv[1]).touch()",
        marker,
    )
    _run_async_command(
        command=command + "; :",
        payload_text="{}",
        cwd=tmp_path,
        env=_command_env(tmp_path),
        timeout=0.3,
        on_complete=done.set,
    )
    assert done.wait(timeout=3)
    time.sleep(1)
    assert not marker.exists()


@pytest.mark.parametrize("read_delay", [0.0, 0.3], ids=["immediate-reader", "delayed-reader"])
@pytest.mark.parametrize(
    "payload",
    ["synthetic-data\n" * 12_000, "", "完整输入 λ ☃\n"],
    ids=["large-input", "empty-input", "unicode-input"],
)
def test_polling_command_preserves_complete_stdin_and_output(tmp_path, read_delay, payload):
    command = _python_command(
        f"import sys,time; time.sleep({read_delay}); data=sys.stdin.read(); time.sleep(0.3); "
        "sys.stdout.write(data); sys.stderr.write('diagnostic')"
    )
    code, stdout, stderr = _run_command_hook(
        command=command,
        payload_text=payload,
        cwd=tmp_path,
        env=_command_env(tmp_path),
        timeout=3,
    )
    assert code == 0, stderr
    assert stdout == payload.strip()
    assert stderr == "diagnostic"


def test_command_input_is_closed_when_process_start_fails(tmp_path, monkeypatch):
    received_input = None

    def fail_to_start(*_args, **kwargs):
        nonlocal received_input
        received_input = kwargs["stdin"]
        raise FileNotFoundError("synthetic process start failure")

    monkeypatch.setattr(command_process.subprocess, "Popen", fail_to_start)
    with pytest.raises(FileNotFoundError, match="synthetic process start failure"):
        _run_command_hook(
            command="synthetic-missing-command",
            payload_text="{}",
            cwd=tmp_path,
            env=_command_env(tmp_path),
            timeout=3,
        )
    assert received_input is not None
    assert received_input.closed


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group cleanup contract")
def test_cancelled_dispatch_stops_foreground_process_and_later_hooks(
    tmp_path, monkeypatch, cancellation_observer
):
    observe, cancellations = cancellation_observer
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    (project / ".koder").mkdir(parents=True)
    ready = tmp_path / "ready"
    late = tmp_path / "late-effect"
    later_hook = tmp_path / "later-hook"
    command = _python_command(
        "import pathlib,sys,time; pathlib.Path(sys.argv[1]).touch(); "
        "time.sleep(0.8); pathlib.Path(sys.argv[2]).touch()",
        ready,
        late,
    )
    (project / ".koder" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {"type": "command", "command": command + "; :", "timeout": 3},
                                {
                                    "type": "command",
                                    "command": _python_command(
                                        "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
                                        later_hook,
                                    ),
                                },
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    async def exercise():
        task = asyncio.create_task(
            observe(dispatch_command_hooks_async(cwd=project, event_name="Stop", payload={}))
        )
        try:
            with fail_after(3):
                while not ready.exists():
                    if task.done():
                        await task
                        pytest.fail("hook dispatch finished before the child started")
                    await asyncio.sleep(0.01)
            task.cancel("cancel-hook")
            with pytest.raises(asyncio.CancelledError):
                await task
            assert [error.args for error in cancellations] == [("cancel-hook",)]
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(1)

    # asyncio.run also joins the default executor, including a leaked worker
    # on the unfixed baseline, so no process/thread crosses the test boundary.
    asyncio.run(exercise())
    assert not late.exists(), "foreground hook continued after cancellation"
    assert not later_hook.exists(), "cancelled dispatch continued to another hook"


def test_cancellation_join_is_bounded_for_non_command_hook_io(
    tmp_path, monkeypatch, cancellation_observer
):
    from koder_agent.harness.hooks import runtime

    observe, cancellations = cancellation_observer
    entered = threading.Event()
    release = threading.Event()

    def blocked_dispatch(**_kwargs):
        entered.set()
        assert release.wait(timeout=3)
        return HookDispatchResult(matched_hooks=1)

    monkeypatch.setattr(runtime, "dispatch_command_hooks", blocked_dispatch)
    monkeypatch.setattr(runtime, "_CANCELLATION_JOIN_TIMEOUT_SECONDS", 0.05, raising=False)

    async def exercise():
        task = asyncio.create_task(
            observe(dispatch_command_hooks_async(cwd=tmp_path, event_name="Stop", payload={}))
        )
        try:
            with fail_after(1):
                while not entered.is_set():
                    await asyncio.sleep(0.01)
            task.cancel("cancel-other-io")
            await asyncio.wait({task}, timeout=0.3)
            assert task.done(), "cancellation waited on uninterruptible hook I/O"
            with pytest.raises(asyncio.CancelledError):
                await task
            assert [error.args for error in cancellations] == [("cancel-other-io",)]
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())
