"""Bounded, cancellable ownership of command-hook subprocesses."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import threading
import time

_CANCELLATION_POLL_SECONDS = 0.1


class HookCommandCancelledError(Exception):
    """The foreground dispatch was cancelled before its command completed."""


def _stop_process(process: subprocess.Popen[str]) -> None:
    """Reap the child, killing its owned process group on POSIX.

    This is lifecycle cleanup, not an OS sandbox: descendants that deliberately
    create another session are outside the group. Windows only kills the child.
    """
    try:
        if os.name == "posix":
            # Every process below is spawned with start_new_session=True, so
            # this never targets Koder's own process group.
            os.killpg(process.pid, signal.SIGKILL)
        else:  # pragma: no cover - Windows requires native validation
            process.kill()
    except ProcessLookupError:
        pass
    process.wait(timeout=1)


def run_command(
    command: str | list[str],
    *,
    input: str,
    cwd: str,
    shell: bool,
    env: dict[str, str],
    timeout: int | float,
    cancel_event: threading.Event | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one command; timeout/cancellation stops its processes before return."""
    if cancel_event is not None and cancel_event.is_set():
        raise HookCommandCancelledError()
    # Older supported CPython releases can stop writing pending stdin after
    # a communicate() timeout. Prepare private, seekable input so output/cancellation
    # polling cannot strand a slow reader. The file is closed on every path,
    # including spawn failure, without another writer thread to detach.
    with tempfile.TemporaryFile(mode="w+") as input_stream:
        input_stream.write(input)
        input_stream.seek(0)
        if cancel_event is not None and cancel_event.is_set():
            raise HookCommandCancelledError()
        process = subprocess.Popen(
            command,
            stdin=input_stream,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            shell=shell,
            env=env,
            start_new_session=os.name == "posix",
        )
        deadline = time.monotonic() + timeout
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise HookCommandCancelledError()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    stdout, stderr = process.communicate(
                        timeout=min(remaining, _CANCELLATION_POLL_SECONDS),
                    )
                    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
                except subprocess.TimeoutExpired:
                    # communicate() preserves captured stdout/stderr on retry.
                    continue
        except BaseException:
            _stop_process(process)
            raise
        finally:
            # Do not drain inherited pipes after killing the group: an escaped
            # descendant could hold them open forever. Timeout output is discarded.
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
