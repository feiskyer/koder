"""An idle private stdin pipe must not keep the auth CLI process alive."""

import os
import selectors
import signal
import subprocess
import sys
import textwrap
import time
from contextlib import suppress

import pytest


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX pipe readiness for a child process")
def test_manual_timeout_does_not_leave_a_blocking_input_thread(tmp_path):
    script = tmp_path / "manual_probe.py"
    script.write_text(
        textwrap.dedent("""
            import asyncio
            import webbrowser
            from koder_agent.harness.auth.commands import _handle_manual_code_flow

            webbrowser.open = lambda _url: False

            async def main():
                print("PROBE_READY", flush=True)
                result = await _handle_manual_code_flow(
                    "https://auth.example.invalid/synthetic", 0.05
                )
                print("PROBE_RETURNED:" + str(result.error), flush=True)

            asyncio.run(main())
            print("PROBE_EXITED", flush=True)
            """),
        encoding="utf-8",
    )
    command = [
        "uv",
        "run",
        "--no-project",
        "--no-env-file",
        "--python",
        sys.executable,
        "python",
        str(script),
    ]
    process = subprocess.Popen(
        command,
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    captured = bytearray()
    exited_without_input = False
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + 15
            while b"PROBE_RETURNED:" not in captured:
                remaining = deadline - time.monotonic()
                assert remaining > 0, f"Child did not return from manual input: {captured!r}"
                assert selector.select(remaining), f"No child output: {captured!r}"
                chunk = os.read(process.stdout.fileno(), 65536)
                assert chunk, f"Child exited before reporting its result: {captured!r}"
                captured.extend(chunk)
        assert b"PROBE_RETURNED:timeout" in captured
        try:
            process.wait(timeout=5)
            exited_without_input = True
        except subprocess.TimeoutExpired:
            pass
    finally:
        if process.poll() is None:
            # Release a regressed legacy reader through this child's private
            # stdin, never through the user's real terminal.
            try:
                process.stdin.write(b"synthetic-cleanup-only\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        with suppress(BrokenPipeError, OSError):
            process.stdin.close()
        captured.extend(process.stdout.read())
        errors = process.stderr.read()
        process.stdout.close()
        process.stderr.close()
    assert exited_without_input, (
        f"Input timed out but its reader prevented shutdown. stdout={captured!r}, stderr={errors!r}"
    )
    assert process.returncode == 0
    assert b"PROBE_EXITED" in captured
