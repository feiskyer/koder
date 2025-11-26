"""Tests for shell tool background mode functionality."""

import gc
import json
import re
import sys
import types

import pytest
import pytest_asyncio

# Stub litellm before importing koder_agent to avoid optional dependency issues
if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

# Ensure project root is on sys.path when running tests directly
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import koder_agent components (guarded for easier debugging if something fails)
try:
    from koder_agent.core.scheduler import AgentScheduler
    from koder_agent.tools.shell import (
        IS_WINDOWS,
        BackgroundShellManager,
        run_shell,
        shell_kill,
        shell_output,
    )
except ModuleNotFoundError as e:  # pragma: no cover - import-time diagnostics
    # Re-raise but with additional context; pytest will still treat this as an error
    raise ModuleNotFoundError(f"Failed to import koder_agent modules: {e}") from e


# Mark all tests as async
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture(autouse=True)
async def cleanup_background_shells():
    """Ensure all background shells are cleaned up after each test."""
    import asyncio

    yield
    # Cleanup any remaining background shells
    for shell_id in list(BackgroundShellManager.get_available_ids()):
        try:
            await BackgroundShellManager.terminate(shell_id)
        except Exception:
            pass
    # Give the event loop time to process subprocess cleanup callbacks
    await asyncio.sleep(0.05)
    gc.collect()


def _make_echo_command(text: str) -> str:
    """Create a simple echo command for the current platform."""
    if IS_WINDOWS:
        return f'Write-Output "{text}"'
    return f'echo "{text}"'


def _make_long_sleep_command(seconds: int) -> str:
    """Create a long-running command for timeout/kill tests."""
    if IS_WINDOWS:
        return f"Start-Sleep -Seconds {seconds}"
    return f"sleep {seconds}"


def _make_multi_line_command(lines: list[str]) -> str:
    """Create a command that prints multiple lines."""
    if IS_WINDOWS:
        # Use separate Write-Output calls so each line is flushed
        parts = [f'Write-Output "{line}"' for line in lines]
        return "; ".join(parts)
    # Use printf for predictable output on Unix
    joined = "\\n".join(lines)
    return f'printf "{joined}\\n"'


def _parse_shell_id(result: str) -> str:
    """Extract shell_id from run_shell background response."""
    match = re.search(r"shell_id: (\w+)", result)
    assert match, f"shell_id not found in result: {result!r}"
    return match.group(1)


def _extract_output_lines(output: str) -> list[str]:
    """Extract non-metadata lines from shell_output/shell_kill response."""
    lines = output.splitlines()
    useful: list[str] = []
    for line in lines:
        if line.startswith("[status]:") or line.startswith("[exit_code]:"):
            continue
        useful.append(line)
    return useful


async def test_run_shell_foreground_default_timeout():
    """Foreground command executes successfully with default timeout."""
    cmd = _make_echo_command("hello-foreground-default")
    # Only provide the required argument so the default timeout is used
    result = await run_shell.on_invoke_tool(None, json.dumps({"command": cmd}))

    assert "hello-foreground-default" in result


async def test_run_shell_foreground_custom_timeout():
    """Foreground command executes with a custom timeout value."""
    cmd = _make_echo_command("hello-foreground-custom")
    result = await run_shell.on_invoke_tool(
        None,
        json.dumps(
            {
                "command": cmd,
                "timeout": 1,
            }
        ),
    )

    assert "hello-foreground-custom" in result


async def test_run_shell_foreground_timeout_expiration():
    """Foreground command respects timeout and reports expiration."""
    cmd = _make_long_sleep_command(5)
    timeout_seconds = 1

    result = await run_shell.on_invoke_tool(
        None,
        json.dumps(
            {
                "command": cmd,
                "timeout": timeout_seconds,
            }
        ),
    )

    assert f"Command timed out after {timeout_seconds} seconds" == result


async def test_run_shell_background_returns_shell_id_and_manager_state():
    """Background command returns a shell_id and is tracked by BackgroundShellManager."""
    import asyncio

    cmd = _make_echo_command("background-test")
    result = await run_shell.on_invoke_tool(
        None,
        json.dumps(
            {
                "command": cmd,
                "run_in_background": True,
            }
        ),
    )

    assert "Command started in background." in result
    shell_id = _parse_shell_id(result)

    # Shell should be registered in the manager
    assert shell_id in BackgroundShellManager.get_available_ids()
    shell = BackgroundShellManager.get(shell_id)
    assert shell is not None
    assert shell.command == cmd

    # Allow the process/monitor to settle, then clean up via shell_kill
    await asyncio.sleep(0.2)
    kill_result = await shell_kill.on_invoke_tool(None, json.dumps({"shell_id": shell_id}))
    assert f"Shell {shell_id} terminated." in kill_result
    assert shell_id not in BackgroundShellManager.get_available_ids()


async def test_shell_output_incremental_output():
    """shell_output returns new output and then '(no new output)' on subsequent call."""
    import asyncio

    lines = ["line1", "line2", "line3"]
    cmd = _make_multi_line_command(lines)

    result = await run_shell.on_invoke_tool(
        None,
        json.dumps(
            {
                "command": cmd,
                "run_in_background": True,
            }
        ),
    )
    shell_id = _parse_shell_id(result)

    # Wait for the background process and monitor to collect all output
    await asyncio.sleep(0.5)

    first = await shell_output.on_invoke_tool(None, json.dumps({"shell_id": shell_id}))
    first_lines = _extract_output_lines(first)
    # All lines should be present on first read
    for line in lines:
        assert line in first_lines

    second = await shell_output.on_invoke_tool(None, json.dumps({"shell_id": shell_id}))
    second_lines = _extract_output_lines(second)
    # Second read should report no new output
    assert "(no new output)" in second_lines

    # Cleanup background shell
    await shell_kill.on_invoke_tool(None, json.dumps({"shell_id": shell_id}))


async def test_shell_output_with_regex_filter():
    """shell_output applies regex filtering to incremental output."""
    import asyncio

    lines = ["info: ok", "ERROR: something bad", "debug: details"]
    cmd = _make_multi_line_command(lines)

    result = await run_shell.on_invoke_tool(
        None,
        json.dumps(
            {
                "command": cmd,
                "run_in_background": True,
            }
        ),
    )
    shell_id = _parse_shell_id(result)

    # Wait for all output to be produced
    await asyncio.sleep(0.5)

    filtered = await shell_output.on_invoke_tool(
        None,
        json.dumps(
            {
                "shell_id": shell_id,
                "filter_str": "ERROR",
            }
        ),
    )
    filtered_lines = _extract_output_lines(filtered)

    assert any("ERROR: something bad" == line for line in filtered_lines)
    # Non-matching lines should not appear in the filtered output
    assert all(
        not (line.startswith("info:") or line.startswith("debug:")) for line in filtered_lines
    )

    # Cleanup background shell
    await shell_kill.on_invoke_tool(None, json.dumps({"shell_id": shell_id}))


async def test_shell_kill_terminates_long_running_process():
    """shell_kill terminates a long-running background process and removes it from the manager."""
    import asyncio

    cmd = _make_long_sleep_command(5)

    result = await run_shell.on_invoke_tool(
        None,
        json.dumps(
            {
                "command": cmd,
                "run_in_background": True,
            }
        ),
    )
    shell_id = _parse_shell_id(result)

    # Give the process a moment to start
    await asyncio.sleep(0.2)

    kill_result = await shell_kill.on_invoke_tool(None, json.dumps({"shell_id": shell_id}))
    lines = kill_result.splitlines()

    assert any(f"Shell {shell_id} terminated." in line for line in lines)
    assert any(line.startswith("[exit_code]:") for line in lines)
    assert shell_id not in BackgroundShellManager.get_available_ids()


async def test_background_shell_manager_status_updates():
    """BackgroundShellManager tracks status and exit_code for completed commands."""
    import asyncio

    cmd = _make_echo_command("manager-status-test")

    result = await run_shell.on_invoke_tool(
        None,
        json.dumps(
            {
                "command": cmd,
                "run_in_background": True,
            }
        ),
    )
    shell_id = _parse_shell_id(result)

    # Wait for process to complete and monitor to update status
    await asyncio.sleep(0.3)

    shell = BackgroundShellManager.get(shell_id)
    assert shell is not None
    assert shell.status in {"completed", "failed"}
    assert shell.exit_code is not None

    # Cleanup via manager terminate to exercise that path directly
    terminated_shell = await BackgroundShellManager.terminate(shell_id)
    assert terminated_shell.shell_id == shell_id
    assert shell_id not in BackgroundShellManager.get_available_ids()


async def test_security_validation_blocks_forbidden_commands():
    """SecurityGuard validation still blocks dangerous commands."""
    # SecurityGuard looks for 'rm -rf' substring, so ensure it's present
    dangerous_command = "rm -rf /tmp/some-path"

    result = await run_shell.on_invoke_tool(
        None,
        json.dumps(
            {
                "command": dangerous_command,
            }
        ),
    )

    assert "Forbidden command pattern detected: rm -rf" in result


async def test_shell_output_and_kill_invalid_shell_id():
    """shell_output and shell_kill handle invalid shell_ids gracefully."""
    invalid_id = "nonexistent123"

    out_result = await shell_output.on_invoke_tool(
        None,
        json.dumps(
            {
                "shell_id": invalid_id,
            }
        ),
    )
    assert f"Shell not found: {invalid_id}" in out_result
    assert "Available:" in out_result

    kill_result = await shell_kill.on_invoke_tool(
        None,
        json.dumps(
            {
                "shell_id": invalid_id,
            }
        ),
    )
    assert f"Shell not found: {invalid_id}" in kill_result
    assert "Available:" in kill_result


async def test_scheduler_cleanup_terminates_background_shells():
    """AgentScheduler.cleanup performs background shell cleanup via BackgroundShellManager."""
    cmd = _make_long_sleep_command(5)

    result = await run_shell.on_invoke_tool(
        None,
        json.dumps(
            {
                "command": cmd,
                "run_in_background": True,
            }
        ),
    )
    shell_id = _parse_shell_id(result)
    assert shell_id in BackgroundShellManager.get_available_ids()

    scheduler = AgentScheduler(session_id="test-shell-cleanup", streaming=False)
    await scheduler.cleanup()

    assert shell_id not in BackgroundShellManager.get_available_ids()
