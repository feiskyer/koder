"""Hook timeout bounding and off-loop dispatch tests.

Covers the availability bug where a hook config omitting ``timeout`` flowed
``None`` into ``subprocess.run``/``urlopen`` (blocking forever), and the
event-loop hang caused by running blocking hook I/O synchronously on the loop.
"""

import asyncio
import json
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from koder_agent.harness.hooks.runtime import (
    _run_command_hook,
    dispatch_command_hooks,
)


def _write_stop_hook_settings(project: Path, hook: dict) -> None:
    (project / ".koder").mkdir(parents=True, exist_ok=True)
    (project / ".koder" / "settings.json").write_text(
        json.dumps({"hooks": {"Stop": [{"hooks": [hook]}]}}),
        encoding="utf-8",
    )


def _spy_subprocess_run(monkeypatch):
    calls: list[dict] = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("koder_agent.harness.hooks.runtime.run_command", fake_run)
    return calls


def _dispatch_stop(project: Path):
    return dispatch_command_hooks(
        cwd=project,
        event_name="Stop",
        match_value=None,
        payload={"event": "Stop"},
    )


# ---------------------------------------------------------------------------
# Timeout defaulting (command hooks)
# ---------------------------------------------------------------------------


def test_command_hook_without_timeout_uses_bounded_default(tmp_path, monkeypatch):
    """A hook that omits ``timeout`` must never run with timeout=None."""
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    _write_stop_hook_settings(project, {"type": "command", "command": "echo ok"})
    calls = _spy_subprocess_run(monkeypatch)

    result = _dispatch_stop(project)

    assert result.matched_hooks == 1
    assert calls[0]["timeout"] == 60


def test_command_hook_with_explicit_timeout_is_respected(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    _write_stop_hook_settings(project, {"type": "command", "command": "echo ok", "timeout": 5})
    calls = _spy_subprocess_run(monkeypatch)

    result = _dispatch_stop(project)

    assert result.matched_hooks == 1
    assert calls[0]["timeout"] == 5


def test_command_hook_with_zero_timeout_falls_back_to_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    _write_stop_hook_settings(project, {"type": "command", "command": "echo ok", "timeout": 0})
    calls = _spy_subprocess_run(monkeypatch)

    result = _dispatch_stop(project)

    assert result.matched_hooks == 1
    assert calls[0]["timeout"] == 60


def test_command_hook_with_negative_timeout_falls_back_to_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    _write_stop_hook_settings(project, {"type": "command", "command": "echo ok", "timeout": -3})
    calls = _spy_subprocess_run(monkeypatch)

    result = _dispatch_stop(project)

    assert result.matched_hooks == 1
    assert calls[0]["timeout"] == 60


# ---------------------------------------------------------------------------
# Timeout defaulting (http hooks)
# ---------------------------------------------------------------------------


def test_http_hook_without_timeout_uses_bounded_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    _write_stop_hook_settings(project, {"type": "http", "url": "http://127.0.0.1:9/hook"})
    seen: dict = {}

    def fake_http(**kwargs):
        seen.update(kwargs)
        return 200, "", ""

    monkeypatch.setattr("koder_agent.harness.hooks.runtime._run_http_hook", fake_http)

    result = _dispatch_stop(project)

    assert result.matched_hooks == 1
    assert seen["timeout"] == 60


# ---------------------------------------------------------------------------
# Async entrypoint: dispatch off the event loop
# ---------------------------------------------------------------------------


def test_dispatch_command_hooks_async_matches_sync_result(tmp_path, monkeypatch):
    from koder_agent.harness.hooks.runtime import dispatch_command_hooks_async

    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    marker = tmp_path / "async-dispatch.json"
    _write_stop_hook_settings(
        project,
        {
            "type": "command",
            "command": (
                f'python -c "import sys, pathlib; '
                f"pathlib.Path(r'{marker}').write_text(sys.stdin.read())\""
            ),
        },
    )

    result = asyncio.run(
        dispatch_command_hooks_async(
            cwd=project,
            event_name="Stop",
            match_value=None,
            payload={"event": "Stop"},
        )
    )

    assert result.matched_hooks == 1
    assert json.loads(marker.read_text(encoding="utf-8"))["event"] == "Stop"


def test_dispatch_command_hooks_async_does_not_block_event_loop(tmp_path, monkeypatch):
    from koder_agent.harness.hooks.runtime import dispatch_command_hooks_async

    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    _write_stop_hook_settings(
        project,
        {"type": "command", "command": 'python -c "import time; time.sleep(0.5)"'},
    )

    async def _main():
        task = asyncio.create_task(
            dispatch_command_hooks_async(
                cwd=project,
                event_name="Stop",
                match_value=None,
                payload={"event": "Stop"},
            )
        )
        # If the hook ran synchronously on the loop, this sleep could not
        # complete until the 0.5s hook finished (and the task would be done).
        await asyncio.sleep(0.05)
        loop_responsive = not task.done()
        result = await task
        return loop_responsive, result

    loop_responsive, result = asyncio.run(_main())

    assert loop_responsive is True
    assert result.matched_hooks == 1


# ---------------------------------------------------------------------------
# Async call sites: PostToolUse dispatch must not block the loop
# ---------------------------------------------------------------------------


def test_on_tool_end_does_not_block_event_loop(tmp_path, monkeypatch):
    from agents import Tool

    from koder_agent.agentic.approval_hooks import ApprovalHooks

    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    (project / ".koder").mkdir(parents=True)
    (project / ".koder" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "PostToolUse": [
                        {
                            "matcher": "read_file",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": 'python -c "import time; time.sleep(0.5)"',
                                }
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    hooks = ApprovalHooks(wrapped_hooks=None)
    tool = MagicMock(spec=Tool)
    tool.name = "read_file"
    ctx = MagicMock()
    ctx.tool_arguments = "{}"

    async def _main():
        task = asyncio.create_task(hooks.on_tool_end(ctx, MagicMock(), tool, "ok"))
        await asyncio.sleep(0.05)
        loop_responsive = not task.done()
        await task
        return loop_responsive

    assert asyncio.run(_main()) is True


# ---------------------------------------------------------------------------
# H11: PreToolUse hook timeout must fail-closed (block)
# ---------------------------------------------------------------------------


class TestPreToolHookTimeoutFailsClosed:
    def test_pretool_hook_timeout_blocks_tool(self):
        """A PreToolUse hook that times out must block the tool call (exit 2)."""
        from unittest.mock import patch

        with patch(
            "koder_agent.harness.hooks.runtime.run_command",
            side_effect=subprocess.TimeoutExpired("cmd", 5),
        ):
            code, stdout, stderr = _run_command_hook(
                command="sleep 100",
                payload_text="{}",
                cwd=Path.cwd(),
                env={},
                timeout=1,
            )
        assert code == 2  # Block signal
        assert "timed out" in stderr.lower()

    def test_pretool_hook_timeout_dispatch_returns_blocked(self, tmp_path, monkeypatch):
        """dispatch_command_hooks returns blocked=True when a PreToolUse hook times out."""
        monkeypatch.setenv("HOME", str(tmp_path))
        project = tmp_path / "project"
        (project / ".koder").mkdir(parents=True)
        (project / ".koder" / "settings.json").write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "sleep 100",
                                        "timeout": 1,
                                    }
                                ]
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

        # Make the owned command runner raise TimeoutExpired.
        def _timeout_run(*args, **kwargs):
            raise subprocess.TimeoutExpired("sleep 100", 1)

        monkeypatch.setattr("koder_agent.harness.hooks.runtime.run_command", _timeout_run)

        result = dispatch_command_hooks(
            cwd=project,
            event_name="PreToolUse",
            payload={"tool_name": "run_shell", "tool_input": {}},
            match_value="run_shell",
        )
        assert result.blocked is True
        assert "timed out" in (result.block_reason or "").lower()
