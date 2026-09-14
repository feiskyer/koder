import json
import os
import shutil
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock

import pytest

if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents import ToolInputGuardrailData

from koder_agent.agentic.hook_guardrail import hook_pretool_guardrail
from koder_agent.harness.hooks.runtime import (
    _once_fired,
    _payload_target,
    _run_agent_hook,
    dispatch_command_hooks,
    list_configured_hooks,
    poll_file_change_hooks,
    update_watch_paths,
)


def test_hook_payload_target_uses_notebook_canonical_path():
    assert (
        _payload_target(
            {
                "tool_name": "notebook_edit",
                "tool_input": {"notebook_path": "/tmp/outside.ipynb"},
            }
        )
        == "/tmp/outside.ipynb"
    )
    assert (
        _payload_target(
            {
                "tool_name": "notebook_edit",
                "tool_input": {
                    "path": "/workspace/decoy.ipynb",
                    "notebook_path": "/tmp/outside.ipynb",
                },
            }
        )
        == ""
    )


def test_agent_hook_excludes_todo_tools_without_a_runtime_identity(monkeypatch):
    captured = {}

    async def fake_create_dev_agent(tools, **_kwargs):
        captured["tool_names"] = {tool.name for tool in tools}
        return object()

    async def fake_run(*_args, **_kwargs):
        return types.SimpleNamespace(final_output='{"ok": true}')

    monkeypatch.setattr(
        "koder_agent.agentic.agent.create_dev_agent",
        fake_create_dev_agent,
    )
    monkeypatch.setattr("agents.Runner.run", fake_run)

    assert _run_agent_hook(prompt_text="check", payload_text="{}", model=None) == '{"ok": true}'
    assert "task_delegate" not in captured["tool_names"]
    assert "todo_read" not in captured["tool_names"]
    assert "todo_write" not in captured["tool_names"]


def test_list_configured_hooks_reads_koder_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    (project / ".koder").mkdir(parents=True)
    (project / ".koder" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "PostToolUse": [
                        {
                            "matcher": "Edit|Write",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "echo ok",
                                }
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    listings = list_configured_hooks(project)
    assert len(listings) == 1
    assert listings[0].event == "PostToolUse"
    assert listings[0].matcher == "Edit|Write"
    assert listings[0].source == "project_settings"


def test_list_configured_hooks_includes_policy_and_plugin_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    (project / ".koder").mkdir(parents=True)
    (tmp_path / ".koder").mkdir(parents=True)
    (tmp_path / ".koder" / "managed-settings.json").write_text(
        json.dumps(
            {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo managed"}]}]}}
        ),
        encoding="utf-8",
    )
    plugin_hooks = tmp_path / ".koder" / "plugins" / "demo" / "hooks"
    plugin_hooks.mkdir(parents=True)
    (plugin_hooks.parent / "plugin.json").write_text(
        json.dumps({"name": "demo", "version": "1.0.0"}),
        encoding="utf-8",
    )
    (plugin_hooks / "hooks.json").write_text(
        json.dumps(
            {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo plugin"}]}]}}
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    listings = list_configured_hooks(project)
    sources = {listing.source for listing in listings}
    assert "policy_settings" in sources
    assert "plugin" in sources


def test_async_plugin_hook_snapshot_outlives_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    project.mkdir()
    plugin_dir = tmp_path / ".koder" / "plugins" / "demo"
    hooks_dir = plugin_dir / "hooks"
    hooks_dir.mkdir(parents=True)
    (plugin_dir / "plugin.json").write_text(
        json.dumps(
            {
                "name": "demo",
                "version": "1.0.0",
                "hooks": "hooks/hooks.json",
            }
        ),
        encoding="utf-8",
    )
    (plugin_dir / "asset.txt").write_text("snapshot-owned", encoding="utf-8")
    (hooks_dir / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "ignored",
                                    "async": True,
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    entered = threading.Event()
    inspect_snapshot = threading.Event()
    inspected = threading.Event()
    finish = threading.Event()
    observed: dict[str, object] = {}
    snapshot: Path | None = None

    def fake_run(*_args, **kwargs):
        snapshot = Path(kwargs["cwd"])
        observed["snapshot"] = snapshot
        observed["plugin_root"] = kwargs["env"].get("KODER_PLUGIN_ROOT")
        entered.set()
        assert inspect_snapshot.wait(timeout=5)
        observed["exists"] = snapshot.is_dir()
        observed["asset"] = (snapshot / "asset.txt").read_text(encoding="utf-8")
        inspected.set()
        assert finish.wait(timeout=5)
        return 0, "", ""

    monkeypatch.setattr("koder_agent.harness.hooks.runtime._run_command_hook", fake_run)

    try:
        result = dispatch_command_hooks(
            cwd=project,
            event_name="Stop",
            payload={"event": "Stop"},
        )
        assert result.matched_hooks == 1
        assert entered.wait(timeout=5)

        shutil.rmtree(plugin_dir)
        inspect_snapshot.set()
        assert inspected.wait(timeout=5)
        snapshot = observed["snapshot"]
        assert isinstance(snapshot, Path)
        assert observed["exists"] is True
        assert observed["asset"] == "snapshot-owned"
        assert observed["plugin_root"] == str(snapshot.resolve())
    finally:
        inspect_snapshot.set()
        finish.set()

    assert snapshot is not None
    deadline = time.monotonic() + 5
    while snapshot.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not snapshot.exists()


def test_dispatch_command_hooks_blocks_user_prompt_submit_on_exit_code_2(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    (project / ".koder").mkdir(parents=True)
    (project / ".koder" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "UserPromptSubmit": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "python -c \"import sys; print('blocked'); raise SystemExit(2)\"",
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    result = dispatch_command_hooks(
        cwd=project,
        event_name="UserPromptSubmit",
        match_value=None,
        payload={"event": "UserPromptSubmit", "prompt": "hello"},
    )

    assert result.blocked is True


def test_hook_pretool_guardrail_rejects_blocked_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    (project / ".koder").mkdir(parents=True)
    (project / ".koder" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "run_shell",
                            "hooks": [
                                {
                                    "type": "command",
                                    "if": "run_shell(rm *)",
                                    "command": 'python -c "import sys; print(\'{\\"decision\\": \\"block\\", \\"reason\\": \\"rm blocked\\"}\')"',
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

    mock_context = MagicMock()
    mock_context.tool_name = "run_shell"
    mock_context.tool_arguments = json.dumps({"command": "rm -rf build"})
    data = MagicMock(spec=ToolInputGuardrailData)
    data.context = mock_context

    # The guardrail is async (runs hook I/O off the event loop); the SDK's
    # ToolInputGuardrail.run awaits awaitable guardrail results.
    result = __import__("asyncio").run(hook_pretool_guardrail(data))
    assert result.behavior["type"] == "reject_content"


def test_dispatch_command_hooks_runs_session_start_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    marker = tmp_path / "session-start.json"
    (project / ".koder").mkdir(parents=True)
    (project / ".koder" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "matcher": "startup",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": f"python -c \"import sys, pathlib; pathlib.Path(r'{marker}').write_text(sys.stdin.read())\"",
                                }
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    result = dispatch_command_hooks(
        cwd=project,
        event_name="SessionStart",
        match_value="startup",
        payload={"event": "SessionStart", "source": "startup"},
    )

    assert result.matched_hooks == 1
    assert json.loads(marker.read_text(encoding="utf-8"))["event"] == "SessionStart"


def test_dispatch_command_hooks_supports_http_prompt_agent_and_async_handlers(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    async_marker = tmp_path / "async.txt"
    http_watch_path = tmp_path / "http-watch"
    received = []
    (project / ".koder").mkdir(parents=True)

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received.append(json.loads(body))
            response = json.dumps(
                {"hookSpecificOutput": {"watchPaths": [str(http_watch_path)]}}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    (project / ".koder" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "http",
                                    "url": url,
                                },
                                {
                                    "type": "prompt",
                                    "prompt": "Decide",
                                },
                                {
                                    "type": "agent",
                                    "prompt": "Investigate",
                                },
                                {
                                    "type": "command",
                                    "command": 'python -c "pass"',
                                },
                                {
                                    "type": "command",
                                    "command": f"python -c \"import pathlib,time; time.sleep(0.1); pathlib.Path(r'{async_marker}').write_text('done')\"",
                                    "async": True,
                                },
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "koder_agent.harness.hooks.runtime._run_prompt_hook",
        lambda **_kwargs: json.dumps({"hookSpecificOutput": {"echo": "prompt"}}),
    )
    monkeypatch.setattr(
        "koder_agent.harness.hooks.runtime._run_agent_hook",
        lambda **_kwargs: json.dumps({"hookSpecificOutput": {"echo": "agent"}}),
    )

    try:
        result = dispatch_command_hooks(
            cwd=project,
            event_name="Stop",
            match_value=None,
            payload={"event": "Stop"},
        )
    finally:
        try:
            server.shutdown()
        finally:
            server.server_close()
            thread.join(timeout=2)

    assert result.matched_hooks == 5
    assert result.watch_paths == [str(http_watch_path)]
    assert len(received) == 1 and received[0]["event"] == "Stop"
    time.sleep(0.2)
    assert async_marker.read_text(encoding="utf-8") == "done"


def test_post_tool_use_failure_hook_dispatches_for_error_results(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    marker = tmp_path / "post-tool-use-failure.json"
    (project / ".koder").mkdir(parents=True)
    (project / ".koder" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "PostToolUseFailure": [
                        {
                            "matcher": "read_file",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": f"python -c \"import sys, pathlib; pathlib.Path(r'{marker}').write_text(sys.stdin.read())\"",
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

    from koder_agent.harness.permissions.service import PermissionService
    from koder_agent.harness.tools.registry import ToolRegistry, ToolSpec

    async def _error_tool(_arguments):
        return {"tool": "read_file", "status": "error", "content": "boom"}

    registry = ToolRegistry.with_permission_service(PermissionService.default())
    registry.register(ToolSpec(name="read_file", invoke=_error_tool))
    result = __import__("asyncio").run(registry.get("read_file").invoke({}))

    assert result["status"] == "error"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["event"] == "PostToolUseFailure"


def test_post_tool_use_hook_can_block_successful_tool_result(tmp_path, monkeypatch):
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
                                    "command": 'python -c "print(\'{\\"decision\\":\\"block\\",\\"reason\\":\\"bad output\\"}\')"',
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

    from koder_agent.harness.permissions.service import PermissionService
    from koder_agent.harness.tools.registry import ToolRegistry, ToolSpec

    async def _ok_tool(_arguments):
        return {"tool": "read_file", "status": "success", "content": "ok"}

    registry = ToolRegistry.with_permission_service(PermissionService.default())
    registry.register(ToolSpec(name="read_file", invoke=_ok_tool))
    result = __import__("asyncio").run(registry.get("read_file").invoke({}))

    assert result["status"] == "error"
    assert result["content"] == "bad output"


def test_post_tool_use_failure_hook_can_override_error_message(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    (project / ".koder").mkdir(parents=True)
    (project / ".koder" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "PostToolUseFailure": [
                        {
                            "matcher": "read_file",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": 'python -c "print(\'{\\"decision\\":\\"block\\",\\"reason\\":\\"override failure\\"}\')"',
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

    from koder_agent.harness.permissions.service import PermissionService
    from koder_agent.harness.tools.registry import ToolRegistry, ToolSpec

    async def _error_tool(_arguments):
        return {"tool": "read_file", "status": "error", "content": "boom"}

    registry = ToolRegistry.with_permission_service(PermissionService.default())
    registry.register(ToolSpec(name="read_file", invoke=_error_tool))
    result = __import__("asyncio").run(registry.get("read_file").invoke({}))

    assert result["status"] == "error"
    assert result["content"] == "override failure"


def test_file_changed_hooks_fire_for_watched_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    watched = tmp_path / "watched.txt"
    marker = tmp_path / "file-changed.json"
    watched.write_text("one", encoding="utf-8")
    (project / ".koder").mkdir(parents=True)
    (project / ".koder" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "matcher": "startup",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": f'python -c "print(\'{{\\"hookSpecificOutput\\":{{\\"watchPaths\\":[\\"{watched}\\"]}}}}\')"',
                                }
                            ],
                        }
                    ],
                    "FileChanged": [
                        {
                            "matcher": watched.name,
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": f"python -c \"import sys, pathlib; pathlib.Path(r'{marker}').write_text(sys.stdin.read())\"",
                                }
                            ],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    session_start = dispatch_command_hooks(
        cwd=project,
        event_name="SessionStart",
        match_value="startup",
        payload={"event": "SessionStart", "source": "startup"},
    )
    update_watch_paths(session_start.watch_paths)
    watched.write_text("two", encoding="utf-8")
    fired = poll_file_change_hooks(project)

    assert fired == 1
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["event"] == "FileChanged"


def test_session_start_hooks_can_persist_environment_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    (project / ".koder").mkdir(parents=True)
    (project / ".koder" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "matcher": "startup",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "python -c \"import os, pathlib; pathlib.Path(os.environ['KODER_ENV_FILE']).write_text('export DEMO_ENV=hooked\\n')\"",
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
    monkeypatch.delenv("DEMO_ENV", raising=False)

    dispatch_command_hooks(
        cwd=project,
        event_name="SessionStart",
        match_value="startup",
        payload={"event": "SessionStart", "source": "startup", "session_id": "demo-session"},
    )

    assert os.environ["DEMO_ENV"] == "hooked"


def test_mcp_tool_matcher_supports_prefixed_tool_names(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    marker = tmp_path / "mcp.json"
    (project / ".koder").mkdir(parents=True)
    (project / ".koder" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "mcp__.*",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": f"python -c \"import sys, pathlib; pathlib.Path(r'{marker}').write_text(sys.stdin.read())\"",
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

    dispatch_command_hooks(
        cwd=project,
        event_name="PreToolUse",
        match_value="mcp__server__tool",
        payload={"event": "PreToolUse", "tool_name": "mcp__server__tool", "tool_input": {}},
    )

    assert json.loads(marker.read_text(encoding="utf-8"))["tool_name"] == "mcp__server__tool"


def test_elicitation_events_support_structured_output(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    marker = tmp_path / "elicitation.json"
    (project / ".koder").mkdir(parents=True)
    (project / ".koder" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "Elicitation": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": 'python -c "print(\'{\\"hookSpecificOutput\\":{\\"action\\":\\"accept\\",\\"content\\":{\\"answer\\":\\"ok\\"}}}\')"',
                                }
                            ]
                        }
                    ],
                    "ElicitationResult": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": f"python -c \"import sys, pathlib; pathlib.Path(r'{marker}').write_text(sys.stdin.read())\"",
                                }
                            ]
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    elicitation = dispatch_command_hooks(
        cwd=project,
        event_name="Elicitation",
        match_value="demo-server",
        payload={"event": "Elicitation", "server_name": "demo-server"},
    )
    dispatch_command_hooks(
        cwd=project,
        event_name="ElicitationResult",
        match_value="demo-server",
        payload={"event": "ElicitationResult", "server_name": "demo-server"},
    )

    assert elicitation.elicitation_action == "accept"
    assert elicitation.elicitation_content == {"answer": "ok"}
    assert json.loads(marker.read_text(encoding="utf-8"))["event"] == "ElicitationResult"


# ---------------------------------------------------------------------------
# H12: Reentrancy guard prevents infinite recursion
# ---------------------------------------------------------------------------


class TestHookReentrancyGuard:
    """dispatch_command_hooks inside a hook dispatch should be a no-op."""

    def test_dispatch_does_not_reenter(self, tmp_path, monkeypatch):
        """Nested dispatch_command_hooks returns empty result (no-op)."""
        from koder_agent.harness.hooks.runtime import _dispatch_guard

        monkeypatch.setenv("HOME", str(tmp_path))
        project = tmp_path / "project"
        (project / ".koder").mkdir(parents=True)
        (project / ".koder" / "settings.json").write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [{"hooks": [{"type": "command", "command": "echo block"}]}]
                    }
                }
            ),
            encoding="utf-8",
        )

        # Simulate already being inside a dispatch
        _dispatch_guard.in_dispatch = True
        try:
            result = dispatch_command_hooks(
                cwd=project,
                event_name="PreToolUse",
                payload={"tool_name": "run_shell"},
                match_value="run_shell",
            )
            # Should be a no-op due to reentrancy guard
            assert result.matched_hooks == 0
            assert result.blocked is False
        finally:
            _dispatch_guard.in_dispatch = False

    def test_dispatch_works_after_guard_released(self, tmp_path, monkeypatch):
        """After a guarded call completes, future dispatch calls work normally."""
        import subprocess

        from koder_agent.harness.hooks.runtime import _dispatch_guard

        monkeypatch.setenv("HOME", str(tmp_path))
        project = tmp_path / "project"
        (project / ".koder").mkdir(parents=True)
        (project / ".koder" / "settings.json").write_text(
            json.dumps(
                {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo ok"}]}]}}
            ),
            encoding="utf-8",
        )

        # Confirm guard is not set
        assert not getattr(_dispatch_guard, "in_dispatch", False)

        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        monkeypatch.setattr("koder_agent.harness.hooks.runtime.run_command", fake_run)

        result = dispatch_command_hooks(
            cwd=project,
            event_name="Stop",
            payload={"event": "Stop"},
            match_value=None,
        )
        assert result.matched_hooks == 1

        # Guard should be released after the call
        assert not getattr(_dispatch_guard, "in_dispatch", False)


def test_once_hook_does_not_double_fire(tmp_path, monkeypatch):
    """Concurrent dispatches with once:true hooks must not double-fire (M8 race fix)."""
    _once_fired.clear()
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    counter = tmp_path / "race-counter"
    counter.write_text("0", encoding="utf-8")

    (project / ".koder").mkdir(parents=True, exist_ok=True)
    (project / ".koder" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": (
                                        f'python -c "'
                                        f"import pathlib, time; "
                                        f"time.sleep(0.05); "
                                        f"p = pathlib.Path(r'{counter}'); "
                                        f'p.write_text(str(int(p.read_text()) + 1))"'
                                    ),
                                    "once": True,
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(project)

    # Fire many concurrent dispatches — only one should actually run the hook.
    errors: list[Exception] = []

    def _dispatch():
        try:
            dispatch_command_hooks(
                cwd=project,
                event_name="SessionStart",
                payload={"event": "SessionStart", "source": "startup"},
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_dispatch) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # The hook should have fired exactly once despite 10 concurrent dispatches.
    assert int(counter.read_text()) == 1
    _once_fired.clear()


@pytest.mark.parametrize("hook_name", ["_run_prompt_hook", "_run_agent_hook"])
@pytest.mark.parametrize("runner_fails", [False, True])
def test_agent_backed_hooks_close_mcp_owner(monkeypatch, hook_name, runner_fails):
    from koder_agent.harness.hooks import runtime as hook_runtime
    from koder_agent.mcp import MCPServerSet

    class Server:
        name = "hook-owned"

        def __init__(self):
            self.cleanup_count = 0

        async def cleanup(self):
            self.cleanup_count += 1

    server = Server()
    owner = MCPServerSet([server])
    agent = types.SimpleNamespace(mcp_servers=[], _koder_mcp_servers=owner)

    async def create_agent(*args, **kwargs):
        return agent

    async def run_agent(*args, **kwargs):
        if runner_fails:
            raise RuntimeError("hook failed")
        return types.SimpleNamespace(final_output="{}")

    monkeypatch.setattr("koder_agent.agentic.agent.create_dev_agent", create_agent)
    monkeypatch.setattr("agents.Runner.run", run_agent)
    monkeypatch.setattr("koder_agent.tools.get_all_tools", lambda: [])

    run_hook = getattr(hook_runtime, hook_name)
    if runner_fails:
        with pytest.raises(RuntimeError, match="hook failed"):
            run_hook(prompt_text="check", payload_text="{}", model=None)
    else:
        assert run_hook(prompt_text="check", payload_text="{}", model=None) == "{}"

    assert server.cleanup_count == 1
    assert agent._koder_mcp_servers is None
