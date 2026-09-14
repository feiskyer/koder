# ruff: noqa: E402

import asyncio
import json
import shlex
import sys
import types
from pathlib import Path

if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from koder_agent.harness.agents.definitions import AgentDefinition
from koder_agent.harness.agents.hooks import SubagentLifecycleHooks
from koder_agent.harness.permissions.service import PermissionService


def _python_command(script):
    return shlex.join(
        [
            "uv",
            "run",
            "--no-project",
            "--no-env-file",
            "--python",
            sys.executable,
            "python",
            "-c",
            script,
        ]
    )


def test_subagent_frontmatter_hooks_run_for_matching_tools(tmp_path):
    pre_path = tmp_path / "pre.json"
    post_path = tmp_path / "post.json"
    agent = AgentDefinition(
        agent_type="reviewer",
        when_to_use="Reviews code",
        system_prompt="You are a reviewer.",
        source="projectSettings",
        hooks={
            "PreToolUse": [
                {
                    "matcher": "read_file",
                    "hooks": [
                        {
                            "type": "command",
                            "command": _python_command(
                                f"import sys, pathlib; pathlib.Path({str(pre_path)!r}).write_text(sys.stdin.read())"
                            ),
                        }
                    ],
                }
            ],
            "PostToolUse": [
                {
                    "matcher": "read_file",
                    "hooks": [
                        {
                            "type": "command",
                            "command": _python_command(
                                f"import sys, pathlib; pathlib.Path({str(post_path)!r}).write_text(sys.stdin.read())"
                            ),
                        }
                    ],
                }
            ],
        },
    )
    hooks = SubagentLifecycleHooks(agent_definition=agent, cwd=tmp_path)

    class _Agent:
        name = "reviewer"

    class _Tool:
        name = "read_file"

    asyncio.run(hooks.on_tool_start(None, _Agent(), _Tool()))
    asyncio.run(hooks.on_tool_end(None, _Agent(), _Tool(), "ok"))

    pre_payload = json.loads(pre_path.read_text(encoding="utf-8"))
    post_payload = json.loads(post_path.read_text(encoding="utf-8"))
    assert pre_payload["event"] == "PreToolUse"
    assert pre_payload["tool_name"] == "read_file"
    assert post_payload["event"] == "PostToolUse"
    assert post_payload["tool_name"] == "read_file"


def test_project_subagent_start_and_stop_hooks_run_from_settings(tmp_path):
    start_path = tmp_path / "start.json"
    stop_path = tmp_path / "stop.json"
    (tmp_path / ".koder").mkdir(parents=True)
    (tmp_path / ".koder" / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SubagentStart": [
                        {
                            "matcher": "reviewer",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": _python_command(
                                        f"import sys, pathlib; pathlib.Path({str(start_path)!r}).write_text(sys.stdin.read())"
                                    ),
                                }
                            ],
                        }
                    ],
                    "SubagentStop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": _python_command(
                                        f"import sys, pathlib; pathlib.Path({str(stop_path)!r}).write_text(sys.stdin.read())"
                                    ),
                                }
                            ],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    agent = AgentDefinition(
        agent_type="reviewer",
        when_to_use="Reviews code",
        system_prompt="You are a reviewer.",
        source="projectSettings",
    )
    hooks = SubagentLifecycleHooks(agent_definition=agent, cwd=tmp_path)

    class _Agent:
        name = "reviewer"

    asyncio.run(hooks.on_agent_start(None, _Agent()))
    asyncio.run(hooks.on_agent_end(None, _Agent(), "done"))

    start_payload = json.loads(start_path.read_text(encoding="utf-8"))
    stop_payload = json.loads(stop_path.read_text(encoding="utf-8"))
    assert start_payload["event"] == "SubagentStart"
    assert stop_payload["event"] == "SubagentStop"


def test_subagent_shell_preflight_does_not_fail_without_tool_arguments(tmp_path):
    agent = AgentDefinition(
        agent_type="reviewer",
        when_to_use="Reviews code",
        system_prompt="You are a reviewer.",
        source="projectSettings",
    )
    hooks = SubagentLifecycleHooks(
        agent_definition=agent,
        cwd=tmp_path,
        permission_service=PermissionService.default(workspace_root=tmp_path),
    )

    class _Agent:
        name = "reviewer"

    class _Tool:
        name = "run_shell"

    asyncio.run(hooks.on_tool_start(None, _Agent(), _Tool()))


def test_subagent_frontmatter_command_hooks_are_timeout_bounded(monkeypatch):
    """Definition-local hooks must supply a finite default to the shared runner."""
    seen = {}

    def fake_run(command, **kwargs):
        seen["timeout"] = kwargs.get("timeout", "MISSING")

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        return _R()

    monkeypatch.setattr("koder_agent.harness.agents.hooks.run_command", fake_run)

    # SubagentLifecycleHooks is already imported (module fully loaded), so reach
    # the private mini-runner via its module without re-triggering package import.
    hooks_mod = sys.modules[SubagentLifecycleHooks.__module__]
    rules = [{"hooks": [{"type": "command", "command": "echo hi"}]}]
    hooks_mod._run_command_hooks(rules, {"event": "PreToolUse"}, cwd=".")

    assert seen["timeout"] not in (None, "MISSING"), "hook runner got an unbounded timeout"
    assert isinstance(seen["timeout"], (int, float)) and seen["timeout"] > 0
