"""Saved authorization retains executable identity and literal command spelling."""

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.tool_context import ToolContext

from koder_agent.harness.permissions import service as service_module
from koder_agent.harness.permissions.persistence import PermissionStore
from koder_agent.harness.permissions.rules import derive_shell_prefix_rule
from koder_agent.harness.permissions.shell_classifier import classify_shell_command
from koder_agent.harness.permissions.shell_segments import parse_shell_segments
from koder_agent.harness.tools import shell_executor
from koder_agent.harness.tools.registry import ToolRegistry
from koder_agent.tools.permission_context import (
    reset_tool_permission_context,
    set_tool_permission_context,
)
from koder_agent.tools.shell import run_shell


@pytest.fixture
def local_policy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for module in (service_module, shell_executor):
        monkeypatch.setattr(
            module, "resolve_sandbox_settings", lambda _cwd: SimpleNamespace(enabled=False)
        )
    return service_module.PermissionService.default()


@pytest.mark.parametrize(
    ("command", "prefix"),
    [
        ("./tools/npm test", "./tools/npm test:*"),
        ("/opt/tooling/npm test", "/opt/tooling/npm test:*"),
        ("./tools/pytest tests/test_app.py", "./tools/pytest:*"),
    ],
)
def test_saved_prefix_keeps_the_approved_executable(command, prefix):
    assert derive_shell_prefix_rule(command) == prefix


@pytest.mark.parametrize(
    "command",
    ["npm test & touch marker", "npm test |& touch marker", "npm test\n touch marker"],
)
def test_background_and_pipe_chains_are_never_widened(command):
    assert derive_shell_prefix_rule(command) is None


def test_quoted_literal_separator_is_not_misread_as_a_command_chain():
    assert derive_shell_prefix_rule("npm test --pattern 'a|b'") == "npm test:*"


@pytest.mark.parametrize(
    "command",
    [
        "ls#probe",
        "<ls ./probe",
        "env <ls ./probe",
        "echo ok && <ls ./probe",
        "ls\u00a0probe",
        "\u00a0ls",
        "ls\vls",
        "ls\rls",
        "ls\fls",
    ],
)
def test_literals_and_redirection_operands_are_not_the_executable(command):
    decision = classify_shell_command(command)
    assert decision.requires_approval and not decision.read_only


@pytest.mark.parametrize(
    "command",
    [
        "echo ok && 'FOO=bar' ls",
        "echo ok && F\\OO=bar ls",
        "command 'FOO=bar' ls",
        "env <ls ./probe",
        "echo ok && <ls ./probe",
    ],
)
def test_rule_matching_does_not_reinterpret_segment_spelling(local_policy, command):
    for rule in ("FOO=bar ls", "ls:*", "echo:*"):
        local_policy.add_rule("run_shell", "allow", rule)
    assert local_policy.evaluate_tool_call("run_shell", {"command": command}).requires_approval


@pytest.mark.parametrize(
    "command",
    [
        "echo ok && FOO=bar ls",
        "FOO=bar ls && echo ok",
        "echo ok && FOO=bar ls # literal assignment",
    ],
)
def test_literal_assignment_rule_still_matches_its_own_spelling(local_policy, command):
    local_policy.add_rule("run_shell", "allow", "FOO=bar ls")
    local_policy.add_rule("run_shell", "allow", "echo:*")
    assert local_policy.evaluate_tool_call("run_shell", {"command": command}).allowed


def test_persisted_path_rule_cannot_authorize_a_different_program(tmp_path, local_policy):
    store = PermissionStore(tmp_path / "rules.json")
    original = service_module.PermissionService.default(store=store, workspace_root=tmp_path)
    original.add_approval_rule("run_shell", {"command": "./tools/npm test"})
    reloaded = service_module.PermissionService.default(store=store, workspace_root=tmp_path)
    assert reloaded.evaluate_tool_call("run_shell", {"command": "./tools/npm test --watch"}).allowed
    for other in ("npm test", "./other/npm test", "/different/npm test"):
        assert reloaded.evaluate_tool_call("run_shell", {"command": other}).requires_approval


@pytest.mark.parametrize("command", ["env time cat data", "command 'if' test", "env '!' ls"])
def test_executable_keywords_do_not_borrow_shell_syntax_rules(local_policy, command):
    for rule in ("time:*", "if:*", "!:*"):
        local_policy.add_rule("run_shell", "allow", rule)
    assert local_policy.evaluate_tool_call("run_shell", {"command": command}).requires_approval


@pytest.mark.parametrize(
    ("command", "segments"),
    [
        ("echo 'a|b' && ls", [("echo 'a|b'", ("echo", "a|b")), ("ls", ("ls",))]),
        ("ls#local # trailing comment", [("ls#local", ("ls#local",))]),
        (
            "echo ok; 'FOO=bar' ls",
            [("echo ok", ("echo", "ok")), ("'FOO=bar' ls", ("FOO=bar", "ls"))],
        ),
        ("< input.txt cat", [("< input.txt cat", ("cat",))]),
        ("0<input.txt cat", [("0<input.txt cat", ("cat",))]),
        ('"0"<input.txt cat', [('"0"<input.txt cat', ("0", "cat"))]),
        ("cat < 'ls' 2>&1", [("cat < 'ls' 2>&1", ("cat",))]),
        ('echo "|" "<" "#" "2"', [('echo "|" "<" "#" "2"', ("echo", "|", "<", "#", "2"))]),
        ("echo a\\#b", [("echo a\\#b", ("echo", "a#b"))]),
        ("echo a\\;b", [("echo a\\;b", ("echo", "a;b"))]),
        ("echo a#b; ls", [("echo a#b", ("echo", "a#b")), ("ls", ("ls",))]),
        ("ls\u00a0local", [("ls\u00a0local", ("ls\u00a0local",))]),
        ("echo one\ncat data", [("echo one", ("echo", "one")), ("cat data", ("cat", "data"))]),
    ],
)
def test_shared_segmenter_preserves_literal_spelling_and_real_argv(command, segments):
    assert [(segment.raw, segment.tokens) for segment in parse_shell_segments(command)] == segments


@pytest.mark.parametrize("command", ["cat <", "cat < ; ls", "echo 'unterminated", "(ls)", "ls\0"])
def test_unsupported_or_incomplete_syntax_does_not_gain_rules(local_policy, command):
    local_policy.add_rule("run_shell", "allow", "cat:*")
    local_policy.add_rule("run_shell", "allow", "echo:*")
    local_policy.add_rule("run_shell", "allow", "ls:*")
    assert local_policy.evaluate_tool_call("run_shell", {"command": command}).requires_approval


@pytest.mark.skipif(os.name != "posix", reason="Uses private POSIX executable fixtures")
@pytest.mark.parametrize("entrypoint", ["sdk", "registry"])
@pytest.mark.parametrize("scenario", ["comment", "redirect", "quoted-assignment"])
@pytest.mark.asyncio
async def test_literal_spelling_cannot_smuggle_a_private_marker_write(
    local_policy, tmp_path, monkeypatch, entrypoint, scenario
):
    marker = tmp_path / "marker.txt"
    program = (
        tmp_path
        / {"comment": "ls#probe", "redirect": "probe", "quoted-assignment": "FOO=bar"}[scenario]
    )
    program.write_text("#!/bin/sh\nprintf executed > marker.txt\n", encoding="utf-8")
    program.chmod(0o700)
    (tmp_path / "ls").write_text("input", encoding="utf-8")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    command = {
        "comment": "ls#probe",
        "redirect": "<ls ./probe",
        "quoted-assignment": "echo ok && 'FOO=bar' ls",
    }[scenario]
    if scenario == "quoted-assignment":
        local_policy.add_rule("run_shell", "allow", "echo:*")
        local_policy.add_rule("run_shell", "allow", "FOO=bar ls")
    if entrypoint == "sdk":
        approver = AsyncMock(return_value=False)
        encoded = json.dumps({"command": command})
        context = ToolContext(
            context=None,
            tool_name="run_shell",
            tool_call_id="spelling-test",
            tool_arguments=encoded,
        )
        token = set_tool_permission_context(local_policy, approver=approver)
        try:
            result = await run_shell.on_invoke_tool(context, encoded)
        finally:
            reset_tool_permission_context(token)
        assert not marker.exists(), f"Unapproved SDK command executed: {result}"
        assert "Permission denied" in result
        approver.assert_awaited_once()
    else:
        registry = ToolRegistry.with_permission_service(local_policy)
        registry.register_module("shell")
        result = await registry.get("run_shell").invoke({"command": command})
        assert not marker.exists(), f"Unapproved registry command executed: {result}"
        assert result["status"] == "approval_required"
