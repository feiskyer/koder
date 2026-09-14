"""An executable's basename is hazard evidence, not readonly authorization."""

import json
import os
import shlex
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.tool_context import ToolContext

from koder_agent.harness.permissions import service as service_module
from koder_agent.harness.permissions.shell_classifier import (
    classify_shell_command,
    normalize_segment_for_rule,
)
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


PATH_COMMANDS = [
    "./ls",
    "../bin/cat data.txt",
    "/tmp/custom/ls",
    "env ./ls",
    "timeout 5 /tmp/custom/cat data.txt",
    "./env ls",
    "/tmp/custom/timeout 5 ls",
]


@pytest.mark.parametrize("command", PATH_COMMANDS)
def test_explicit_paths_do_not_inherit_readonly_classification(command):
    decision = classify_shell_command(command)
    assert decision.allowed
    assert decision.requires_approval
    assert not decision.read_only


@pytest.mark.parametrize(
    ("command", "normalized"),
    [
        ("./ls", None),
        ("./env ls", None),
        ("env ./ls", "./ls"),
        ("./tool=value ls", None),
        ("timeout 5 /tmp/custom/cat data.txt", "/tmp/custom/cat data.txt"),
    ],
)
def test_allow_normalization_preserves_program_identity(command, normalized):
    assert normalize_segment_for_rule(shlex.split(command)) == normalized


@pytest.mark.parametrize(
    "command", ["./ls", "./env ls", "env ./ls", "echo ok && ./ls", "./tool=value ls"]
)
def test_bare_allow_rules_cannot_authorize_another_executable(local_policy, command):
    local_policy.add_rule("run_shell", "allow", "ls:*")
    local_policy.add_rule("run_shell", "allow", "echo:*")
    result = local_policy.evaluate_tool_call("run_shell", {"command": command})
    assert result.requires_approval


@pytest.mark.parametrize(
    "command", ["./ls", "/tmp/custom/ls", "./env ls", "env ./ls", "echo ok && ./ls"]
)
def test_basename_deny_rules_remain_conservative(local_policy, command):
    local_policy.add_rule("run_shell", "deny", "ls:*")
    local_policy.add_rule("run_shell", "allow", "*")
    result = local_policy.evaluate_tool_call("run_shell", {"command": command})
    assert not result.allowed and not result.requires_approval
    assert result.matched_rule == "ls:*"


@pytest.mark.parametrize(
    "command",
    [
        "/usr/bin/sudo id",
        "./sudo id",
        "./env sudo id",
        "env /usr/bin/sudo id",
        "/tmp/wrapper/env /bin/rm -rf /",
    ],
)
def test_explicit_paths_do_not_weaken_privileged_command_denials(command):
    decision = classify_shell_command(command)
    assert not decision.allowed
    assert decision.destructive


@pytest.mark.parametrize("command", ["ls", "env ls", "timeout 5 ls", "nice -n 1 cat data.txt"])
def test_existing_bare_readonly_commands_remain_supported(command):
    decision = classify_shell_command(command)
    assert decision.read_only and not decision.requires_approval


def test_unresolved_wrapper_chain_cannot_gain_an_allow_rule(local_policy):
    command = "env " * 8 + "ls"
    local_policy.add_rule("run_shell", "allow", "ls:*")
    assert normalize_segment_for_rule(shlex.split(command)) is None
    assert local_policy.evaluate_tool_call("run_shell", {"command": command}).requires_approval


def test_unparseable_command_does_not_inherit_a_raw_allow_prefix(local_policy):
    local_policy.add_rule("run_shell", "allow", "echo:*")
    command = 'echo ok; touch unapproved.txt; "unterminated'
    assert local_policy.evaluate_tool_call("run_shell", {"command": command}).requires_approval


@pytest.mark.skipif(os.name != "posix", reason="Uses an executable POSIX shell fixture")
@pytest.mark.parametrize("binary_name", ["ls", "env"])
@pytest.mark.parametrize("with_rule", [False, True])
@pytest.mark.asyncio
async def test_sdk_function_tool_preserves_the_executable_approval_boundary(
    local_policy, tmp_path, binary_name, with_rule
):
    program = tmp_path / binary_name
    marker = tmp_path / "sdk-executed.txt"
    program.write_text("#!/bin/sh\nprintf executed > sdk-executed.txt\n", encoding="utf-8")
    program.chmod(0o700)
    command = "./ls" if binary_name == "ls" else "./env ls"
    if with_rule:
        local_policy.add_rule("run_shell", "allow", "ls:*")
    approver = AsyncMock(return_value=False)
    encoded = json.dumps({"command": command})
    context = ToolContext(
        context=None, tool_name="run_shell", tool_call_id="identity-test", tool_arguments=encoded
    )
    token = set_tool_permission_context(local_policy, approver=approver)
    try:
        result = await run_shell.on_invoke_tool(context, encoded)
    finally:
        reset_tool_permission_context(token)
    assert not marker.exists(), f"Unapproved SDK executable ran: {result}"
    assert "Permission denied" in result
    approver.assert_awaited_once()


@pytest.mark.skipif(os.name != "posix", reason="Uses an executable POSIX shell fixture")
@pytest.mark.parametrize("binary_name", ["ls", "env"])
@pytest.mark.parametrize("with_rule", [False, True])
@pytest.mark.asyncio
async def test_unapproved_program_never_reaches_execution(
    local_policy, tmp_path, binary_name, with_rule
):
    program = tmp_path / binary_name
    marker = tmp_path / "executed.txt"
    program.write_text("#!/bin/sh\nprintf executed > executed.txt\n", encoding="utf-8")
    program.chmod(0o700)
    command = "./ls" if binary_name == "ls" else "./env ls"
    if with_rule:
        local_policy.add_rule("run_shell", "allow", "ls:*")
    registry = ToolRegistry.with_permission_service(local_policy)
    registry.register_module("shell")

    result = await registry.get("run_shell").invoke({"command": command})

    assert not marker.exists(), f"Unapproved executable ran: {result}"
    assert result["status"] == "approval_required"


@pytest.mark.skipif(os.name != "posix", reason="Uses an executable POSIX shell fixture")
@pytest.mark.parametrize("command", ["./ls", "env ./ls"])
@pytest.mark.asyncio
async def test_explicit_program_rule_can_authorize_its_execution(local_policy, tmp_path, command):
    program = tmp_path / "ls"
    marker = tmp_path / "executed.txt"
    program.write_text("#!/bin/sh\nprintf executed > executed.txt\n", encoding="utf-8")
    program.chmod(0o700)
    local_policy.add_rule("run_shell", "allow", "./ls:*")
    registry = ToolRegistry.with_permission_service(local_policy)
    registry.register_module("shell")

    result = await registry.get("run_shell").invoke({"command": command})

    assert result["status"] == "success"
    assert marker.read_text(encoding="utf-8") == "executed"
