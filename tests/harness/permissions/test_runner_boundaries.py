"""A wrapper's inner-command projection must not authorize other execution."""

import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.tool_context import ToolContext

from koder_agent.harness.permissions import service as service_module
from koder_agent.harness.permissions.shell_classifier import classify_shell_command
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


OPAQUE_COMMANDS = [
    "env -S './probe' ls",
    "env --split-string='./probe' ls",
    "env -S./probe ls",
    "env -P ls ./probe",
    "env --unknown-option ls ./probe",
    "env -u PATH ls",
    "env -i ls",
    "env PATH=. ls",
    "env LD_PRELOAD=./plugin.so cat sample.txt",
    "env FOO=bar ls",
    "nice FOO=bar ls",
    "nice --unknown-option ls ./probe",
    "timeout --unknown-option 1 ls",
    "timeout --signal ls 1 ./probe",
    "stdbuf --unknown-option ls ./probe",
    "command --unknown-option ls ./probe",
    "env -- -S ./probe ls",
    "nice -n",
    "timeout --signal",
    "timeout 5",
    "stdbuf --output",
    "setsid ls",
    "nohup echo ok",
    "xargs sort",
    "xargs -a ls ./probe",
    "echo -o marker.txt sample.txt | xargs sort",
]


@pytest.mark.parametrize("command", OPAQUE_COMMANDS)
def test_opaque_wrappers_do_not_inherit_readonly_classification(command):
    result = classify_shell_command(command)
    assert result.allowed and result.requires_approval and not result.read_only


@pytest.mark.parametrize(
    "command",
    OPAQUE_COMMANDS
    + [
        "PATH=. ls",
        "FOO=bar ls",
        "'FOO=bar' ls",
        '"FOO=bar" ls',
        r"F\OO=bar ls",
        "echo ok && 'FOO=bar' ls",
        "env 'ls suffix'",
        "echo ok && 'ls suffix'",
    ],
)
def test_opaque_wrappers_cannot_borrow_an_inner_allow_rule(local_policy, command):
    for program in ("ls", "echo", "sort", "cat"):
        local_policy.add_rule("run_shell", "allow", program + ":*")
    result = local_policy.evaluate_tool_call("run_shell", {"command": command})
    assert result.requires_approval, result


@pytest.mark.skipif(os.name != "posix", reason="Uses private executable POSIX fixtures")
@pytest.mark.parametrize("entrypoint", ["sdk", "registry"])
@pytest.mark.parametrize(
    "scenario", ["split", "path", "xargs", "quoted-assignment", "spaced-program"]
)
@pytest.mark.asyncio
async def test_unapproved_wrappers_cannot_execute_synthetic_writes(
    local_policy, tmp_path, monkeypatch, entrypoint, scenario
):
    marker = tmp_path / "marker.txt"
    programs = {"path": "ls", "quoted-assignment": "FOO=bar", "spaced-program": "ls suffix"}
    program = tmp_path / programs.get(scenario, "probe")
    program.write_text("#!/bin/sh\nprintf executed > marker.txt\n", encoding="utf-8")
    program.chmod(0o700)
    (tmp_path / "sample.txt").write_text("executed", encoding="utf-8")
    if scenario == "split":
        command = "env -S './probe' ls"
    elif scenario == "path":
        command = "env PATH=. ls"
    elif scenario == "xargs":
        command = "echo -o marker.txt sample.txt | xargs sort"
    else:
        monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
        command = "env 'ls suffix'" if scenario == "spaced-program" else "'FOO=bar' ls"
        local_policy.add_rule("run_shell", "allow", "ls:*")

    if entrypoint == "sdk":
        approver = AsyncMock(return_value=False)
        encoded = json.dumps({"command": command})
        context = ToolContext(
            context=None, tool_name="run_shell", tool_call_id="wrapper-test", tool_arguments=encoded
        )
        token = set_tool_permission_context(local_policy, approver=approver)
        try:
            result = await run_shell.on_invoke_tool(context, encoded)
        finally:
            reset_tool_permission_context(token)
        assert not marker.exists(), f"Unapproved SDK wrapper executed: {result}"
        assert "Permission denied" in result
        approver.assert_awaited_once()
    else:
        registry = ToolRegistry.with_permission_service(local_policy)
        registry.register_module("shell")
        result = await registry.get("run_shell").invoke({"command": command})
        assert not marker.exists(), f"Unapproved registry wrapper executed: {result}"
        assert result["status"] == "approval_required"


@pytest.mark.parametrize(
    "command",
    [
        "env ls",
        "env -- ls",
        "command -p -- ls",
        "timeout 5 ls",
        "timeout --signal=TERM --kill-after=1 5 ls",
        "timeout -sTERM -k1 5 ls",
        "nice -n 1 cat sample.txt",
        "nice --adjustment=1 cat sample.txt",
        "nice -n1 cat sample.txt",
        "stdbuf -oL cat sample.txt",
        "stdbuf --output=L cat sample.txt",
    ],
)
def test_explicitly_supported_runner_forms_remain_readonly(command):
    assert classify_shell_command(command).read_only


@pytest.mark.parametrize("command", ["env ls", "timeout 5 ls", "nice -n 1 ls", "command -- ls"])
def test_supported_runners_keep_inner_allow_rules(local_policy, command):
    local_policy.add_rule("run_shell", "allow", "ls:*")
    assert local_policy.evaluate_tool_call("run_shell", {"command": command}).allowed


@pytest.mark.parametrize(
    "command", ["env FOO=bar /usr/bin/sudo id", "timeout 5 sudo id", "./env sudo id"]
)
def test_conservative_hazard_projection_still_denies_privilege(command):
    assert not classify_shell_command(command).allowed


@pytest.mark.parametrize("command", ["FOO=bar ls", "env FOO=bar ls", "nohup ls"])
def test_explicit_full_command_rules_can_authorize_wrappers(local_policy, command):
    local_policy.add_rule("run_shell", "allow", command)
    assert local_policy.evaluate_tool_call("run_shell", {"command": command}).allowed
