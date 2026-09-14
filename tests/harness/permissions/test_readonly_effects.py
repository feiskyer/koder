"""Read-only program names do not authorize effectful options or expanded argv."""

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


@pytest.mark.parametrize(
    "command",
    [
        "rg --pre ./probe needle sample.txt",
        "rg --pre=./probe needle sample.txt",
        "rg --hostname-bin ./probe needle sample.txt",
        "rg --hostname-bin=./probe needle sample.txt",
        "file -C -m magic.rules",
        "file --compile -m magic.rules",
        "file --comp -m magic.rules",
        "file -bC -m magic.rules",
        "sort --compress-program=./probe sample.txt",
        "sort --compress-program ./probe sample.txt",
        "sort --comp=./probe sample.txt",
        "sort $READONLY_ARGS",
        'sort "$READONLY_ARG" sample.txt',
        "sort *",
        "rg $READONLY_ARGS",
        "find . $PREDICATE",
        "file ${FILE_OPTIONS}",
        "git diff $DIFF_OPTIONS",
        "git tag -lrelease*",
        "git diff --ext-diff",
        "git show --textconv HEAD",
        "echo ${PROMPT@P}",
    ],
)
def test_effectful_or_expanded_arguments_require_approval(command):
    result = classify_shell_command(command)
    assert result.requires_approval and not result.read_only


@pytest.mark.parametrize(
    "command",
    [
        "rg -n needle sample.txt",
        "rg -e --pre sample.txt",
        "rg --regexp --pre sample.txt",
        "rg -- --pre sample.txt",
        "rg --pre-glob '*.txt' needle sample.txt",
        "file -m magic.rules sample.txt",
        "file -- -C",
        "sort -k1 sample.txt",
        "sort --key 1 sample.txt",
        "sort -- -o",
        "git diff --no-ext-diff",
        "echo '$LITERAL'",
        "rg '${LITERAL}' sample.txt",
        "rg '\\$NAME' sample.txt",
    ],
)
def test_literal_read_operations_remain_available(command):
    assert classify_shell_command(command).read_only


@pytest.mark.skipif(os.name != "posix", reason="Uses private POSIX command fixtures")
@pytest.mark.parametrize("entrypoint", ["sdk", "registry"])
@pytest.mark.parametrize("scenario", ["rg-pre", "rg-hostname", "file-compile", "sort-expansion"])
@pytest.mark.asyncio
async def test_unapproved_read_program_cannot_execute_or_publish(
    local_policy, tmp_path, monkeypatch, entrypoint, scenario
):
    marker = tmp_path / ("magic.rules.mgc" if scenario == "file-compile" else "marker.txt")
    (tmp_path / "sample.txt").write_text("needle\n", encoding="utf-8")
    (tmp_path / "magic.rules").write_text("0 string KODER synthetic\n", encoding="utf-8")
    probe = tmp_path / "probe"
    probe.write_text(
        "#!/bin/sh\nprintf executed > marker.txt\nprintf 'needle\\n'\n", encoding="utf-8"
    )
    probe.chmod(0o700)
    commands = {
        "rg-pre": "rg --pre ./probe needle sample.txt",
        "rg-hostname": "rg --hostname-bin ./probe --hyperlink-format 'file://{host}{path}' --color=always -H needle sample.txt",
        "file-compile": "file -C -m magic.rules",
        "sort-expansion": "sort $READONLY_ARGS",
    }
    monkeypatch.setenv("READONLY_ARGS", "-o marker.txt sample.txt")
    command = commands[scenario]
    if entrypoint == "sdk":
        approver = AsyncMock(return_value=False)
        encoded = json.dumps({"command": command})
        context = ToolContext(
            context=None, tool_name="run_shell", tool_call_id="effects-test", tool_arguments=encoded
        )
        token = set_tool_permission_context(local_policy, approver=approver)
        try:
            result = await run_shell.on_invoke_tool(context, encoded)
        finally:
            reset_tool_permission_context(token)
        assert not marker.exists(), f"Unapproved command published output: {result}"
        assert "Permission denied" in result
        approver.assert_awaited_once()
    else:
        registry = ToolRegistry.with_permission_service(local_policy)
        registry.register_module("shell")
        result = await registry.get("run_shell").invoke({"command": command})
        assert not marker.exists(), f"Unapproved command published output: {result}"
        assert result["status"] == "approval_required"
