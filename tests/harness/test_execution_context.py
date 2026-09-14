"""Directory snapshots preserve task isolation and the main workspace policy."""

import asyncio
import json
from pathlib import Path

import pytest

from koder_agent.harness.execution_context import (
    execution_directory,
    execution_path,
    execution_workspace_root,
    get_execution_cwd,
    scoped_execution_cwd,
)
from koder_agent.harness.permissions.modes import PermissionMode
from koder_agent.harness.permissions.service import PermissionService
from koder_agent.tools.file import write_file
from koder_agent.tools.permission_context import (
    reset_tool_permission_context,
    set_tool_permission_context,
)


def test_directory_scope_restores_outer_scope_on_failure(tmp_path):
    outer, inner = tmp_path / "outer", tmp_path / "inner"
    outer.mkdir()
    inner.mkdir()
    assert scoped_execution_cwd() is None
    with execution_directory(outer):
        with pytest.raises(ValueError, match="synthetic failure"):
            with execution_directory(inner):
                assert execution_path("file.txt") == inner / "file.txt"
                assert execution_workspace_root() == inner
                raise ValueError("synthetic failure")
        assert get_execution_cwd() == outer
    assert scoped_execution_cwd() is None


@pytest.mark.asyncio
async def test_directory_propagates_to_thread_without_chdir(tmp_path):
    origin = Path.cwd()
    with execution_directory(tmp_path):
        assert await asyncio.to_thread(get_execution_cwd) == tmp_path
        assert await asyncio.to_thread(Path.cwd) == origin
    assert get_execution_cwd() == origin


@pytest.mark.asyncio
@pytest.mark.parametrize("deny", [False, True])
async def test_main_subdirectory_targets_use_cwd_and_keep_workspace_root(
    tmp_path, monkeypatch, deny
):
    subdirectory = tmp_path / "src"
    subdirectory.mkdir()
    monkeypatch.chdir(subdirectory)
    permissions = PermissionService(
        mode=PermissionMode.ACCEPT_EDITS,
        workspace_root=tmp_path,
        rules={"write_file": {"deny": [str(subdirectory / "file.txt")]}} if deny else {},
    )
    token = set_tool_permission_context(permissions)
    try:
        output = await write_file.on_invoke_tool(
            None, json.dumps({"path": "file.txt", "content": "synthetic content"})
        )
        assert ("Permission denied" in output) is deny
        assert (subdirectory / "file.txt").exists() is not deny
        # The main session can still access a sibling of cwd inside its
        # originally authorized workspace, unlike an isolated child agent.
        sibling = await write_file.on_invoke_tool(
            None, json.dumps({"path": "../sibling.txt", "content": "sibling"})
        )
        assert "Permission denied" not in sibling
        assert (tmp_path / "sibling.txt").read_text() == "sibling"
    finally:
        reset_tool_permission_context(token)
    assert permissions.workspace_root == tmp_path
    assert scoped_execution_cwd() is None


@pytest.mark.asyncio
async def test_harness_registry_authorizes_actual_subdirectory_target(tmp_path, monkeypatch):
    from koder_agent.harness.tools.registry import ToolRegistry

    subdirectory = tmp_path / "src"
    subdirectory.mkdir()
    monkeypatch.chdir(subdirectory)
    permissions = PermissionService(
        mode=PermissionMode.ACCEPT_EDITS,
        workspace_root=tmp_path,
        rules={"write_file": {"deny": [str(subdirectory / "private.txt")]}},
    )
    registry = ToolRegistry.with_core_tools(categories={"file"}, permission_service=permissions)
    tool = registry.get("write_file")
    result = await tool.invoke({"path": "private.txt", "content": "must not be written"})
    assert result["status"] == "error"
    assert not (subdirectory / "private.txt").exists()
    assert scoped_execution_cwd() is None
