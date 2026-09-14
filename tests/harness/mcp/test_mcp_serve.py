"""Tests for koder MCP server mode (``koder mcp serve``)."""

from __future__ import annotations

import asyncio
from argparse import Namespace
from contextlib import AsyncExitStack, asynccontextmanager
from unittest.mock import AsyncMock, patch

import anyio
import mcp

from koder_agent.mcp.serve import _build_tool_list, create_mcp_server


@asynccontextmanager
async def connected_session():
    """Drive real server dispatch through SDK streams and initialization."""
    server = create_mcp_server()
    to_server, server_read = anyio.create_memory_object_stream(10)
    to_client, client_read = anyio.create_memory_object_stream(10)

    @asynccontextmanager
    async def transport():
        yield client_read, to_server

    async with to_server, server_read, to_client, client_read, anyio.create_task_group() as group:
        group.start_soon(server.run, server_read, to_client, server.create_initialization_options())
        async with AsyncExitStack() as stack:
            if hasattr(mcp, "Client"):
                client = await stack.enter_async_context(
                    mcp.Client(transport(), mode="legacy", cache=None, read_timeout_seconds=5)
                )
                session = client.session
            else:
                session = await stack.enter_async_context(mcp.ClientSession(client_read, to_server))
                await session.initialize()
            yield session
        group.cancel_scope.cancel()


# ---------------------------------------------------------------------------
# Tool list construction
# ---------------------------------------------------------------------------


class TestBuildToolList:
    """Verify that ``_build_tool_list`` correctly translates koder tools."""

    def test_returns_tools_and_map(self):
        mcp_tools, tool_map = _build_tool_list()
        assert len(mcp_tools) > 0
        assert len(tool_map) > 0
        # Every MCP tool must also appear in the mapping
        for t in mcp_tools:
            assert t.name in tool_map

    def test_excluded_tools_absent(self):
        """Agent-internal tools must not be exposed via MCP."""
        mcp_tools, tool_map = _build_tool_list()
        exposed_names = {t.name for t in mcp_tools}
        for excluded in (
            "task_delegate",
            "send_message",
            "team_create",
            "team_delete",
            "agent_tool",
            "todo_read",
            "todo_write",
        ):
            assert excluded not in exposed_names
            assert excluded not in tool_map

    def test_core_tools_present(self):
        """Key user-facing tools must be exposed."""
        mcp_tools, _ = _build_tool_list()
        exposed_names = {t.name for t in mcp_tools}
        for expected in (
            "read_file",
            "write_file",
            "edit_file",
            "glob_search",
            "grep_search",
            "run_shell",
        ):
            assert expected in exposed_names

    def test_tool_has_name_and_schema(self):
        """Every tool must have a name and a valid inputSchema dict."""
        mcp_tools, _ = _build_tool_list()
        for t in mcp_tools:
            assert t.name
            assert isinstance(t.model_dump(by_alias=True)["inputSchema"], dict)


# ---------------------------------------------------------------------------
# Server creation
# ---------------------------------------------------------------------------


class TestCreateMcpServer:
    """Verify that ``create_mcp_server`` produces a valid MCP Server."""

    def test_server_has_handlers(self):
        server = create_mcp_server()
        assert server.create_initialization_options().capabilities.tools is not None


# ---------------------------------------------------------------------------
# CLI parser acceptance
# ---------------------------------------------------------------------------


class TestCliParserAcceptsServe:
    def test_mcp_serve_parses(self):
        from koder_agent.cli import _build_cli_parser

        parser = _build_cli_parser("mcp")
        args = parser.parse_args(["mcp", "serve"])
        assert args.command == "mcp"
        assert args.mcp_action == "serve"


# ---------------------------------------------------------------------------
# Handler dispatch
# ---------------------------------------------------------------------------


class TestHandleMcpSubcommandServe:
    def test_serve_action_dispatches(self):
        """``handle_mcp_subcommand`` must call ``start_mcp_server`` for serve."""
        mock_start = AsyncMock()
        args = Namespace(mcp_action="serve")

        with patch(
            "koder_agent.mcp.serve.start_mcp_server",
            mock_start,
        ):
            from koder_agent.harness.mcp.commands import handle_mcp_subcommand

            result = asyncio.run(handle_mcp_subcommand(args))

        assert result == 0
        mock_start.assert_awaited_once()


# ---------------------------------------------------------------------------
# End-to-end tool invocation through the server handlers
# ---------------------------------------------------------------------------


class TestToolCallViaServer:
    """Exercise the ``call_tool`` handler registered on the MCP server."""

    def test_call_known_tool(self):
        """Calling a known tool (list_directory) should succeed."""

        async def _run():
            async with connected_session() as session:
                result = await session.call_tool("list_directory", {"path": "."})
                assert result.model_dump(by_alias=True).get("isError") is not True
                assert len(result.content) > 0
                assert result.content[0].type == "text"

        asyncio.run(_run())

    def test_call_unknown_tool_errors(self):
        """Calling an unknown tool should return an error result."""

        async def _run():
            async with connected_session() as session:
                result = await session.call_tool("nonexistent_tool_xyz", {})
                assert result.model_dump(by_alias=True)["isError"] is True

        asyncio.run(_run())

    def test_todo_tools_are_not_callable_without_an_mcp_request_identity(self):
        async def _run():
            async with connected_session() as session:
                responses = await asyncio.gather(
                    *(session.call_tool(name, {}) for name in ("todo_read", "todo_write"))
                )
                assert all(response.model_dump(by_alias=True)["isError"] for response in responses)

        asyncio.run(_run())

    def test_list_tools_handler(self):
        """The list_tools handler should return all exposed tools."""

        async def _run():
            async with connected_session() as session:
                result = await session.list_tools()
                names = {t.name for t in result.tools}
                assert "read_file" in names
                assert "task_delegate" not in names

        asyncio.run(_run())
