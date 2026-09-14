"""MCP server mode -- expose koder tools as an MCP server over stdio."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from agents.tool_context import ToolContext
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

logger = logging.getLogger(__name__)

# Tools that should be excluded when serving koder as an MCP server.
# These are agent-internal primitives that don't make sense for external callers.
_EXCLUDED_TOOLS = frozenset(
    {
        "task_delegate",
        "send_message",
        "team_create",
        "team_delete",
        "agent_tool",
        "todo_read",
        "todo_write",
    }
)


def _get_koder_version() -> str:
    """Return the installed koder package version."""
    try:
        from importlib.metadata import version

        return version("koder")
    except Exception:
        return "0.0.0"


def _build_tool_list() -> tuple[list[types.Tool], dict[str, Any]]:
    """Build the MCP tool list and a name-to-tool mapping from koder tools.

    Returns a tuple of (mcp_tools, koder_tool_map) where *koder_tool_map*
    maps tool name to the original koder ``FunctionTool`` object so we can
    dispatch ``tools/call`` requests.
    """
    from koder_agent.tools import get_all_tools

    koder_tools = get_all_tools()
    mcp_tools: list[types.Tool] = []
    tool_map: dict[str, Any] = {}

    for tool in koder_tools:
        name: str = getattr(tool, "name", "")
        if not name or name in _EXCLUDED_TOOLS:
            continue

        description: str = getattr(tool, "description", name)
        schema: dict[str, Any] = {}
        if hasattr(tool, "params_json_schema"):
            schema = tool.params_json_schema

        mcp_tools.append(
            types.Tool(
                name=name,
                description=description,
                inputSchema=schema or {"type": "object", "properties": {}},
            )
        )
        tool_map[name] = tool

    return mcp_tools, tool_map


def create_mcp_server() -> Server:
    """Create and configure the MCP ``Server`` instance."""
    mcp_tools, tool_map = _build_tool_list()

    async def handle_list_tools() -> list[types.Tool]:
        return mcp_tools

    async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        koder_tool = tool_map.get(name)
        if koder_tool is None:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True,
            )

        try:
            raw_arguments = json.dumps(arguments or {})
            context = ToolContext(
                context=None,
                tool_name=name,
                tool_call_id=f"mcp-{uuid.uuid4().hex}",
                tool_arguments=raw_arguments,
            )
            result = await koder_tool.on_invoke_tool(context, raw_arguments)
            return types.CallToolResult(content=[types.TextContent(type="text", text=str(result))])
        except Exception as exc:
            logger.exception("Tool %s raised an error", name)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Tool error: {exc}")],
                isError=True,
            )

    if not hasattr(Server, "list_tools"):
        # MCP 2 uses constructor-based registration and parsed request params.
        async def on_list_tools(_context, _params):
            return types.ListToolsResult(tools=mcp_tools)

        async def on_call_tool(_context, params):
            return await handle_call_tool(params.name, params.arguments)

        return Server(
            "koder",
            version=_get_koder_version(),
            on_list_tools=on_list_tools,
            on_call_tool=on_call_tool,
        )

    server = Server("koder", version=_get_koder_version())
    server.list_tools()(handle_list_tools)
    server.call_tool()(handle_call_tool)
    return server


async def start_mcp_server() -> None:
    """Start koder as an MCP server over stdio transport.

    This function blocks until the client disconnects (stdin closes).
    """
    server = create_mcp_server()
    init_options = server.create_initialization_options()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)
