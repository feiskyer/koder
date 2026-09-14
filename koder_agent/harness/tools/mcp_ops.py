"""Harness wrappers for MCP discovery tools."""

from __future__ import annotations

import json
from typing import Any

from koder_agent.harness.execution_context import get_execution_cwd
from koder_agent.mcp.server_manager import MCPServerManager

from .registry import ToolRegistry, ToolSpec, build_tool_result

DISCOVERABLE_TOOL_METADATA = {
    "read_file": "Read file contents from the working tree with optional offsets.",
    "write_file": "Write full file contents and return a diff summary.",
    "edit_file": "Apply a textual diff to an existing file.",
    "glob_search": "Find files by glob pattern under a workspace path.",
    "grep_search": "Search file contents by regex pattern.",
    "web_fetch": "Fetch a URL and summarize content for a prompt.",
    "web_search": "Search the public web for a query.",
    "list_mcp_resources": "List MCP resources exposed by configured servers.",
    "read_mcp_resource": "Read a specific MCP resource by server and URI.",
    "tool_search": "Search available deferred tools by keyword.",
}


def _server_description(transport_type: Any) -> str:
    value = getattr(transport_type, "value", transport_type)
    return f"{value} MCP server configuration"


async def invoke_list_mcp_resources(arguments: dict[str, Any]) -> dict[str, Any]:
    target_server = arguments.get("server")
    manager = MCPServerManager()
    servers = await manager.list_servers(cwd=str(get_execution_cwd()))
    filtered = [server for server in servers if not target_server or server.name == target_server]

    if target_server and not filtered:
        return build_tool_result(
            "list_mcp_resources",
            f'Server "{target_server}" not found',
            status="error",
        )

    content = [
        {
            "uri": f"config://{server.name}",
            "name": server.name,
            "mimeType": "application/x-koder-mcp-config",
            "description": _server_description(server.transport_type),
            "server": server.name,
        }
        for server in filtered
    ]
    return build_tool_result("list_mcp_resources", content, status="success")


async def invoke_read_mcp_resource(arguments: dict[str, Any]) -> dict[str, Any]:
    server_name = arguments.get("server")
    uri = arguments.get("uri")
    if not isinstance(server_name, str) or not server_name:
        return build_tool_result(
            "read_mcp_resource",
            "Missing required argument: server",
            status="error",
        )
    if not isinstance(uri, str) or not uri:
        return build_tool_result(
            "read_mcp_resource",
            "Missing required argument: uri",
            status="error",
        )

    manager = MCPServerManager()
    server = await manager.get_server(server_name, cwd=str(get_execution_cwd()))
    if server is None:
        return build_tool_result(
            "read_mcp_resource",
            f'Server "{server_name}" not found',
            status="error",
        )

    expected_uri = f"config://{server_name}"
    if uri != expected_uri:
        return build_tool_result(
            "read_mcp_resource",
            f'Unsupported resource URI "{uri}" for server "{server_name}"',
            status="error",
        )

    content = {
        "contents": [
            {
                "uri": expected_uri,
                "mimeType": "application/json",
                "text": json.dumps(
                    {
                        "name": server.name,
                        "transport_type": getattr(
                            server.transport_type, "value", server.transport_type
                        ),
                        "command": server.command,
                        "args": server.args,
                        "url": server.url,
                    },
                    sort_keys=True,
                ),
            }
        ]
    }
    return build_tool_result("read_mcp_resource", content, status="success")


async def invoke_tool_search(arguments: dict[str, Any]) -> dict[str, Any]:
    query = arguments.get("query")
    max_results = arguments.get("max_results", 5)
    if not isinstance(query, str) or not query.strip():
        return build_tool_result("tool_search", "Missing required argument: query", status="error")
    if not isinstance(max_results, int) or max_results <= 0:
        max_results = 5

    lowered = query.lower().strip()
    matches = [
        name
        for name, description in DISCOVERABLE_TOOL_METADATA.items()
        if lowered in name.lower() or lowered in description.lower()
    ]
    content = {
        "matches": matches[:max_results],
        "query": query,
        "total_deferred_tools": len(DISCOVERABLE_TOOL_METADATA),
    }
    return build_tool_result("tool_search", content, status="success")


def register_tools(registry: ToolRegistry) -> None:
    registry.register_many(
        [
            ToolSpec(
                name="list_mcp_resources",
                invoke=invoke_list_mcp_resources,
                category="mcp",
            ),
            ToolSpec(
                name="read_mcp_resource",
                invoke=invoke_read_mcp_resource,
                category="mcp",
            ),
            ToolSpec(name="tool_search", invoke=invoke_tool_search, category="mcp"),
        ],
        source=__name__,
    )
