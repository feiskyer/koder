"""MCP resource tools — list and read resources from connected MCP servers."""

from __future__ import annotations

import json
import logging

from ..harness.execution_context import get_execution_cwd
from ..mcp.server_manager import MCPServerManager
from .compat import function_tool

_logger = logging.getLogger(__name__)


@function_tool
async def list_mcp_resources(server: str = "") -> str:
    """List available MCP server resources.

    Returns a list of resources provided by connected MCP servers.
    Optionally filter by a specific server name.

    Args:
        server: Optional server name to filter results.
    """
    try:
        manager = MCPServerManager()
        servers = await manager.list_servers(cwd=str(get_execution_cwd()))
        filtered = [s for s in servers if not server or s.name == server]

        if server and not filtered:
            return f'Server "{server}" not found. Use "mcp list" to see connected servers.'

        entries = [
            {
                "uri": f"config://{s.name}",
                "name": s.name,
                "description": f"{getattr(s.transport_type, 'value', s.transport_type)} MCP server",
            }
            for s in filtered
        ]
        return json.dumps(entries, indent=2) if entries else "No MCP resources currently available."
    except Exception as exc:
        _logger.debug("list_mcp_resources failed: %s", exc)
        return "No MCP resources currently available. Use 'mcp list' to see connected servers."


@function_tool
async def read_mcp_resource(server_name: str, uri: str) -> str:
    """Read a resource from an MCP server.

    Args:
        server_name: Name of the MCP server providing the resource.
        uri: URI of the resource to read (e.g. config://server-name).
    """
    if not server_name:
        return "Missing required argument: server_name"
    if not uri:
        return "Missing required argument: uri"

    try:
        manager = MCPServerManager()
        server = await manager.get_server(server_name, cwd=str(get_execution_cwd()))
        if server is None:
            return f'Server "{server_name}" not found.'

        expected_uri = f"config://{server_name}"
        if uri != expected_uri:
            return f'Unsupported resource URI "{uri}" for server "{server_name}".'

        content = json.dumps(
            {
                "name": server.name,
                "transport_type": getattr(server.transport_type, "value", server.transport_type),
                "command": server.command,
                "args": server.args,
                "url": server.url,
            },
            sort_keys=True,
            indent=2,
        )
        return content
    except Exception as exc:
        _logger.debug("read_mcp_resource failed: %s", exc)
        return f"Failed to read MCP resource. Server: {server_name}, URI: {uri}"
