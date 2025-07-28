"""MCP (Model Context Protocol) support for Koder."""

from typing import List

from agents.mcp import MCPServer

from .server_config import MCPServerConfig, MCPServerType
from .server_factory import MCPServerFactory
from .server_manager import MCPServerManager


async def load_mcp_servers() -> List[MCPServer]:
    """Load and create MCP server instances from configuration."""
    try:
        manager = MCPServerManager()
        configs = await manager.list_servers()

        if not configs:
            return []

        # Create server instances
        servers = await MCPServerFactory.create_servers_from_configs(configs)

        return servers

    except Exception as e:
        raise RuntimeError(f"Failed to load MCP servers: {e}") from e


__all__ = [
    "load_mcp_servers",
    "MCPServerConfig",
    "MCPServerType",
    "MCPServerManager",
    "MCPServerFactory",
]
