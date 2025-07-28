"""Factory for creating MCP server instances."""

import logging
from typing import List, Optional

from agents.mcp import (
    MCPServer,
    MCPServerSse,
    MCPServerSseParams,
    MCPServerStdio,
    MCPServerStdioParams,
    MCPServerStreamableHttp,
    MCPServerStreamableHttpParams,
    create_static_tool_filter,
)

from .server_config import MCPServerConfig, MCPServerType

logger = logging.getLogger(__name__)


class MCPServerFactory:
    """Factory for creating MCP server instances from configurations."""

    @staticmethod
    def create_server(
        config: MCPServerConfig,
    ) -> MCPServer:
        """Create an MCP server instance from configuration."""
        try:
            # Create tool filter if specified
            tool_filter = None
            if config.allowed_tools or config.blocked_tools:
                tool_filter = create_static_tool_filter(
                    allowed_tool_names=config.allowed_tools,
                    blocked_tool_names=config.blocked_tools,
                )

            if config.transport_type == MCPServerType.STDIO:
                return MCPServerFactory._create_stdio_server(config, tool_filter)
            elif config.transport_type == MCPServerType.SSE:
                return MCPServerFactory._create_sse_server(config, tool_filter)
            elif config.transport_type == MCPServerType.HTTP:
                return MCPServerFactory._create_http_server(config, tool_filter)
            else:
                raise ValueError(f"Unsupported transport type: {config.transport_type}")

        except Exception as e:
            logger.error(f"Failed to create MCP server '{config.name}': {e}")
            raise

    @staticmethod
    def _create_stdio_server(config: MCPServerConfig, tool_filter) -> MCPServerStdio:
        """Create a stdio MCP server."""
        if not config.command:
            raise ValueError("stdio server requires a command")

        params = MCPServerStdioParams(
            command=config.command,
            args=config.args or [],
            env=config.env_vars or {},
        )
        return MCPServerStdio(
            params=params,
            client_session_timeout_seconds=300,
            tool_filter=tool_filter,
            cache_tools_list=config.cache_tools_list,
        )

    @staticmethod
    def _create_sse_server(config: MCPServerConfig, tool_filter) -> MCPServerSse:
        """Create an SSE MCP server."""
        if not config.url:
            raise ValueError("SSE server requires a URL")

        params = MCPServerSseParams(
            url=config.url,
            headers=config.headers or {},
            timeout=300,
        )

        return MCPServerSse(
            params=params,
            tool_filter=tool_filter,
            cache_tools_list=config.cache_tools_list,
        )

    @staticmethod
    def _create_http_server(config: MCPServerConfig, tool_filter) -> MCPServerStreamableHttp:
        """Create an HTTP MCP server."""
        if not config.url:
            raise ValueError("HTTP server requires a URL")

        params = MCPServerStreamableHttpParams(
            url=config.url,
            headers=config.headers or {},
            timeout=300,
        )

        return MCPServerStreamableHttp(
            params=params,
            tool_filter=tool_filter,
            cache_tools_list=config.cache_tools_list,
        )

    @staticmethod
    async def create_servers_from_configs(
        configs: List[MCPServerConfig],
    ) -> List[MCPServer]:
        """Create multiple MCP server instances from configurations."""
        servers = []
        for config in configs:
            try:
                server = MCPServerFactory.create_server(config)
                await server.connect()
                servers.append(server)
                logger.info(f"Created MCP server '{config.name}' ({config.transport_type})")
            except Exception as e:
                logger.error(f"Failed to create MCP server '{config.name}': {e}")
                # Continue with other servers even if one fails
                continue

        return servers

    @staticmethod
    def validate_config(config: MCPServerConfig) -> Optional[str]:
        """Validate an MCP server configuration."""
        try:
            if config.transport_type == MCPServerType.STDIO:
                if not config.command:
                    return "stdio servers must have a command"
            elif config.transport_type in [MCPServerType.SSE, MCPServerType.HTTP]:
                if not config.url:
                    return f"{config.transport_type} servers must have a URL"
                if not config.url.startswith(("http://", "https://")):
                    return "URL must start with http:// or https://"
            else:
                return f"Unsupported transport type: {config.transport_type}"

            # Validate tool lists don't overlap
            if config.allowed_tools and config.blocked_tools:
                overlap = set(config.allowed_tools) & set(config.blocked_tools)
                if overlap:
                    return f"Tools cannot be in both allowed and blocked lists: {list(overlap)}"

            return None
        except Exception as e:
            return f"Configuration validation error: {e}"
