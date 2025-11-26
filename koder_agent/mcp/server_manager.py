"""MCP server configuration management using YAML config file."""

import asyncio
from pathlib import Path
from typing import List, Optional

import aiosqlite

from ..config import MCPServerConfigYaml, get_config_manager
from .server_config import MCPServerConfig, MCPServerType


class MCPServerManager:
    """Manages MCP server configurations in the YAML config file."""

    def __init__(self):
        """Initialize the MCP server manager."""
        self.config_manager = get_config_manager()
        self._migration_lock = asyncio.Lock()
        self._migration_completed = False

    @staticmethod
    def _legacy_db_path() -> Path:
        """Path to the legacy SQLite database."""
        return Path.home() / ".koder" / "koder.db"

    async def _ensure_legacy_migration(self) -> None:
        """Import MCP servers from the legacy SQLite database if needed."""
        if self._migration_completed:
            return

        async with self._migration_lock:
            if self._migration_completed:
                return

            db_path = self._legacy_db_path()
            if not db_path.exists():
                self._migration_completed = True
                return

            try:
                async with aiosqlite.connect(db_path) as conn:
                    cursor = await conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='mcp_servers'"
                    )
                    if not await cursor.fetchone():
                        self._migration_completed = True
                        return

                    conn.row_factory = aiosqlite.Row
                    data_cursor = await conn.execute("SELECT * FROM mcp_servers")
                    rows = await data_cursor.fetchall()
            except Exception:
                # If we cannot read the legacy DB, skip migration silently
                self._migration_completed = True
                return

            if not rows:
                self._migration_completed = True
                return

            koder_config = self.config_manager.load()
            existing_names = {server.name for server in koder_config.mcp_servers}
            migrated = 0

            for row in rows:
                try:
                    config = MCPServerConfig.from_db_dict(dict(row))
                except Exception:
                    continue
                if config.name in existing_names:
                    continue
                koder_config.mcp_servers.append(self._mcp_config_to_yaml(config))
                existing_names.add(config.name)
                migrated += 1

            if migrated:
                self.config_manager.save(koder_config)

            self._migration_completed = True

    def _yaml_to_mcp_config(self, yaml_config: MCPServerConfigYaml) -> MCPServerConfig:
        """Convert YAML config model to MCPServerConfig.

        Args:
            yaml_config: The YAML configuration model.

        Returns:
            MCPServerConfig instance.
        """
        return MCPServerConfig(
            name=yaml_config.name,
            transport_type=MCPServerType(yaml_config.transport_type),
            command=yaml_config.command,
            args=yaml_config.args or [],
            env_vars=yaml_config.env_vars or {},
            url=yaml_config.url,
            headers=yaml_config.headers or {},
            cache_tools_list=yaml_config.cache_tools_list,
            allowed_tools=yaml_config.allowed_tools,
            blocked_tools=yaml_config.blocked_tools,
        )

    def _mcp_config_to_yaml(self, config: MCPServerConfig) -> MCPServerConfigYaml:
        """Convert MCPServerConfig to YAML config model.

        Args:
            config: The MCPServerConfig instance.

        Returns:
            MCPServerConfigYaml instance.
        """
        return MCPServerConfigYaml(
            name=config.name,
            transport_type=config.transport_type.value,
            command=config.command,
            args=config.args or [],
            env_vars=config.env_vars or {},
            url=config.url,
            headers=config.headers or {},
            cache_tools_list=config.cache_tools_list,
            allowed_tools=config.allowed_tools,
            blocked_tools=config.blocked_tools,
        )

    async def add_server(self, config: MCPServerConfig) -> None:
        """Add a new MCP server configuration.

        Args:
            config: The server configuration to add.
        """
        await self._ensure_legacy_migration()
        koder_config = self.config_manager.load()
        yaml_config = self._mcp_config_to_yaml(config)
        koder_config.mcp_servers.append(yaml_config)
        self.config_manager.save(koder_config)

    async def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """Get an MCP server configuration by name.

        Args:
            name: The server name to look up.

        Returns:
            MCPServerConfig if found, None otherwise.
        """
        await self._ensure_legacy_migration()
        koder_config = self.config_manager.load()
        for server in koder_config.mcp_servers:
            if server.name == name:
                return self._yaml_to_mcp_config(server)
        return None

    async def list_servers(self) -> List[MCPServerConfig]:
        """List all MCP server configurations.

        Returns:
            List of all configured MCP servers.
        """
        await self._ensure_legacy_migration()
        koder_config = self.config_manager.load()
        return [self._yaml_to_mcp_config(s) for s in koder_config.mcp_servers]

    async def update_server(self, config: MCPServerConfig) -> bool:
        """Update an existing MCP server configuration.

        Args:
            config: The updated server configuration.

        Returns:
            True if server was found and updated, False otherwise.
        """
        await self._ensure_legacy_migration()
        koder_config = self.config_manager.load()
        yaml_config = self._mcp_config_to_yaml(config)

        for i, server in enumerate(koder_config.mcp_servers):
            if server.name == config.name:
                koder_config.mcp_servers[i] = yaml_config
                self.config_manager.save(koder_config)
                return True
        return False

    async def remove_server(self, name: str) -> bool:
        """Remove an MCP server configuration.

        Args:
            name: The name of the server to remove.

        Returns:
            True if server was found and removed, False otherwise.
        """
        await self._ensure_legacy_migration()
        koder_config = self.config_manager.load()
        initial_count = len(koder_config.mcp_servers)
        koder_config.mcp_servers = [s for s in koder_config.mcp_servers if s.name != name]

        if len(koder_config.mcp_servers) < initial_count:
            self.config_manager.save(koder_config)
            return True
        return False

    async def server_exists(self, name: str) -> bool:
        """Check if a server with the given name exists.

        Args:
            name: The server name to check.

        Returns:
            True if server exists, False otherwise.
        """
        await self._ensure_legacy_migration()
        koder_config = self.config_manager.load()
        return any(s.name == name for s in koder_config.mcp_servers)

    async def get_servers_by_type(self, transport_type: str) -> List[MCPServerConfig]:
        """Get all servers of a specific transport type.

        Args:
            transport_type: The transport type to filter by (stdio, sse, http).

        Returns:
            List of servers matching the transport type.
        """
        await self._ensure_legacy_migration()
        koder_config = self.config_manager.load()
        return [
            self._yaml_to_mcp_config(s)
            for s in koder_config.mcp_servers
            if s.transport_type == transport_type
        ]
