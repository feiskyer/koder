"""MCP server configuration management."""

import os
from typing import List, Optional

import aiosqlite

from .server_config import MCPServerConfig


class MCPServerManager:
    """Manages MCP server configurations in the database."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Use the same path as ContextManager
            home_dir = os.path.expanduser("~")
            koder_dir = os.path.join(home_dir, ".koder")
            os.makedirs(koder_dir, exist_ok=True)
            self.db_path = os.path.join(koder_dir, "koder.db")
        else:
            self.db_path = db_path

    async def _ensure_mcp_table(self, conn: aiosqlite.Connection) -> None:
        """Ensure the MCP servers table exists."""
        await conn.execute(
            """CREATE TABLE IF NOT EXISTS mcp_servers (
                name TEXT PRIMARY KEY,
                transport_type TEXT NOT NULL,
                command TEXT,
                args TEXT,
                env_vars TEXT,
                url TEXT,
                headers TEXT,
                cache_tools_list INTEGER DEFAULT 0,
                allowed_tools TEXT,
                blocked_tools TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        await conn.commit()

    async def add_server(self, config: MCPServerConfig) -> None:
        """Add a new MCP server configuration."""
        data = config.to_db_dict()
        async with aiosqlite.connect(self.db_path) as conn:
            await self._ensure_mcp_table(conn)
            await conn.execute(
                """INSERT INTO mcp_servers
                   (name, transport_type, command, args, env_vars, url, headers,
                    cache_tools_list, allowed_tools, blocked_tools)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["name"],
                    data["transport_type"],
                    data["command"],
                    data["args"],
                    data["env_vars"],
                    data["url"],
                    data["headers"],
                    data["cache_tools_list"],
                    data["allowed_tools"],
                    data["blocked_tools"],
                ),
            )
            await conn.commit()

    async def get_server(self, name: str) -> Optional[MCPServerConfig]:
        """Get an MCP server configuration by name."""
        async with aiosqlite.connect(self.db_path) as conn:
            await self._ensure_mcp_table(conn)
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM mcp_servers WHERE name = ?", (name,))
            row = await cursor.fetchone()
            if row:
                return MCPServerConfig.from_db_dict(dict(row))
            return None

    async def list_servers(self) -> List[MCPServerConfig]:
        """List all MCP server configurations."""
        async with aiosqlite.connect(self.db_path) as conn:
            await self._ensure_mcp_table(conn)
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute("SELECT * FROM mcp_servers ORDER BY name")
            rows = await cursor.fetchall()
            return [MCPServerConfig.from_db_dict(dict(row)) for row in rows]

    async def update_server(self, config: MCPServerConfig) -> bool:
        """Update an existing MCP server configuration."""
        data = config.to_db_dict()
        async with aiosqlite.connect(self.db_path) as conn:
            await self._ensure_mcp_table(conn)
            cursor = await conn.execute(
                """UPDATE mcp_servers SET
                   transport_type = ?, command = ?, args = ?, env_vars = ?,
                   url = ?, headers = ?, cache_tools_list = ?,
                   allowed_tools = ?, blocked_tools = ?
                   WHERE name = ?""",
                (
                    data["transport_type"],
                    data["command"],
                    data["args"],
                    data["env_vars"],
                    data["url"],
                    data["headers"],
                    data["cache_tools_list"],
                    data["allowed_tools"],
                    data["blocked_tools"],
                    data["name"],
                ),
            )
            await conn.commit()
            return cursor.rowcount > 0

    async def remove_server(self, name: str) -> bool:
        """Remove an MCP server configuration."""
        async with aiosqlite.connect(self.db_path) as conn:
            await self._ensure_mcp_table(conn)
            cursor = await conn.execute("DELETE FROM mcp_servers WHERE name = ?", (name,))
            await conn.commit()
            return cursor.rowcount > 0

    async def server_exists(self, name: str) -> bool:
        """Check if a server with the given name exists."""
        async with aiosqlite.connect(self.db_path) as conn:
            await self._ensure_mcp_table(conn)
            cursor = await conn.execute("SELECT 1 FROM mcp_servers WHERE name = ? LIMIT 1", (name,))
            row = await cursor.fetchone()
            return row is not None

    async def get_servers_by_type(self, transport_type: str) -> List[MCPServerConfig]:
        """Get all servers of a specific transport type."""
        async with aiosqlite.connect(self.db_path) as conn:
            await self._ensure_mcp_table(conn)
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT * FROM mcp_servers WHERE transport_type = ? ORDER BY name",
                (transport_type,),
            )
            rows = await cursor.fetchall()
            return [MCPServerConfig.from_db_dict(dict(row)) for row in rows]
