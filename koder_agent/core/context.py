"""Context management for conversation history."""

import json
import os
from typing import Dict, List, Optional

import aiosqlite
import tiktoken


class ContextManager:
    """Manages conversation context and history with token limits."""

    def __init__(
        self, session_id: str = "default", db_path: Optional[str] = None, max_tokens: int = 50000
    ):
        self.session_id = session_id
        if db_path is None:
            # Use $HOME/.koder/koder.db as default
            home_dir = os.path.expanduser("~")
            koder_dir = os.path.join(home_dir, ".koder")
            os.makedirs(koder_dir, exist_ok=True)
            self.db_path = os.path.join(koder_dir, "koder.db")
        else:
            self.db_path = db_path
        self.max_tokens = max_tokens
        self.encoder = tiktoken.encoding_for_model("gpt-4o")

    async def _ensure_table(self, conn: aiosqlite.Connection) -> None:
        """Ensure the context table exists."""
        await conn.execute(
            """CREATE TABLE IF NOT EXISTS ctx (
                sid TEXT PRIMARY KEY,
                msgs TEXT
            )"""
        )
        await conn.commit()

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

    async def load(self) -> List[Dict[str, str]]:
        """Load conversation history from database."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await self._ensure_table(conn)
                cursor = await conn.execute(
                    "SELECT msgs FROM ctx WHERE sid = ?", (self.session_id,)
                )
                row = await cursor.fetchone()
                if row and row[0]:
                    return json.loads(row[0])
                return []
        except Exception as e:
            print(f"Error loading context: {e}")
            return []

    async def save(self, messages: List[Dict[str, str]]) -> None:
        """Save conversation history to database."""
        try:
            # Check token count and compress if needed
            total_tokens = sum(len(self.encoder.encode(msg["content"])) for msg in messages)

            if total_tokens > self.max_tokens:
                messages = self._compress_messages(messages)

            async with aiosqlite.connect(self.db_path) as conn:
                await self._ensure_table(conn)
                await conn.execute(
                    "REPLACE INTO ctx VALUES(?, ?)",
                    (self.session_id, json.dumps(messages, ensure_ascii=False)),
                )
                await conn.commit()
        except Exception as e:
            print(f"Error saving context: {e}")

    def _compress_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Compress messages to fit within token limit."""
        if not messages:
            return messages

        # Keep first message (system) if it exists
        result = []
        system_msg = None
        if messages and messages[0].get("role") == "system":
            system_msg = messages[0]
            result.append(system_msg)
            remaining_messages = messages[1:]
        else:
            remaining_messages = messages

        # Calculate tokens for system message
        system_tokens = len(self.encoder.encode(system_msg["content"])) if system_msg else 0
        available_tokens = self.max_tokens - system_tokens

        # Work backwards from the end, adding messages until we hit token limit
        current_tokens = 0
        messages_to_keep = []

        for msg in reversed(remaining_messages):
            msg_tokens = len(self.encoder.encode(msg["content"]))
            if current_tokens + msg_tokens > available_tokens:
                break
            messages_to_keep.insert(0, msg)
            current_tokens += msg_tokens

        # Add summary if we dropped messages
        dropped_count = len(remaining_messages) - len(messages_to_keep)
        if dropped_count > 0:
            summary = {
                "role": "system",
                "content": f"[Note: {dropped_count} previous messages were truncated to stay within token limits]",
            }
            result.append(summary)

        result.extend(messages_to_keep)
        return result

    async def clear(self) -> None:
        """Clear conversation history for the current session."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await self._ensure_table(conn)
                await conn.execute("DELETE FROM ctx WHERE sid = ?", (self.session_id,))
                await conn.commit()
        except Exception as e:
            print(f"Error clearing context: {e}")

    async def list_sessions(self) -> List[str]:
        """List all available sessions."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await self._ensure_table(conn)
                cursor = await conn.execute("SELECT sid FROM ctx")
                rows = await cursor.fetchall()
                return [row[0] for row in rows]
        except Exception:
            return []
