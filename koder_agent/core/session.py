"""Enhanced SQLiteSession with Koder-specific features.

This module extends the official agents.SQLiteSession to add:
- Session titles with LLM-based generation
- Token-aware message summarization
- Separate metadata storage for extensibility
"""

import json
import os
from typing import Dict, List, Optional

import aiosqlite
import tiktoken
from agents import SQLiteSession
from agents.items import TResponseInputItem

from ..utils.client import get_model_name, llm_completion
from ..utils.model_info import get_summarization_threshold


async def migrate_legacy_sessions(db_path: str) -> None:
    """Migrate legacy sessions from ctx table to new SQLiteSession format.

    This function:
    1. Checks if the old `ctx` table exists
    2. Checks if migration has already been performed
    3. Migrates all sessions and titles to the new format
    4. Marks migration as complete

    The old `ctx` table is kept for a grace period as backup.

    Args:
        db_path: Path to the SQLite database
    """
    async with aiosqlite.connect(db_path) as conn:
        # Check if legacy ctx table exists
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ctx'"
        )
        if not await cursor.fetchone():
            return  # No legacy data to migrate

        # Check if migration already done
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='migration_status'"
        )
        if await cursor.fetchone():
            return  # Already migrated

        # Create session_metadata table if not exists
        await conn.execute(
            """CREATE TABLE IF NOT EXISTS session_metadata (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )

        # Get all legacy sessions
        cursor = await conn.execute("SELECT sid, msgs, title FROM ctx")
        sessions = await cursor.fetchall()

        # First, close this connection to avoid locks
        await conn.commit()

    # Now migrate each session with separate connections
    migrated_count = 0
    for session_id, msgs_json, title in sessions:
        try:
            # Parse messages
            messages = json.loads(msgs_json) if msgs_json else []

            if messages:
                # Create SQLiteSession instance and add items
                # This will open its own connection
                session = SQLiteSession(session_id, db_path)
                await session.add_items(messages)

            # Migrate title to session_metadata table (separate connection)
            if title:
                async with aiosqlite.connect(db_path) as title_conn:
                    await title_conn.execute(
                        """INSERT OR REPLACE INTO session_metadata
                        (session_id, title) VALUES (?, ?)""",
                        (session_id, title),
                    )
                    await title_conn.commit()

            migrated_count += 1

        except Exception:
            continue

    # Mark migration as complete (separate connection)
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """CREATE TABLE migration_status (
                migrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                migrated_sessions INTEGER
            )"""
        )
        await conn.execute(
            "INSERT INTO migration_status (migrated_sessions) VALUES (?)", (migrated_count,)
        )
        await conn.commit()


class EnhancedSQLiteSession(SQLiteSession):
    """Extended SQLiteSession with title management and auto-summarization.

    This class wraps the official SQLiteSession and adds:
    1. Session titles stored in a separate metadata table
    2. Automatic message summarization when token threshold is exceeded
    3. LLM-based title generation from first user message

    The underlying SQLiteSession handles all conversation history persistence,
    while this class adds the custom features in a non-intrusive way.
    """

    def __init__(
        self,
        session_id: str,
        db_path: Optional[str] = None,
        summarization_threshold: Optional[int] = None,
    ):
        """Initialize enhanced session.

        Args:
            session_id: Unique identifier for this session
            db_path: Path to SQLite database file (default: ~/.koder/koder.db)
            summarization_threshold: Token count threshold for summarization
                                   (default: model-specific from config)
        """
        # Set up database path
        if db_path is None:
            home_dir = os.path.expanduser("~")
            koder_dir = os.path.join(home_dir, ".koder")
            os.makedirs(koder_dir, exist_ok=True)
            db_path = os.path.join(koder_dir, "koder.db")

        # Initialize base SQLiteSession
        super().__init__(session_id, db_path)

        # Store configuration
        self.summarization_threshold = summarization_threshold
        self._title: Optional[str] = None

        # Initialize tiktoken encoder for accurate token counting
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            try:
                self.encoder = tiktoken.encoding_for_model("gpt-4o")
            except Exception:
                # Fallback: approximate tokens using UTF-8 bytes
                class _NaiveEncoder:
                    def encode(self, text: str) -> list[int]:
                        return list(text.encode("utf-8"))

                self.encoder = _NaiveEncoder()

    async def _ensure_metadata_table(self) -> None:
        """Ensure the session_metadata table exists."""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute(
                """CREATE TABLE IF NOT EXISTS session_metadata (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )"""
            )
            await conn.commit()

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """Add items to session with automatic summarization if needed.

        This override intercepts the items before they're saved and:
        1. Estimates the total token count
        2. If threshold exceeded, summarizes assistant responses
        3. Passes processed items to the base class

        Args:
            items: List of message dictionaries to add
        """
        if not items:
            await super().add_items(items)
            return

        # Get current session history
        existing_items = await self.get_items()

        # Combine existing and new items for token calculation
        all_items = existing_items + items

        # Estimate token count
        total_tokens = self._estimate_tokens(all_items)

        # Check if summarization is needed
        threshold = self.summarization_threshold
        if threshold is None:
            model = get_model_name()
            threshold = get_summarization_threshold(model)

        # Summarize if threshold exceeded
        if total_tokens > threshold:
            print(f"\n[Session] Token count: {total_tokens} exceeds threshold: {threshold}")
            print("[Session] Triggering message history summarization...")
            all_items = await self._summarize_messages(all_items, total_tokens)
            new_tokens = self._estimate_tokens(all_items)
            print(f"[Session] Summarization complete: {total_tokens} -> {new_tokens} tokens")

            # Clear existing items and replace with summarized version
            await self.clear_session()
            await super().add_items(all_items)
        else:
            # No summarization needed, just add new items
            await super().add_items(items)

    def _estimate_tokens(self, messages: List[Dict]) -> int:
        """Accurately calculate token count for message history.

        Args:
            messages: List of message dictionaries

        Returns:
            Estimated total token count
        """
        total_tokens = 0

        for msg in messages:
            if not isinstance(msg, dict):
                total_tokens += len(self.encoder.encode(str(msg)))
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                total_tokens += len(self.encoder.encode(content))
            elif isinstance(content, list):
                # Handle multimodal messages (e.g., images + text)
                for block in content:
                    if isinstance(block, dict):
                        total_tokens += len(
                            self.encoder.encode(json.dumps(block, ensure_ascii=False))
                        )
                    else:
                        total_tokens += len(self.encoder.encode(str(block)))
            else:
                total_tokens += len(self.encoder.encode(str(content)))

            # Metadata overhead per message (~4 tokens)
            total_tokens += 4

        return total_tokens

    async def _summarize_messages(self, messages: List[Dict], current_tokens: int) -> List[Dict]:
        """Summarize message history using LLM when tokens exceed threshold.

        Strategy:
        - Keep all user messages (these represent user intents)
        - Summarize assistant responses between user messages
        - Structure: system -> user1 -> summary1 -> user2 -> summary2 -> ...

        Args:
            messages: Original message list
            current_tokens: Current token count

        Returns:
            Summarized message list
        """
        # Find all user message indices
        user_indices = [i for i, msg in enumerate(messages) if msg.get("role") == "user"]

        if len(user_indices) < 1:
            print("[Session] Insufficient messages, cannot summarize")
            return messages

        # Build new message list
        new_messages = []
        summary_count = 0

        # Keep system message if present
        if messages and messages[0].get("role") == "system":
            new_messages.append(messages[0])

        # Iterate through each user message and summarize execution after it
        for i, user_idx in enumerate(user_indices):
            # Add current user message
            new_messages.append(messages[user_idx])

            # Determine message range to summarize
            if i < len(user_indices) - 1:
                next_user_idx = user_indices[i + 1]
            else:
                next_user_idx = len(messages)

            # Extract execution messages for this round
            execution_messages = messages[user_idx + 1 : next_user_idx]

            # If there are execution messages, summarize them
            if execution_messages:
                summary_text = await self._create_summary(execution_messages, i + 1)
                if summary_text:
                    summary_message = {
                        "role": "assistant",
                        "content": f"[Previous Response Summary]\n\n{summary_text}",
                    }
                    new_messages.append(summary_message)
                    summary_count += 1

        print(f"[Session] Structure: {len(user_indices)} user messages + {summary_count} summaries")
        return new_messages

    async def _create_summary(self, messages: List[Dict], round_num: int) -> str:
        """Create a summary for one execution round using LLM.

        Args:
            messages: List of messages to summarize
            round_num: Round number for logging

        Returns:
            Summary text, or empty string if summarization fails
        """
        if not messages:
            return ""

        # Build content to summarize
        summary_content = f"Round {round_num} execution:\n\n"
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Normalize content to string
            if isinstance(content, (dict, list)):
                try:
                    content_str = json.dumps(content, ensure_ascii=False)
                except TypeError:
                    content_str = str(content)
            else:
                content_str = str(content)

            if role == "assistant":
                summary_content += f"Assistant: {content_str}\n"
            elif role == "tool":
                tool_name = msg.get("name", "unknown")
                if len(content_str) > 500:
                    content_preview = content_str[:500] + "..."
                else:
                    content_preview = content_str
                summary_content += f"Tool ({tool_name}): {content_preview}\n"
            elif role != "system":
                summary_content += f"{role.capitalize()}: {content_str}\n"

        # Call LLM to generate concise summary
        try:
            summary_prompt = f"""Please provide a concise summary of the following Agent execution process:

{summary_content}

Requirements:
1. Focus on what tasks were completed and which tools were called
2. Keep key execution results and important findings
3. Be concise and clear, within 500 words
4. Use the same language as the original content
5. Only summarize the Agent's execution process, not user requests"""

            summary_text = await llm_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant skilled at summarizing Agent execution processes concisely.",
                    },
                    {"role": "user", "content": summary_prompt},
                ]
            )

            print(f"[Session] Summary for round {round_num} generated successfully")
            return summary_text

        except Exception as e:
            print(f"[Session] Summary generation failed for round {round_num}: {e}")
            # Fallback: use truncated original content
            truncated = (
                summary_content[:1000] + "..." if len(summary_content) > 1000 else summary_content
            )
            return f"[Summary generation failed - truncated content]\n{truncated}"

    async def get_title(self) -> Optional[str]:
        """Get the title for this session.

        Returns:
            Session title or None if not set
        """
        try:
            await self._ensure_metadata_table()
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute(
                    "SELECT title FROM session_metadata WHERE session_id = ?", (self.session_id,)
                )
                row = await cursor.fetchone()
                return row[0] if row and row[0] else None
        except Exception:
            return None

    async def set_title(self, title: str) -> None:
        """Set the title for this session.

        Args:
            title: Title to set
        """
        try:
            await self._ensure_metadata_table()
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute(
                    """INSERT INTO session_metadata (session_id, title, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(session_id) DO UPDATE SET
                        title = excluded.title,
                        updated_at = CURRENT_TIMESTAMP""",
                    (self.session_id, title),
                )
                await conn.commit()
        except Exception as e:
            print(f"[Session] Error setting title: {e}")

    async def generate_title(self, user_message: str) -> str:
        """Generate a concise session title from the first user message.

        Uses LLM to create a 3-6 word title that captures the main intent.

        Args:
            user_message: First user message in the session

        Returns:
            Generated title (fallback to truncated message if generation fails)
        """
        if not user_message or not user_message.strip():
            return "New session"

        prompt = f"""Generate a very short, descriptive title (3-6 words max) for this coding session based on the user's first message. The title should capture the main intent or task. Only return the title text, nothing else.

User message: {user_message[:500]}

Examples of good titles:
- Fix authentication bug
- Add dark mode toggle
- Refactor database queries
- Setup CI/CD pipeline"""

        try:
            title = await llm_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a concise title generator. Return only the title, no quotes or extra text.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            # Clean up the title
            title = title.strip().strip("\"'").strip()
            # Limit length
            if len(title) > 50:
                title = title[:47] + "..."
            return title
        except Exception:
            # Fallback: use first 50 chars of user message
            fallback = user_message[:50].strip()
            if len(user_message) > 50:
                fallback += "..."
            return fallback

    async def get_display_name(self) -> str:
        """Get display name for session: 'title - YYYY-MM-DD HH:MM' or session ID.

        Returns:
            Formatted display name
        """
        title = await self.get_title()
        if title:
            from ..utils.sessions import parse_session_dt

            _, dt = parse_session_dt(self.session_id)
            if dt:
                date_suffix = dt.strftime("%Y-%m-%d %H:%M")
                return f"{title} - {date_suffix}"
            return title
        return self.session_id

    @staticmethod
    async def list_sessions_with_titles(
        db_path: Optional[str] = None,
    ) -> list[tuple[str, Optional[str]]]:
        """List all sessions with their titles from the database.

        This is a static method that queries all sessions, not just the current one.

        Args:
            db_path: Path to database (default: ~/.koder/koder.db)

        Returns:
            List of (session_id, title) tuples
        """
        if db_path is None:
            home_dir = os.path.expanduser("~")
            db_path = os.path.join(home_dir, ".koder", "koder.db")

        try:
            async with aiosqlite.connect(db_path) as conn:
                # Ensure metadata table exists
                await conn.execute(
                    """CREATE TABLE IF NOT EXISTS session_metadata (
                        session_id TEXT PRIMARY KEY,
                        title TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )"""
                )

                # Get all sessions from the SQLiteSession table
                # SQLiteSession stores data in an 'items' table
                session_ids = set()

                # Check if there's an 'items' table (SQLiteSession's storage table)
                cursor = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='items'"
                )
                if await cursor.fetchone():
                    cursor = await conn.execute("SELECT DISTINCT session_id FROM items")
                    rows = await cursor.fetchall()
                    session_ids.update(row[0] for row in rows)

                # Also get sessions from metadata table
                cursor = await conn.execute("SELECT session_id FROM session_metadata")
                rows = await cursor.fetchall()
                session_ids.update(row[0] for row in rows)

                # Get titles for all sessions
                result = []
                for session_id in session_ids:
                    cursor = await conn.execute(
                        "SELECT title FROM session_metadata WHERE session_id = ?", (session_id,)
                    )
                    row = await cursor.fetchone()
                    title = row[0] if row and row[0] else None
                    result.append((session_id, title))

                return result

        except Exception as e:
            print(f"[Session] Error listing sessions: {e}")
            return []
