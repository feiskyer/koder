"""Context management for conversation history with LLM-based summarization."""

import json
import os
from typing import Dict, List, Optional, Tuple

import aiosqlite
import tiktoken

from ..utils.client import get_model_name, llm_completion
from ..utils.model_info import get_context_window_size, get_summarization_threshold


class ContextManager:
    """Manages conversation context and history with intelligent summarization."""

    def __init__(
        self,
        session_id: str = "default",
        db_path: Optional[str] = None,
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

        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback to a model-specific encoding if available,
            # otherwise use a naive character-based encoder.
            try:
                self.encoder = tiktoken.encoding_for_model("gpt-4o")
            except Exception:
                # Final fallback: approximate tokens using UTF-8 bytes length
                class _NaiveEncoder:
                    def encode(self, text: str) -> list[int]:
                        return list(text.encode("utf-8"))

                self.encoder = _NaiveEncoder()

    async def _ensure_table(self, conn: aiosqlite.Connection) -> None:
        """Ensure the context table exists with all columns."""
        await conn.execute(
            """CREATE TABLE IF NOT EXISTS ctx (
                sid TEXT PRIMARY KEY,
                msgs TEXT,
                title TEXT DEFAULT NULL
            )"""
        )
        # Migration: add title column if not exists (for existing databases)
        cursor = await conn.execute("PRAGMA table_info(ctx)")
        columns = [row[1] for row in await cursor.fetchall()]
        if "title" not in columns:
            await conn.execute("ALTER TABLE ctx ADD COLUMN title TEXT DEFAULT NULL")
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

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Accurately calculate token count for message history using tiktoken.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            Estimated total token count
        """
        total_tokens = 0

        for msg in messages:
            # Handle unexpected message formats defensively
            if not isinstance(msg, dict):
                total_tokens += len(self.encoder.encode(str(msg)))
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                total_tokens += len(self.encoder.encode(content))
            elif isinstance(content, list):
                # Handle list content (e.g., multimodal messages)
                for block in content:
                    if isinstance(block, dict):
                        total_tokens += len(
                            self.encoder.encode(json.dumps(block, ensure_ascii=False))
                        )
                    else:
                        total_tokens += len(self.encoder.encode(str(block)))
            else:
                total_tokens += len(self.encoder.encode(str(content)))

            # Metadata overhead per message (approximately 4 tokens)
            total_tokens += 4

        return total_tokens

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

    async def save(
        self, messages: List[Dict[str, str]], current_token_count: int | None = None
    ) -> None:
        """Save conversation history to database, with summarization if needed.

        Args:
            messages: List of message dictionaries to save
            current_token_count: Actual token count from API response.
                               If None (first chat), skip summarization.
        """
        try:
            # Check token count and summarize if needed
            # Skip summarization if no token count provided (first chat)
            if current_token_count is not None:
                model = get_model_name()
                threshold = get_summarization_threshold(model)
                if current_token_count > threshold:
                    messages = await self._summarize_messages(messages, current_token_count, model)

            async with aiosqlite.connect(self.db_path) as conn:
                await self._ensure_table(conn)
                # Use INSERT ON CONFLICT to preserve existing title
                await conn.execute(
                    """INSERT INTO ctx (sid, msgs) VALUES (?, ?)
                    ON CONFLICT(sid) DO UPDATE SET msgs = excluded.msgs""",
                    (self.session_id, json.dumps(messages, ensure_ascii=False)),
                )
                await conn.commit()
        except Exception as e:
            print(f"Error saving context: {e}")

    async def _summarize_messages(
        self, messages: List[Dict[str, str]], current_tokens: int, model: str
    ) -> List[Dict[str, str]]:
        """
        Summarize message history using LLM when tokens exceed threshold.

        Strategy:
        - Keep all user messages (these represent user intents)
        - Summarize assistant responses between user messages
        - Structure: system -> user1 -> summary1 -> user2 -> summary2 -> ...

        Args:
            messages: Original message list
            current_tokens: Current token count (from API or estimated)
            model: Model name for context window calculation

        Returns:
            Summarized message list
        """
        threshold = get_summarization_threshold(model)
        context_window = get_context_window_size(model)

        print(
            f"\n[Context] Token count: {current_tokens}/{context_window} (threshold: {threshold})"
        )
        print("[Context] Triggering message history summarization...")

        # Find all user message indices (skip system prompt at index 0)
        user_indices = [i for i, msg in enumerate(messages) if msg.get("role") == "user"]

        # Need at least 1 user message to perform summary
        if len(user_indices) < 1:
            print("[Context] Insufficient messages, cannot summarize")
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

            # If there are execution messages in this round, summarize them
            if execution_messages:
                summary_text = await self._create_summary(execution_messages, i + 1)
                if summary_text:
                    summary_message = {
                        "role": "assistant",
                        "content": f"[Previous Response Summary]\n\n{summary_text}",
                    }
                    new_messages.append(summary_message)
                    summary_count += 1

        # Calculate new token count
        new_tokens = self._estimate_tokens(new_messages)
        print(f"[Context] Summary completed: {current_tokens} -> {new_tokens} tokens")
        print(
            f"[Context] Structure: system + {len(user_indices)} user messages + {summary_count} summaries"
        )

        return new_messages

    async def _create_summary(self, messages: List[Dict[str, str]], round_num: int) -> str:
        """
        Create a summary for one execution round using LLM.

        Args:
            messages: List of messages to summarize (assistant responses, tool outputs, etc.)
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

            # Normalize content to string for robust handling
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
                # Tool results - include brief preview
                tool_name = msg.get("name", "unknown")
                if len(content_str) > 500:
                    content_preview = content_str[:500] + "..."
                else:
                    content_preview = content_str
                summary_content += f"Tool ({tool_name}): {content_preview}...\n"
            elif role == "system":
                # Skip system messages in execution summary
                continue
            else:
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

            # Use unified llm_completion (reuses agent's model/api_key config)
            summary_text = await llm_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant skilled at summarizing Agent execution processes concisely.",
                    },
                    {"role": "user", "content": summary_prompt},
                ]
            )

            print(f"[Context] Summary for round {round_num} generated successfully")
            return summary_text

        except Exception as e:
            print(f"[Context] Summary generation failed for round {round_num}: {e}")
            # Use truncated original content on failure
            truncated = (
                summary_content[:1000] + "..." if len(summary_content) > 1000 else summary_content
            )
            return f"[Summary generation failed - truncated content]\n{truncated}"

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

    async def get_title(self) -> Optional[str]:
        """Get the title for the current session."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await self._ensure_table(conn)
                cursor = await conn.execute(
                    "SELECT title FROM ctx WHERE sid = ?", (self.session_id,)
                )
                row = await cursor.fetchone()
                return row[0] if row and row[0] else None
        except Exception:
            return None

    async def set_title(self, title: str) -> None:
        """Set the title for the current session."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await self._ensure_table(conn)
                # Use upsert to handle race condition where session may not exist yet
                await conn.execute(
                    """INSERT INTO ctx (sid, msgs, title) VALUES (?, '[]', ?)
                    ON CONFLICT(sid) DO UPDATE SET title = excluded.title""",
                    (self.session_id, title),
                )
                await conn.commit()
        except Exception as e:
            print(f"Error setting title: {e}")

    async def get_display_name(self) -> str:
        """Get display name: 'title - YYYY-MM-DD HH:MM' or session ID if no title."""
        title = await self.get_title()
        if title:
            from ..utils.sessions import parse_session_dt

            _, dt = parse_session_dt(self.session_id)
            if dt:
                date_suffix = dt.strftime("%Y-%m-%d %H:%M")
                return f"{title} - {date_suffix}"
            return title
        return self.session_id

    async def list_sessions_with_titles(self) -> List[Tuple[str, Optional[str]]]:
        """List all sessions with their titles."""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                await self._ensure_table(conn)
                cursor = await conn.execute("SELECT sid, title FROM ctx")
                rows = await cursor.fetchall()
                return [(row[0], row[1]) for row in rows]
        except Exception:
            return []

    async def generate_title(self, user_message: str) -> str:
        """Generate a concise session title (3-6 words) from user message using LLM."""
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
