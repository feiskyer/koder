"""Enhanced SQLiteSession with Koder-specific features.

This module extends the official agents.SQLiteSession to add:
- Session titles with LLM-based generation
- Token estimation helpers
- Separate metadata storage for extensibility

Conversation summarization is owned by the scheduler. Storage owns append-time
tool-output truncation and the read evidence invalidated by history mutations.
"""

import asyncio
import json
import os
import sqlite3
import threading
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

import tiktoken
from agents import SQLiteSession
from agents.items import TResponseInputItem

from ..utils.client import llm_completion
from ..utils.sqlite_connections import sqlite_connection
from .legacy_sessions import migrate_legacy_sessions as migrate_legacy_sessions

if TYPE_CHECKING:
    from aiosqlite import Connection

    from ..tools.file_state import ReadFileState


_METADATA_ADDITIONAL_COLUMNS = ("cwd", "agent", "tag", "color")


async def _metadata_columns(conn: "Connection") -> set[str]:
    cursor = await conn.execute("PRAGMA table_info(session_metadata)")
    try:
        return {row[1] for row in await cursor.fetchall()}
    finally:
        await cursor.close()


async def _ensure_metadata_schema(conn: "Connection") -> None:
    """Upgrade on a fresh, caller-owned connection; close rolls back failure.

    The fast path is read-only. If migration is needed, acquire the writer
    transaction *before* rechecking columns, so another opener cannot race
    between the schema inspection and ALTER TABLE.
    """
    if set(_METADATA_ADDITIONAL_COLUMNS).issubset(await _metadata_columns(conn)):
        return
    await conn.execute("BEGIN IMMEDIATE")
    await conn.execute("""CREATE TABLE IF NOT EXISTS session_metadata (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            tag TEXT,
            color TEXT,
            agent TEXT,
            cwd TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
    columns = await _metadata_columns(conn)
    for name in _METADATA_ADDITIONAL_COLUMNS:
        if name not in columns:
            # Identifiers come only from the fixed migration list above.
            await conn.execute(f"ALTER TABLE session_metadata ADD COLUMN {name} TEXT")
    await conn.commit()


class _ReplacementCancelledError(Exception):
    """Internal signal that a cancelled async replacement rolled back."""


class _ReplacementOutcome(Enum):
    """Final transaction outcome observed by the async caller."""

    CANCELLED = "cancelled"
    COMMITTED = "committed"


class _ReplacementCoordinator:
    """Linearize async cancellation against the worker's commit decision."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancellation_requested = False
        self._commit_started = False

    def request_cancellation(self) -> bool:
        """Request rollback unless the worker already won the commit race."""
        with self._lock:
            if self._commit_started:
                return False
            self._cancellation_requested = True
            return True

    def raise_if_cancellation_requested(self) -> None:
        with self._lock:
            if self._cancellation_requested:
                raise _ReplacementCancelledError()

    def begin_commit(self) -> None:
        """Claim the operation for commit or observe a prior cancellation."""
        with self._lock:
            if self._cancellation_requested:
                raise _ReplacementCancelledError()
            self._commit_started = True


class EnhancedSQLiteSession(SQLiteSession):
    """Extended SQLiteSession with title and metadata management.

    This class wraps the official SQLiteSession and adds:
    1. Session titles stored in a separate metadata table
    2. LLM-based title generation from first user message
    3. Token estimation helpers used by the scheduler

    Conversation summarization is owned by the scheduler
    (``AutoCompactManager`` + ``llm_compact_messages``); ``add_items`` does not
    summarize. Ephemeral file-read evidence belongs to this actual Session and
    is invalidated when its stored context changes or the Session closes.
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
            summarization_threshold: Deprecated/inert. Kept only for backwards
                                   compatibility with callers that still set it
                                   (e.g. the scheduler's legacy suppression
                                   hack). It no longer triggers any behavior.
        """
        # Read-before-write evidence is ephemeral conversation state, not
        # process-global state or persisted authorization. Initialize it lazily
        # so metadata-only Session users do not import the tool registry.
        self._file_read_state: Optional["ReadFileState"] = None

        # Set up database path
        if db_path is None:
            home_dir = os.path.expanduser("~")
            koder_dir = os.path.join(home_dir, ".koder")
            os.makedirs(koder_dir, exist_ok=True)
            db_path = os.path.join(koder_dir, "koder.db")

        # Initialize base SQLiteSession
        super().__init__(session_id, db_path)

        # Inert attribute kept for backwards compatibility (see docstring).
        self.summarization_threshold = summarization_threshold
        self._title: Optional[str] = None
        self._tag: Optional[str] = None
        self._color: Optional[str] = None

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

    @property
    def file_read_state(self) -> "ReadFileState":
        """Return this live Session's read evidence, shared across its turns."""
        with self._lock:
            if self._closed:
                raise RuntimeError("SQLiteSession is closed")
            if self._file_read_state is None:
                from ..tools.file_state import ReadFileState

                self._file_read_state = ReadFileState()
            return self._file_read_state

    def _invalidate_file_read_state(self) -> None:
        if self._file_read_state is not None:
            self._file_read_state.invalidate_all()

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._invalidate_file_read_state()

    async def clear_session(self) -> None:
        # SDK mutations can finish on their worker after cancellation. Invalidate
        # before admission rather than leaving stale evidence in that case.
        self._invalidate_file_read_state()
        await super().clear_session()

    async def pop_item(self) -> TResponseInputItem | None:
        self._invalidate_file_read_state()
        return await super().pop_item()

    async def _ensure_metadata_table(self) -> None:
        """Ensure the session_metadata table exists."""
        async with sqlite_connection(self.db_path) as conn:
            await _ensure_metadata_schema(conn)

    @classmethod
    async def collect_local_stats(cls, db_path: Optional[str] = None) -> dict[str, object]:
        resolved_db_path = cls._resolve_db_path(db_path)
        session = cls(session_id="stats-probe", db_path=resolved_db_path)
        summary = {
            "total_sessions": 0,
            "total_messages": 0,
            "active_days": 0,
            "first_session_date": None,
            "last_session_date": None,
            "peak_day": None,
        }

        try:
            await session._ensure_metadata_table()
            try:
                async with sqlite_connection(resolved_db_path) as conn:
                    cursor = await conn.execute("""
                        SELECT session_id, created_at, updated_at
                        FROM session_metadata
                        ORDER BY created_at ASC, session_id ASC
                        """)
                    rows = await cursor.fetchall()
                    if rows:
                        total_sessions = len(rows)
                        created_dates = [str(row[1])[:10] for row in rows if row[1]]
                        updated_dates = [str(row[2])[:10] for row in rows if row[2]]
                        all_dates = [*created_dates, *updated_dates]
                        peak_day = None
                        if created_dates:
                            counts = Counter(created_dates)
                            peak_day = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
                                0
                            ][0]
                        summary.update(
                            {
                                "total_sessions": total_sessions,
                                "active_days": len(set(all_dates)),
                                "first_session_date": (
                                    min(created_dates) if created_dates else None
                                ),
                                "last_session_date": (
                                    max(updated_dates or created_dates)
                                    if (updated_dates or created_dates)
                                    else None
                                ),
                                "peak_day": peak_day,
                            }
                        )

                    for table_name in ("agent_messages", "items"):
                        cursor = await conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
                            (table_name,),
                        )
                        if await cursor.fetchone():
                            cursor = await conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                            row = await cursor.fetchone()
                            summary["total_messages"] = (
                                int(row[0]) if row and row[0] is not None else 0
                            )
                            break
            except Exception:
                return summary

            return summary
        finally:
            session.close()

    @staticmethod
    def _resolve_db_path(db_path: Optional[str] = None) -> str:
        if db_path is not None:
            return db_path
        home_dir = os.path.expanduser("~")
        os.makedirs(os.path.join(home_dir, ".koder"), exist_ok=True)
        return os.path.join(home_dir, ".koder", "koder.db")

    @classmethod
    async def record_session_cwd(
        cls,
        session_id: str,
        cwd: str,
        db_path: Optional[str] = None,
    ) -> None:
        resolved_db_path = cls._resolve_db_path(db_path)
        session = cls(session_id=session_id, db_path=resolved_db_path)
        try:
            await session._ensure_metadata_table()
            async with sqlite_connection(resolved_db_path) as conn:
                await conn.execute(
                    """INSERT INTO session_metadata (session_id, cwd, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(session_id) DO UPDATE SET
                        cwd = excluded.cwd,
                        updated_at = CURRENT_TIMESTAMP""",
                    (session_id, cwd),
                )
                await conn.commit()
        finally:
            session.close()

    @classmethod
    async def get_most_recent_session_for_cwd(
        cls,
        cwd: str,
        db_path: Optional[str] = None,
    ) -> Optional[str]:
        resolved_db_path = cls._resolve_db_path(db_path)
        session = cls(session_id="metadata-probe", db_path=resolved_db_path)
        try:
            await session._ensure_metadata_table()
            async with sqlite_connection(resolved_db_path) as conn:
                cursor = await conn.execute(
                    """SELECT session_id
                    FROM session_metadata
                    WHERE cwd = ?
                    ORDER BY updated_at DESC, created_at DESC, session_id DESC
                    LIMIT 1""",
                    (cwd,),
                )
                row = await cursor.fetchone()
                return row[0] if row else None
        except Exception:
            return None
        finally:
            session.close()

    @classmethod
    async def record_session_agent(
        cls,
        session_id: str,
        agent: str,
        db_path: Optional[str] = None,
    ) -> None:
        resolved_db_path = cls._resolve_db_path(db_path)
        session = cls(session_id=session_id, db_path=resolved_db_path)
        try:
            await session._ensure_metadata_table()
            async with sqlite_connection(resolved_db_path) as conn:
                await conn.execute(
                    """INSERT INTO session_metadata (session_id, agent, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(session_id) DO UPDATE SET
                        agent = excluded.agent,
                        updated_at = CURRENT_TIMESTAMP""",
                    (session_id, agent),
                )
                await conn.commit()
        finally:
            session.close()

    @classmethod
    async def get_session_agent(
        cls,
        session_id: str,
        db_path: Optional[str] = None,
    ) -> Optional[str]:
        resolved_db_path = cls._resolve_db_path(db_path)
        session = cls(session_id=session_id, db_path=resolved_db_path)
        try:
            return await session.get_agent()
        finally:
            session.close()

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """Persist items to the session storage.

        This session is a pure storage layer for *full* (LLM-based) compaction,
        which is owned entirely by the scheduler's ``AutoCompactManager`` +
        ``llm_compact_messages`` path. The legacy per-user-turn summarizer that
        previously lived here has been removed (it competed with the scheduler's
        modern path and used bare ``print()`` which corrupted the Rich Live UI).
        The ``summarization_threshold`` attribute is kept for backwards
        compatibility with callers that still set it, but it is now inert.

        ``add_items`` *is* the interception point for **micro-compaction**: the
        openai-agents Runner feeds tool outputs back into the conversation and
        then persists them here, so this is the one place we can truncate a
        single oversized tool result before it is written to disk (and re-read
        into future turns). Micro-compaction only *truncates* individual
        oversized outputs -- it never drops items or breaks
        tool_call/tool_result pairing, so item count and call ids are preserved.
        It is a no-op for small outputs, so normal operation is unchanged.

        Args:
            items: List of message dictionaries to add
        """
        if not items:
            await super().add_items(items)
            return

        compacted_items = self._apply_micro_compaction(items)
        if compacted_items != items:
            self._invalidate_file_read_state()
        await super().add_items(compacted_items)

    async def replace_items(self, items: list[TResponseInputItem]) -> None:
        """Atomically replace this session's ordered conversation items.

        The delete and inserts use the same SQLite connection inside one
        ``BEGIN IMMEDIATE`` transaction. Readers on other connections therefore
        observe either the previous committed history or the complete
        replacement, never an empty intermediate state. Existing serialized
        items retain their database timestamps, and the session metadata row is
        updated without replacing its original creation timestamp.

        Cancellation is coordinated with the worker thread at the commit
        boundary. If cancellation wins, the transaction rolls back before this
        coroutine raises. If commit wins, this coroutine retrieves the committed
        outcome and returns successfully instead of reporting cancellation.
        """
        # Replacement is also the exact-restore primitive used by scheduler
        # rollback paths.  Do not run append-time micro-compaction here: the
        # supplied ordered snapshot must be persisted byte-for-byte.
        replacement_items = list(items)
        coordinator = _ReplacementCoordinator()
        worker = asyncio.create_task(
            asyncio.to_thread(
                self._replace_items_sync,
                replacement_items,
                coordinator,
            )
        )
        while True:
            try:
                outcome = await asyncio.shield(worker)
                break
            except asyncio.CancelledError:
                coordinator.request_cancellation()

        if outcome is _ReplacementOutcome.CANCELLED:
            raise asyncio.CancelledError

        # Exact replacement reports success only once commit has won. Clear the
        # owner's evidence here so compaction, rewind and repair share a boundary.
        self._invalidate_file_read_state()

    def _replace_items_sync(
        self,
        items: list[TResponseInputItem],
        coordinator: _ReplacementCoordinator,
    ) -> _ReplacementOutcome:
        with self._replacement_connection() as conn:
            try:
                coordinator.raise_if_cancellation_requested()
                conn.execute("BEGIN IMMEDIATE")
                cursor = conn.execute(
                    f"""SELECT message_data, created_at
                    FROM {self.messages_table}
                    WHERE session_id = ?
                    ORDER BY id ASC""",
                    (self.session_id,),
                )
                preserved_timestamps: dict[str, deque[str]] = defaultdict(deque)
                for message_data, created_at in cursor.fetchall():
                    preserved_timestamps[message_data].append(created_at)

                conn.execute(
                    f"DELETE FROM {self.messages_table} WHERE session_id = ?",
                    (self.session_id,),
                )
                self._before_replace_insert(conn)
                # Preserve the legacy insertion seam used by context-preflight
                # rollback tests. Production replacement inserts remain below
                # so existing message timestamps can be restored precisely.
                self._insert_items(conn, [])
                coordinator.raise_if_cancellation_requested()

                conn.execute(
                    f"INSERT OR IGNORE INTO {self.sessions_table} (session_id) VALUES (?)",
                    (self.session_id,),
                )
                for item in items:
                    message_data = json.dumps(item)
                    timestamps = preserved_timestamps.get(message_data)
                    created_at = timestamps.popleft() if timestamps else None
                    conn.execute(
                        f"""INSERT INTO {self.messages_table}
                        (session_id, message_data, created_at)
                        VALUES (?, ?, COALESCE(?, CURRENT_TIMESTAMP))""",
                        (self.session_id, message_data, created_at),
                    )

                coordinator.raise_if_cancellation_requested()
                conn.execute(
                    f"""UPDATE {self.sessions_table}
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?""",
                    (self.session_id,),
                )
                self._before_replace_commit(conn)
                coordinator.begin_commit()
                self._commit_replace_transaction(conn)
                return _ReplacementOutcome.COMMITTED
            except _ReplacementCancelledError:
                self._rollback_replace_transaction_safely(conn)
                return _ReplacementOutcome.CANCELLED
            except BaseException:
                self._rollback_replace_transaction_safely(conn)
                raise

    @contextmanager
    def _replacement_connection(self):
        """Yield a transaction connection whose lifetime is owned by this worker.

        The SDK's normal file-backed path caches one connection in thread-local
        storage. ``replace_items`` runs through ``asyncio.to_thread``, whose
        worker can retire immediately after the call; caching there leaves an
        unclosed sqlite connection for garbage collection. Exact replacement
        still uses the SDK's shared per-database lock, but file-backed workers
        now close their dedicated connection before returning.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("SQLiteSession is closed")
            if self._is_memory_db:
                yield self._shared_connection
                return

            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                yield conn
            finally:
                conn.close()

    def _insert_items(self, conn, items: list[TResponseInputItem]) -> None:
        """Insert raw items for failure-injection compatibility.

        Exact replacement uses the timestamp-preserving loop in
        :meth:`_replace_items_sync`; this helper remains as a transaction-local
        seam so partial-insert failures can be injected and proven to roll back.
        """
        super()._insert_items(conn, items)

    def _before_replace_insert(self, _conn) -> None:
        """Test seam invoked after delete and before replacement inserts."""

    def _before_replace_commit(self, _conn) -> None:
        """Test seam invoked immediately before cancellation/commit linearization."""

    def _commit_replace_transaction(self, conn) -> None:
        """Commit seam used to exercise cancellation after commit has won."""
        conn.commit()

    def _rollback_replace_transaction(self, conn) -> None:
        """Rollback seam used to verify failures are surfaced, never hidden."""
        conn.rollback()

    def _rollback_replace_transaction_safely(self, conn) -> None:
        """Rollback and force the connection out of a failed transaction.

        The overridable rollback helper is a test seam and may itself fail. In
        that case, issue a direct connection rollback before surfacing the
        helper error so this session's thread-local connection cannot retain an
        uncommitted delete or an open failed transaction.
        """
        try:
            self._rollback_replace_transaction(conn)
        except BaseException:
            conn.rollback()
            if conn.in_transaction:
                raise RuntimeError("Forced session replacement rollback did not complete")
            raise

    def _apply_micro_compaction(self, items: list[TResponseInputItem]) -> list[TResponseInputItem]:
        """Truncate oversized tool outputs before persisting.

        Reuses ``micro_compact_messages`` (which handles both ``role=='tool'``
        content and ``type=='function_call_output'`` output shapes). Threshold
        and toggle come from environment variables via
        ``MicroCompactConfig.from_env``. This is intentionally defensive: any
        failure falls back to the original items so persistence is never broken.
        """
        try:
            from ..harness.memory.micro_compact import (
                MicroCompactConfig,
                micro_compact_messages,
            )

            config = MicroCompactConfig.from_env()
            if not config.enabled:
                return items
            return micro_compact_messages(items, config=config)
        except Exception:
            return items

    def _estimate_tokens(self, messages: List[Dict]) -> int:
        """Estimate complete stored items, including tool arguments and output.

        Args:
            messages: List of message dictionaries

        Returns:
            Estimated total token count
        """
        from ..harness.memory.budget import estimate_serialized_tokens

        return sum(estimate_serialized_tokens(message) for message in messages)

    async def get_title(self) -> Optional[str]:
        """Get the title for this session.

        Returns:
            Session title or None if not set
        """
        try:
            await self._ensure_metadata_table()
            async with sqlite_connection(self.db_path) as conn:
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
            async with sqlite_connection(self.db_path) as conn:
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

    async def get_tag(self) -> Optional[str]:
        """Get the tag for this session."""
        try:
            await self._ensure_metadata_table()
            async with sqlite_connection(self.db_path) as conn:
                cursor = await conn.execute(
                    "SELECT tag FROM session_metadata WHERE session_id = ?",
                    (self.session_id,),
                )
                row = await cursor.fetchone()
                tag = row[0] if row and row[0] else None
                self._tag = tag
                return tag
        except Exception:
            return self._tag

    async def set_tag(self, tag: Optional[str]) -> None:
        """Set or clear the tag for this session."""
        try:
            await self._ensure_metadata_table()
            normalized = tag.strip() if tag and tag.strip() else None
            async with sqlite_connection(self.db_path) as conn:
                await conn.execute(
                    """INSERT INTO session_metadata (session_id, tag, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(session_id) DO UPDATE SET
                        tag = excluded.tag,
                        updated_at = CURRENT_TIMESTAMP""",
                    (self.session_id, normalized),
                )
                await conn.commit()
            self._tag = normalized
        except Exception as e:
            print(f"[Session] Error setting tag: {e}")

    async def get_color(self) -> Optional[str]:
        """Get the display color for this session."""
        try:
            await self._ensure_metadata_table()
            async with sqlite_connection(self.db_path) as conn:
                cursor = await conn.execute(
                    "SELECT color FROM session_metadata WHERE session_id = ?",
                    (self.session_id,),
                )
                row = await cursor.fetchone()
                color = row[0] if row and row[0] else None
                self._color = color
                return color
        except Exception:
            return self._color

    async def set_color(self, color: Optional[str]) -> None:
        """Set or clear the display color for this session."""
        try:
            await self._ensure_metadata_table()
            normalized = color.strip() if color and color.strip() else None
            async with sqlite_connection(self.db_path) as conn:
                await conn.execute(
                    """INSERT INTO session_metadata (session_id, color, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(session_id) DO UPDATE SET
                        color = excluded.color,
                        updated_at = CURRENT_TIMESTAMP""",
                    (self.session_id, normalized),
                )
                await conn.commit()
            self._color = normalized
        except Exception as e:
            print(f"[Session] Error setting color: {e}")

    async def get_cwd(self) -> Optional[str]:
        """Get the recorded working directory for this session."""
        try:
            await self._ensure_metadata_table()
            async with sqlite_connection(self.db_path) as conn:
                cursor = await conn.execute(
                    "SELECT cwd FROM session_metadata WHERE session_id = ?",
                    (self.session_id,),
                )
                row = await cursor.fetchone()
                return row[0] if row and row[0] else None
        except Exception:
            return None

    async def get_agent(self) -> Optional[str]:
        """Get the agent identity recorded for this session."""
        try:
            await self._ensure_metadata_table()
            async with sqlite_connection(self.db_path) as conn:
                cursor = await conn.execute(
                    "SELECT agent FROM session_metadata WHERE session_id = ?",
                    (self.session_id,),
                )
                row = await cursor.fetchone()
                return row[0] if row and row[0] else None
        except Exception:
            return None

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
                ],
                use_small=True,
                response_reserve=128,
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
            async with sqlite_connection(db_path) as conn:
                await _ensure_metadata_schema(conn)

                # Titles/cwd are optional metadata. Discover SDK session records
                # and stored messages even when title generation never ran.
                # Keep legacy items-only databases discoverable as well.
                history_tables = ("agent_sessions", "agent_messages", "items")
                cursor = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name IN (?, ?, ?)",
                    history_tables,
                )
                available_tables = {row[0] for row in await cursor.fetchall()}
                session_ids = set()
                for table_name in history_tables:
                    if table_name not in available_tables:
                        continue
                    # The identifier comes only from the fixed tuple above.
                    cursor = await conn.execute(f"SELECT DISTINCT session_id FROM {table_name}")
                    rows = await cursor.fetchall()
                    session_ids.update(row[0] for row in rows)

                # One metadata read retains metadata-only sessions and avoids
                # issuing a separate title query for every stored session.
                cursor = await conn.execute("SELECT session_id, title FROM session_metadata")
                titles = {row[0]: row[1] or None for row in await cursor.fetchall()}
                session_ids.update(titles)

                return [(session_id, titles.get(session_id)) for session_id in sorted(session_ids)]

        except Exception as e:
            print(f"[Session] Error listing sessions: {e}")
            return []
