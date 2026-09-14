"""Standalone synchronous transcript store with legacy DB preservation.

The CLI uses EnhancedSQLiteSession and the Agents SDK tables instead. This
explicit-call helper is not the main runtime persistence or recovery backend.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .legacy_db import LegacyDB
from .models import TranscriptMessage, TranscriptSession
from .writer_lock import TranscriptWriterLock


class TranscriptStore:
    """Owns an explicit caller's synchronous transcript DB, separate from legacy data."""

    def __init__(self, runtime_db_path: str | Path, legacy_db_path: str | Path):
        self.runtime_db_path = Path(runtime_db_path)
        self._legacy_db = LegacyDB(legacy_db_path)
        self._writer_lock = TranscriptWriterLock.for_path(self.runtime_db_path)
        self._connection = sqlite3.connect(self.runtime_db_path)
        try:
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._init_schema()
        except BaseException:
            self._connection.close()
            raise

    @classmethod
    def for_test(cls, base_dir: str | Path) -> "TranscriptStore":
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        return cls(
            runtime_db_path=base / "runtime_transcripts.db",
            legacy_db_path=base / "legacy_koder.db",
        )

    def _init_schema(self) -> None:
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS runtime_sessions (
                session_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS runtime_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_id) REFERENCES runtime_sessions(session_id) ON DELETE CASCADE
            )
            """)
        self._connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_runtime_messages_session_order
            ON runtime_messages(session_id, id)
            """)
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def legacy_db(self) -> LegacyDB:
        return self._legacy_db

    def create_session(self, name: str) -> str:
        session_id = f"runtime-{uuid.uuid4().hex[:12]}"
        self._connection.execute(
            "INSERT INTO runtime_sessions(session_id, name) VALUES (?, ?)",
            (session_id, name),
        )
        self._connection.commit()
        return session_id

    def append_user_message(self, session_id: str, content: str) -> None:
        self.append_message(session_id, "user", content)

    def append_assistant_message(self, session_id: str, content: str) -> None:
        self.append_message(session_id, "assistant", content)

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata = metadata or {}
        metadata_json = json.dumps(metadata, sort_keys=True)

        async def _write():
            async with self._writer_lock.acquire():
                cursor = self._connection.cursor()
                try:
                    cursor.execute("BEGIN IMMEDIATE")
                    cursor.execute(
                        """
                        INSERT INTO runtime_messages(session_id, role, content, metadata_json)
                        VALUES (?, ?, ?, ?)
                        """,
                        (session_id, role, content, metadata_json),
                    )
                    cursor.execute(
                        """
                        UPDATE runtime_sessions
                        SET updated_at = CURRENT_TIMESTAMP
                        WHERE session_id = ?
                        """,
                        (session_id,),
                    )
                    self._connection.commit()
                except Exception:
                    self._connection.rollback()
                    raise

        import asyncio

        asyncio.run(_write())

    def read_messages(self, session_id: str) -> list[TranscriptMessage]:
        rows = self._connection.execute(
            """
            SELECT id, session_id, role, content, metadata_json, created_at
            FROM runtime_messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()
        return [
            TranscriptMessage(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def read_session(self, session_id: str) -> TranscriptSession | None:
        row = self._connection.execute(
            """
            SELECT session_id, name, created_at, updated_at
            FROM runtime_sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return TranscriptSession(
            session_id=row["session_id"],
            name=row["name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
