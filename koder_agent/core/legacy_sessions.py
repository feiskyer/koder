"""Atomic, retryable import of the original ``ctx`` conversation table."""

import asyncio
import json
import sqlite3
from contextlib import closing

from agents import SQLiteSession

from ..utils.async_tasks import await_owned_task


class LegacySessionMigrationError(ValueError):
    """Legacy data needs repair before it can be imported safely."""


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        is not None
    )


def _migration_completed(conn: sqlite3.Connection) -> bool:
    return _table_exists(conn, "migration_status") and (
        conn.execute("SELECT 1 FROM migration_status LIMIT 1").fetchone() is not None
    )


def _legacy_messages(raw: str | None) -> list[dict]:
    try:
        messages = json.loads(raw) if raw else []
    except (ValueError, TypeError):
        raise LegacySessionMigrationError(
            "Invalid legacy transcript JSON; migration was not committed"
        ) from None
    if not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages):
        raise LegacySessionMigrationError(
            "Invalid legacy transcript shape; expected a list of message objects"
        )
    return messages


def _migrate_legacy_sessions_sync(db_path: str) -> int:
    with closing(sqlite3.connect(db_path)) as conn:
        if not _table_exists(conn, "ctx") or _migration_completed(conn):
            return 0

        # Let the installed SDK create its own schema. Keep the table names from
        # that owner, but write through one connection so data and marker commit
        # together. No temporary agent session row is created by construction.
        schema_owner = SQLiteSession("__legacy_migration_schema__", db_path)
        try:
            sessions_table = schema_owner.sessions_table
            messages_table = schema_owner.messages_table
        finally:
            schema_owner.close()

        with conn:
            conn.execute("BEGIN IMMEDIATE")
            # Another process may have completed the import during schema setup.
            if _migration_completed(conn):
                return 0
            conn.execute("""CREATE TABLE IF NOT EXISTS session_metadata (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    tag TEXT,
                    color TEXT,
                    agent TEXT,
                    cwd TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )""")
            rows = conn.execute("SELECT sid, msgs, title FROM ctx ORDER BY sid").fetchall()
            for session_id, raw, title in rows:
                if not isinstance(session_id, str) or not session_id:
                    raise LegacySessionMigrationError(
                        "Invalid legacy session identifier; migration was not committed"
                    )
                messages = _legacy_messages(raw)
                existing = [
                    json.loads(row[0])
                    for row in conn.execute(
                        f"SELECT message_data FROM {messages_table} "
                        "WHERE session_id = ? ORDER BY id",
                        (session_id,),
                    ).fetchall()
                ]
                if existing:
                    # Earlier versions could stop after appending a transcript
                    # but before recording completion. Keep newer suffixes and
                    # never append that same legacy prefix again.
                    if existing[: len(messages)] != messages:
                        raise LegacySessionMigrationError(
                            "Legacy migration conflicts with existing session history; "
                            "no histories were changed"
                        )
                else:
                    conn.execute(
                        f"INSERT OR IGNORE INTO {sessions_table} (session_id) VALUES (?)",
                        (session_id,),
                    )
                    conn.executemany(
                        f"INSERT INTO {messages_table} (session_id, message_data) VALUES (?, ?)",
                        [(session_id, json.dumps(item)) for item in messages],
                    )
                conn.execute(
                    """INSERT INTO session_metadata (session_id, title) VALUES (?, ?)
                       ON CONFLICT(session_id) DO UPDATE SET
                           title = COALESCE(session_metadata.title, excluded.title)""",
                    (session_id, title),
                )

            conn.execute("""CREATE TABLE IF NOT EXISTS migration_status (
                    migrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    migrated_sessions INTEGER
                )""")
            conn.execute(
                "INSERT INTO migration_status (migrated_sessions) VALUES (?)", (len(rows),)
            )
        return len(rows)


async def migrate_legacy_sessions(db_path: str) -> int:
    """Import legacy sessions and their completion marker in one transaction.

    Invalid or conflicting histories abort the import without changing either
    history. The original ``ctx`` table remains intact. Cancellation waits for
    the owned worker to commit or roll back; it never abandons a partial import.
    """
    return await await_owned_task(
        asyncio.create_task(asyncio.to_thread(_migrate_legacy_sessions_sync, db_path))
    )
