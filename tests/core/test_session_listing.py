"""Session discovery must include stored history without optional metadata."""

import json
import sqlite3
from contextlib import closing

import pytest

from koder_agent.core.session import EnhancedSQLiteSession


async def _store_history(database, session_id):
    session = EnhancedSQLiteSession(session_id, str(database))
    items = [{"role": "user", "content": "Stored conversation without a generated title."}]
    try:
        await session.add_items(items)
        assert await session.get_items() == items
    finally:
        session.close()


@pytest.mark.asyncio
async def test_listing_includes_sdk_history_without_metadata(tmp_path):
    database = tmp_path / "sessions.db"
    await _store_history(database, "untitled-history")

    assert await EnhancedSQLiteSession.list_sessions_with_titles(str(database)) == [
        ("untitled-history", None)
    ]


@pytest.mark.asyncio
async def test_listing_includes_persisted_empty_sdk_session(tmp_path):
    database = tmp_path / "sessions.db"
    session = EnhancedSQLiteSession("empty-workflow", str(database))
    try:
        await session.replace_items([])
    finally:
        session.close()
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT session_id FROM agent_sessions").fetchall() == [
            ("empty-workflow",)
        ]

    assert await EnhancedSQLiteSession.list_sessions_with_titles(str(database)) == [
        ("empty-workflow", None)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("history_table", ["items", "agent_messages"])
async def test_listing_discovers_history_only_database(tmp_path, history_table):
    database = tmp_path / "history-only.db"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            f"CREATE TABLE {history_table} (session_id TEXT NOT NULL, message_data TEXT NOT NULL)"
        )
        connection.executemany(
            f"INSERT INTO {history_table} VALUES (?, ?)",
            [("history-only", "{}"), ("history-only", "{}")],
        )
        connection.commit()

    assert await EnhancedSQLiteSession.list_sessions_with_titles(str(database)) == [
        ("history-only", None)
    ]


@pytest.mark.asyncio
async def test_listing_merges_storage_formats_and_preserves_titles(tmp_path):
    database = tmp_path / "mixed.db"
    await _store_history(database, "current-only")
    await _store_history(database, "shared' session")
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE TABLE items (session_id TEXT, message_data TEXT)")
        connection.executemany(
            "INSERT INTO items VALUES (?, ?)",
            [
                ("shared' session", json.dumps({"role": "user", "content": "legacy copy"})),
                ("legacy-only", "{}"),
                ("legacy-only", "{}"),
            ],
        )
        connection.execute(
            "CREATE TABLE session_metadata (session_id TEXT PRIMARY KEY, title TEXT)"
        )
        connection.executemany(
            "INSERT INTO session_metadata VALUES (?, ?)",
            [
                ("shared' session", "Preserved title"),
                ("metadata-only", "Created before the first message"),
                ("legacy-only", ""),
            ],
        )
        connection.commit()

    sessions = await EnhancedSQLiteSession.list_sessions_with_titles(str(database))
    assert len(sessions) == 4
    assert dict(sessions) == {
        "current-only": None,
        "shared' session": "Preserved title",
        "legacy-only": None,
        "metadata-only": "Created before the first message",
    }


@pytest.mark.asyncio
async def test_listing_empty_database_is_empty(tmp_path):
    assert await EnhancedSQLiteSession.list_sessions_with_titles(str(tmp_path / "empty.db")) == []
