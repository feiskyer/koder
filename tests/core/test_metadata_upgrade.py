"""Concurrent metadata users must share one transactional schema migration."""

import asyncio
import sqlite3
from contextlib import asynccontextmanager, closing

import pytest

from koder_agent.core import session as session_module
from koder_agent.core.session import EnhancedSQLiteSession


def create_legacy(database):
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("""CREATE TABLE session_metadata (
                session_id TEXT PRIMARY KEY, title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )""")
        connection.commit()


@pytest.mark.asyncio
async def test_concurrent_legacy_metadata_migration_keeps_all_cwd_writes(tmp_path):
    database = tmp_path / "legacy.db"
    create_legacy(database)
    count = 16
    outcomes = await asyncio.gather(
        *(
            EnhancedSQLiteSession.record_session_cwd(
                f"session-{index}", str(tmp_path / f"cwd-{index}"), db_path=str(database)
            )
            for index in range(count)
        ),
        return_exceptions=True,
    )
    assert not any(isinstance(outcome, BaseException) for outcome in outcomes), outcomes
    with closing(sqlite3.connect(database)) as connection:
        rows = connection.execute("SELECT session_id, cwd FROM session_metadata").fetchall()
    assert dict(rows) == {
        f"session-{index}": str(tmp_path / f"cwd-{index}") for index in range(count)
    }


@pytest.mark.asyncio
async def test_session_listing_uses_the_same_complete_schema(tmp_path):
    database = tmp_path / "legacy.db"
    create_legacy(database)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            "INSERT INTO session_metadata (session_id, title) VALUES (?, ?)",
            ("legacy-session", "keep this title"),
        )
        connection.commit()
    assert await EnhancedSQLiteSession.list_sessions_with_titles(str(database)) == [
        ("legacy-session", "keep this title")
    ]
    with closing(sqlite3.connect(database)) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(session_metadata)")}
    assert {"agent", "cwd", "tag", "color"}.issubset(columns)


@pytest.mark.asyncio
async def test_cancelled_schema_upgrade_rolls_back_before_next_writer(tmp_path, monkeypatch):
    database = tmp_path / "legacy.db"
    create_legacy(database)
    original_connect = session_module.sqlite_connection
    altered = asyncio.Event()
    release = asyncio.Event()

    @asynccontextmanager
    async def paused_connect(*args, **kwargs):
        async with original_connect(*args, **kwargs) as conn:
            execute = conn.execute

            async def paused_execute(sql, *parameters, **options):
                cursor = await execute(sql, *parameters, **options)
                if sql.startswith("ALTER TABLE session_metadata"):
                    altered.set()
                    try:
                        await release.wait()
                    except BaseException:
                        await cursor.close()
                        raise
                return cursor

            monkeypatch.setattr(conn, "execute", paused_execute)
            yield conn

    monkeypatch.setattr(session_module, "sqlite_connection", paused_connect)
    task = asyncio.create_task(
        EnhancedSQLiteSession.record_session_cwd("cancelled", "/synthetic", str(database))
    )
    try:
        await asyncio.wait_for(altered.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    with closing(sqlite3.connect(database)) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(session_metadata)")}
        assert "cwd" not in columns
        assert connection.execute("SELECT COUNT(*) FROM session_metadata").fetchone() == (0,)
    await EnhancedSQLiteSession.record_session_cwd("next", "/synthetic", str(database))
    with closing(sqlite3.connect(database)) as connection:
        assert connection.execute("SELECT session_id, cwd FROM session_metadata").fetchall() == [
            ("next", "/synthetic")
        ]
