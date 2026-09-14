"""Legacy history imports must be atomic, retryable and non-destructive."""

import asyncio
import json
import sqlite3
import threading
from contextlib import closing

import pytest
from agents import SQLiteSession

from koder_agent.core import legacy_sessions
from koder_agent.core.session import migrate_legacy_sessions

LEGACY = [{"role": "user", "content": "legacy request"}]


def _seed(path, records):
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE ctx (sid TEXT PRIMARY KEY, msgs TEXT, title TEXT)")
        conn.executemany("INSERT INTO ctx VALUES (?, ?, ?)", records)


def _messages(path, session_id):
    with closing(sqlite3.connect(path)) as conn, conn:
        return [
            json.loads(row[0])
            for row in conn.execute(
                "SELECT message_data FROM agent_messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        ]


def _completed(path):
    with closing(sqlite3.connect(path)) as conn, conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='migration_status'"
        ).fetchone()
        return bool(exists and conn.execute("SELECT 1 FROM migration_status").fetchone())


@pytest.mark.asyncio
async def test_migration_rolls_back_all_sessions_when_a_later_write_fails(tmp_path):
    path = tmp_path / "sessions.db"
    _seed(path, [("first", json.dumps(LEGACY), "first"), ("second", json.dumps(LEGACY), "second")])
    schema = SQLiteSession("schema", path)
    schema.close()
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("""CREATE TRIGGER fail_import BEFORE INSERT ON agent_messages
               WHEN NEW.session_id = 'second'
               BEGIN SELECT RAISE(ABORT, 'injected migration failure'); END""")

    with pytest.raises(sqlite3.DatabaseError, match="injected migration failure"):
        await migrate_legacy_sessions(str(path))
    assert _messages(path, "first") == []
    assert _messages(path, "second") == []
    assert not _completed(path)

    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("DROP TRIGGER fail_import")
    assert await migrate_legacy_sessions(str(path)) == 2
    assert _messages(path, "first") == LEGACY
    assert _messages(path, "second") == LEGACY
    assert _completed(path)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["{broken", "{}", "[1]", '"not a transcript"'])
async def test_invalid_legacy_payload_does_not_mark_migration_complete(tmp_path, invalid):
    path = tmp_path / "sessions.db"
    _seed(path, [("first", json.dumps(LEGACY), "first"), ("bad", invalid, "bad")])
    with pytest.raises(ValueError, match="legacy|Legacy"):
        await migrate_legacy_sessions(str(path))
    assert not _completed(path)
    with closing(sqlite3.connect(path)) as conn, conn:
        assert conn.execute("SELECT msgs FROM ctx WHERE sid='bad'").fetchone()[0] == invalid


@pytest.mark.asyncio
async def test_retry_preserves_already_imported_history_and_newer_title(tmp_path):
    path = tmp_path / "sessions.db"
    _seed(path, [("legacy", json.dumps(LEGACY), "old title")])
    existing = SQLiteSession("legacy", path)
    newer = {"role": "assistant", "content": "newer reply"}
    try:
        # Simulate old import code stopping after messages, before its marker.
        await existing.add_items([*LEGACY, newer])
    finally:
        existing.close()
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE session_metadata (session_id TEXT PRIMARY KEY, title TEXT)")
        conn.execute("INSERT INTO session_metadata VALUES ('legacy', 'renamed by user')")

    assert await migrate_legacy_sessions(str(path)) == 1
    assert await migrate_legacy_sessions(str(path)) == 0
    assert _messages(path, "legacy") == [*LEGACY, newer]
    with closing(sqlite3.connect(path)) as conn, conn:
        assert conn.execute("SELECT title FROM session_metadata").fetchone()[0] == "renamed by user"
        assert conn.execute("SELECT COUNT(*) FROM ctx").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_conflicting_existing_history_is_not_overwritten_or_appended(tmp_path):
    path = tmp_path / "sessions.db"
    _seed(path, [("legacy", json.dumps(LEGACY), "old title")])
    current = [{"role": "user", "content": "different modern history"}]
    existing = SQLiteSession("legacy", path)
    try:
        await existing.add_items(current)
    finally:
        existing.close()

    with pytest.raises(ValueError, match="conflict|Conflict"):
        await migrate_legacy_sessions(str(path))
    assert _messages(path, "legacy") == current
    assert not _completed(path)


@pytest.mark.asyncio
async def test_empty_old_completion_table_is_retried(tmp_path):
    path = tmp_path / "sessions.db"
    _seed(path, [("legacy", json.dumps(LEGACY), "old title")])
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE migration_status (migrated_sessions INTEGER)")

    assert await migrate_legacy_sessions(str(path)) == 1
    assert _messages(path, "legacy") == LEGACY
    assert _completed(path)


@pytest.mark.asyncio
async def test_concurrent_importers_do_not_duplicate_history(tmp_path):
    path = tmp_path / "sessions.db"
    _seed(path, [("legacy", json.dumps(LEGACY), "old title")])
    results = await asyncio.gather(
        migrate_legacy_sessions(str(path)),
        migrate_legacy_sessions(str(path)),
    )
    assert sorted(results) == [0, 1]
    assert _messages(path, "legacy") == LEGACY


@pytest.mark.asyncio
async def test_migration_releases_its_sdk_schema_owner(tmp_path):
    path = tmp_path / "sessions.db"
    _seed(path, [("legacy", json.dumps(LEGACY), "old title")])
    await migrate_legacy_sessions(str(path))
    assert path.resolve() not in SQLiteSession._file_lock_counts


@pytest.mark.asyncio
async def test_database_without_legacy_rows_needs_no_import(tmp_path):
    path = tmp_path / "sessions.db"
    assert await migrate_legacy_sessions(str(path)) == 0
    assert not _completed(path)


@pytest.mark.asyncio
async def test_empty_legacy_session_is_preserved(tmp_path):
    path = tmp_path / "sessions.db"
    _seed(path, [("empty", None, "empty conversation")])
    assert await migrate_legacy_sessions(str(path)) == 1
    with closing(sqlite3.connect(path)) as conn, conn:
        assert conn.execute("SELECT session_id FROM agent_sessions").fetchall() == [("empty",)]
    assert _messages(path, "empty") == []


@pytest.mark.asyncio
async def test_cancellation_joins_the_migration_worker_before_returning(tmp_path, monkeypatch):
    path = tmp_path / "sessions.db"
    _seed(path, [("legacy", json.dumps(LEGACY), "old title")])
    real_connect = sqlite3.connect
    write_started = asyncio.Event()
    release_write = threading.Event()
    loop = asyncio.get_running_loop()

    class BlockingConnection(sqlite3.Connection):
        def executemany(self, sql, parameters):
            cursor = super().executemany(sql, parameters)
            if "INSERT INTO agent_messages" in sql:
                loop.call_soon_threadsafe(write_started.set)
                if not release_write.wait(timeout=5):
                    raise RuntimeError("test did not release migration worker")
            return cursor

    def connect(*args, **kwargs):
        return real_connect(*args, **kwargs, factory=BlockingConnection)

    monkeypatch.setattr(legacy_sessions.sqlite3, "connect", connect)
    migration = asyncio.create_task(migrate_legacy_sessions(str(path)))
    try:
        await asyncio.wait_for(write_started.wait(), timeout=2)
        migration.cancel()
        await asyncio.sleep(0)
        migration.cancel()
        await asyncio.sleep(0)
        assert not migration.done()
        release_write.set()
        with pytest.raises(asyncio.CancelledError):
            await migration
        assert _messages(path, "legacy") == LEGACY
        assert _completed(path)
    finally:
        release_write.set()
        await asyncio.gather(migration, return_exceptions=True)
