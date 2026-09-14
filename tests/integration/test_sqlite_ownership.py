"""Metadata callers must own SQLite workers through opening and shutdown."""

import asyncio
import json
import sqlite3
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import pytest

from koder_agent.core.session import EnhancedSQLiteSession
from koder_agent.mcp import server_manager


async def _eventually(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


def _lookup_task(operation, database, monkeypatch):
    if operation == "sessions":
        return asyncio.create_task(EnhancedSQLiteSession.list_sessions_with_titles(str(database)))
    monkeypatch.setattr(server_manager, "get_config_manager", lambda: SimpleNamespace())
    manager = server_manager.MCPServerManager()
    monkeypatch.setattr(manager, "_legacy_db_path", lambda: Path(database))
    return asyncio.create_task(manager._ensure_legacy_migration())


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["sessions", "legacy-mcp"])
@pytest.mark.parametrize("cancel_cleanup", [False, True], ids=["error", "cancel-cleanup"])
async def test_failed_metadata_connect_waits_for_its_worker(
    tmp_path, monkeypatch, operation, cancel_cleanup
):
    entered = threading.Event()
    release = threading.Event()
    connections = []
    connect = aiosqlite.connect

    def controlled_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        stop = connection.stop

        def hold_worker():
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test did not release SQLite shutdown")

        def controlled_stop():
            connection._tx.put_nowait((None, hold_worker))
            return stop()

        connection.stop = controlled_stop
        connections.append(connection)
        return connection

    monkeypatch.setattr(aiosqlite, "connect", controlled_connect)
    # An existing directory passes legacy migration's existence preflight,
    # but SQLite cannot open it as a database.
    task = _lookup_task(operation, tmp_path, monkeypatch)
    try:
        await _eventually(entered.is_set)
        if cancel_cleanup:
            task.cancel("cancel-failed-open-cleanup")
            await asyncio.sleep(0)
            task.cancel("cancel-failed-open-cleanup-again")
            await asyncio.sleep(0)
        assert not task.done(), "metadata lookup returned while its failed-open worker was alive"
        release.set()
        if cancel_cleanup:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert await task == ([] if operation == "sessions" else None)
        assert all(not connection._thread.is_alive() for connection in connections)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await _eventually(lambda: all(not conn._thread.is_alive() for conn in connections))


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["sessions", "legacy-mcp"])
async def test_cancelled_metadata_open_keeps_ownership_until_close(
    tmp_path, monkeypatch, operation
):
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    connections = []
    raw_connections = []
    connect = aiosqlite.connect

    class ObservedConnection(sqlite3.Connection):
        def close(self):
            super().close()
            closed.set()

    def controlled_connect(*args, **kwargs):
        # Cross-thread access is only for cleanup on the broken baseline.
        # Application queries and close still run on the real SQLite worker.
        connection = connect(*args, **kwargs, factory=ObservedConnection, check_same_thread=False)
        connector = connection._connector

        def controlled_connector():
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test did not release SQLite opening")
            raw = connector()
            raw_connections.append(raw)
            return raw

        connection._connector = controlled_connector
        connections.append(connection)
        return connection

    monkeypatch.setattr(aiosqlite, "connect", controlled_connect)
    database = tmp_path / "sessions.sqlite"
    database.touch()
    task = _lookup_task(operation, database, monkeypatch)
    try:
        await _eventually(entered.is_set)
        task.cancel("cancel-metadata")
        await asyncio.sleep(0)
        task.cancel("cancel-metadata-again")
        await asyncio.sleep(0)
        assert not task.done(), "metadata cancellation detached its opening database worker"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set(), "the opened SQLite connection was not closed"
        assert all(not connection._thread.is_alive() for connection in connections)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await _eventually(lambda: all(not conn._thread.is_alive() for conn in connections))
        for raw in raw_connections:
            raw.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["sessions", "legacy-mcp"])
async def test_metadata_close_survives_repeated_cancellation(tmp_path, monkeypatch, operation):
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    connections = []
    connect = aiosqlite.connect

    class SlowClosingConnection(sqlite3.Connection):
        def close(self):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test did not release SQLite close")
            super().close()
            closed.set()

    def controlled_connect(*args, **kwargs):
        connection = connect(*args, **kwargs, factory=SlowClosingConnection)
        connections.append(connection)
        return connection

    monkeypatch.setattr(aiosqlite, "connect", controlled_connect)
    database = tmp_path / "sessions.sqlite"
    database.touch()
    task = _lookup_task(operation, database, monkeypatch)
    try:
        await _eventually(entered.is_set)
        task.cancel("cancel-metadata-close")
        await asyncio.sleep(0)
        task.cancel("cancel-metadata-close-again")
        await asyncio.sleep(0)
        assert not task.done(), "metadata cancellation abandoned its closing SQLite worker"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
        assert all(not connection._thread.is_alive() for connection in connections)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await _eventually(lambda: all(not conn._thread.is_alive() for conn in connections))


@pytest.mark.asyncio
async def test_owned_session_metadata_preserves_roundtrip_results(tmp_path, monkeypatch):
    connections = []
    connect = aiosqlite.connect

    def observed_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(aiosqlite, "connect", observed_connect)
    database = str(tmp_path / "sessions.sqlite")
    session = EnhancedSQLiteSession("owned-metadata", database)
    try:
        await session.set_title("Owned metadata")
        await session.set_tag("review")
        await session.set_color("blue")
        await EnhancedSQLiteSession.record_session_cwd("owned-metadata", str(tmp_path), database)
        await EnhancedSQLiteSession.record_session_agent("owned-metadata", "worker", database)
        assert await session.get_title() == "Owned metadata"
        assert await session.get_tag() == "review"
        assert await session.get_color() == "blue"
        assert await session.get_cwd() == str(tmp_path)
        assert await session.get_agent() == "worker"
        assert (
            await EnhancedSQLiteSession.get_most_recent_session_for_cwd(str(tmp_path), database)
            == "owned-metadata"
        )
        assert await EnhancedSQLiteSession.get_session_agent("owned-metadata", database) == "worker"
        assert await EnhancedSQLiteSession.list_sessions_with_titles(database) == [
            ("owned-metadata", "Owned metadata")
        ]
        summary = await EnhancedSQLiteSession.collect_local_stats(database)
        assert summary["total_sessions"] == 1
        assert summary["total_messages"] == 0
        assert connections
        assert all(not connection._thread.is_alive() for connection in connections)
    finally:
        session.close()
        for connection in connections:
            await connection.close()
        await _eventually(lambda: all(not conn._thread.is_alive() for conn in connections))


def test_finished_metadata_lookup_cannot_callback_after_loop_close(tmp_path):
    """Closing a finished caller's loop must not crash an escaped SQLite worker."""
    script = textwrap.dedent("""
        import asyncio
        import json
        import sys
        import threading
        import aiosqlite
        from koder_agent.core.session import EnhancedSQLiteSession

        release = threading.Event()
        stopping = threading.Event()
        connections = []
        errors = []
        connect = aiosqlite.connect

        def exception_hook(args):
            errors.append(f"{type(args.exc_value).__name__}: {args.exc_value}")

        threading.excepthook = exception_hook

        def controlled_connect(*args, **kwargs):
            connection = connect(*args, **kwargs)
            original_stop = connection.stop
            def hold_shutdown():
                stopping.set()
                if not release.wait(10):
                    raise TimeoutError("test did not release SQLite shutdown")
            def stop():
                connection._tx.put_nowait((None, hold_shutdown))
                return original_stop()
            connection.stop = stop
            connections.append(connection)
            return connection

        aiosqlite.connect = controlled_connect

        async def start_lookup():
            task = asyncio.create_task(
                EnhancedSQLiteSession.list_sessions_with_titles(sys.argv[1])
            )
            async with asyncio.timeout(5):
                while not stopping.is_set():
                    await asyncio.sleep(0.001)
            return task

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            task = loop.run_until_complete(start_lookup())
            if not task.done():
                release.set()
                loop.run_until_complete(asyncio.wait_for(task, 5))
            # The public lookup has finished, so its caller can close the loop.
            loop.close()
        finally:
            release.set()
            if not loop.is_closed():
                loop.close()
            for connection in connections:
                connection._thread.join(5)
        print(json.dumps({
            "closed_loop_errors": errors,
            "workers_exited": all(not c._thread.is_alive() for c in connections),
        }))
        """)
    result = subprocess.run(
        [
            "uv",
            "run",
            "--quiet",
            "--no-project",
            "--no-env-file",
            "--python",
            sys.executable,
            "python",
            "-c",
            script,
            str(tmp_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["workers_exited"], report
    assert report["closed_loop_errors"] == [], report
