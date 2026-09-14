"""Cancellation while SQLite opens must not abandon its connection worker."""

import asyncio
import sqlite3
import threading

import pytest

from koder_agent.core import goals


@pytest.mark.asyncio
async def test_cancelled_goal_connection_waits_for_open_and_close(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    actual_connect = goals.aiosqlite.connect
    connections = []

    def slow_connect(path):
        connection = actual_connect(path)
        connector = connection._connector

        def open_database():
            entered.set()
            if not release.wait(5):
                raise TimeoutError("synthetic connector was not released")
            return connector()

        connection._connector = open_database
        connections.append(connection)
        return connection

    monkeypatch.setattr(goals.aiosqlite, "connect", slow_connect)
    store = goals.GoalStore(str(tmp_path / "goals.db"))
    task = asyncio.create_task(store.get_goal("cancelled-open"))
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        task.cancel()
        for _ in range(3):
            await asyncio.sleep(0)
        assert not task.done(), "connection cleanup was detached from the cancelled owner"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert store._conn is None
        assert not connections[0]._thread.is_alive()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await store.close()
        for connection in connections:
            await asyncio.to_thread(connection._thread.join, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_failed_open_waits_for_worker_shutdown(tmp_path, monkeypatch, cancelled):
    opening, release_open = threading.Event(), threading.Event()
    stopping, release_stop = threading.Event(), threading.Event()
    actual_connect = goals.aiosqlite.connect
    connections = []

    def broken_connect(path):
        connection = actual_connect(path)
        original_stop = connection.stop

        def fail_open():
            opening.set()
            if not release_open.wait(5):
                raise TimeoutError("synthetic open was not released")
            raise sqlite3.OperationalError("synthetic database open failure")

        def hold_shutdown():
            stopping.set()
            if not release_stop.wait(5):
                raise TimeoutError("synthetic shutdown was not released")

        def stop():
            # Keep the real worker alive before its real stop marker. No database
            # or connection mock can establish whether that worker was joined.
            connection._tx.put_nowait((None, hold_shutdown))
            return original_stop()

        connection._connector = fail_open
        connection.stop = stop
        connections.append(connection)
        return connection

    monkeypatch.setattr(goals.aiosqlite, "connect", broken_connect)
    store = goals.GoalStore(str(tmp_path / "failed.db"))
    task = asyncio.create_task(store.get_goal("failed-open"))
    try:
        assert await asyncio.to_thread(opening.wait, 3)
        if cancelled:
            task.cancel()
        release_open.set()
        assert await asyncio.to_thread(stopping.wait, 3)
        for _ in range(3):
            await asyncio.sleep(0)
        assert not task.done(), "failed connect returned before its worker stopped"
        release_stop.set()
        if cancelled:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(sqlite3.OperationalError, match="synthetic database open failure"):
                await task
        assert store._conn is None
        assert not connections[0]._thread.is_alive()
    finally:
        release_open.set()
        release_stop.set()
        await asyncio.gather(task, return_exceptions=True)
        await store.close()
        for connection in connections:
            await asyncio.to_thread(connection._thread.join, 3)


@pytest.mark.asyncio
async def test_repeated_cancellation_keeps_new_reader_behind_connection_cleanup(
    tmp_path, monkeypatch
):
    opening, release_open = threading.Event(), threading.Event()
    closing, release_close = asyncio.Event(), asyncio.Event()
    actual_connect = goals.aiosqlite.connect
    connections = []

    def slow_first_connect(path):
        connection = actual_connect(path)
        if not connections:
            original_connector = connection._connector
            original_close = connection.close

            def open_database():
                opening.set()
                if not release_open.wait(5):
                    raise TimeoutError("synthetic connector was not released")
                return original_connector()

            async def close_database():
                closing.set()
                await release_close.wait()
                await original_close()

            connection._connector = open_database
            connection.close = close_database
        connections.append(connection)
        return connection

    monkeypatch.setattr(goals.aiosqlite, "connect", slow_first_connect)
    store = goals.GoalStore(str(tmp_path / "readers.db"))
    task = asyncio.create_task(store.get_goal("cancelled"))
    reader = None
    try:
        assert await asyncio.to_thread(opening.wait, 3)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release_open.set()
        await asyncio.wait_for(closing.wait(), 3)
        reader = asyncio.create_task(store.get_goal("next-reader"))
        task.cancel()
        for _ in range(3):
            await asyncio.sleep(0)
        assert not task.done()
        assert not reader.done()
        assert len(connections) == 1
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await reader is None
        assert len(connections) == 2
        assert store._conn is connections[1]
        assert not connections[0]._thread.is_alive()
    finally:
        release_open.set()
        release_close.set()
        await asyncio.gather(
            *(pending for pending in (task, reader) if pending is not None),
            return_exceptions=True,
        )
        await store.close()
        for connection in connections:
            await asyncio.to_thread(connection._thread.join, 3)
