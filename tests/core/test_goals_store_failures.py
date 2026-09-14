"""Goal storage must own failed transactions and their cleanup."""

import asyncio
import sqlite3
import threading

import aiosqlite
import pytest
import pytest_asyncio

from koder_agent.core.goals import (
    GoalAccountingMode,
    GoalStatus,
    GoalStore,
    GoalUpdate,
)

SESSION = "interrupted-goal"


@pytest_asyncio.fixture
async def store(tmp_path):
    instance = GoalStore(str(tmp_path / "goals.db"))
    try:
        yield instance
    finally:
        await instance.close()


async def _mutate(store, operation):
    if operation == "replace":
        return await store.replace_goal(SESSION, "replacement", GoalStatus.ACTIVE, None)
    if operation == "insert":
        return await store.insert_goal(SESSION, "replacement", GoalStatus.ACTIVE, None)
    if operation == "delete":
        return await store.delete_goal(SESSION)
    if operation == "pause":
        return await store.pause_active_goal(SESSION)
    if operation == "usage_limit":
        return await store.usage_limit_active_goal(SESSION)
    if operation == "account":
        return await store.account_usage(SESSION, 5, 30, GoalAccountingMode.ACTIVE_ONLY)
    updates = {
        "objective": GoalUpdate(objective="changed"),
        "status": GoalUpdate(status=GoalStatus.PAUSED),
        "budget": GoalUpdate(token_budget=40),
        "status_and_budget": GoalUpdate(status=GoalStatus.PAUSED, token_budget=40),
    }
    return await store.update_goal(SESSION, updates[operation])


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "cancel"])
@pytest.mark.parametrize(
    "operation",
    [
        "replace",
        "insert",
        "delete",
        "pause",
        "usage_limit",
        "account",
        "objective",
        "status",
        "budget",
        "status_and_budget",
    ],
)
async def test_failed_write_cannot_be_committed_by_a_later_write(
    store, monkeypatch, failure, operation
):
    original = await store.replace_goal(
        SESSION,
        "original",
        GoalStatus.COMPLETE if operation == "insert" else GoalStatus.ACTIVE,
        100,
    )
    commit_started = asyncio.Event()
    release_commit = asyncio.Event()

    async def interrupted_commit():
        if failure == "error":
            raise sqlite3.OperationalError("injected commit failure")
        commit_started.set()
        await release_commit.wait()

    with monkeypatch.context() as patch:
        patch.setattr(store._conn, "commit", interrupted_commit)
        if failure == "error":
            with pytest.raises(sqlite3.OperationalError, match="injected commit failure"):
                await _mutate(store, operation)
        else:
            mutation = asyncio.create_task(_mutate(store, operation))
            try:
                await asyncio.wait_for(commit_started.wait(), timeout=2)
                mutation.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await mutation
            finally:
                release_commit.set()
                await asyncio.gather(mutation, return_exceptions=True)

    # A different successful write must not seal the interrupted goal change.
    await store.insert_goal("unrelated", "successful write", GoalStatus.ACTIVE, None)
    observer = GoalStore(store.db_path)
    try:
        assert await observer.get_goal(SESSION) == original
        assert await store.get_goal(SESSION) == original
    finally:
        await observer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["objective", "pause", "usage_limit"])
async def test_update_returns_its_own_snapshot_after_a_concurrent_replacement(
    store, monkeypatch, operation
):
    original = await store.replace_goal(SESSION, "original", GoalStatus.ACTIVE, 100)
    other_store = GoalStore(store.db_path)
    real_commit = store._conn.commit
    replacement = None

    async def commit_then_replace():
        nonlocal replacement
        await real_commit()
        replacement = await other_store.replace_goal(
            SESSION, "another writer", GoalStatus.ACTIVE, None
        )

    try:
        monkeypatch.setattr(store._conn, "commit", commit_then_replace)
        result = await _mutate(store, operation)
        assert result.goal_id == original.goal_id
        assert await store.get_goal(SESSION) == replacement
    finally:
        await other_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "cancel"])
async def test_failed_connection_setup_closes_unpublished_connection(
    tmp_path, monkeypatch, failure
):
    store = GoalStore(str(tmp_path / "setup.db"))
    real_connect = aiosqlite.connect
    opened = []
    setup_started = asyncio.Event()
    release_setup = asyncio.Event()

    async def interrupted_setup():
        if failure == "error":
            raise sqlite3.OperationalError("injected setup failure")
        setup_started.set()
        await release_setup.wait()

    def connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connection.commit = interrupted_setup
        opened.append(connection)
        return connection

    monkeypatch.setattr(aiosqlite, "connect", connect)
    task = asyncio.create_task(store.get_goal(SESSION))
    try:
        if failure == "cancel":
            await asyncio.wait_for(setup_started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(sqlite3.OperationalError, match="injected setup failure"):
                await task
        assert store._conn is None
        with pytest.raises(ValueError, match="closed|active connection"):
            await opened[0].execute("SELECT 1")
    finally:
        release_setup.set()
        await asyncio.gather(task, return_exceptions=True)
        await store.close()
        for connection in opened:
            await connection.close()


@pytest.mark.asyncio
async def test_rollback_keeps_lock_through_repeated_cancellation(store, monkeypatch):
    original = await store.replace_goal(SESSION, "original", GoalStatus.ACTIVE, 100)
    connection = store._conn
    real_rollback = connection.rollback
    rollback_started = asyncio.Event()
    release_rollback = asyncio.Event()

    async def failed_commit():
        raise sqlite3.OperationalError("injected commit failure")

    async def blocked_rollback():
        rollback_started.set()
        await release_rollback.wait()
        await real_rollback()

    monkeypatch.setattr(connection, "commit", failed_commit)
    monkeypatch.setattr(connection, "rollback", blocked_rollback)
    mutation = asyncio.create_task(_mutate(store, "account"))
    reader = None
    try:
        await asyncio.wait_for(rollback_started.wait(), timeout=2)
        mutation.cancel()
        # Give both cancellations and the competing reader a turn on the loop.
        await asyncio.sleep(0)
        mutation.cancel()
        reader = asyncio.create_task(store.get_goal(SESSION))
        await asyncio.sleep(0)
        assert not mutation.done()
        assert not reader.done()
        release_rollback.set()
        with pytest.raises(asyncio.CancelledError):
            await mutation
        assert await reader == original
        assert not connection.in_transaction
    finally:
        release_rollback.set()
        await asyncio.gather(
            *(task for task in (mutation, reader) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_failed_rollback_discards_connection_before_next_operation(store, monkeypatch):
    original = await store.replace_goal(SESSION, "original", GoalStatus.ACTIVE, 100)
    connection = store._conn

    async def fail():
        raise sqlite3.OperationalError("injected database failure")

    monkeypatch.setattr(connection, "commit", fail)
    monkeypatch.setattr(connection, "rollback", fail)
    with pytest.raises(sqlite3.OperationalError, match="injected database failure"):
        await _mutate(store, "account")

    assert store._conn is None
    with pytest.raises(ValueError, match="closed|active connection"):
        await connection.execute("SELECT 1")
    assert await store.get_goal(SESSION) == original


@pytest.mark.asyncio
async def test_cancelled_sqlite_worker_write_is_rolled_back_before_reuse(store):
    original = await store.replace_goal(SESSION, "original", GoalStatus.ACTIVE, 100)
    statement_started = asyncio.Event()
    release_statement = threading.Event()
    loop = asyncio.get_running_loop()

    def block_worker():
        loop.call_soon_threadsafe(statement_started.set)
        if not release_statement.wait(timeout=5):
            raise RuntimeError("test did not release the SQLite worker")
        return 0

    await store._conn.create_function("block_worker", 0, block_worker)
    async with store._conn.execute(
        """CREATE TRIGGER block_goal_update BEFORE UPDATE ON session_goals
           BEGIN SELECT block_worker(); END"""
    ):
        pass
    mutation = asyncio.create_task(_mutate(store, "account"))
    reader = None
    try:
        await asyncio.wait_for(statement_started.wait(), timeout=2)
        mutation.cancel()
        reader = asyncio.create_task(store.get_goal(SESSION))
        await asyncio.sleep(0)
        assert not reader.done()
        release_statement.set()
        with pytest.raises(asyncio.CancelledError):
            await mutation
        assert await reader == original
        assert not store._conn.in_transaction
    finally:
        release_statement.set()
        await asyncio.gather(
            *(task for task in (mutation, reader) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_close_finishes_before_reopening_after_repeated_cancellation(store, monkeypatch):
    original = await store.replace_goal(SESSION, "original", GoalStatus.ACTIVE, 100)
    connection = store._conn
    real_close = connection.close
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def blocked_close():
        close_started.set()
        await release_close.wait()
        await real_close()

    monkeypatch.setattr(connection, "close", blocked_close)
    closing = asyncio.create_task(store.close())
    reader = None
    try:
        await asyncio.wait_for(close_started.wait(), timeout=2)
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        reader = asyncio.create_task(store.get_goal(SESSION))
        await asyncio.sleep(0)
        assert not closing.done()
        assert not reader.done()
        release_close.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert await reader == original
        assert store._conn is not connection
    finally:
        release_close.set()
        await asyncio.gather(
            *(task for task in (closing, reader) if task is not None),
            return_exceptions=True,
        )
