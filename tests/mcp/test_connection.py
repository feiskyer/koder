"""Same-task AnyIO ownership survives independent callers and cancellation."""

import asyncio
from contextlib import asynccontextmanager

import anyio
import pytest

from koder_agent.mcp.connection import MCPConnection


@pytest.mark.asyncio
async def test_connection_enters_and_exits_on_same_task_with_concurrent_cleanup():
    tasks = []

    @asynccontextmanager
    async def context():
        async with anyio.create_task_group():
            tasks.append(asyncio.current_task())
            yield "connected"
            tasks.append(asyncio.current_task())

    connection = MCPConnection(context)
    assert await connection.connect() == "connected"
    await asyncio.gather(connection.close(), connection.close())
    assert len(tasks) == 2 and tasks[0] is tasks[1]
    assert tasks[0] is not asyncio.current_task()
    with pytest.raises(ConnectionError, match="closed"):
        await connection.connect()


@pytest.mark.asyncio
async def test_repeated_caller_cancellation_joins_startup_cleanup():
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    finished = []

    @asynccontextmanager
    async def context():
        async with anyio.create_task_group():
            try:
                started.set()
                await asyncio.Event().wait()
                yield
            finally:
                cleaning.set()
                await release.wait()
                finished.append(True)

    connection = MCPConnection(context)
    caller = asyncio.create_task(connection.connect())
    await started.wait()
    caller.cancel("original cancellation")
    await cleaning.wait()
    caller.cancel("repeated cancellation")
    closer = asyncio.create_task(connection.close())
    await asyncio.sleep(0)
    assert not caller.done() and not closer.done()
    release.set()
    with pytest.raises(asyncio.CancelledError, match="original cancellation"):
        await caller
    await closer
    assert finished == [True]


@pytest.mark.asyncio
async def test_initialization_error_is_propagated_after_cleanup():
    failure = ValueError("handshake rejected")
    finished = []

    @asynccontextmanager
    async def context():
        try:
            raise failure
            yield
        finally:
            finished.append(True)

    connection = MCPConnection(context)
    with pytest.raises(ValueError) as raised:
        await connection.connect()
    assert raised.value is failure
    assert finished == [True]
    await connection.close()


@pytest.mark.asyncio
async def test_cleanup_before_owner_starts_settles_connect():
    @asynccontextmanager
    async def context():
        raise AssertionError("must not open resources after retirement")
        yield

    connection = MCPConnection(context)
    await connection.close()
    with pytest.raises(ConnectionError, match="closed"):
        await connection.connect()
