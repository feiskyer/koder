"""Owned lifetimes for aiosqlite connections and their worker threads."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

from .async_tasks import await_owned_task


async def close_sqlite_connection(connection: aiosqlite.Connection) -> None:
    """Close a connection and join its worker while the event loop still exists."""
    try:
        await connection.close()
    finally:
        # aiosqlite 0.22.1 queues stop on failed connect, but close() then
        # returns early without joining it. Even a successful stop callback
        # can wake the caller just before the worker actually exits.
        # Poll exit without starting an additional executor thread.
        while connection._thread.is_alive():
            await asyncio.sleep(0.001)


@asynccontextmanager
async def sqlite_connection(
    database: str | Path, **kwargs: Any
) -> AsyncIterator[aiosqlite.Connection]:
    """Own opening, queued work and shutdown through repeated cancellation.

    Cancelling the connection's awaitable directly can lose a database that its
    worker is still opening. Retain the opening task before the first await,
    then close and join it on every exit, including failed opening. Transaction
    commit behavior remains the caller's responsibility.
    """
    connection = aiosqlite.connect(database, **kwargs)
    try:
        await await_owned_task(asyncio.ensure_future(connection))
        yield connection
    finally:
        await await_owned_task(asyncio.create_task(close_sqlite_connection(connection)))
