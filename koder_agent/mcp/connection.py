"""Keep MCP transport/session context managers on one lifetime-owning task."""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from typing import Any, Callable


class MCPConnection:
    """Separate transport ownership from connect/cleanup callers.

    AnyIO scopes must exit on the task that entered them. Live handles,
    reconnection retirement, and cancellation cleanup legitimately run on other
    tasks; they signal and join this owner instead of unwinding its stack.
    """

    def __init__(self, context: Callable[[], AbstractAsyncContextManager[Any]]) -> None:
        self._context = context
        self._ready = asyncio.get_running_loop().create_future()
        self._stop = asyncio.Event()
        self._cancel_requested = False
        self._task = asyncio.create_task(self._run(), name="mcp-connection-owner")
        self._task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task) -> None:
        # Also settle connect if cancellation happened before _run even started.
        error = None if task.cancelled() else task.exception()
        if not self._ready.done():
            if task.cancelled():
                self._ready.cancel()
            else:
                self._ready.set_exception(error or ConnectionError("MCP owner exited"))

    async def _run(self) -> None:
        try:
            async with self._context() as connection:
                if self._ready.cancelled():
                    return
                self._ready.set_result(connection)
                await self._stop.wait()
        except BaseException as exc:
            # Context unwinding has completed before an initialization failure
            # reaches the caller. There is no transport left for it to clean.
            if not self._ready.done():
                self._ready.set_exception(exc)
            else:
                raise

    async def connect(self) -> Any:
        if self._stop.is_set():
            raise ConnectionError("MCP connection is closed")
        try:
            return await asyncio.shield(self._ready)
        except BaseException:
            self._ready.cancel()
            try:
                await self.close()
            except BaseException:
                # Preserve the entering initialization failure or cancellation.
                # A cleanup failure stays on the owner for retirement to retry.
                pass
            raise

    async def close(self) -> None:
        self._stop.set()
        if (
            (not self._ready.done() or self._ready.cancelled())
            and not self._task.done()
            and not self._cancel_requested
        ):
            # Multiple retiring/cancelled callers must not interrupt the owner's
            # finally blocks with repeated native Task.cancel() injections.
            self._cancel_requested = True
            self._task.cancel()
        cancellation = None
        while not self._task.done():
            try:
                await asyncio.shield(self._task)
            except asyncio.CancelledError as exc:
                if not self._task.cancelled():
                    cancellation = cancellation or exc
            except BaseException:
                break
        if not self._task.cancelled():
            self._task.result()
        if cancellation is not None:
            raise cancellation
