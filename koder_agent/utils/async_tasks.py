"""Ownership helpers for asynchronous resource cleanup."""

import asyncio
import contextlib
from typing import Callable, ParamSpec, TypeVar

_T = TypeVar("_T")
_P = ParamSpec("_P")


async def await_owned_task(task: asyncio.Future[_T]) -> _T:
    """Join owned work through repeated cancellation, then propagate cancellation.

    Shielding alone leaves the work running when its caller exits. Resource owners
    must also wait for completion before releasing their lock or dropping state.
    """
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            break

    try:
        result = task.result()
    except BaseException:
        if cancelled:
            raise asyncio.CancelledError from None
        raise
    if cancelled:
        raise asyncio.CancelledError
    return result


async def run_sync_owned(function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
    """Run blocking work off-loop and join it before propagating cancellation.

    Cancelling a thread waiter cannot stop the thread. Retain the work until
    completion so a storage write or lock acquisition cannot become detached.
    """
    work = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError as original:
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await await_owned_task(work)
        raise original
