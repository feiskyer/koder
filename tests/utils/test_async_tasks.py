"""Cancellation cannot detach cleanup from its resource owner."""

import asyncio
import threading
from contextvars import ContextVar

import pytest

from koder_agent.utils.async_tasks import await_owned_task, run_sync_owned


@pytest.mark.asyncio
async def test_owned_task_returns_result():
    async def work():
        return 42

    assert await await_owned_task(asyncio.create_task(work())) == 42


@pytest.mark.asyncio
async def test_owned_task_propagates_failure():
    async def work():
        raise ValueError("cleanup failed")

    with pytest.raises(ValueError, match="cleanup failed"):
        await await_owned_task(asyncio.create_task(work()))


@pytest.mark.asyncio
async def test_owned_task_propagates_child_cancellation():
    async def work():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await await_owned_task(asyncio.create_task(work()))


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_caller_cancellation_waits_for_work_even_when_work_fails(fail):
    started = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()

    async def work():
        started.set()
        await release.wait()
        completed.set()
        if fail:
            raise ValueError("cleanup failed")
        return 42

    child = asyncio.create_task(work())
    owner = asyncio.create_task(await_owned_task(child))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        owner.cancel()
        await asyncio.sleep(0)
        owner.cancel()
        await asyncio.sleep(0)
        assert not owner.done()
        assert not child.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert completed.is_set()
    finally:
        release.set()
        await asyncio.gather(owner, child, return_exceptions=True)


@pytest.mark.asyncio
async def test_sync_work_preserves_arguments_and_context_without_blocking_loop(
    event_loop_progress_probe,
):
    context = ContextVar("sync-work-fixture", default="outside")
    reset = context.set("request")
    loop_thread = threading.get_ident()
    wrap, progress = event_loop_progress_probe

    def work(value, *, extra):
        assert threading.get_ident() != loop_thread
        assert context.get() == "request"
        context.set("worker-only")
        return value + extra

    try:
        assert await run_sync_owned(wrap(work), 40, extra=2) == 42
        assert progress == [True]
        assert context.get() == "request"
    finally:
        context.reset(reset)


@pytest.mark.asyncio
async def test_sync_work_forwards_original_failure():
    failure = OSError("synthetic storage failure")

    def work():
        raise failure

    with pytest.raises(OSError) as caught:
        await run_sync_owned(work)
    assert caught.value is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_sync_work_joins_repeated_cancellation_before_returning(fail, cancellation_observer):
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    completed = threading.Event()
    observe, cancellations = cancellation_observer

    def work():
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5), "test did not release synthetic I/O"
        completed.set()
        if fail:
            raise OSError("synthetic storage failure")
        return 42

    owner = asyncio.create_task(observe(run_sync_owned(work)))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        owner.cancel("first cancellation")
        await asyncio.sleep(0)
        owner.cancel("second cancellation")
        await asyncio.sleep(0)
        assert not owner.done()
        assert not completed.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert completed.is_set()
        assert [error.args for error in cancellations] == [("first cancellation",)]
    finally:
        release.set()
        await asyncio.gather(owner, return_exceptions=True)
