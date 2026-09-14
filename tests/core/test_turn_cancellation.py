"""Cancellation owns and joins both provider work and its signal waiter."""

import asyncio
import inspect

import pytest

from koder_agent.core.turn_cancellation import (
    TurnCancellationScope,
    await_with_turn_cancellation,
    reset_turn_cancellation_scope,
    set_turn_cancellation_scope,
)


async def _drain_new_tasks(baseline):
    pending = [task for task in asyncio.all_tasks() - baseline if not task.done()]
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_owner", [False, True])
async def test_cancel_joins_provider_and_signal_waiter_before_return(cancel_owner):
    baseline = asyncio.all_tasks()
    started = asyncio.Event()
    closed = asyncio.Event()
    scope = TurnCancellationScope()
    token = set_turn_cancellation_scope(scope)

    async def provider():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    try:
        owner = asyncio.create_task(await_with_turn_cancellation(provider()))
        await asyncio.wait_for(started.wait(), timeout=2)
        if cancel_owner:
            owner.cancel("owner stopped")
        else:
            scope.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert closed.is_set(), "provider outlived its owner"
        assert not [task for task in asyncio.all_tasks() - baseline if not task.done()]
    finally:
        reset_turn_cancellation_scope(token)
        await _drain_new_tasks(baseline)


@pytest.mark.asyncio
async def test_pre_cancelled_scope_closes_unstarted_coroutine():
    scope = TurnCancellationScope()
    scope.cancel()
    token = set_turn_cancellation_scope(scope)
    started = False

    async def provider():
        nonlocal started
        started = True
        return "unexpected"

    coroutine = provider()
    try:
        with pytest.raises(asyncio.CancelledError):
            await await_with_turn_cancellation(coroutine)
        assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
        assert not started
    finally:
        coroutine.close()
        reset_turn_cancellation_scope(token)


@pytest.mark.asyncio
async def test_pre_cancelled_scope_joins_already_started_task():
    baseline = asyncio.all_tasks()
    started = asyncio.Event()
    closed = asyncio.Event()
    scope = TurnCancellationScope()
    token = set_turn_cancellation_scope(scope)

    async def provider():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    try:
        work = asyncio.create_task(provider())
        await asyncio.wait_for(started.wait(), timeout=2)
        scope.cancel()
        with pytest.raises(asyncio.CancelledError):
            await await_with_turn_cancellation(work)
        assert work.done()
        assert closed.is_set()
    finally:
        reset_turn_cancellation_scope(token)
        await _drain_new_tasks(baseline)


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_success_or_failure_does_not_leave_signal_waiter(fails):
    baseline = asyncio.all_tasks()
    scope = TurnCancellationScope()
    token = set_turn_cancellation_scope(scope)
    error = RuntimeError("synthetic provider failure")

    async def provider():
        if fails:
            raise error
        return "done"

    try:
        if fails:
            with pytest.raises(RuntimeError) as raised:
                await await_with_turn_cancellation(provider())
            assert raised.value is error
        else:
            assert await await_with_turn_cancellation(provider()) == "done"
        assert not [task for task in asyncio.all_tasks() - baseline if not task.done()]
    finally:
        reset_turn_cancellation_scope(token)
        await _drain_new_tasks(baseline)


@pytest.mark.asyncio
async def test_repeated_owner_cancel_waits_for_provider_cleanup(cancellation_observer):
    observe, cancellations = cancellation_observer
    baseline = asyncio.all_tasks()
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    closed = asyncio.Event()
    scope = TurnCancellationScope()
    token = set_turn_cancellation_scope(scope)

    async def provider():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await cleanup_release.wait()
            closed.set()

    try:
        owner = asyncio.create_task(observe(await_with_turn_cancellation(provider())))
        await asyncio.wait_for(started.wait(), timeout=2)
        owner.cancel("first cancellation")
        await asyncio.wait_for(cleanup_started.wait(), timeout=2)
        owner.cancel("second cancellation")
        await asyncio.sleep(0)
        assert not owner.done(), "owner returned before provider cleanup"
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert [error.args for error in cancellations] == [("first cancellation",)]
        assert closed.is_set()
    finally:
        cleanup_release.set()
        reset_turn_cancellation_scope(token)
        await _drain_new_tasks(baseline)
