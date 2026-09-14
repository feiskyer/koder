"""Turn-wide cancellation shared by setup, auxiliary calls, and model execution."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from typing import Awaitable, Callable, TypeVar

from koder_agent.utils.async_tasks import await_owned_task

from .keyboard_listener import CancellationToken

T = TypeVar("T")


class TurnCancellationScope:
    """One cancellation signal spanning the entire interactive turn."""

    def __init__(self) -> None:
        self.token = CancellationToken()
        self._callbacks: list[Callable[[], None]] = []

    @property
    def is_cancelled(self) -> bool:
        return self.token.is_cancelled

    def cancel(self) -> None:
        if self.is_cancelled:
            return
        self.token.cancel()
        for callback in list(self._callbacks):
            try:
                callback()
            except Exception:
                # Cancellation remains best-effort across independent provider,
                # runner, and UI callbacks; one cleanup must not block another.
                continue

    def add_callback(self, callback: Callable[[], None]) -> Callable[[], None]:
        if self.is_cancelled:
            callback()
            return lambda: None
        self._callbacks.append(callback)

        def remove() -> None:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

        return remove

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise asyncio.CancelledError


_CURRENT_SCOPE: ContextVar[TurnCancellationScope | None] = ContextVar(
    "koder_turn_cancellation_scope",
    default=None,
)


def current_turn_cancellation_scope() -> TurnCancellationScope | None:
    return _CURRENT_SCOPE.get()


def set_turn_cancellation_scope(scope: TurnCancellationScope) -> Token:
    return _CURRENT_SCOPE.set(scope)


def reset_turn_cancellation_scope(token: Token) -> None:
    _CURRENT_SCOPE.reset(token)


async def await_with_turn_cancellation(awaitable: Awaitable[T]) -> T:
    """Own work until it settles, even when the caller itself is cancelled."""
    scope = current_turn_cancellation_scope()
    if scope is None:
        return await awaitable

    # Take ownership before checking the signal: the caller may already have
    # created a coroutine or started a task that must not be abandoned.
    work_task = asyncio.ensure_future(awaitable)
    cancel_task = asyncio.create_task(scope.token.wait())
    primary_error: BaseException | None = None
    try:
        scope.raise_if_cancelled()
        done, _ = await asyncio.wait(
            {work_task, cancel_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancel_task in done:
            raise asyncio.CancelledError
        return work_task.result()
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        for task in (work_task, cancel_task):
            if not task.done():
                task.cancel()
        settlement = asyncio.gather(work_task, cancel_task, return_exceptions=True)
        try:
            await await_owned_task(settlement)
        except asyncio.CancelledError:
            # Repeated cancellation must not interrupt provider cleanup or
            # replace the original failure/cancellation (including its message).
            if primary_error is None:
                raise
