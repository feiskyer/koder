"""Structured ownership for async iterators forwarded by MCP proxies."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from typing import TypeVar

_T = TypeVar("_T")


async def close_async_iterator(iterator: AsyncIterator[object]) -> None:
    """Close an owned iterator when supported, including async generators."""
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


@asynccontextmanager
async def closing_async_iterator(source: AsyncIterable[_T]) -> AsyncIterator[AsyncIterator[_T]]:
    """Propagate early close before a proxy releases its transport/admission."""
    iterator = aiter(source)
    try:
        yield iterator
    finally:
        await close_async_iterator(iterator)
