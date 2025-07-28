"""Async message queue for inter-component communication."""

import asyncio
from collections import deque
from typing import Any, Deque, Optional


class AsyncMessageQueue:
    """Async message queue for managing communication between components."""

    def __init__(self, maxsize: int = 100):
        self.queue: Deque[Any] = deque(maxlen=maxsize)
        self.event = asyncio.Event()
        self.lock = asyncio.Lock()

    async def put(self, item: Any) -> None:
        """Put an item in the queue."""
        async with self.lock:
            self.queue.append(item)
            self.event.set()

    async def get(self) -> Any:
        """Get an item from the queue, waiting if necessary."""
        while True:
            async with self.lock:
                if self.queue:
                    item = self.queue.popleft()
                    if not self.queue:
                        self.event.clear()
                    return item
            await self.event.wait()

    async def get_nowait(self) -> Optional[Any]:
        """Get an item from the queue without waiting."""
        async with self.lock:
            if self.queue:
                item = self.queue.popleft()
                if not self.queue:
                    self.event.clear()
                return item
            return None

    def qsize(self) -> int:
        """Return the approximate size of the queue."""
        return len(self.queue)

    def empty(self) -> bool:
        """Return True if the queue is empty."""
        return len(self.queue) == 0

    def full(self) -> bool:
        """Return True if the queue is full."""
        return len(self.queue) == self.queue.maxlen

    async def clear(self) -> None:
        """Clear all items from the queue."""
        async with self.lock:
            self.queue.clear()
            self.event.clear()
