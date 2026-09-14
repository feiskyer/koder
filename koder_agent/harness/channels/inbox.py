"""Bounded, runtime-owned disk staging for incoming channel messages.

Only small entry descriptors remain in memory. Admission never waits for a
model turn or a free consumer slot. Retained files are for explicit review,
not automatic replay of possibly effectful interrupted turns.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import stat
import tempfile
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

from koder_agent.harness.paths import harness_home_dir
from koder_agent.utils.async_tasks import await_owned_task, run_sync_owned

logger = logging.getLogger(__name__)


class ChannelInboxRejectedError(RuntimeError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Channel message not accepted: {reason}")


class ChannelInboxStorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChannelInboxLimits:
    max_messages: int = 4096
    max_bytes: int = 64 * 1024 * 1024
    max_message_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("Channel inbox limits must be positive integers")
        if self.max_message_bytes > self.max_bytes:
            raise ValueError("The channel message limit cannot exceed the total inbox limit")

    @classmethod
    def from_env(cls) -> ChannelInboxLimits:
        defaults = cls()
        return cls(
            max_messages=int(
                os.environ.get("KODER_CHANNEL_MAX_PENDING_MESSAGES", defaults.max_messages)
            ),
            max_bytes=int(os.environ.get("KODER_CHANNEL_MAX_PENDING_BYTES", defaults.max_bytes)),
            max_message_bytes=int(
                os.environ.get("KODER_CHANNEL_MAX_MESSAGE_BYTES", defaults.max_message_bytes)
            ),
        )


@dataclass(frozen=True)
class ChannelMessage:
    id: int
    source: str
    content: str


@dataclass(frozen=True)
class _Entry:
    id: int
    size: int
    digest: str
    state: str = "pending"
    reason: str | None = None

    @property
    def filename(self) -> str:
        return f"{self.id:016d}.{self.state}.json"


class ChannelInbox:
    """Stage payloads off-loop, retaining bounded metadata and explicit outcomes."""

    def __init__(self, *, root: Path | None = None, limits: ChannelInboxLimits | None = None):
        self.root = root if root is not None else harness_home_dir() / "channel-inbox"
        self.limits = limits if limits is not None else ChannelInboxLimits.from_env()
        self.directory: Path | None = None
        self._entries: dict[int, _Entry] = {}
        self._pending: deque[int] = deque()
        self._reservations: dict[int, int] = {}
        self._reserved_bytes = 0
        self._admissions: set[asyncio.Task[int]] = set()
        self._lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self._open_task: asyncio.Task | None = None
        self._close_task: asyncio.Task | None = None
        self._closing = False
        self._closed = False
        self._fault: str | None = None
        self._next_id = 1
        self._bytes = 0
        self._accepted = 0
        self._completed = 0
        self._rejections: dict[str, int] = {}

    @property
    def rejected(self) -> int:
        return sum(self._rejections.values())

    def snapshot(self) -> dict:
        states = ("pending", "running", "failed", "cancelled", "interrupted")
        return {
            "status": (
                "closed"
                if self._closed
                else (
                    "closing"
                    if self._closing
                    else (
                        "faulted"
                        if self._fault
                        else "open"
                        if self.directory is not None
                        else "initializing"
                    )
                )
            ),
            "path": str(self.directory) if self.directory is not None else None,
            "accepted": self._accepted,
            "completed": self._completed,
            "rejected": self.rejected,
            "rejections": dict(self._rejections),
            "retained_messages": len(self._entries),
            "retained_bytes": self._bytes,
            "admitting": len(self._reservations),
            "admitting_bytes": self._reserved_bytes,
            "limits": vars(self.limits).copy(),
            "fault": self._fault,
            **{
                state: sum(entry.state == state for entry in self._entries.values())
                for state in states
            },
        }

    def _reject(self, reason: str) -> NoReturn:
        if not self._closed:
            self._rejections[reason] = self._rejections.get(reason, 0) + 1
        raise ChannelInboxRejectedError(reason)

    def _ensure_open(self) -> None:
        if self._closing or self._closed:
            self._reject("closed")
        if self._fault is not None:
            self._reject("storage_unavailable")
        if self.directory is None:
            raise RuntimeError("Open the channel inbox before use")

    def _storage_failed(self, error: Exception) -> ChannelInboxStorageError:
        self._fault = type(error).__name__
        self._ready.set()
        return ChannelInboxStorageError(
            f"Channel inbox storage unavailable ({self._fault}); retained at {self.directory}"
        )

    def _allocate_directory(self) -> Path:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="run-", dir=self.root))

    async def open(self) -> ChannelInbox:
        if self._closing:
            self._reject("closed")
        if self._open_task is None:
            self._open_task = asyncio.create_task(self._initialize())
        await await_owned_task(self._open_task)
        return self

    async def _initialize(self) -> None:
        try:
            self.directory = await run_sync_owned(self._allocate_directory)
        except Exception as error:
            raise self._storage_failed(error) from error

    @staticmethod
    def _publish(path: Path, data: bytes) -> None:
        # Do not resolve destination symlinks: this namespace is private and
        # runtime-owned, not a user-editable symlinked configuration file.
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        descriptor, staged = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
        committed = False
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staged, path)
            committed = True
        finally:
            if not committed:
                try:
                    Path(staged).unlink(missing_ok=True)
                except OSError:
                    logger.warning("Unable to remove incomplete channel staging file")

    async def put(self, source: str, content: str) -> int:
        self._ensure_open()
        # Reserve before the first await. Otherwise every concurrent connection
        # could retain another payload while waiting for the disk-operation lock.
        if len(self._entries) + len(self._reservations) >= self.limits.max_messages:
            self._reject("message_capacity")
        if not isinstance(source, str) or not isinstance(content, str):
            self._reject("invalid_message")
        if len(source) + len(content) > self.limits.max_message_bytes:
            self._reject("message_too_large")
        identifier = self._next_id
        payload = {
            "schema_version": 1,
            "id": identifier,
            "source": source,
            "received_at": datetime.now(timezone.utc).isoformat(),
            "content": content,
        }
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except UnicodeError:
            self._reject("invalid_encoding")
        if len(data) > self.limits.max_message_bytes:
            self._reject("message_too_large")
        if self._bytes + self._reserved_bytes + len(data) > self.limits.max_bytes:
            self._reject("byte_capacity")
        entry = _Entry(identifier, len(data), hashlib.sha256(data).hexdigest())
        self._next_id += 1
        self._reservations[identifier] = entry.size
        self._reserved_bytes += entry.size
        admission = self._admit_reserved(entry, data)
        try:
            operation = asyncio.create_task(admission)
        except BaseException:
            admission.close()
            self._release_reservation(identifier)
            raise
        self._admissions.add(operation)
        operation.add_done_callback(self._admission_done)
        return await await_owned_task(operation)

    def _release_reservation(self, identifier: int) -> None:
        self._reserved_bytes -= self._reservations.pop(identifier, 0)

    def _admission_done(self, operation: asyncio.Task[int]) -> None:
        self._admissions.discard(operation)
        if not operation.cancelled():
            operation.exception()

    async def _admit_reserved(self, entry: _Entry, data: bytes) -> int:
        # Reservations made before closure remain owned until publication or a
        # reported storage failure, even if their receive waiter is cancelled.
        try:
            async with self._lock:
                if self._fault is not None:
                    self._reject("storage_unavailable")
                return await self._store(entry, data)
        finally:
            self._release_reservation(entry.id)

    async def _store(self, entry: _Entry, data: bytes) -> int:
        try:
            await run_sync_owned(self._publish, self.directory / entry.filename, data)
        except Exception as error:
            self._rejections["storage_unavailable"] = (
                self._rejections.get("storage_unavailable", 0) + 1
            )
            raise self._storage_failed(error) from error
        self._release_reservation(entry.id)
        self._entries[entry.id] = entry
        self._pending.append(entry.id)
        self._bytes += entry.size
        self._accepted += 1
        self._ready.set()
        return entry.id

    def _move(self, entry: _Entry, state: str) -> _Entry:
        changed = replace(entry, state=state)
        target = self.directory / changed.filename
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        os.replace(self.directory / entry.filename, target)
        return changed

    def _read_and_claim(self, entry: _Entry) -> tuple[_Entry, ChannelMessage]:
        path = self.directory / entry.filename
        if path.is_symlink():
            raise ValueError("Channel payload must not be a symlink")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Channel payload must be a regular file")
            data = stream.read(self.limits.max_message_bytes + 1)
        if len(data) != entry.size or hashlib.sha256(data).hexdigest() != entry.digest:
            raise ValueError("Channel payload integrity check failed")
        payload = json.loads(data)
        if (
            payload.get("id") != entry.id
            or payload.get("schema_version") != 1
            or not isinstance(payload.get("source"), str)
            or not isinstance(payload.get("content"), str)
        ):
            raise ValueError("Invalid channel payload record")
        changed = self._move(entry, "running")
        return changed, ChannelMessage(entry.id, payload["source"], payload["content"])

    async def get(self) -> ChannelMessage | None:
        if self.directory is None and not self._closing:
            raise RuntimeError("Open the channel inbox before use")
        while True:
            async with self._lock:
                if self._closing:
                    return None
                if self._fault is not None:
                    raise ChannelInboxStorageError(f"Channel inbox faulted ({self._fault})")
                if self._pending:
                    entry = self._entries[self._pending[0]]
                    return await await_owned_task(asyncio.create_task(self._claim(entry)))
                self._ready.clear()
            await self._ready.wait()

    async def _claim(self, entry: _Entry) -> ChannelMessage:
        try:
            changed, message = await run_sync_owned(self._read_and_claim, entry)
        except Exception as error:
            raise self._storage_failed(error) from error
        self._pending.popleft()
        self._entries[entry.id] = changed
        return message

    async def finish(self, identifier: int, outcome: str, *, reason: str | None = None) -> None:
        if outcome not in {"completed", "failed", "cancelled", "interrupted"}:
            raise ValueError("Invalid channel delivery outcome")
        async with self._lock:
            if self._closing:
                raise ChannelInboxRejectedError("closed")
            entry = self._entries[identifier]
            if entry.state != "running":
                raise ValueError("Only a running delivery can be finished")
            await await_owned_task(asyncio.create_task(self._finish(entry, outcome, reason)))

    async def _finish(self, entry: _Entry, outcome: str, reason: str | None) -> None:
        try:
            if outcome == "completed":
                await run_sync_owned((self.directory / entry.filename).unlink)
                self._entries.pop(entry.id)
                self._bytes -= entry.size
                self._completed += 1
            else:
                changed = await run_sync_owned(self._move, entry, outcome)
                self._entries[entry.id] = replace(
                    changed, reason=reason[:128] if reason is not None else None
                )
        except Exception as error:
            raise self._storage_failed(error) from error

    async def aclose(self) -> None:
        if self._close_task is None:
            self._closing = True
            self._ready.set()
            self._close_task = asyncio.create_task(self._close())
        await await_owned_task(self._close_task)

    async def _close(self) -> None:
        if self._open_task is not None:
            try:
                await self._open_task
            except Exception:
                pass
        # No new reservation can be made after _closing is set. Join the bounded
        # set before acquiring the same lock used by their disk writes.
        if self._admissions:
            await asyncio.gather(*tuple(self._admissions), return_exceptions=True)
        async with self._lock:
            try:
                if self.directory is None:
                    return
                for entry in list(self._entries.values()):
                    if entry.state == "running":
                        changed = await run_sync_owned(self._move, entry, "interrupted")
                        self._entries[entry.id] = changed
                if not self._entries and not self.rejected and self._fault is None:
                    await run_sync_owned(self.directory.rmdir)
                    self.directory = None
                    return
                manifest = {
                    "schema_version": 1,
                    **self.snapshot(),
                    "status": "closed",
                    "automatic_replay": False,
                    "entries": [
                        {
                            "id": entry.id,
                            "file": entry.filename,
                            "state": entry.state,
                            "sha256": entry.digest,
                            "reason": entry.reason,
                        }
                        for entry in self._entries.values()
                    ],
                }
                await run_sync_owned(
                    self._publish,
                    self.directory / "manifest.json",
                    json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
                )
            except Exception as error:
                self._fault = type(error).__name__
                logger.error("Channel inbox finalization failed (%s)", self._fault)
                raise
            finally:
                self._closed = True
                if self.directory is not None:
                    logger.warning(
                        "Channel inbox retained at %s: %d unfinished messages, %d rejected",
                        self.directory,
                        len(self._entries),
                        self.rejected,
                    )
