"""Track file read state for read-before-write enforcement."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from koder_agent.harness.execution_context import execution_path


@dataclass
class _FileReadRecord:
    timestamp: float  # os.stat mtime at read time
    size: Optional[int] = None  # os.stat st_size at read time
    content_hash: Optional[str] = None  # sha256 of on-disk bytes at read time
    content: Optional[str] = None
    is_partial: bool = False


def _hash_file(path: str) -> Optional[str]:
    """Return a sha256 hex digest of the file's bytes, or None on error."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


class ReadFileState:
    """Tracks which files have been read and detects staleness."""

    def __init__(self) -> None:
        self._records: dict[str, _FileReadRecord] = {}

    def _normalize(self, path: str) -> str:
        return str(execution_path(path).resolve())

    def record_read(
        self, path: str, *, content: Optional[str] = None, is_partial: bool = False
    ) -> None:
        norm = self._normalize(path)
        try:
            stat = os.stat(norm)
            mtime = stat.st_mtime
            size: Optional[int] = stat.st_size
        except OSError:
            mtime = 0.0
            size = None
        self._records[norm] = _FileReadRecord(
            timestamp=mtime,
            size=size,
            content_hash=_hash_file(norm),
            content=content,
            is_partial=is_partial,
        )

    def has_been_read(self, path: str) -> bool:
        return self._normalize(path) in self._records

    def is_partial_view(self, path: str) -> bool:
        record = self._records.get(self._normalize(path))
        return record.is_partial if record else False

    def get_full_content(self, path: str) -> Optional[str]:
        """Return the cached full (non-partial) content, or None if unavailable.

        Returns None when the file has not been read, was only read partially,
        or no content was recorded.
        """
        record = self._records.get(self._normalize(path))
        if record is None or record.is_partial:
            return None
        return record.content

    def is_stale(self, path: str) -> bool:
        norm = self._normalize(path)
        record = self._records.get(norm)
        if record is None:
            return False
        try:
            stat = os.stat(norm)
            current_mtime = stat.st_mtime
            current_size = stat.st_size
        except OSError:
            # Cannot stat the file now. If it existed at read time (we captured a
            # size) it has since been removed/became unreadable -- conservatively
            # stale. If it never existed at read time, preserve prior behavior and
            # report not-stale (nothing was ever read to go stale).
            return record.size is not None

        if current_mtime > record.timestamp:
            # mtime advanced -- verify with content comparison for full reads.
            if record.content is not None and not record.is_partial:
                try:
                    current_content = Path(norm).read_text(encoding="utf-8")
                    if current_content == record.content:
                        return False
                except OSError:
                    pass
            return True

        # mtime is equal-or-older (same-second / coarse-mtime filesystems, or an
        # in-place edit that preserved mtime). A size change is a definitive edit.
        if record.size is not None and current_size != record.size:
            return True

        # Size unchanged but mtime not advanced: fall back to a content hash so a
        # same-size, same-second edit is still caught. Compare against the hash we
        # captured at read time; if either hash is unavailable, treat as fresh
        # (best effort) since mtime did not advance.
        current_hash = _hash_file(norm)
        if record.content_hash is not None and current_hash is not None:
            if current_hash != record.content_hash:
                return True

        return False

    def invalidate_all(self) -> None:
        """Clear all state after compaction (context no longer has file contents)."""
        self._records.clear()

    def clear(self, path: Optional[str] = None) -> None:
        if path:
            self._records.pop(self._normalize(path), None)
        else:
            self._records.clear()
