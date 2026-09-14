"""Atomic text replacement for small, runtime-owned state files."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


def write_text_atomic(path: Path, content: str) -> None:
    """Publish a complete UTF-8 snapshot, preserving existing modes and symlinks.

    New files are private (0600). The temporary file lives beside the destination
    so replacement stays atomic. Callers performing read-modify-write must hold
    their own lock across the read and this write.
    """
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = stat.S_IMODE(destination.stat().st_mode)
    except FileNotFoundError:
        mode = 0o600
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            os.chmod(temporary, mode)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
