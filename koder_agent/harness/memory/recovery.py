"""Standalone SQLite snapshot helpers, not automatic recovery for main Sessions."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecoveryResult:
    """Result of a transcript recovery attempt."""

    recovered: bool
    reason: str


@dataclass(frozen=True)
class BackupResult:
    """Result of a transcript backup attempt."""

    created: bool
    reason: str


def _sqlite_db_is_healthy(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        conn = sqlite3.connect(path)
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()
            return bool(result and result[0] == "ok")
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return False


def default_backup_path(runtime_db_path: str | Path) -> Path:
    """Return the conventional ``<db>.bak`` path used by recovery."""
    runtime_db = Path(runtime_db_path)
    return runtime_db.with_suffix(runtime_db.suffix + ".bak")


def create_backup(
    runtime_db_path: str | Path,
    backup_db_path: str | Path | None = None,
) -> BackupResult:
    """Snapshot a HEALTHY runtime DB to ``<db>.bak`` so recovery has something to restore.

    Only explicit callers create these snapshots. Main CLI persistence uses
    ``EnhancedSQLiteSession`` and the SDK store, not this backup/restore path.
    Integrators must choose their own backup cadence and recovery ownership.

    Uses SQLite's online backup API to capture a transactionally-consistent copy
    even under WAL mode / concurrent source access. Each call owns an independent
    staging directory beside the destination. The candidate is checked before
    atomic ``os.replace`` publication, so a rejected snapshot leaves the previous
    backup intact. Concurrent publishers may replace one another's complete
    snapshots; they never share or delete one another's staging database.

    The standalone ``TranscriptStore`` does not automatically invoke this
    helper either. A helper test is not evidence of main-session backup or
    recovery integration.
    """
    runtime_db = Path(runtime_db_path)
    backup_db = (
        Path(backup_db_path) if backup_db_path is not None else default_backup_path(runtime_db)
    )

    if not runtime_db.exists():
        return BackupResult(created=False, reason="no runtime database to back up")

    if not _sqlite_db_is_healthy(runtime_db):
        # Never overwrite a known-good backup with a corrupt source.
        return BackupResult(created=False, reason="runtime database is not healthy")

    try:
        backup_db.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{backup_db.name}.", dir=backup_db.parent
        ) as staging:
            tmp_path = Path(staging) / "snapshot.db"
            source = sqlite3.connect(runtime_db)
            try:
                dest = sqlite3.connect(tmp_path)
                try:
                    source.backup(dest)
                finally:
                    dest.close()
            finally:
                source.close()
            if not _sqlite_db_is_healthy(tmp_path):
                return BackupResult(created=False, reason="backup failed integrity check")
            os.replace(tmp_path, backup_db)
    except (sqlite3.DatabaseError, OSError):
        return BackupResult(created=False, reason="failed to write backup")

    return BackupResult(created=True, reason="backup created")


def recover_partial_write(
    runtime_db_path: str | Path,
    backup_db_path: str | Path | None = None,
) -> RecoveryResult:
    """Restore a complete backup snapshot to a quiescent, damaged primary.

    Callers must close primary connections and coordinate its writers before
    recovery. This helper does not replace an actively used SQLite database.
    The backup itself is read through SQLite's snapshot API, including committed
    WAL data, and a validated candidate is published without truncating the
    existing target in place.
    """
    runtime_db = Path(runtime_db_path)
    backup_db = (
        Path(backup_db_path)
        if backup_db_path is not None
        else runtime_db.with_suffix(runtime_db.suffix + ".bak")
    )

    if _sqlite_db_is_healthy(runtime_db):
        return RecoveryResult(recovered=False, reason="primary database is healthy")

    if not backup_db.exists():
        return RecoveryResult(recovered=False, reason="no backup database available")

    if not _sqlite_db_is_healthy(backup_db):
        return RecoveryResult(recovered=False, reason="backup database is not healthy")

    restored = create_backup(backup_db, runtime_db.resolve())
    if restored.created:
        return RecoveryResult(recovered=True, reason="restored from backup")
    return RecoveryResult(recovered=False, reason=f"restore failed: {restored.reason}")
