"""Recovery publishes a complete SQLite snapshot without truncating its target."""

import sqlite3
from contextlib import closing
from pathlib import Path

from koder_agent.harness.memory import recovery


def _database(path, value):
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("CREATE TABLE marker (value TEXT)")
        connection.execute("INSERT INTO marker VALUES (?)", (value,))


def _value(path):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        return connection.execute("SELECT value FROM marker").fetchone()[0]


def test_failed_recovery_publication_keeps_target_and_backup(tmp_path, monkeypatch):
    target = tmp_path / "runtime.db"
    backup = tmp_path / "backup.db"
    original = b"original damaged database"
    target.write_bytes(original)
    _database(backup, "recoverable")
    backup_before = backup.read_bytes()

    def fail_publication(*_args):
        raise OSError("synthetic publication failure")

    with monkeypatch.context() as failure:
        failure.setattr(recovery.os, "replace", fail_publication)
        result = recovery.recover_partial_write(target, backup)

    assert not result.recovered
    assert target.read_bytes() == original
    assert backup.read_bytes() == backup_before
    assert _value(backup) == "recoverable"
    assert recovery.recover_partial_write(target, backup).recovered
    assert _value(target) == "recoverable"
    assert backup.read_bytes() == backup_before


def test_rejected_recovery_candidate_keeps_original_target(tmp_path, monkeypatch):
    target = tmp_path / "runtime.db"
    backup = tmp_path / "backup.db"
    original = b"original damaged database"
    target.write_bytes(original)
    _database(backup, "recoverable")
    monkeypatch.setattr(recovery, "_sqlite_db_is_healthy", lambda path: Path(path) == backup)

    result = recovery.recover_partial_write(target, backup)

    assert not result.recovered
    assert target.read_bytes() == original


def test_recovery_includes_committed_backup_wal(tmp_path):
    target = tmp_path / "runtime.db"
    backup = tmp_path / "backup.db"
    target.write_bytes(b"damaged")
    with closing(sqlite3.connect(backup)) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE marker (value TEXT)")
        writer.execute("INSERT INTO marker VALUES ('committed in WAL')")
        writer.commit()
        assert backup.with_name(backup.name + "-wal").exists()

        result = recovery.recover_partial_write(target, backup)

        assert result.recovered
        assert _value(target) == "committed in WAL"
        assert _value(backup) == "committed in WAL"
