"""Backup publication must preserve good data and own its staging files."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
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


def test_backup_does_not_consume_a_preexisting_staging_name(tmp_path):
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    foreign = backup.with_suffix(backup.suffix + ".tmp")
    _database(source, "new")
    foreign.write_bytes(b"owned by another operation")

    result = recovery.create_backup(source, backup)

    assert foreign.exists(), "backup removed a staging file it did not create"
    assert foreign.read_bytes() == b"owned by another operation"
    assert result.created
    assert _value(backup) == "new"


def test_rejected_snapshot_does_not_replace_the_previous_backup(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    _database(source, "new")
    _database(backup, "previous good backup")
    previous = backup.read_bytes()

    # Model the integrity checker rejecting the candidate. The existing
    # destination must not be replaced before that decision is made.
    monkeypatch.setattr(recovery, "_sqlite_db_is_healthy", lambda path: Path(path) == source)
    result = recovery.create_backup(source, backup)

    assert not result.created
    assert backup.read_bytes() == previous
    assert _value(backup) == "previous good backup"


def test_concurrent_backups_use_independent_snapshots(tmp_path, monkeypatch):
    first = tmp_path / "first.db"
    second = tmp_path / "second.db"
    backup = tmp_path / "backup.db"
    _database(first, "first")
    _database(second, "second")
    staged = []
    reached_publication = threading.Barrier(2, timeout=5)
    replace = recovery.os.replace

    def publish(source, destination):
        staged.append(Path(source))
        reached_publication.wait()
        return replace(source, destination)

    monkeypatch.setattr(recovery.os, "replace", publish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(recovery.create_backup, path, backup) for path in (first, second)]
        results = [future.result(timeout=10) for future in futures]

    assert len(set(staged)) == 2, "concurrent writers shared a temporary database"
    assert all(result.created for result in results)
    assert _value(backup) in {"first", "second"}
    assert all(not path.exists() for path in staged)
