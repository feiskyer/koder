"""Task graph snapshots must survive concurrent readers and interrupted writes."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import replace
from pathlib import Path

import pytest

from koder_agent.harness.tasks.storage import TaskStorage


def test_dependency_recovers_both_records_after_second_write_fails(tmp_path, monkeypatch):
    storage = TaskStorage(tmp_path / "tasks")
    first = storage.create("blocker")
    second = storage.create("blocked")
    real_write = storage._write_task

    def fail_second(task):
        if task.id == second.id:
            raise OSError("injected second-record failure")
        real_write(task)

    monkeypatch.setattr(storage, "_write_task", fail_second)
    with pytest.raises(OSError, match="second-record"):
        storage.add_block(blocker_id=first.id, blocked_id=second.id)

    recovered = TaskStorage(storage.root)
    assert recovered.get(first.id).blocks == [second.id]
    assert recovered.get(second.id).blocked_by == [first.id]


def test_delete_recovers_all_dependency_removals_after_write_failure(tmp_path, monkeypatch):
    storage = TaskStorage(tmp_path / "tasks")
    blocker = storage.create("blocker")
    blocked = storage.create("blocked")
    storage.add_block(blocker_id=blocker.id, blocked_id=blocked.id)

    def fail_write(_task):
        raise OSError("injected unlink cleanup failure")

    monkeypatch.setattr(storage, "_write_task", fail_write)
    with pytest.raises(OSError, match="unlink cleanup"):
        storage.delete(blocker.id)

    recovered = TaskStorage(storage.root)
    assert recovered.get(blocker.id) is None
    assert recovered.get(blocked.id).blocked_by == []


def test_dependency_readers_wait_for_a_complete_graph_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "tasks"
    writer = TaskStorage(root)
    reader = TaskStorage(root)
    first = writer.create("blocker")
    second = writer.create("blocked")
    first_written = threading.Event()
    release_write = threading.Event()
    real_write = writer._write_task

    def hold_first(task):
        real_write(task)
        if task.id == first.id:
            first_written.set()
            assert release_write.wait(5)

    monkeypatch.setattr(writer, "_write_task", hold_first)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writing = pool.submit(writer.add_block, blocker_id=first.id, blocked_id=second.id)
        try:
            assert first_written.wait(5)
            reading = pool.submit(reader.list_all)
            with pytest.raises(TimeoutError):
                reading.result(timeout=0.1)
        finally:
            release_write.set()
        assert writing.result(timeout=5)
        snapshot = {task.id: task for task in reading.result(timeout=5)}
    assert snapshot[first.id].blocks == [second.id]
    assert snapshot[second.id].blocked_by == [first.id]


def test_task_cannot_depend_on_itself(tmp_path):
    storage = TaskStorage(tmp_path / "tasks")
    task = storage.create("self")

    assert storage.add_block(blocker_id=task.id, blocked_id=task.id) is False
    assert storage.get(task.id) == task


def test_missing_highwatermark_does_not_overwrite_live_tasks(tmp_path):
    storage = TaskStorage(tmp_path / "tasks")
    original = storage.create("must survive")
    (storage.root / ".highwatermark").unlink()

    created = storage.create("new")

    assert created.id != original.id
    assert storage.get(original.id) == original


def _probe(root, mode, *, writer="worker"):
    return subprocess.Popen(
        [
            "uv",
            "run",
            "--no-project",
            "--no-config",
            sys.executable,
            str(Path(__file__).with_name("_process_probe.py")),
            str(root),
            mode,
            "--writer",
            writer,
        ],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _finish(process, *, expected_code=0):
    try:
        output, error = process.communicate(timeout=25)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
    assert process.returncode == expected_code, error
    if expected_code:
        return []
    result = next(line for line in output.splitlines() if line.startswith("TASK_IDS:"))
    return json.loads(result.removeprefix("TASK_IDS:"))


@pytest.mark.parametrize("mode", ["add-crash", "delete-crash"])
def test_graph_transaction_recovers_after_actual_process_exit(tmp_path, mode):
    storage = TaskStorage(tmp_path / "tasks")
    first = storage.create("blocker")
    second = storage.create("blocked")
    if mode == "delete-crash":
        storage.add_block(blocker_id=first.id, blocked_id=second.id)

    _finish(_probe(storage.root, mode), expected_code=24)
    assert (storage.root / ".transaction").is_file()
    recovered = TaskStorage(storage.root)
    if mode == "add-crash":
        assert recovered.get(first.id).blocks == [second.id]
        assert recovered.get(second.id).blocked_by == [first.id]
    else:
        assert recovered.get(first.id) is None
        assert recovered.get(second.id).blocked_by == []
    assert not (storage.root / ".transaction").exists()
    recovered.update(second.id, title="after recovery")
    assert TaskStorage(storage.root).get(second.id).title == "after recovery"


def test_independent_processes_allocate_unique_task_ids(tmp_path):
    storage = TaskStorage(tmp_path / "tasks")
    workers = [_probe(storage.root, "create", writer=writer) for writer in ("a", "b")]
    try:
        deadline = time.monotonic() + 20
        while not all((storage.root / f"{writer}.ready").exists() for writer in ("a", "b")):
            assert all(worker.poll() is None for worker in workers)
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        (storage.root / "start").write_text("start", encoding="utf-8")
        results = [_finish(worker) for worker in workers]
    ids = [task_id for group in results for task_id in group]
    assert len(ids) == len(set(ids)) == 24
    assert {task.id for task in storage.list_all()} == set(ids)


def test_invalid_journal_destination_is_rejected_before_any_replay(tmp_path):
    storage = TaskStorage(tmp_path / "tasks")
    first = storage.create("original")
    valid = replace(first, title="must not publish").to_dict()
    invalid = replace(first, id="../outside").to_dict()
    journal = storage.root / ".transaction"
    journal.write_text(
        json.dumps({"version": 1, "writes": [valid, invalid], "deletes": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="task ID"):
        storage.list_all()

    # Inspect only this synthetic fixture; the public API correctly fails closed.
    assert json.loads((storage.root / f"{first.id}.json").read_text()) == first.to_dict()
    assert journal.exists()


def test_failed_journal_publication_leaves_both_tasks_unchanged(tmp_path, monkeypatch):
    storage = TaskStorage(tmp_path / "tasks")
    first = storage.create("blocker")
    second = storage.create("blocked")

    def fail_publish(*_args, **_kwargs):
        raise OSError("injected journal publication failure")

    monkeypatch.setattr("koder_agent.harness.tasks.storage.write_text_atomic", fail_publish)
    with pytest.raises(OSError, match="journal publication"):
        storage.add_block(blocker_id=first.id, blocked_id=second.id)

    assert storage.get(first.id) == first
    assert storage.get(second.id) == second
    assert not (storage.root / ".transaction").exists()
