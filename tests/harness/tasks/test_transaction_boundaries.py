"""Locked storage contract for logical task requests and legacy callers."""

import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pytest

from koder_agent.harness.tasks.storage import TaskStorage


@pytest.mark.parametrize("metadata", [[], "text", {"invalid": float("nan")}])
def test_invalid_metadata_cannot_allocate_or_mutate(tmp_path, metadata):
    storage = TaskStorage(tmp_path / "tasks")
    with pytest.raises((TypeError, ValueError)):
        storage.create("invalid", metadata=metadata)
    assert not (storage.root / ".highwatermark").exists()
    original = storage.create("original")
    with pytest.raises((TypeError, ValueError)):
        storage.update_with_dependencies(original.id, title="invalid", metadata=metadata)
    assert storage.get(original.id) == original


def test_legacy_single_operations_keep_return_and_owner_contracts(tmp_path):
    storage = TaskStorage(tmp_path / "tasks")
    first = storage.create("first")
    second = storage.create("second")
    assert storage.update(first.id, owner="agent").owner == "agent"
    assert storage.update(first.id, title="renamed").owner == "agent"
    assert storage.update(first.id, owner=None).owner is None
    assert storage.update("999", title="missing") is None
    assert storage.add_block(blocker_id=first.id, blocked_id="999") is False
    assert storage.add_block(blocker_id="999", blocked_id=second.id) is False
    assert storage.add_block(blocker_id=first.id, blocked_id=first.id) is False
    assert storage.add_block(blocker_id=first.id, blocked_id=second.id) is True
    assert storage.delete(first.id) is True
    assert storage.delete(first.id) is False
    assert storage.get(second.id).blocked_by == []


def test_overlapping_endpoints_are_merged_not_overwritten(tmp_path):
    storage = TaskStorage(tmp_path / "tasks")
    first = storage.create("first")
    second = storage.create("second")
    # General cycles were already allowed; only self-dependency is rejected.
    result = storage.update_with_dependencies(
        first.id,
        title="new",
        add_blocks=(second.id, second.id),
        add_blocked_by=(second.id, second.id),
    )
    assert result.before == first
    assert result.after.title == "new"
    for task in storage.list_all():
        other = second.id if task.id == first.id else first.id
        assert task.blocks == [other]
        assert task.blocked_by == [other]


def test_late_invalid_incoming_edge_rejects_fields_and_outgoing_edges(tmp_path):
    storage = TaskStorage(tmp_path / "tasks")
    first = storage.create("first")
    second = storage.create("second")
    with pytest.raises(ValueError, match="not found"):
        storage.update_with_dependencies(
            first.id,
            title="rejected",
            add_blocks=(second.id,),
            add_blocked_by=("999",),
        )
    assert storage.list_all() == [first, second]
    assert not (storage.root / ".transaction").exists()


def test_pending_replay_failure_prevents_a_new_request(tmp_path, monkeypatch):
    storage = TaskStorage(tmp_path / "tasks")
    first = storage.create("first")
    second = storage.create("second")

    def fail(_task):
        raise OSError("injected recovery failure")

    monkeypatch.setattr(storage, "_write_task", fail)
    with pytest.raises(OSError):
        storage.update_with_dependencies(first.id, title="committed", add_blocks=(second.id,))
    journal = storage.root / ".transaction"
    pending_intent = journal.read_bytes()
    with pytest.raises(OSError):
        storage.update_with_dependencies(second.id, title="must not overwrite intent")
    assert journal.read_bytes() == pending_intent
    recovered = TaskStorage(storage.root)
    assert recovered.get(first.id).title == "committed"
    assert recovered.get(first.id).blocks == [second.id]
    assert recovered.get(second.id).title == "second"
    assert recovered.get(second.id).blocked_by == [first.id]


def test_batch_readers_and_writers_wait_for_the_whole_request(tmp_path, monkeypatch):
    writer = TaskStorage(tmp_path / "tasks")
    reader = TaskStorage(writer.root)
    peer = TaskStorage(writer.root)
    first = writer.create("first")
    second = writer.create("second")
    third = writer.create("third")
    first_written = threading.Event()
    release_write = threading.Event()
    real_write = writer._write_task

    def hold_first(task):
        real_write(task)
        if task.id == first.id:
            first_written.set()
            assert release_write.wait(5)

    monkeypatch.setattr(writer, "_write_task", hold_first)
    with ThreadPoolExecutor(max_workers=3) as pool:
        writing = pool.submit(
            writer.update_with_dependencies,
            first.id,
            title="new",
            add_blocks=(second.id,),
            add_blocked_by=(third.id,),
        )
        try:
            assert first_written.wait(5)
            reading = pool.submit(reader.list_all)
            peer_writing = pool.submit(peer.update, second.id, title="peer update")
            with pytest.raises(TimeoutError):
                reading.result(timeout=0.1)
            with pytest.raises(TimeoutError):
                peer_writing.result(timeout=0.1)
        finally:
            release_write.set()
        assert writing.result(timeout=5).after.title == "new"
        snapshot = {task.id: task for task in reading.result(timeout=5)}
        assert peer_writing.result(timeout=5).title == "peer update"
    assert snapshot[first.id].title == "new"
    assert snapshot[first.id].blocks == [second.id]
    assert snapshot[first.id].blocked_by == [third.id]
    assert snapshot[second.id].blocked_by == [first.id]
    assert snapshot[third.id].blocks == [first.id]
    assert reader.get(second.id).title == "peer update"
    assert reader.get(second.id).blocked_by == [first.id]
