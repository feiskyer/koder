"""Exercise the registered FunctionTools against isolated storage transactions."""

import json

import pytest
from filelock import Timeout

from koder_agent.harness.tasks.storage import TaskStorage
from koder_agent.tools.task_lifecycle import (
    _set_task_storage,
    task_create_tool,
    task_update_tool,
)


@pytest.fixture
def storage(tmp_path):
    store = TaskStorage(tmp_path / "tasks")
    _set_task_storage(store)
    yield store
    _set_task_storage(None)


async def invoke(tool, **arguments):
    output = await tool.on_invoke_tool(None, json.dumps(arguments))
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"unstructured_error": output}


def assert_failure(result):
    assert result.get("success") is False, result
    assert result.get("error"), result
    assert not result.get("updated_fields"), result
    assert "status_change" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["add_blocks", "add_blocked_by"])
@pytest.mark.parametrize("bad_id", ["999", "1", "../outside"])
async def test_rejected_dependency_rejects_entire_tool_update(storage, direction, bad_id):
    first = storage.create("original", metadata={"keep": 1})
    second = storage.create("peer")
    before = storage.list_all()

    result = await invoke(
        task_update_tool,
        task_id=first.id,
        subject="must not persist",
        status="in_progress",
        metadata='{"keep": null, "new": 2}',
        **{direction: [second.id, bad_id]},
    )

    # Check both the persisted state and the model-facing success claim.
    assert storage.list_all() == before
    assert_failure(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata", ["{", "[]", "null", '"text"', "123", '{"value": NaN}', '{"value": 1e999}']
)
@pytest.mark.parametrize("operation", ["create", "update"])
async def test_metadata_must_be_a_json_object(storage, metadata, operation):
    original = storage.create("original")
    before = storage.list_all()
    if operation == "create":
        result = await invoke(
            task_create_tool, subject="rejected", description="", metadata=metadata
        )
    else:
        result = await invoke(
            task_update_tool,
            task_id=original.id,
            subject="rejected",
            metadata=metadata,
        )
    assert storage.list_all() == before
    assert_failure(result)


@pytest.mark.asyncio
async def test_invalid_status_rejects_all_fields(storage):
    original = storage.create("original")
    result = await invoke(
        task_update_tool, task_id=original.id, subject="rejected", status="unknown"
    )
    assert storage.get(original.id) == original
    assert_failure(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [{"subject": "ignored"}, {"add_blocks": ["999"]}])
async def test_delete_cannot_silently_ignore_other_mutations(storage, extra):
    original = storage.create("original")
    result = await invoke(task_update_tool, task_id=original.id, status="deleted", **extra)
    assert storage.get(original.id) == original
    assert_failure(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "deleted"])
async def test_absent_subject_is_a_failed_tool_request(storage, status):
    result = await invoke(task_update_tool, task_id="999", status=status)
    assert_failure(result)
    assert storage.list_all() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "deleted"])
async def test_missing_storage_receipt_is_not_success(storage, monkeypatch, status):
    original = storage.create("original")
    # Old caller ignored these failure receipts. The integrated caller must
    # likewise refuse a missing receipt from its single transaction operation.
    monkeypatch.setattr(storage, "update", lambda *_a, **_kw: None)
    monkeypatch.setattr(storage, "delete", lambda *_a, **_kw: False)
    monkeypatch.setattr(storage, "update_with_dependencies", lambda *_a, **_kw: None, raising=False)
    result = await invoke(task_update_tool, task_id=original.id, status=status)
    assert_failure(result)
    assert storage.get(original.id) == original


@pytest.mark.asyncio
async def test_one_tool_update_commits_fields_and_both_edge_directions(storage):
    first = storage.create("original", metadata={"keep": 1, "remove": 2})
    second = storage.create("second")
    third = storage.create("third")
    result = await invoke(
        task_update_tool,
        task_id=first.id,
        subject="new",
        description="details",
        status="in_progress",
        owner="agent",
        metadata='{"remove": null, "new": 3}',
        add_blocks=[second.id, second.id],
        add_blocked_by=[third.id],
    )
    assert result["success"] is True
    assert result["status_change"] == {"from": "pending", "to": "in_progress"}
    assert set(result["updated_fields"]) == {
        "subject",
        "description",
        "status",
        "owner",
        "metadata",
        "blocks",
        "blocked_by",
    }
    current = storage.get(first.id)
    assert (current.title, current.description, current.owner) == ("new", "details", "agent")
    assert current.metadata == {"keep": 1, "new": 3}
    assert current.blocks == [second.id]
    assert current.blocked_by == [third.id]
    assert storage.get(second.id).blocked_by == [first.id]
    assert storage.get(third.id).blocks == [first.id]


@pytest.mark.asyncio
async def test_delete_tool_cleans_edges_and_keeps_id_allocation(storage):
    first = storage.create("first")
    second = storage.create("second")
    storage.add_block(blocker_id=first.id, blocked_id=second.id)
    result = await invoke(task_update_tool, task_id=first.id, status="deleted")
    assert result["success"] is True
    assert result["status_change"] == {"from": "pending", "to": "deleted"}
    assert storage.get(first.id) is None
    assert storage.get(second.id).blocked_by == []
    created = await invoke(task_create_tool, subject="third", description="details")
    assert created["task"] == {"id": "3", "subject": "third"}


@pytest.mark.asyncio
async def test_journal_publication_failure_cannot_leave_a_field_only_update(storage, monkeypatch):
    first = storage.create("original")
    second = storage.create("second")
    before = storage.list_all()
    from koder_agent.harness.tasks import storage as module

    real_write = module.write_text_atomic

    def fail_journal(path, text):
        if path.name == ".transaction":
            raise OSError("injected journal publication failure")
        return real_write(path, text)

    monkeypatch.setattr(module, "write_text_atomic", fail_journal)
    result = await invoke(task_update_tool, task_id=first.id, subject="new", add_blocks=[second.id])
    assert storage.list_all() == before
    assert_failure(result)
    assert not (storage.root / ".transaction").exists()


@pytest.mark.asyncio
async def test_interrupted_tool_update_recovers_the_whole_request(storage, monkeypatch):
    first = storage.create("original")
    second = storage.create("second")
    third = storage.create("third")
    real_write = storage._write_task

    def fail_second(task):
        if task.id == second.id:
            raise OSError("injected second write failure")
        real_write(task)

    monkeypatch.setattr(storage, "_write_task", fail_second)
    result = await invoke(
        task_update_tool,
        task_id=first.id,
        subject="new",
        status="completed",
        add_blocks=[second.id],
        add_blocked_by=[third.id],
    )
    assert_failure(result)
    assert result["outcome"] == "unknown"
    assert "recover" in result["error"].lower()
    recovered = TaskStorage(storage.root)
    current = recovered.get(first.id)
    assert current.title == "new"
    assert current.status == "completed"
    assert current.blocks == [second.id]
    assert current.blocked_by == [third.id]
    assert recovered.get(second.id).blocked_by == [first.id]
    assert recovered.get(third.id).blocks == [first.id]
    assert not (storage.root / ".transaction").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "delete"])
async def test_storage_exceptions_are_reported_without_success(storage, monkeypatch, operation):
    original = storage.create("original")

    def fail(*_args, **_kwargs):
        raise OSError("injected persistence failure")

    if operation == "create":
        monkeypatch.setattr(storage, "_write_task", fail)
        result = await invoke(task_create_tool, subject="new", description="")
    else:
        monkeypatch.setattr("koder_agent.harness.tasks.storage.write_text_atomic", fail)
        result = await invoke(task_update_tool, task_id=original.id, status="deleted")
    assert_failure(result)
    assert storage.list_all() == [original]


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["add_blocks", "add_blocked_by"])
@pytest.mark.parametrize("bad_id", ["1", "999"])
async def test_false_dependency_result_is_not_success(storage, direction, bad_id):
    original = storage.create("original")
    result = await invoke(task_update_tool, task_id=original.id, **{direction: [bad_id]})
    assert_failure(result)
    assert storage.get(original.id) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "deleted"])
async def test_status_receipt_uses_snapshot_after_lock_acquisition(storage, monkeypatch, status):
    original = storage.create("original")
    peer = TaskStorage(storage.root)
    real_with_lock = storage._with_lock

    def peer_changes_status_before_lock(fn):
        peer.update(original.id, status="in_progress")
        return real_with_lock(fn)

    monkeypatch.setattr(storage, "_with_lock", peer_changes_status_before_lock)
    result = await invoke(task_update_tool, task_id=original.id, status=status)
    assert result["success"] is True
    assert result["status_change"] == {"from": "in_progress", "to": status}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "deleted"])
async def test_subject_disappearing_before_lock_is_not_success(storage, monkeypatch, status):
    original = storage.create("original")
    peer = TaskStorage(storage.root)
    real_with_lock = storage._with_lock

    def peer_deletes_before_lock(fn):
        peer.delete(original.id)
        return real_with_lock(fn)

    monkeypatch.setattr(storage, "_with_lock", peer_deletes_before_lock)
    result = await invoke(task_update_tool, task_id=original.id, status=status)
    assert_failure(result)
    assert peer.list_all() == []


@pytest.mark.asyncio
async def test_repeated_update_reports_no_changes_and_does_not_write(storage, monkeypatch):
    first = storage.create("first", metadata={"keep": 1})
    second = storage.create("second")
    storage.add_block(blocker_id=first.id, blocked_id=second.id)
    before = storage.list_all()

    def fail(*_args, **_kwargs):
        raise AssertionError("No-op must not publish a journal or task write")

    monkeypatch.setattr("koder_agent.harness.tasks.storage.write_text_atomic", fail)
    result = await invoke(
        task_update_tool,
        task_id=first.id,
        subject=first.title,
        status="pending",
        metadata='{"keep": 1}',
        add_blocks=[second.id, second.id],
    )
    assert result == {"success": True, "task_id": first.id, "updated_fields": []}
    assert storage.list_all() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_task_status_domain_remains_usable(storage, status):
    first = storage.create("first")
    result = await invoke(task_update_tool, task_id=first.id, status=status)
    assert result["success"] is True
    assert storage.get(first.id).status == status


@pytest.mark.asyncio
async def test_lock_timeout_is_not_success(storage, monkeypatch):
    original = storage.create("original")

    def fail(_fn):
        raise Timeout(str(storage.root / ".lock"))

    monkeypatch.setattr(storage, "_with_lock", fail)
    result = await invoke(task_update_tool, task_id=original.id, subject="new")
    assert_failure(result)
    assert TaskStorage(storage.root).get(original.id) == original


@pytest.mark.asyncio
async def test_failed_delete_recovery_does_not_claim_success(storage, monkeypatch):
    first = storage.create("first")
    second = storage.create("second")
    third = storage.create("third")
    storage.add_block(blocker_id=first.id, blocked_id=second.id)
    storage.add_block(blocker_id=third.id, blocked_id=first.id)

    def fail(_task):
        raise OSError("injected deletion cleanup failure")

    monkeypatch.setattr(storage, "_write_task", fail)
    result = await invoke(task_update_tool, task_id=first.id, status="deleted")
    assert_failure(result)
    assert result["outcome"] == "unknown"
    recovered = TaskStorage(storage.root)
    assert recovered.get(first.id) is None
    assert recovered.get(second.id).blocked_by == []
    assert recovered.get(third.id).blocks == []


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_request", [{"add_blocks": ["1"]}, {"metadata": "[]"}])
async def test_error_payload_is_recognized_by_existing_display_classifier(storage, bad_request):
    from rich.console import Console

    from koder_agent.core.streaming_display import StreamingDisplayManager

    first = storage.create("first")
    result = await invoke(task_update_tool, task_id=first.id, **bad_request)
    assert_failure(result)
    display = StreamingDisplayManager(Console())
    # Pure classifier contract, not a claim of real terminal acceptance.
    assert display._is_error_output(json.dumps(result), "task_update")
