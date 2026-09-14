import json
import logging
import os
import stat
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import patch

import pytest

from koder_agent.harness.memory.candidates import (
    CandidateStore,
    approve_candidate,
    memory_storage_scope,
    reject_candidate,
)
from koder_agent.harness.memory.governance import MAX_MEMORY_CONTENT_CHARS
from koder_agent.harness.memory.retrieval import retrieve_relevant_memories


def _memory_payload(content: str = "Prefer focused tests.") -> dict:
    return {
        "type": "project",
        "content": content,
        "description": "Testing preference",
    }


def _skill_payload() -> dict:
    return {
        "name": "focused-verification",
        "description": "Verify changes with focused tests first",
        "instructions": "Run focused tests before broader verification.",
    }


def _project_root(tmp_path: Path) -> Path:
    project_root = tmp_path / "origin-project"
    project_root.mkdir(parents=True, exist_ok=True)
    return project_root.resolve()


def _stage_memory(
    store: CandidateStore,
    tmp_path: Path,
    payload: dict | None = None,
    *,
    session_id: str = "candidate-test-session",
):
    candidate_payload = payload or _memory_payload()
    return store.stage(
        candidate_payload,
        storage_scope=memory_storage_scope(candidate_payload),
        origin_project_root=_project_root(tmp_path),
        origin_session_id=session_id,
    )


def _stage_skill(
    store: CandidateStore,
    tmp_path: Path,
    payload: dict | None = None,
    *,
    session_id: str = "candidate-test-session",
):
    return store.stage(
        payload or _skill_payload(),
        storage_scope="user",
        origin_project_root=_project_root(tmp_path),
        origin_session_id=session_id,
    )


def _claim_worker(root: str, candidate_id: str, ready, start, results) -> None:
    try:
        store = CandidateStore(Path(root), kind="memory")
        ready.set()
        if not start.wait(30):
            raise TimeoutError("candidate claim worker was not started")
        record = store.claim(candidate_id)
        results.put(record.state if record else "missing")
    except Exception as exc:  # pragma: no cover - asserted via process result
        results.put(f"error:{type(exc).__name__}:{exc}")


def _approve_worker(
    root: str,
    candidate_id: str,
    skill_draft_dir: str,
    ready,
    start,
    results,
) -> None:
    try:
        store = CandidateStore(Path(root), kind="memory")
        ready.set()
        if not start.wait(30):
            raise TimeoutError("candidate approval worker was not started")
        result = approve_candidate(
            candidate_id,
            memory_store=store,
            skill_store=None,
            skill_draft_dir=Path(skill_draft_dir),
        )
        results.put(result.status)
    except Exception as exc:  # pragma: no cover - asserted via process result
        results.put(f"error:{type(exc).__name__}:{exc}")


def _run_candidate_race(target, arguments):
    # The full suite has SQLite/executor threads: do not inherit their state
    # through fork. Wait for both fresh interpreters before testing contention.
    context = get_context("spawn")
    start = context.Event()
    results = context.Queue()
    readiness = [context.Event() for _ in range(2)]
    processes = [
        context.Process(target=target, args=(*arguments, ready, start, results))
        for ready in readiness
    ]
    try:
        for process in processes:
            process.start()
        for ready in readiness:
            assert ready.wait(10), "candidate worker did not finish startup"
        start.set()
        outcomes = [results.get(timeout=5) for _ in processes]
        for process in processes:
            process.join(timeout=5)
        assert all(process.exitcode == 0 for process in processes)
        return outcomes
    finally:
        unreaped = []
        for process in processes:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                if process.is_alive():
                    unreaped.append(process.pid)
                    continue
            process.close()
        results.close()
        results.join_thread()
        assert not unreaped, f"candidate workers did not stop: {unreaped}"


def test_candidate_list_show_reject_and_restart(tmp_path: Path):
    root = tmp_path / "memory-candidates"
    store = CandidateStore(root, kind="memory")
    candidate = _stage_memory(store, tmp_path)

    restarted = CandidateStore(root, kind="memory")
    assert [item.id for item in restarted.list()] == [candidate.id]
    assert restarted.get(candidate.id).payload["content"] == "Prefer focused tests."

    assert reject_candidate(candidate.id, memory_store=restarted, skill_store=None)
    assert restarted.list() == []
    assert restarted.get(candidate.id) is None


def test_memory_approval_is_restart_safe_and_not_duplicated(tmp_path: Path):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    candidate = _stage_memory(store, tmp_path)
    memory_dir = _project_root(tmp_path) / ".koder" / "memory"

    first = approve_candidate(
        candidate.id,
        memory_store=store,
        skill_store=None,
        skill_draft_dir=tmp_path / "skill-drafts",
    )
    second = approve_candidate(
        candidate.id,
        memory_store=CandidateStore(store.root, kind="memory"),
        skill_store=None,
        skill_draft_dir=tmp_path / "skill-drafts",
    )

    assert first.status == "approved"
    assert second.status == "already-approved"
    assert len(list(memory_dir.glob("*.md"))) == 1


def test_approval_resumes_claimed_candidate_after_restart(tmp_path: Path):
    root = tmp_path / "memory-candidates"
    store = CandidateStore(root, kind="memory")
    candidate = _stage_memory(store, tmp_path)
    claimed = store.claim(candidate.id)
    assert claimed is not None
    assert claimed.state == "approving"

    restarted = CandidateStore(root, kind="memory")
    result = approve_candidate(
        candidate.id,
        memory_store=restarted,
        skill_store=None,
        skill_draft_dir=tmp_path / "skill-drafts",
    )

    assert result.status == "approved"
    assert restarted.list() == []


def test_restart_hides_processing_record_when_approval_receipt_exists(tmp_path: Path):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    candidate = _stage_memory(store, tmp_path)

    with patch(
        "koder_agent.harness.memory.candidates.unlink_trusted_file",
        side_effect=OSError("interrupted cleanup"),
    ):
        with pytest.raises(OSError, match="interrupted cleanup"):
            approve_candidate(
                candidate.id,
                memory_store=store,
                skill_store=None,
                skill_draft_dir=tmp_path / "skill-drafts",
            )

    restarted = CandidateStore(store.root, kind="memory")
    assert restarted.list() == []
    result = approve_candidate(
        candidate.id,
        memory_store=restarted,
        skill_store=None,
        skill_draft_dir=tmp_path / "skill-drafts",
    )
    assert result.status == "already-approved"


def test_skill_candidate_is_separate_and_approval_creates_disabled_draft(tmp_path: Path):
    memory_store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    skill_store = CandidateStore(tmp_path / "skill-candidates", kind="skill")
    candidate = _stage_skill(skill_store, tmp_path)

    result = approve_candidate(
        candidate.id,
        memory_store=memory_store,
        skill_store=skill_store,
        skill_draft_dir=tmp_path / "skill-drafts",
    )

    assert result.status == "approved"
    assert result.output_path is not None
    assert result.output_path.name == "SKILL.md"
    draft = result.output_path.read_text(encoding="utf-8")
    assert "user-invocable: false" in draft
    assert "disable-model-invocation: true" in draft
    assert "allowed-tools: []" in draft
    assert "hooks:" not in draft
    assert not (_project_root(tmp_path) / ".koder" / "memory").exists()


def test_atomic_stage_failure_leaves_no_candidate_or_temp_file(tmp_path: Path):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")

    with patch("koder_agent.harness.memory.approved_writer.os.link", side_effect=OSError("boom")):
        with pytest.raises(OSError, match="boom"):
            _stage_memory(store, tmp_path)

    assert list(store.root.rglob("*.json")) == []
    assert list(store.root.rglob("*.tmp")) == []


def test_candidate_store_rejects_symlink_root_and_unsafe_ids(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "memory-candidates"
    root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        CandidateStore(root, kind="memory").list()

    safe = CandidateStore(tmp_path / "safe-candidates", kind="memory")
    with pytest.raises(ValueError, match="candidate id"):
        safe.get("../../escape")


def test_approval_rejects_symlink_output_root(tmp_path: Path):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    candidate = _stage_memory(store, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    memory_dir = _project_root(tmp_path) / ".koder" / "memory"
    memory_dir.parent.mkdir(parents=True)
    memory_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        approve_candidate(
            candidate.id,
            memory_store=store,
            skill_store=None,
            skill_draft_dir=tmp_path / "skill-drafts",
        )

    assert list(outside.iterdir()) == []


def test_candidate_logs_do_not_include_content_or_secrets(tmp_path: Path, caplog):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    secret = "sk-live-super-secret-value"
    store.pending_dir.mkdir(parents=True)
    (store.pending_dir / f"{secret}.json").write_text("{bad json", encoding="utf-8")

    with caplog.at_level(logging.INFO):
        with pytest.raises(ValueError, match="secret"):
            _stage_memory(store, tmp_path, _memory_payload(secret))
        store.list()

    assert secret not in caplog.text


def test_candidate_payload_change_invalidates_candidate_id(tmp_path: Path):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    candidate = _stage_memory(store, tmp_path)
    path = store.pending_dir / f"{candidate.id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["payload"]["content"] = "Changed after review."
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match candidate id"):
        store.get(candidate.id)
    with pytest.raises(ValueError, match="does not match candidate id"):
        approve_candidate(
            candidate.id,
            memory_store=store,
            skill_store=None,
            skill_draft_dir=tmp_path / "skill-drafts",
        )


def test_concurrent_claim_is_serialized(tmp_path: Path):
    root = tmp_path / "memory-candidates"
    candidate = _stage_memory(CandidateStore(root, kind="memory"), tmp_path)
    states = _run_candidate_race(_claim_worker, (str(root), candidate.id))

    assert states == ["approving", "approving"]
    assert not (root / "pending" / f"{candidate.id}.json").exists()
    assert (root / "processing" / f"{candidate.id}.json").exists()


def test_concurrent_approval_creates_one_receipt_and_one_output(tmp_path: Path):
    root = tmp_path / "memory-candidates"
    candidate = _stage_memory(CandidateStore(root, kind="memory"), tmp_path)
    statuses = _run_candidate_race(
        _approve_worker, (str(root), candidate.id, str(tmp_path / "skill-drafts"))
    )

    assert sorted(statuses) == ["already-approved", "approved"]
    assert len(list((_project_root(tmp_path) / ".koder" / "memory").glob("*.md"))) == 1
    assert len(list((root / "approved").glob("*.json"))) == 1


def test_duplicate_id_across_kinds_is_rejected(tmp_path: Path):
    memory_store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    skill_store = CandidateStore(tmp_path / "skill-candidates", kind="skill")
    candidate = _stage_memory(memory_store, tmp_path)
    skill_store.pending_dir.mkdir(parents=True)
    (skill_store.pending_dir / f"{candidate.id}.json").write_bytes(
        (memory_store.pending_dir / f"{candidate.id}.json").read_bytes()
    )

    with pytest.raises(ValueError, match="duplicate candidate id"):
        approve_candidate(
            candidate.id,
            memory_store=memory_store,
            skill_store=skill_store,
            skill_draft_dir=tmp_path / "skill-drafts",
        )


def test_memory_approval_cannot_modify_hardlinked_destination(tmp_path: Path):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    candidate = _stage_memory(store, tmp_path)
    outside = tmp_path / "outside-memory.md"
    outside.write_text("outside", encoding="utf-8")
    memory_dir = _project_root(tmp_path) / ".koder" / "memory"
    memory_dir.mkdir(parents=True)
    destination = memory_dir / f"auto-dream-{candidate.id}.md"
    os.link(outside, destination)

    with pytest.raises(FileExistsError, match="approved output conflict"):
        approve_candidate(
            candidate.id,
            memory_store=store,
            skill_store=None,
            skill_draft_dir=tmp_path / "skill-drafts",
        )

    assert outside.read_text(encoding="utf-8") == "outside"


def test_skill_approval_cannot_modify_hardlinked_destination(tmp_path: Path):
    store = CandidateStore(tmp_path / "skill-candidates", kind="skill")
    candidate = _stage_skill(store, tmp_path)
    outside = tmp_path / "outside-skill.md"
    outside.write_text("outside", encoding="utf-8")
    draft_dir = tmp_path / "skill-drafts"
    destination_dir = draft_dir / f"focused-verification-{candidate.id}"
    destination_dir.mkdir(parents=True)
    os.link(outside, destination_dir / "SKILL.md")

    with pytest.raises(FileExistsError, match="approved output conflict"):
        approve_candidate(
            candidate.id,
            memory_store=None,
            skill_store=store,
            skill_draft_dir=draft_dir,
        )

    assert outside.read_text(encoding="utf-8") == "outside"


def test_approval_parent_symlink_swap_cannot_escape_root(tmp_path: Path, monkeypatch):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    candidate = _stage_memory(store, tmp_path)
    memory_dir = _project_root(tmp_path) / ".koder" / "memory"
    memory_dir.mkdir(parents=True)
    moved = tmp_path / "trusted-moved"
    outside = tmp_path / "outside"
    outside.mkdir()

    from koder_agent.harness.memory import approved_writer

    original = approved_writer._validate_destination
    swapped = False

    def swap_after_parent_open(parent_fd, name, *, exclusive):
        nonlocal swapped
        if not swapped and name.endswith(".md"):
            memory_dir.rename(moved)
            memory_dir.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original(parent_fd, name, exclusive=exclusive)

    monkeypatch.setattr(approved_writer, "_validate_destination", swap_after_parent_open)
    approve_candidate(
        candidate.id,
        memory_store=store,
        skill_store=None,
        skill_draft_dir=tmp_path / "skill-drafts",
    )

    assert list(outside.iterdir()) == []
    assert len(list(moved.glob("*.md"))) == 1


def test_approved_output_is_atomic_and_private(tmp_path: Path):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    candidate = _stage_memory(store, tmp_path)
    result = approve_candidate(
        candidate.id,
        memory_store=store,
        skill_store=None,
        skill_draft_dir=tmp_path / "skill-drafts",
    )

    assert result.output_path is not None
    assert stat.S_IMODE(result.output_path.stat().st_mode) == 0o600
    assert "Prefer focused tests." in result.output_path.read_text(encoding="utf-8")
    assert list(result.output_path.parent.glob(".*.tmp")) == []


def test_rejected_candidate_is_not_restaged(tmp_path: Path):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    candidate = _stage_memory(store, tmp_path)
    assert store.reject(candidate.id)

    restaged = _stage_memory(store, tmp_path)

    assert restaged.state == "rejected"
    assert store.list() == []


def test_failed_processing_candidate_can_be_rejected(tmp_path: Path):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    candidate = _stage_memory(store, tmp_path)
    with patch(
        "koder_agent.harness.memory.candidates._write_memory_candidate",
        side_effect=OSError("persistence failed"),
    ):
        with pytest.raises(OSError, match="persistence failed"):
            approve_candidate(
                candidate.id,
                memory_store=store,
                skill_store=None,
                skill_draft_dir=tmp_path / "skill-drafts",
            )

    assert store.get(candidate.id).state == "approving"
    assert store.reject(candidate.id)
    assert store.list() == []


def test_rejection_survives_restart(tmp_path: Path):
    root = tmp_path / "memory-candidates"
    store = CandidateStore(root, kind="memory")
    candidate = _stage_memory(store, tmp_path)
    assert store.reject(candidate.id)

    restarted = CandidateStore(root, kind="memory")
    assert restarted.rejection_receipt(candidate.id) is not None
    assert _stage_memory(restarted, tmp_path).state == "rejected"


def test_oversized_candidate_is_rejected_before_write(tmp_path: Path):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    payload = _memory_payload("x" * (MAX_MEMORY_CONTENT_CHARS + 1))

    with pytest.raises(ValueError, match="exceeds limit"):
        _stage_memory(store, tmp_path, payload)

    assert not store.root.exists()


def test_oversized_candidate_file_is_not_loaded(tmp_path: Path):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    candidate_id = "a" * 64
    store.pending_dir.mkdir(parents=True)
    (store.pending_dir / f"{candidate_id}.json").write_bytes(b"{" + b"x" * 70_000)

    with pytest.raises(ValueError, match="size limit"):
        store.get(candidate_id)


def test_malformed_candidate_fields_are_rejected(tmp_path: Path):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")

    with pytest.raises(ValueError, match="malformed fields"):
        _stage_memory(store, tmp_path, {**_memory_payload(), "unknown": "value"})
    with pytest.raises(ValueError, match="must be a string"):
        _stage_memory(store, tmp_path, {**_memory_payload(), "content": 123})


def test_candidate_identity_binds_scope_project_and_session_provenance(tmp_path: Path):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    payload = _memory_payload("Origin-bound fact")
    project_a = _project_root(tmp_path)
    project_b = tmp_path / "other-project"
    project_b.mkdir()

    first = store.stage(
        payload,
        storage_scope="project",
        origin_project_root=project_a,
        origin_session_id="session-a",
    )
    other_session = store.stage(
        payload,
        storage_scope="project",
        origin_project_root=project_a,
        origin_session_id="session-b",
    )
    other_project = store.stage(
        payload,
        storage_scope="project",
        origin_project_root=project_b,
        origin_session_id="session-a",
    )

    assert len({first.id, other_session.id, other_project.id}) == 3
    persisted = json.loads((store.pending_dir / f"{first.id}.json").read_text(encoding="utf-8"))
    assert persisted["storage_scope"] == "project"
    assert persisted["origin_project_root"] == str(project_a)
    assert persisted["origin_session_id"] == "session-a"
    assert store.get(first.id).origin_project_root == str(project_a)

    with pytest.raises(ValueError, match="storage scope does not match"):
        store.stage(
            payload,
            storage_scope="user",
            origin_project_root=project_a,
            origin_session_id="session-c",
        )


def test_project_approval_stays_in_origin_workspace_and_user_memory_is_global(
    tmp_path: Path,
    monkeypatch,
):
    home = tmp_path / "home"
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"
    home.mkdir()
    project_a.mkdir()
    project_b.mkdir()
    monkeypatch.setenv("HOME", str(home))
    store = CandidateStore(home / ".koder" / "memory-candidates", kind="memory")
    project_payload = {
        "type": "project",
        "content": "projectonlyzxq",
        "description": "Origin project marker",
    }
    user_payload = {
        "type": "user",
        "content": "userglobalqvx",
        "description": "Global user marker",
    }
    project_candidate = store.stage(
        project_payload,
        storage_scope="project",
        origin_project_root=project_a,
        origin_session_id="project-a-session",
    )
    user_candidate = store.stage(
        user_payload,
        storage_scope="user",
        origin_project_root=project_a,
        origin_session_id="project-a-session",
    )

    project_result = approve_candidate(
        project_candidate.id,
        memory_store=store,
        skill_store=None,
    )
    user_result = approve_candidate(
        user_candidate.id,
        memory_store=store,
        skill_store=None,
    )

    assert project_result.output_path == (
        project_a / ".koder" / "memory" / f"auto-dream-{project_candidate.id}.md"
    )
    assert user_result.output_path == (
        home / ".koder" / "memory" / f"auto-dream-{user_candidate.id}.md"
    )
    assert not list((home / ".koder" / "memory").glob(f"*{project_candidate.id}*.md"))

    project_a_result = retrieve_relevant_memories(
        "projectonlyzxq",
        [project_a / ".koder" / "memory", home / ".koder" / "memory"],
        max_tokens=1000,
    )
    project_b_result = retrieve_relevant_memories(
        "projectonlyzxq",
        [project_b / ".koder" / "memory", home / ".koder" / "memory"],
        max_tokens=1000,
    )
    user_from_project_b = retrieve_relevant_memories(
        "userglobalqvx",
        [project_b / ".koder" / "memory", home / ".koder" / "memory"],
        max_tokens=1000,
    )
    assert [memory.path for memory in project_a_result.memories] == [project_result.output_path]
    assert project_b_result.memories == []
    assert [memory.path for memory in user_from_project_b.memories] == [user_result.output_path]


def test_distinct_skill_candidates_with_colliding_prefixes_do_not_overwrite(
    tmp_path: Path,
    monkeypatch,
):
    store = CandidateStore(tmp_path / "skill-candidates", kind="skill")
    first_payload = _skill_payload()
    second_payload = {**_skill_payload(), "instructions": "Run the second verification path."}
    first_id = "deadbeef" + "0" * 56
    second_id = "deadbeef" + "1" * 56

    def fake_candidate_id(_self, payload, **_provenance):
        return first_id if payload["instructions"] == first_payload["instructions"] else second_id

    monkeypatch.setattr(CandidateStore, "_candidate_id", fake_candidate_id)
    first = _stage_skill(store, tmp_path, first_payload)
    second = _stage_skill(store, tmp_path, second_payload)

    first_result = approve_candidate(
        first.id,
        memory_store=None,
        skill_store=store,
        skill_draft_dir=tmp_path / "skill-drafts",
    )
    first_content = first_result.output_path.read_text(encoding="utf-8")
    second_result = approve_candidate(
        second.id,
        memory_store=None,
        skill_store=store,
        skill_draft_dir=tmp_path / "skill-drafts",
    )

    assert first.id[:8] == second.id[:8]
    assert first_result.output_path != second_result.output_path
    assert first_result.output_path.parent.name.endswith(first.id)
    assert second_result.output_path.parent.name.endswith(second.id)
    assert first_result.output_path.read_text(encoding="utf-8") == first_content
    assert "Run the second verification path." in second_result.output_path.read_text(
        encoding="utf-8"
    )


def test_existing_identical_full_id_artifact_is_verified_after_interruption(tmp_path: Path):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    candidate = _stage_memory(store, tmp_path)

    with patch.object(
        store,
        "_finish_approval_unlocked",
        side_effect=OSError("receipt interrupted"),
    ):
        with pytest.raises(OSError, match="receipt interrupted"):
            approve_candidate(
                candidate.id,
                memory_store=store,
                skill_store=None,
            )

    existing = list((_project_root(tmp_path) / ".koder" / "memory").glob("*.md"))
    assert len(existing) == 1
    before = existing[0].read_bytes()
    result = approve_candidate(
        candidate.id,
        memory_store=CandidateStore(store.root, kind="memory"),
        skill_store=None,
    )

    assert result.status == "approved"
    assert result.output_path == existing[0]
    assert result.output_path.read_bytes() == before


def test_existing_full_id_artifact_with_unsafe_mode_is_not_accepted(tmp_path: Path):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    candidate = _stage_memory(store, tmp_path)

    with patch.object(
        store,
        "_finish_approval_unlocked",
        side_effect=OSError("receipt interrupted"),
    ):
        with pytest.raises(OSError, match="receipt interrupted"):
            approve_candidate(candidate.id, memory_store=store, skill_store=None)

    artifact = next((_project_root(tmp_path) / ".koder" / "memory").glob("*.md"))
    artifact.chmod(0o644)
    with pytest.raises(FileExistsError, match="approved output conflict"):
        approve_candidate(
            candidate.id,
            memory_store=CandidateStore(store.root, kind="memory"),
            skill_store=None,
        )


def test_approved_receipt_cleanup_failure_is_reaped_and_excluded_from_queue(
    tmp_path: Path,
    monkeypatch,
):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    candidate = _stage_memory(store, tmp_path)

    with patch(
        "koder_agent.harness.memory.candidates.unlink_trusted_file",
        side_effect=OSError("cleanup interrupted"),
    ):
        with pytest.raises(OSError, match="cleanup interrupted"):
            approve_candidate(candidate.id, memory_store=store, skill_store=None)

    processing = store.processing_dir / f"{candidate.id}.json"
    assert processing.exists()
    assert (store.approved_dir / f"{candidate.id}.json").exists()
    monkeypatch.setattr("koder_agent.harness.memory.candidates.MAX_CANDIDATE_COUNT", 1)

    with patch(
        "koder_agent.harness.memory.candidates.unlink_trusted_file",
        side_effect=OSError("cleanup still interrupted"),
    ):
        restarted = CandidateStore(store.root, kind="memory")
        replacement = _stage_memory(
            restarted,
            tmp_path,
            _memory_payload("replacement pending candidate"),
            session_id="replacement-session",
        )
        assert processing.exists()
        assert restarted._queue_usage()[0] == 1
        assert (
            approve_candidate(candidate.id, memory_store=restarted, skill_store=None).status
            == "already-approved"
        )
        assert replacement.state == "pending"

    assert [record.id for record in restarted.list()] == [replacement.id]
    assert not processing.exists()


def test_rejected_receipt_cleanup_failure_is_reaped_and_excluded_from_queue(
    tmp_path: Path,
    monkeypatch,
):
    store = CandidateStore(tmp_path / "memory-candidates", kind="memory")
    candidate = _stage_memory(store, tmp_path)

    with patch(
        "koder_agent.harness.memory.candidates.unlink_trusted_file",
        side_effect=OSError("cleanup interrupted"),
    ):
        with pytest.raises(OSError, match="cleanup interrupted"):
            store.reject(candidate.id)

    pending = store.pending_dir / f"{candidate.id}.json"
    assert pending.exists()
    assert (store.rejected_dir / f"{candidate.id}.json").exists()
    monkeypatch.setattr("koder_agent.harness.memory.candidates.MAX_CANDIDATE_COUNT", 1)

    with patch(
        "koder_agent.harness.memory.candidates.unlink_trusted_file",
        side_effect=OSError("cleanup still interrupted"),
    ):
        restarted = CandidateStore(store.root, kind="memory")
        replacement = _stage_memory(
            restarted,
            tmp_path,
            _memory_payload("replacement after rejection"),
            session_id="replacement-session",
        )
        assert pending.exists()
        assert restarted._queue_usage()[0] == 1
        assert reject_candidate(candidate.id, memory_store=restarted, skill_store=None)
        assert replacement.state == "pending"

    assert [record.id for record in restarted.list()] == [replacement.id]
    assert not pending.exists()
