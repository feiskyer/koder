"""Filesystem-backed shared task lists for agent teams."""

from __future__ import annotations

import json
import shutil
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ContextManager

from filelock import FileLock

from koder_agent.harness.agents.hooks import dispatch_project_hook_event
from koder_agent.utils.atomic_file import write_text_atomic

from .runtime import default_tasks_root

TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
TASK_STATES = TERMINAL_STATES | {"pending", "in_progress"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)


@dataclass(frozen=True)
class TeamTaskRecord:
    """A persisted team task entry."""

    id: str
    subject: str
    description: str
    status: str
    owner: str | None
    blocks: list[str]
    blocked_by: list[str]
    active_form: str | None
    metadata: dict[str, Any] | None
    created_at: str
    updated_at: str
    claim_id: str | None = None
    revision: str = ""

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        subject: str,
        description: str = "",
        status: str = "pending",
        owner: str | None = None,
        blocks: list[str] | None = None,
        blocked_by: list[str] | None = None,
        active_form: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "TeamTaskRecord":
        timestamp = _utc_now_iso()
        return cls(
            id=task_id,
            subject=subject,
            description=description,
            status=status,
            owner=owner,
            blocks=list(blocks or []),
            blocked_by=list(blocked_by or []),
            active_form=active_form,
            metadata=dict(metadata or {}) or None,
            created_at=timestamp,
            updated_at=timestamp,
        )


@dataclass(frozen=True)
class ClaimTaskResult:
    """Outcome of claiming a team task."""

    success: bool
    reason: str | None = None
    task: TeamTaskRecord | None = None
    blocked_by_tasks: list[str] | None = None
    busy_with_tasks: list[str] | None = None


class TeamTaskService:
    """Source-backed shared task list stored under `~/.koder/tasks/<team>/`."""

    def __init__(
        self,
        team_name: str,
        *,
        root: Path | None = None,
        cwd: str | Path | None = None,
        lifecycle_guard: Callable[[], ContextManager] | None = None,
        claimant_is_active: Callable[[str], bool] | None = None,
    ):
        self.team_name = team_name
        self.root = (root or default_tasks_root()).expanduser()
        self.cwd = Path(cwd or Path.cwd())
        storage_name = _sanitize(team_name)
        if not storage_name:
            raise ValueError("Invalid team identifier")
        self.task_dir = self.root / storage_name
        # Keep the lock inode outside the removable directory.
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.root / f".{storage_name}.lock"
        self._lock = FileLock(str(self.lock_path), timeout=5)
        self._lifecycle_guard = lifecycle_guard or nullcontext
        self._claimant_is_active = claimant_is_active
        with self._lifecycle_guard(), self._lock:
            self.task_dir.mkdir(parents=True, exist_ok=True)
            generation_path = self.task_dir / ".generation"
            if not generation_path.exists():
                write_text_atomic(generation_path, uuid.uuid4().hex)
            self._generation = generation_path.read_text(encoding="utf-8")

    @classmethod
    def for_test(
        cls,
        team_name: str,
        *,
        root: Path,
        cwd: str | Path | None = None,
    ) -> "TeamTaskService":
        return cls(team_name, root=root, cwd=cwd or root)

    def cleanup(self) -> None:
        with self._transaction():
            shutil.rmtree(self.task_dir)

    @contextmanager
    def _transaction(self):
        with self._lifecycle_guard(), self._lock:
            try:
                generation = (self.task_dir / ".generation").read_text(encoding="utf-8")
            except FileNotFoundError:
                raise KeyError(self.team_name) from None
            if generation != self._generation:
                raise KeyError(self.team_name)
            yield

    def _task_path(self, task_id: str) -> Path:
        return self.task_dir / f"{_sanitize(task_id)}.json"

    def _read_task(self, path: Path) -> TeamTaskRecord | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return TeamTaskRecord(**data)

    def _write_task(self, task: TeamTaskRecord) -> TeamTaskRecord:
        task = replace(task, revision=uuid.uuid4().hex)
        write_text_atomic(
            self._task_path(task.id), json.dumps(task.__dict__, indent=2, ensure_ascii=False)
        )
        return task

    def get_task(self, task_id: str) -> TeamTaskRecord | None:
        with self._transaction():
            return self._read_task(self._task_path(task_id))

    def list_tasks(self) -> list[TeamTaskRecord]:
        with self._transaction():
            return self._list_tasks()

    def _list_tasks(self) -> list[TeamTaskRecord]:
        tasks: list[TeamTaskRecord] = []
        for path in sorted(self.task_dir.glob("*.json"), key=lambda item: int(item.stem)):
            task = self._read_task(path)
            if task is not None:
                tasks.append(task)
        return tasks

    def create_task(
        self,
        subject: str,
        *,
        description: str = "",
        blocked_by: list[str] | None = None,
        active_form: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TeamTaskRecord:
        with self._transaction():
            existing = self._list_tasks()
            hwm_path = self.task_dir / ".highwatermark"
            hwm = int(hwm_path.read_text()) if hwm_path.exists() else 0
            next_id = str(max([hwm, *(int(task.id) for task in existing)]) + 1)
            # Reserve the ID, not a runnable task, before invoking external hooks.
            write_text_atomic(hwm_path, next_id)
            task = TeamTaskRecord.create(
                task_id=next_id,
                subject=subject,
                description=description,
                blocked_by=blocked_by,
                active_form=active_form,
                metadata=metadata,
            )
        hook_result = dispatch_project_hook_event(
            cwd=self.cwd,
            event_name="TaskCreated",
            match_value=task.id,
            payload={
                "event": "TaskCreated",
                "team_name": self.team_name,
                "task_id": task.id,
                "subject": task.subject,
                "description": task.description,
                "blocked_by": task.blocked_by,
            },
        )
        if getattr(hook_result, "blocked", False):
            raise RuntimeError(hook_result.block_reason or "Task creation blocked by hook")
        with self._transaction():
            return self._write_task(task)

    def update_task(
        self,
        task_id: str,
        *,
        subject: str | None = None,
        description: str | None = None,
        status: str | None = None,
        owner: str | None = None,
        blocks: list[str] | None = None,
        blocked_by: list[str] | None = None,
        active_form: str | None = None,
        metadata: dict[str, Any] | None = None,
        expected_claim: tuple[str, str] | None = None,
    ) -> TeamTaskRecord:
        """Edit a task; pending explicitly retries/releases its previous attempt.

        Omitted/None owner retains ownership except on an explicit pending reset,
        which clears it. A supplied owner reserves the pending task for that owner.
        Completion without expected_claim remains the administrative public API;
        executors must supply the owner and claim ID they actually ran.
        """
        if status is not None and status not in TASK_STATES:
            raise ValueError(f"Invalid task status: {status}")
        with self._transaction():
            existing = self._read_task(self._task_path(task_id))
            if existing is None:
                raise KeyError(task_id)
            if expected_claim is not None and (
                existing.status != "in_progress"
                or (existing.owner, existing.claim_id) != expected_claim
            ):
                raise RuntimeError("Task claim is no longer current")
            if existing.status in TERMINAL_STATES and status not in {
                None,
                existing.status,
                "pending",
            }:
                raise ValueError("Terminal task must be reset to pending before retry")
            next_status = status if status is not None else existing.status
            next_owner = owner if owner is not None else existing.owner
            claim_id = existing.claim_id
            if status == "pending":
                next_owner = owner or None
                claim_id = None
            elif owner is not None and owner != existing.owner:
                # Reassignment invalidates an executing attempt, never its outcome.
                claim_id = None
                if existing.status == "in_progress" and next_status == "in_progress":
                    next_status = "pending"
            if next_status == "in_progress" and claim_id is None:
                claim_id = uuid.uuid4().hex
            if next_status in TERMINAL_STATES:
                claim_id = None
            updated = replace(
                existing,
                subject=subject if subject is not None else existing.subject,
                description=description if description is not None else existing.description,
                status=next_status,
                owner=next_owner,
                claim_id=claim_id,
                blocks=list(blocks if blocks is not None else existing.blocks),
                blocked_by=list(blocked_by if blocked_by is not None else existing.blocked_by),
                active_form=active_form if active_form is not None else existing.active_form,
                metadata=dict(metadata if metadata is not None else (existing.metadata or {}))
                or None,
                updated_at=_utc_now_iso(),
            )
            if existing.status == "completed" or updated.status != "completed":
                return self._write_task(updated)
        # Hooks see the old state. A rejection never rolls back somebody else's
        # newer write; an accepted hook still needs a fresh snapshot comparison.
        hook_result = dispatch_project_hook_event(
            cwd=self.cwd,
            event_name="TaskCompleted",
            match_value=task_id,
            payload={
                "event": "TaskCompleted",
                "team_name": self.team_name,
                "task_id": updated.id,
                "subject": updated.subject,
                "owner": updated.owner,
            },
        )
        if getattr(hook_result, "blocked", False):
            raise RuntimeError(hook_result.block_reason or "Task completion blocked by hook")
        with self._transaction():
            if self._read_task(self._task_path(task_id)) != existing:
                raise RuntimeError("Task changed during completion hook")
            return self._write_task(updated)

    def finish_claim(self, task_id: str, owner: str, claim_id: str, status: str) -> TeamTaskRecord:
        """Publish only the terminal outcome of this exact execution attempt."""
        if status not in TERMINAL_STATES:
            raise ValueError("Claim outcome must be terminal")
        if not claim_id:
            raise ValueError("Claim outcome requires a claim ID")
        return self.update_task(task_id, status=status, expected_claim=(owner, claim_id))

    def cancel_owner_tasks(self, owner: str) -> None:
        """Retain revoked work as cancelled; a new attempt requires explicit retry."""
        with self._transaction():
            for task in self._list_tasks():
                if task.owner == owner and task.status == "in_progress":
                    self._write_task(
                        replace(task, status="cancelled", claim_id=None, updated_at=_utc_now_iso())
                    )

    def update_status(self, task_id: str, status: str) -> TeamTaskRecord:
        return self.update_task(task_id, status=status)

    def block_task(self, from_task_id: str, to_task_id: str) -> bool:
        with self._transaction():
            source = self._read_task(self._task_path(from_task_id))
            target = self._read_task(self._task_path(to_task_id))
            if source is None or target is None:
                return False
            if to_task_id not in source.blocks:
                source = replace(
                    source,
                    blocks=[*source.blocks, to_task_id],
                    updated_at=_utc_now_iso(),
                )
                self._write_task(source)
            if from_task_id not in target.blocked_by:
                target = replace(
                    target,
                    blocked_by=[*target.blocked_by, from_task_id],
                    updated_at=_utc_now_iso(),
                )
                self._write_task(target)
            return source is not None

    def claim_task(
        self,
        task_id: str,
        claimant_agent_id: str,
        *,
        check_agent_busy: bool = False,
    ) -> ClaimTaskResult:
        with self._transaction():
            if self._claimant_is_active is not None and not self._claimant_is_active(
                claimant_agent_id
            ):
                return ClaimTaskResult(success=False, reason="inactive_member")
            tasks = self._list_tasks()
            task = next((item for item in tasks if item.id == task_id), None)
            if task is None:
                return ClaimTaskResult(success=False, reason="task_not_found")
            if task.status in TERMINAL_STATES:
                return ClaimTaskResult(success=False, reason="already_resolved", task=task)
            if task.owner and task.owner != claimant_agent_id:
                return ClaimTaskResult(success=False, reason="already_claimed", task=task)
            if task.status == "in_progress" and task.claim_id and task.owner == claimant_agent_id:
                # Public claims are idempotent, not implicit retries/new attempts.
                return ClaimTaskResult(success=True, task=task)
            unresolved = {item.id for item in tasks if item.status != "completed"}
            blockers = [item for item in task.blocked_by if item in unresolved]
            if blockers:
                return ClaimTaskResult(
                    success=False,
                    reason="blocked",
                    task=task,
                    blocked_by_tasks=blockers,
                )
            if check_agent_busy:
                busy = [
                    item.id
                    for item in tasks
                    if item.owner == claimant_agent_id
                    and item.id != task_id
                    and item.status == "in_progress"
                ]
                if busy:
                    return ClaimTaskResult(
                        success=False,
                        reason="agent_busy",
                        task=task,
                        busy_with_tasks=busy,
                    )
            claimed = replace(
                task,
                status="in_progress",
                owner=claimant_agent_id,
                claim_id=uuid.uuid4().hex,
                updated_at=_utc_now_iso(),
            )
            claimed = self._write_task(claimed)
            return ClaimTaskResult(success=True, task=claimed)
