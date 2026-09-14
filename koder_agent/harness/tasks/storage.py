"""File-based task persistence under ~/.koder/tasks/."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_args

from filelock import FileLock

from koder_agent.utils.atomic_file import write_text_atomic

from .models import TaskRecord, TaskStatus


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_metadata(metadata: dict[str, Any] | None) -> None:
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError("Task metadata must be a JSON object")
        # Validate before allocating an ID or publishing any part of an update.
        json.dumps(metadata, allow_nan=False)


@dataclass(frozen=True)
class TaskMutation:
    """Authoritative snapshots from one locked update (after=None for deletion)."""

    before: TaskRecord
    after: TaskRecord | None


class TaskStorage:
    """JSON-file-per-task storage with file locking and recoverable graph edits.

    Directory layout:
        <root>/
            <id>.json          # One file per task
            .highwatermark     # Max ID ever assigned
            .transaction       # Committed multi-record intent awaiting replay

    Reads and writes participate in the same lock. A graph edit commits when its
    journal is published; interrupted edits roll forward before the next public
    operation. Callers must not infer rollback from an I/O exception after that
    point. Individual task files remain the storage format.
    """

    def __init__(self, root: Path):
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self._root / ".lock"), timeout=5)

    @property
    def root(self) -> Path:
        return self._root

    def _hwm_path(self) -> Path:
        return self._root / ".highwatermark"

    def _read_hwm(self) -> int:
        p = self._hwm_path()
        if p.exists():
            return int(p.read_text().strip())
        return 0

    def _write_hwm(self, value: int) -> None:
        write_text_atomic(self._hwm_path(), str(value))

    def _next_id(self) -> str:
        live_ids = [
            int(path.stem)
            for path in self._root.glob("*.json")
            if re.fullmatch(r"[1-9][0-9]*", path.stem)
        ]
        hwm = max(self._read_hwm(), *live_ids, 0) + 1
        self._write_hwm(hwm)
        return str(hwm)

    def _task_path(self, task_id: str) -> Path:
        if not isinstance(task_id, str) or re.fullmatch(r"[1-9][0-9]*", task_id) is None:
            raise ValueError("Invalid task ID: expected a positive ASCII integer")
        path = self._root / f"{task_id}.json"
        if path.is_symlink():
            raise ValueError("Task storage does not allow symbolic link targets")
        return path

    def _read_task(self, task_id: str) -> TaskRecord | None:
        p = self._task_path(task_id)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        task = TaskRecord.from_dict(data)
        if task.id != task_id:
            raise ValueError("Task ID does not match its storage path")
        return task

    def _write_task(self, task: TaskRecord) -> None:
        p = self._task_path(task.id)
        write_text_atomic(p, json.dumps(task.to_dict(), indent=2, ensure_ascii=False))

    def _with_lock(self, fn):
        """Execute fn while holding an exclusive lock on the storage dir."""
        with self._lock:
            self._recover_transaction()
            return fn()

    def _recover_transaction(self) -> None:
        journal = self._root / ".transaction"
        if journal.is_symlink():
            raise ValueError("Task transaction does not allow symbolic link targets")
        try:
            data = json.loads(journal.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        if (
            not isinstance(data, dict)
            or data.get("version") != 1
            or not isinstance(data.get("writes"), list)
            or not isinstance(data.get("deletes"), list)
        ):
            raise ValueError("Invalid task transaction journal")
        tasks = [TaskRecord.from_dict(item) for item in data["writes"]]
        # Validate the entire intent before touching any of its destinations.
        for task in tasks:
            self._task_path(task.id)
        deleted_paths = [self._task_path(task_id) for task_id in data["deletes"]]
        for task in tasks:
            self._write_task(task)
        for path in deleted_paths:
            path.unlink(missing_ok=True)
        journal.unlink()

    def _commit_graph_edit(self, tasks: list[TaskRecord], *, deletes: list[str]) -> None:
        if not tasks and not deletes:
            return
        for task in tasks:
            self._task_path(task.id)
        for task_id in deletes:
            self._task_path(task_id)
        data = {
            "version": 1,
            "writes": [task.to_dict() for task in tasks],
            "deletes": deletes,
        }
        write_text_atomic(self._root / ".transaction", json.dumps(data, ensure_ascii=False))
        self._recover_transaction()

    def create(
        self,
        title: str,
        *,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord:
        _validate_metadata(metadata)

        def _do():
            task_id = self._next_id()
            task = TaskRecord.create(
                task_id=task_id,
                title=title,
                description=description,
                metadata=metadata,
            )
            self._write_task(task)
            return task

        return self._with_lock(_do)

    def get(self, task_id: str) -> TaskRecord | None:
        return self._with_lock(lambda: self._read_task(task_id))

    def list_all(self, *, filter_resolved_blockers: bool = False) -> list[TaskRecord]:
        return self._with_lock(
            lambda: self._list_all(filter_resolved_blockers=filter_resolved_blockers)
        )

    def _list_all(self, *, filter_resolved_blockers: bool = False) -> list[TaskRecord]:
        tasks = []
        for p in sorted(self._root.glob("*.json")):
            task = self._read_task(p.stem)
            if task is not None:
                tasks.append(task)

        if filter_resolved_blockers:
            completed_ids = {t.id for t in tasks if t.status == "completed"}
            filtered = []
            for t in tasks:
                if t.blocked_by:
                    active_blockers = [b for b in t.blocked_by if b not in completed_ids]
                    t = replace(t, blocked_by=active_blockers)
                filtered.append(t)
            tasks = filtered

        return tasks

    def update(
        self,
        task_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        status: TaskStatus | None = None,
        owner: str | None = ...,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord | None:
        result = self.update_with_dependencies(
            task_id,
            title=title,
            description=description,
            status=status,
            owner=owner,
            metadata=metadata,
        )
        return result.after if result is not None else None

    def update_with_dependencies(
        self,
        task_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        status: TaskStatus | None = None,
        owner: str | None = ...,
        metadata: dict[str, Any] | None = None,
        add_blocks: tuple[str, ...] = (),
        add_blocked_by: tuple[str, ...] = (),
        delete: bool = False,
    ) -> TaskMutation | None:
        """Validate then commit an entire logical request under one directory lock.

        None means the subject is absent; invalid requests raise before mutation.
        The returned snapshots describe this operation, not a separate racy read.
        A published intent rolls forward after I/O failure, exactly as for
        add_block/delete. Such exceptions do not imply rollback.
        """
        _validate_metadata(metadata)
        if status is not None and status not in get_args(TaskStatus):
            raise ValueError(f"Invalid task status: {status}")
        if delete and (
            title is not None
            or description is not None
            or status is not None
            or owner is not ...
            or metadata is not None
            or add_blocks
            or add_blocked_by
        ):
            raise ValueError("Task deletion cannot be combined with other changes")

        def _do():
            task = self._read_task(task_id)
            if task is None:
                return None
            if delete:
                self._delete_task(task_id)
                return TaskMutation(task, None)

            # Load every endpoint before constructing or publishing any changes.
            records = {task_id: task}
            edges = [(task_id, other) for other in add_blocks]
            edges.extend((other, task_id) for other in add_blocked_by)
            for blocker_id, blocked_id in edges:
                if blocker_id == blocked_id:
                    raise ValueError("A task cannot depend on itself")
                for endpoint in (blocker_id, blocked_id):
                    if endpoint not in records:
                        other = self._read_task(endpoint)
                        if other is None:
                            raise ValueError(f"Dependency task not found: {endpoint}")
                        records[endpoint] = other

            changes: dict[str, Any] = {}
            if title is not None:
                changes["title"] = title
            if description is not None:
                changes["description"] = description
            if status is not None:
                changes["status"] = status
            if owner is not ...:
                changes["owner"] = owner
            if metadata is not None:
                merged = dict(task.metadata)
                for k, v in metadata.items():
                    if v is None:
                        merged.pop(k, None)
                    else:
                        merged[k] = v
                changes["metadata"] = merged

            updated = dict(records)
            updated[task_id] = replace(task, **changes)
            for blocker_id, blocked_id in edges:
                blocker = updated[blocker_id]
                blocked = updated[blocked_id]
                if blocked_id not in blocker.blocks:
                    updated[blocker_id] = replace(blocker, blocks=[*blocker.blocks, blocked_id])
                if blocker_id not in blocked.blocked_by:
                    updated[blocked_id] = replace(
                        blocked, blocked_by=[*blocked.blocked_by, blocker_id]
                    )
            timestamp = _utc_now_iso()
            writes = []
            for key, record in updated.items():
                if record != records[key]:
                    updated[key] = replace(record, updated_at=timestamp)
                    writes.append(updated[key])
            self._commit_graph_edit(writes, deletes=[])
            return TaskMutation(task, updated[task_id])

        return self._with_lock(_do)

    def delete(self, task_id: str) -> bool:
        return self.update_with_dependencies(task_id, delete=True) is not None

    def _delete_task(self, task_id: str) -> None:
        """Delete a known task and its edges while the caller owns the lock."""
        hwm = self._read_hwm()
        tid = int(task_id)
        if tid > hwm:
            self._write_hwm(tid)
        updates = []
        for other_path in self._root.glob("*.json"):
            other = self._read_task(other_path.stem)
            if other is None or other.id == task_id:
                continue
            if task_id in other.blocks or task_id in other.blocked_by:
                updates.append(
                    replace(
                        other,
                        blocks=[item for item in other.blocks if item != task_id],
                        blocked_by=[item for item in other.blocked_by if item != task_id],
                        updated_at=_utc_now_iso(),
                    )
                )
        self._commit_graph_edit(updates, deletes=[task_id])

    def add_block(self, *, blocker_id: str, blocked_id: str) -> bool:
        def _do():
            blocker = self._read_task(blocker_id)
            blocked = self._read_task(blocked_id)
            if blocker is None or blocked is None or blocker_id == blocked_id:
                return False

            updates = []
            if blocked_id not in blocker.blocks:
                updated_blocker = replace(
                    blocker,
                    blocks=[*blocker.blocks, blocked_id],
                    updated_at=_utc_now_iso(),
                )
                updates.append(updated_blocker)
            if blocker_id not in blocked.blocked_by:
                updated_blocked = replace(
                    blocked,
                    blocked_by=[*blocked.blocked_by, blocker_id],
                    updated_at=_utc_now_iso(),
                )
                updates.append(updated_blocked)
            self._commit_graph_edit(updates, deletes=[])
            return True

        return self._with_lock(_do)
