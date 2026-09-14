"""JSON-file cron job persistence."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from koder_agent.utils.atomic_file import write_text_atomic


@dataclass
class CronClaim:
    """Own a delivery until acknowledgement or release.

    The OS lock covers queueing and execution, not just the JSON mutation. A
    process exit releases it; the pending receipt remains available for retry.
    """

    storage: CronStorage
    job: dict[str, Any]
    minute: int
    _lock: FileLock

    def is_current(self) -> bool:
        return self._lock.is_locked and self.storage.get(str(self.job["id"])) == self.job

    def complete(self, *, delete_one_shot: bool = True) -> None:
        if not self._lock.is_locked:
            raise RuntimeError("Cron claim is no longer active")
        self.storage._complete_claim(self, delete_one_shot=delete_one_shot)

    def release(self) -> None:
        self._lock.release()


class CronStorage:
    """Stores cron jobs in a single JSON file.

    File format:
        {"tasks": [...], "runs": {"<id>": {"pending_minute": ..., "completed_minute": ...}}}

    Receipts are private to storage; list/get keep returning the job definition.
    Recovery is at-least-once for claimed deliveries, not exactly-once external
    effects. Minutes never observed by a running scheduler are not backfilled.
    """

    def __init__(self, path: Path, *, max_jobs: int = 50):
        self._path = path.resolve()
        self._max_jobs = max_jobs
        self._lock = FileLock(str(self._path) + ".lock", timeout=5)

    def _read_document(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"tasks": [], "runs": {}}
        data = json.loads(self._path.read_text(encoding="utf-8"))
        tasks = data.get("tasks", []) if isinstance(data, dict) else None
        if not isinstance(tasks, list) or any(not isinstance(task, dict) for task in tasks):
            raise ValueError(f"Invalid cron storage at {self._path}: expected a tasks list")
        data.setdefault("tasks", [])
        runs = data.setdefault("runs", {})
        if not isinstance(runs, dict) or any(
            not isinstance(state, dict) or any(type(value) is not int for value in state.values())
            for state in runs.values()
        ):
            raise ValueError(f"Invalid cron storage at {self._path}: invalid run receipts")
        return data

    def _read(self) -> list[dict[str, Any]]:
        return self._read_document()["tasks"]

    def _write_document(self, data: dict[str, Any]) -> None:
        write_text_atomic(self._path, json.dumps(data, indent=2, ensure_ascii=False))

    def _write(self, tasks: list[dict[str, Any]]) -> None:
        data = self._read_document()
        data["tasks"] = tasks
        ids = {task.get("id") for task in tasks}
        data["runs"] = {key: state for key, state in data["runs"].items() if key in ids}
        self._write_document(data)

    def pending_ids(self) -> set[str]:
        """Return previously claimed deliveries needing an acknowledgement."""
        return {
            job_id
            for job_id, state in self._read_document()["runs"].items()
            if "pending_minute" in state
        }

    def claim(self, job_id: str, *, minute: int | None = None) -> CronClaim | None:
        """Non-blockingly claim a stored job, recovering a pending occurrence first."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256(job_id.encode("utf-8")).hexdigest()
        execution_lock = FileLock(f"{self._path}.{digest}.run.lock", timeout=0)
        try:
            execution_lock.acquire()
        except Timeout:
            return None
        claimed = False
        try:
            with self._lock:
                data = self._read_document()
                job = next((job for job in data["tasks"] if job.get("id") == job_id), None)
                if job is None:
                    return None
                state = data["runs"].get(job_id, {})
                if minute is None:
                    minute = int(datetime.now(timezone.utc).timestamp()) // 60
                if "pending_minute" in state:
                    minute = state["pending_minute"]
                elif state.get("completed_minute", -1) >= minute:
                    return None
                data["runs"][job_id] = {**state, "pending_minute": minute}
                self._write_document(data)
            claimed = True
            return CronClaim(self, dict(job), minute, execution_lock)
        finally:
            if not claimed:
                execution_lock.release()

    def _complete_claim(self, claim: CronClaim, *, delete_one_shot: bool) -> None:
        with self._lock:
            data = self._read_document()
            job_id = str(claim.job["id"])
            job = next((job for job in data["tasks"] if job.get("id") == job_id), None)
            state = data["runs"].get(job_id, {})
            if job != claim.job or state.get("pending_minute") != claim.minute:
                # Deletion wins; a stale acknowledgement must not resurrect a job.
                return
            if delete_one_shot and not job.get("recurring", True):
                data["tasks"].remove(job)
                data["runs"].pop(job_id, None)
            else:
                data["runs"][job_id] = {"completed_minute": claim.minute}
            self._write_document(data)

    def create(
        self,
        *,
        cron: str,
        prompt: str,
        recurring: bool = True,
    ) -> dict[str, Any]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            tasks = self._read()
            if len(tasks) >= self._max_jobs:
                raise ValueError(
                    f"Job limit reached ({self._max_jobs}). Delete existing jobs first."
                )

            job: dict[str, Any] = {
                "id": uuid.uuid4().hex[:8],
                "cron": cron,
                "prompt": prompt,
                "recurring": recurring,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            tasks.append(job)
            self._write(tasks)
        return job

    def list_all(self) -> list[dict[str, Any]]:
        return self._read()

    def delete(self, job_id: str) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            tasks = self._read()
            filtered = [t for t in tasks if t.get("id") != job_id]
            if len(filtered) == len(tasks):
                return False
            self._write(filtered)
            return True

    def get(self, job_id: str) -> dict[str, Any] | None:
        for t in self._read():
            if t.get("id") == job_id:
                return t
        return None


_default_storage: CronStorage | None = None


def default_cron_storage() -> CronStorage:
    global _default_storage
    if _default_storage is None:
        root = Path.home() / ".koder"
        _default_storage = CronStorage(root / "scheduled_tasks.json")
    return _default_storage


def set_default_cron_storage(storage: CronStorage | None) -> None:
    global _default_storage
    _default_storage = storage
