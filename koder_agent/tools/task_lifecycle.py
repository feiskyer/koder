"""TaskCreate, TaskUpdate, TaskGet, TaskList tools."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional

from filelock import Timeout
from pydantic import BaseModel, Field

from koder_agent.harness.tasks.storage import TaskStorage

from .compat import function_tool

# --- Storage singleton ---

_storage: TaskStorage | None = None


def _get_task_storage() -> TaskStorage:
    global _storage
    if _storage is None:
        list_id = os.environ.get("KODER_TASK_LIST_ID", "default")
        root = Path.home() / ".koder" / "tasks" / list_id
        _storage = TaskStorage(root)
    return _storage


def _set_task_storage(storage: TaskStorage | None) -> None:
    """Override storage for testing."""
    global _storage
    _storage = storage


# --- Pydantic models ---


class TaskCreateModel(BaseModel):
    subject: str = Field(..., description="Brief task title")
    description: str = Field(..., description="What needs to be done")
    metadata: Optional[str] = Field(default=None, description="JSON string of key-value pairs")


class TaskUpdateModel(BaseModel):
    task_id: str = Field(..., description="Task ID to update")
    subject: Optional[str] = Field(default=None, description="Change task title")
    description: Optional[str] = Field(default=None, description="Change description")
    status: Optional[str] = Field(
        default=None, description="pending | in_progress | completed | failed | cancelled | deleted"
    )
    owner: Optional[str] = Field(default=None, description="Agent name for assignment")
    add_blocks: Optional[List[str]] = Field(
        default=None, description="Task IDs this task should block"
    )
    add_blocked_by: Optional[List[str]] = Field(
        default=None, description="Task IDs that should block this task"
    )
    metadata: Optional[str] = Field(
        default=None, description="JSON string to merge into existing metadata"
    )


class TaskGetModel(BaseModel):
    task_id: str = Field(..., description="Task ID to retrieve")


class TaskListModel(BaseModel):
    pass


# --- Plain implementations (directly callable, used by tests) ---


def _parse_metadata(metadata: str | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    parsed = json.loads(metadata)
    if not isinstance(parsed, dict):
        raise ValueError("Task metadata must be a JSON object")
    return parsed


def _task_error(error: Exception | str, *, task_id: str | None = None) -> str:
    # Keep the existing display classifier's textual error marker as well as
    # the machine-readable failure flag; no display-layer special case needed.
    result: dict[str, Any] = {"success": False, "error": f"Error: {error}"}
    if task_id is not None:
        result.update(task_id=task_id, updated_fields=[])
    if isinstance(error, (OSError, Timeout)):
        # The existing redo journal may already be committed. Do not encourage
        # blind retries (especially task_create) or claim that nothing changed.
        result["outcome"] = "unknown"
        result["error"] = (
            f"Storage operation failed: {error}. Changes may be pending recovery; "
            "inspect the task list before retrying."
        )
    return json.dumps(result)


def task_create(
    subject: str,
    description: str,
    metadata: Optional[str] = None,
) -> str:
    """Create a new task. Returns the task ID and subject.

    Args:
        subject: Brief task title.
        description: What needs to be done.
        metadata: Optional JSON string of key-value pairs, e.g. '{"priority": "high"}'.
    """
    try:
        meta_dict = _parse_metadata(metadata)
        storage = _get_task_storage()
        task = storage.create(subject, description=description, metadata=meta_dict)
    except (ValueError, TypeError, OSError, Timeout) as exc:
        return _task_error(exc)
    return json.dumps({"task": {"id": task.id, "subject": task.title}})


def task_update(
    task_id: str,
    subject: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
    owner: Optional[str] = None,
    add_blocks: Optional[List[str]] = None,
    add_blocked_by: Optional[List[str]] = None,
    metadata: Optional[str] = None,
) -> str:
    """Atomically update a task's fields, status, owner, and dependencies.

    Invalid metadata or dependencies reject the entire update. Deletion must
    be requested alone. Storage errors may require recovery; inspect task state
    before retrying. updated_fields lists only actual changes.

    Args:
        task_id: Task ID to update.
        subject: Change task title.
        description: Change description.
        status: pending | in_progress | completed | failed | cancelled | deleted.
        owner: Agent name for assignment.
        add_blocks: Task IDs this task should block.
        add_blocked_by: Task IDs that should block this task.
        metadata: JSON string to merge into existing metadata, e.g. '{"key": "value"}'.
    """
    try:
        meta_dict = _parse_metadata(metadata)
        storage = _get_task_storage()
        mutation = storage.update_with_dependencies(
            task_id,
            title=subject,
            description=description,
            status=None if status == "deleted" else status,
            owner=owner if owner is not None else ...,
            metadata=meta_dict,
            add_blocks=tuple(add_blocks or ()),
            add_blocked_by=tuple(add_blocked_by or ()),
            delete=status == "deleted",
        )
    except (ValueError, TypeError, OSError, Timeout) as exc:
        return _task_error(exc, task_id=task_id)
    if mutation is None:
        return _task_error("Task not found", task_id=task_id)

    before, after = mutation.before, mutation.after
    updated_fields = (
        ["status"]
        if after is None
        else [
            public_name
            for public_name, attribute in (
                ("subject", "title"),
                ("description", "description"),
                ("status", "status"),
                ("owner", "owner"),
                ("metadata", "metadata"),
                ("blocks", "blocks"),
                ("blocked_by", "blocked_by"),
            )
            if getattr(before, attribute) != getattr(after, attribute)
        ]
    )

    result: dict[str, Any] = {
        "success": True,
        "task_id": task_id,
        "updated_fields": updated_fields,
    }
    if "status" in updated_fields:
        result["status_change"] = {
            "from": before.status,
            "to": after.status if after is not None else "deleted",
        }

    return json.dumps(result)


def task_get(task_id: str) -> str:
    """Get a single task's details by ID.

    Args:
        task_id: ID of the task to fetch (as returned by task_create/task_list)
    """
    storage = _get_task_storage()
    task = storage.get(task_id)
    if task is None:
        return json.dumps({"task": None})

    return json.dumps(
        {
            "task": {
                "id": task.id,
                "subject": task.title,
                "description": task.description,
                "status": task.status,
                "owner": task.owner,
                "blocks": task.blocks,
                "blocked_by": task.blocked_by,
            }
        }
    )


def task_list() -> str:
    """List all tasks with their current status."""
    storage = _get_task_storage()
    tasks = storage.list_all(filter_resolved_blockers=True)
    return json.dumps(
        {
            "tasks": [
                {
                    "id": t.id,
                    "subject": t.title,
                    "status": t.status,
                    "owner": t.owner,
                    "blocked_by": t.blocked_by,
                }
                for t in tasks
            ]
        }
    )


# --- @function_tool wrappers for agent registration ---

task_create_tool = function_tool(task_create)
task_update_tool = function_tool(task_update)
task_get_tool = function_tool(task_get)
task_list_tool = function_tool(task_list)
