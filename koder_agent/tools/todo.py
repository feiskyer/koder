"""Todo list management tools."""

from typing import List

from agents import function_tool
from pydantic import BaseModel


class TodoModel(BaseModel):
    pass


class TodoItem(BaseModel):
    content: str
    status: str
    priority: str
    id: str


class TodoWriteModel(BaseModel):
    todos: List[TodoItem]


# Global todo list storage (in production, this should be persistent)
_todos: List[dict] = []


@function_tool
def todo_read() -> str:
    """Read all todos from the list."""
    global _todos

    if not _todos:
        return "No todos found. The list is empty."

    result = []
    for todo in _todos:
        status = todo.get("status", "pending")
        content = todo.get("content", "")

        # Format with plain text symbols (no Rich markup to avoid color leaking)
        if status == "completed":
            # Completed tasks: checkmark
            result.append(f"  ✓ {content}")
        elif status == "in_progress":
            # Current task: arrow indicator
            result.append(f"  ▶ {content}")
        else:
            # Pending tasks: empty box
            result.append(f"  □ {content}")

    return "\n".join(result)


@function_tool
def todo_write(todos: List[TodoItem]) -> str:
    """Write/update the todo list."""
    global _todos

    # Convert TodoItem objects to dictionaries
    _todos = [todo.model_dump() for todo in todos]

    # Count items by status
    status_counts = {}
    for todo in _todos:
        status = todo.get("status", "pending")
        status_counts[status] = status_counts.get(status, 0) + 1

    # Create summary
    summary_parts = []
    for status, count in status_counts.items():
        summary_parts.append(f"{count} {status}")

    summary = f"Updated {len(todos)} todos: " + ", ".join(summary_parts)

    return summary
