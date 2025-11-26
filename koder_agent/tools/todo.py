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


class TodoStore:
    """Singleton store for todos to ensure shared state across all agents."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._todos = []  # Initialize on first creation only
        return cls._instance

    @property
    def todos(self) -> List[dict]:
        return self._todos

    @todos.setter
    def todos(self, value: List[dict]):
        self._todos = value


# Global singleton instance - created once at module load
_store = TodoStore()


@function_tool
def todo_read() -> str:
    """Read all todos from the list."""
    if not _store.todos:
        return "No todos found. The list is empty."

    return _format_todo_list(_store.todos)


def _format_todo_list(todos: List[dict]) -> str:
    """Format todo list with emoji indicators."""
    result = []
    for todo in todos:
        status = todo.get("status", "pending")
        content = todo.get("content", "")

        if status == "completed":
            # Completed: green checkmark
            result.append(f"✅ {content}")
        elif status == "in_progress":
            # In progress: spinning/working indicator
            result.append(f"🔄 {content}")
        else:
            # Pending: waiting/todo indicator
            result.append(f"⏳ {content}")

    return "\n".join(result)


@function_tool
def todo_write(todos: List[TodoItem]) -> str:
    """Write/update the todo list."""
    # Convert TodoItem objects to dictionaries
    _store.todos = [todo.model_dump() for todo in todos]

    if not _store.todos:
        return "Todo list cleared."

    return _format_todo_list(_store.todos)
