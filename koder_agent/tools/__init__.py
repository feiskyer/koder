"""Tool implementations for Koder Agent."""

from typing import List

from agents import Tool

from .engine import ToolEngine
from .file_ops import (
    FileReadModel,
    FileWriteModel,
    LSModel,
    append_file,
    list_directory,
    read_file,
    write_file,
)
from .search_ops import GlobModel, GrepModel, glob_search, grep_search
from .shell_ops import GitModel, ShellModel, git_command, run_shell
from .task_ops import TaskDelegateModel, TaskModel, task_delegate
from .todo_ops import TodoModel, TodoWriteModel, todo_read, todo_write
from .web_ops import SearchModel, WebFetchModel, web_fetch, web_search

# Create the global tool engine
tool_engine = ToolEngine()

# Register all tools
tool_engine.register(FileReadModel)(read_file)
tool_engine.register(FileWriteModel)(write_file)
tool_engine.register(FileWriteModel)(append_file)
tool_engine.register(ShellModel)(run_shell)
tool_engine.register(SearchModel)(web_search)
tool_engine.register(GlobModel)(glob_search)
tool_engine.register(GrepModel)(grep_search)
tool_engine.register(LSModel)(list_directory)
tool_engine.register(TodoModel)(todo_read)
tool_engine.register(TodoWriteModel)(todo_write)
tool_engine.register(WebFetchModel)(web_fetch)
tool_engine.register(TaskDelegateModel)(task_delegate)
tool_engine.register(GitModel)(git_command)


def get_all_tools() -> List[Tool]:
    """Get all registered tools as a list."""
    # Collect all @function_tool decorated functions directly
    tools = [
        read_file,
        write_file,
        append_file,
        run_shell,
        git_command,
        web_search,
        web_fetch,
        glob_search,
        grep_search,
        list_directory,
        todo_read,
        todo_write,
        task_delegate,
    ]

    # Filter to only include properly decorated tools
    return [tool for tool in tools if hasattr(tool, "name")]


__all__ = [
    "tool_engine",
    "get_all_tools",
    # Models
    "FileReadModel",
    "FileWriteModel",
    "LSModel",
    "ShellModel",
    "GitModel",
    "SearchModel",
    "WebFetchModel",
    "GlobModel",
    "GrepModel",
    "TodoModel",
    "TodoWriteModel",
    "TaskModel",
    "TaskDelegateModel",
    # Functions
    "read_file",
    "write_file",
    "append_file",
    "run_shell",
    "git_command",
    "web_search",
    "web_fetch",
    "glob_search",
    "grep_search",
    "list_directory",
    "todo_read",
    "todo_write",
    "task_delegate",
]
