"""Tool implementations for Koder Agent."""

from typing import List

from agents import FunctionTool, Tool

from ..agentic.skill_guardrail import skill_restriction_guardrail
from .engine import ToolEngine
from .file import (
    FileEditModel,
    FileReadModel,
    FileWriteModel,
    LSModel,
    append_file,
    edit_file,
    list_directory,
    read_file,
    write_file,
)
from .search import GlobModel, GrepModel, glob_search, grep_search
from .shell import (
    BackgroundShellManager,
    GitModel,
    ShellKillModel,
    ShellModel,
    ShellOutputModel,
    git_command,
    run_shell,
    shell_kill,
    shell_output,
)
from .skill import Skill, SkillLoader, SkillModel, get_skill
from .skill_context import (
    SkillRestrictions,
    add_skill_restrictions,
    clear_restrictions,
    get_active_restrictions,
    has_active_restrictions,
)
from .task import TaskDelegateModel, TaskModel, task_delegate
from .todo import TodoModel, TodoWriteModel, todo_read, todo_write
from .web import SearchModel, WebFetchModel, web_fetch, web_search

# Create the global tool engine
tool_engine = ToolEngine()

# Register all tools
tool_engine.register(FileReadModel)(read_file)
tool_engine.register(FileWriteModel)(write_file)
tool_engine.register(FileWriteModel)(append_file)
tool_engine.register(FileEditModel)(edit_file)
tool_engine.register(ShellModel)(run_shell)
tool_engine.register(ShellOutputModel)(shell_output)
tool_engine.register(ShellKillModel)(shell_kill)
tool_engine.register(SearchModel)(web_search)
tool_engine.register(GlobModel)(glob_search)
tool_engine.register(GrepModel)(grep_search)
tool_engine.register(LSModel)(list_directory)
# TODO tools are already registered via @function_tool decorator
# Removing duplicate registration to avoid naming conflicts
tool_engine.register(SkillModel)(get_skill)
tool_engine.register(WebFetchModel)(web_fetch)
tool_engine.register(TaskDelegateModel)(task_delegate)
tool_engine.register(GitModel)(git_command)


def get_all_tools() -> List[Tool]:
    """Get all registered tools as a list.

    Each FunctionTool is configured with the skill_restriction_guardrail
    to enforce skill-based tool restrictions when skills are active.
    """
    # Collect all @function_tool decorated functions directly
    tools = [
        read_file,
        write_file,
        append_file,
        edit_file,
        run_shell,
        shell_output,
        shell_kill,
        git_command,
        web_search,
        web_fetch,
        glob_search,
        grep_search,
        list_directory,
        todo_read,
        todo_write,
        task_delegate,
        get_skill,
    ]

    # Filter to only include properly decorated tools and attach guardrails
    result = []
    for tool in tools:
        if hasattr(tool, "name"):
            # Attach skill restriction guardrail to each FunctionTool
            if isinstance(tool, FunctionTool):
                if tool.tool_input_guardrails is None:
                    tool.tool_input_guardrails = [skill_restriction_guardrail]
                elif skill_restriction_guardrail not in tool.tool_input_guardrails:
                    tool.tool_input_guardrails.append(skill_restriction_guardrail)
            result.append(tool)
    return result


__all__ = [
    "tool_engine",
    "get_all_tools",
    # Manager for cleanup
    "BackgroundShellManager",
    # Models
    "FileEditModel",
    "FileReadModel",
    "FileWriteModel",
    "LSModel",
    "ShellModel",
    "ShellOutputModel",
    "ShellKillModel",
    "GitModel",
    "SearchModel",
    "WebFetchModel",
    "GlobModel",
    "GrepModel",
    "Skill",
    "SkillLoader",
    "SkillModel",
    "SkillRestrictions",
    "add_skill_restrictions",
    "clear_restrictions",
    "get_active_restrictions",
    "has_active_restrictions",
    "TodoModel",
    "TodoWriteModel",
    "TaskModel",
    "TaskDelegateModel",
    # Functions
    "read_file",
    "write_file",
    "append_file",
    "edit_file",
    "run_shell",
    "shell_output",
    "shell_kill",
    "git_command",
    "web_search",
    "web_fetch",
    "glob_search",
    "grep_search",
    "list_directory",
    "todo_read",
    "todo_write",
    "task_delegate",
    "get_skill",
]
