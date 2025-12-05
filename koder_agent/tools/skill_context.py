"""Skill context manager for tracking active skill restrictions.

This module provides async-safe state management for skill-based tool restrictions
using Python's contextvars. When a skill with `allowed_tools` is loaded, only
those tools (plus always-allowed tools) can be used.

The restriction model uses UNION semantics:
- Multiple skills with `allowed_tools` accumulate their allowed tools
- Loading a skill without `allowed_tools` clears all restrictions

Pattern syntax for allowed_tools:
- "read_file"           - Exact tool name match
- "run_shell:git *"     - Shell commands matching glob pattern
- "run_shell:*"         - All shell commands allowed
- "*"                   - Wildcard, all tools allowed

Note on empty `allowed_tools`:
- A skill with `allowed_tools: []` (empty list) is treated as "no restrictions"
- This is intentional: empty means "didn't specify restrictions", not "block all"
- To block all tools, you would need explicit tooling support (not yet implemented)
"""

from __future__ import annotations

import fnmatch
import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Optional

if TYPE_CHECKING:
    from .skill import Skill

# Context variable to track active skill restrictions (async-safe)
_active_restrictions: ContextVar[Optional["SkillRestrictions"]] = ContextVar(
    "active_skill_restrictions", default=None
)


@dataclass
class SkillRestrictions:
    """Tracks tool restrictions from active skills.

    Uses union semantics: tools from multiple loaded skills are combined.

    Pattern syntax for allowed_tools:
    - "read_file"           - Exact tool name match
    - "run_shell:git *"     - Shell commands matching glob pattern
    - "run_shell:*"         - All shell commands allowed
    - "*"                   - Wildcard, all tools allowed
    """

    # Names of skills that contributed to the current restrictions
    loaded_skills: list[str] = field(default_factory=list)

    # Union of all allowed tools from loaded skills (may include patterns)
    allowed_tools: set[str] = field(default_factory=set)

    # Tools that should always be allowed regardless of skill restrictions
    # - get_skill: Must be able to load different skills to change/escape restrictions
    # - todo_read, todo_write: Task management shouldn't be blocked
    ALWAYS_ALLOWED: ClassVar[frozenset[str]] = frozenset({"get_skill", "todo_read", "todo_write"})

    def is_tool_allowed(self, tool_name: str, tool_args: Optional[str] = None) -> bool:
        """Check if a tool is allowed under current restrictions.

        Supports pattern matching:
        - Exact match: "read_file" matches tool_name="read_file"
        - Wildcard: "*" matches any tool
        - Command pattern: "run_shell:git *" matches run_shell with command starting with "git "

        Args:
            tool_name: The name of the tool to check
            tool_args: JSON string of tool arguments (for command pattern matching)

        Returns:
            True if the tool is allowed, False otherwise
        """
        # Always-allowed tools bypass restrictions
        if tool_name in SkillRestrictions.ALWAYS_ALLOWED:
            return True

        # If no restrictions defined, allow all
        if not self.allowed_tools:
            return True

        # Check each allowed pattern
        for pattern in self.allowed_tools:
            if self._matches_pattern(pattern, tool_name, tool_args):
                return True

        return False

    def _matches_pattern(
        self, pattern: str, tool_name: str, tool_args: Optional[str] = None
    ) -> bool:
        """Check if a tool call matches an allowed pattern.

        Args:
            pattern: The allowed pattern (e.g., "read_file", "run_shell:git *", "*")
            tool_name: The actual tool name being called
            tool_args: JSON string of tool arguments

        Returns:
            True if the pattern matches the tool call
        """
        # Universal wildcard - matches everything
        if pattern == "*":
            return True

        # Check for command pattern syntax: "tool_name:command_pattern"
        if ":" in pattern:
            pattern_tool, command_pattern = pattern.split(":", 1)

            # Tool name must match exactly
            if pattern_tool != tool_name:
                return False

            # For run_shell, match against the command argument
            if tool_name == "run_shell" and tool_args:
                return self._matches_shell_command(command_pattern, tool_args)

            # For git_command, match against the args argument
            if tool_name == "git_command" and tool_args:
                return self._matches_git_command(command_pattern, tool_args)

            # Pattern with ":" but no matching logic - treat as no match
            return False

        # Exact tool name match (or glob pattern on tool name)
        return fnmatch.fnmatch(tool_name, pattern)

    def _matches_shell_command(self, pattern: str, tool_args: str) -> bool:
        """Match a shell command against a glob pattern.

        Args:
            pattern: Glob pattern to match (e.g., "git *", "cat *", "*")
            tool_args: JSON string containing {"command": "..."}

        Returns:
            True if the command matches the pattern
        """
        try:
            args = json.loads(tool_args)
            if not isinstance(args, dict):
                return False
            command = args.get("command", "")
            return fnmatch.fnmatch(command, pattern)
        except (json.JSONDecodeError, TypeError, AttributeError):
            return False

    def _matches_git_command(self, pattern: str, tool_args: str) -> bool:
        """Match a git command against a glob pattern.

        Args:
            pattern: Glob pattern to match (e.g., "status", "commit *", "*")
            tool_args: JSON string containing {"args": "..."}

        Returns:
            True if the git args match the pattern
        """
        try:
            args = json.loads(tool_args)
            if not isinstance(args, dict):
                return False
            git_args = args.get("args", "")
            return fnmatch.fnmatch(git_args, pattern)
        except (json.JSONDecodeError, TypeError, AttributeError):
            return False

    def add_skill(self, skill_name: str, tools: list[str]) -> None:
        """Add a skill's allowed tools to the union.

        Args:
            skill_name: Name of the skill being added
            tools: List of tools the skill allows
        """
        if skill_name not in self.loaded_skills:
            self.loaded_skills.append(skill_name)
        self.allowed_tools.update(tools)


def get_active_restrictions() -> Optional[SkillRestrictions]:
    """Get the currently active skill restrictions.

    Returns:
        SkillRestrictions instance if restrictions are active, None otherwise
    """
    return _active_restrictions.get()


def clear_restrictions() -> None:
    """Clear any active skill restrictions.

    Called when a skill without `allowed_tools` is loaded, or to reset state.
    """
    _active_restrictions.set(None)


def add_skill_restrictions(skill: "Skill") -> None:
    """Add tool restrictions from a loaded skill.

    Uses union semantics: if restrictions already exist, the skill's
    allowed tools are added to the existing set.

    Args:
        skill: The skill whose restrictions should be added
    """
    if not skill.allowed_tools:
        return

    current = _active_restrictions.get()

    if current is None:
        # First skill with restrictions
        current = SkillRestrictions()
        _active_restrictions.set(current)

    current.add_skill(skill.name, skill.allowed_tools)


def has_active_restrictions() -> bool:
    """Check if any skill restrictions are currently active.

    Returns:
        True if restrictions are active, False otherwise
    """
    restrictions = _active_restrictions.get()
    return restrictions is not None and bool(restrictions.allowed_tools)
