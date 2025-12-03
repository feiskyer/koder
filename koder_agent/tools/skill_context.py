"""Skill context manager for tracking active skill restrictions.

This module provides async-safe state management for skill-based tool restrictions
using Python's contextvars. When a skill with `allowed_tools` is loaded, only
those tools (plus always-allowed tools) can be used.

The restriction model uses UNION semantics:
- Multiple skills with `allowed_tools` accumulate their allowed tools
- Loading a skill without `allowed_tools` clears all restrictions

Note on empty `allowed_tools`:
- A skill with `allowed_tools: []` (empty list) is treated as "no restrictions"
- This is intentional: empty means "didn't specify restrictions", not "block all"
- To block all tools, you would need explicit tooling support (not yet implemented)
"""

from __future__ import annotations

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
    """

    # Names of skills that contributed to the current restrictions
    loaded_skills: list[str] = field(default_factory=list)

    # Union of all allowed tools from loaded skills
    allowed_tools: set[str] = field(default_factory=set)

    # Tools that should always be allowed regardless of skill restrictions
    # - get_skill: Must be able to load different skills to change/escape restrictions
    # - todo_read, todo_write: Task management shouldn't be blocked
    ALWAYS_ALLOWED: ClassVar[frozenset[str]] = frozenset({"get_skill", "todo_read", "todo_write"})

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed under current restrictions.

        Args:
            tool_name: The name of the tool to check

        Returns:
            True if the tool is allowed, False otherwise
        """
        # Always-allowed tools bypass restrictions
        if tool_name in SkillRestrictions.ALWAYS_ALLOWED:
            return True

        # If no restrictions defined, allow all
        if not self.allowed_tools:
            return True

        return tool_name in self.allowed_tools

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
