"""Skill tool and loader for progressive disclosure of agent skills.

This module implements a 3-level progressive disclosure system:
- Level 1: Skill metadata (name + description) in system prompt at startup
- Level 2: Full skill content loaded on-demand via get_skill tool
- Level 3: Supplementary resources accessed via read_file tool
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from agents import function_tool
from pydantic import BaseModel

# Validation constants (per Claude Code spec)
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
# Name must be lowercase letters, numbers, and hyphens only
NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?$")


@dataclass
class Skill:
    """A loaded skill with its metadata and content."""

    name: str
    description: str
    content: str
    allowed_tools: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    skill_path: Optional[Path] = None

    def to_prompt(self) -> str:
        """Format this skill as a prompt fragment for injection.

        Returns a structured representation including metadata and
        the full skill content for Level 2 disclosure.
        """
        lines: list[str] = [
            f"Skill Name: {self.name}",
            f"Description: {self.description}",
        ]

        if self.allowed_tools:
            tools = ", ".join(self.allowed_tools)
            lines.append(f"Allowed tools: {tools}")

        if self.metadata:
            lines.append("Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"- {key}: {value}")

        lines.append("")
        lines.append("Skill Content:")
        lines.append(self.content.strip())

        return "\n".join(lines).strip()


def _validate_skill_name(name: str, skill_path: Path) -> list[str]:
    """Validate skill name and return list of warning messages.

    Per Claude Code spec:
    - Must be lowercase letters, numbers, and hyphens only
    - Maximum 64 characters

    Args:
        name: The skill name to validate
        skill_path: Path to the skill file (for error messages)

    Returns:
        List of warning messages (empty if valid)
    """
    warnings: list[str] = []

    if len(name) > MAX_NAME_LENGTH:
        warnings.append(
            f"Skill name '{name}' exceeds {MAX_NAME_LENGTH} characters "
            f"(length: {len(name)}) in {skill_path}"
        )

    if not NAME_PATTERN.match(name):
        warnings.append(
            f"Skill name '{name}' should contain only lowercase letters, "
            f"numbers, and hyphens in {skill_path}"
        )

    return warnings


def _validate_skill_description(description: str, skill_path: Path) -> list[str]:
    """Validate skill description and return list of warning messages.

    Per Claude Code spec:
    - Maximum 1024 characters

    Args:
        description: The skill description to validate
        skill_path: Path to the skill file (for error messages)

    Returns:
        List of warning messages (empty if valid)
    """
    warnings: list[str] = []

    if len(description) > MAX_DESCRIPTION_LENGTH:
        warnings.append(
            f"Skill description exceeds {MAX_DESCRIPTION_LENGTH} characters "
            f"(length: {len(description)}) in {skill_path}"
        )

    return warnings


class SkillLoader:
    """Loader for discovering and parsing skill definitions from SKILL.md files."""

    # Pattern to extract YAML frontmatter and body content
    FRONTMATTER_RE = re.compile(r"^---\s*\n(?P<yaml>.*?)\n---\s*\n?(?P<body>.*)$", re.DOTALL)
    # Pattern to match markdown links: [text](target)
    LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def __init__(self, skills_dir: str | Path):
        """Initialize the loader with a skills directory.

        Args:
            skills_dir: Path to directory containing skill subdirectories
        """
        self.skills_dir = Path(skills_dir).expanduser()
        self._cache: dict[str, Skill] = {}
        self._discovered = False

    def load_skill(self, skill_path: Path) -> Optional[Skill]:
        """Load a single skill from a SKILL.md file.

        Args:
            skill_path: Path to SKILL.md file

        Returns:
            Skill instance or None if loading fails
        """
        try:
            raw = skill_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"Warning: failed to read skill file {skill_path}: {exc}")
            return None

        # Strip BOM if present
        text = raw.lstrip("\ufeff")

        # Parse frontmatter
        match = self.FRONTMATTER_RE.match(text)
        meta: dict[str, Any] = {}
        body = text

        if match:
            yaml_text = match.group("yaml")
            body = match.group("body")
            try:
                loaded = yaml.safe_load(yaml_text) or {}
                if isinstance(loaded, dict):
                    meta = loaded
                else:
                    print(f"Warning: frontmatter in {skill_path} must be a mapping")
            except yaml.YAMLError as exc:
                print(f"Warning: invalid YAML in {skill_path}: {exc}")
        else:
            print(f"Warning: no frontmatter found in {skill_path}")

        # Extract standard fields with defaults
        name = str(meta.get("name") or skill_path.stem)
        description = str(meta.get("description") or "")

        # Validate name and description (warn but still load - graceful degradation)
        for warning in _validate_skill_name(name, skill_path):
            print(f"Warning: {warning}")
        for warning in _validate_skill_description(description, skill_path):
            print(f"Warning: {warning}")

        # Handle allowed_tools - support both hyphenated (Claude Code) and underscored
        # Claude Code uses "allowed-tools", Koder historically uses "allowed_tools"
        # Priority: hyphenated key always wins if present (even if empty)
        tools_raw = meta["allowed-tools"] if "allowed-tools" in meta else meta.get("allowed_tools")
        allowed_tools: Optional[list[str]] = None
        if isinstance(tools_raw, list):
            allowed_tools = [str(t) for t in tools_raw]
        elif tools_raw:
            allowed_tools = [str(tools_raw)]

        # Collect remaining fields as metadata
        reserved = {"name", "description", "allowed_tools", "allowed-tools"}
        extra = {k: v for k, v in meta.items() if k not in reserved}
        extra_meta = extra if extra else None

        # Process body content
        body = body.lstrip("\n")
        body = self._resolve_paths(body, skill_path.parent)

        return Skill(
            name=name,
            description=description,
            content=body,
            allowed_tools=allowed_tools,
            metadata=extra_meta,
            skill_path=skill_path,
        )

    def discover_skills(self) -> list[Skill]:
        """Discover all skills under the skills directory.

        Returns:
            List of discovered Skill instances
        """
        self._cache.clear()

        if not self.skills_dir.exists():
            print(f"Warning: skills directory does not exist: {self.skills_dir}")
            self._discovered = True
            return []

        for skill_file in self.skills_dir.rglob("SKILL.md"):
            skill = self.load_skill(skill_file)
            if not skill:
                continue

            if skill.name in self._cache:
                print(f"Warning: duplicate skill name '{skill.name}' in {skill_file}")
                continue

            self._cache[skill.name] = skill

        self._discovered = True
        return list(self._cache.values())

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name.

        Args:
            name: The skill name to look up

        Returns:
            Skill instance or None if not found
        """
        self._ensure_discovered()
        return self._cache.get(name)

    def list_skills(self) -> list[str]:
        """List available skill names.

        Returns:
            Sorted list of skill names
        """
        self._ensure_discovered()
        return sorted(self._cache.keys())

    def get_skills_metadata_prompt(self) -> str:
        """Return Level 1 metadata prompt for all skills.

        This provides just names and descriptions for initial system prompt
        injection, minimizing token usage until skills are actually needed.

        Returns:
            Formatted metadata string
        """
        self._ensure_discovered()

        if not self._cache:
            return "No skills are currently available."

        lines = ["Available skills:", ""]
        for skill in sorted(self._cache.values(), key=lambda s: s.name.lower()):
            lines.append(f"- {skill.name}: {skill.description.strip()}")

        return "\n".join(lines)

    def _resolve_paths(self, content: str, skill_dir: Path) -> str:
        """Convert relative paths in markdown links to absolute paths.

        This enables Level 3 progressive disclosure where the agent can
        use read_file to access supplementary resources.

        Args:
            content: Skill body content
            skill_dir: Directory containing the skill

        Returns:
            Content with resolved absolute paths
        """

        def replace_link(m: re.Match[str]) -> str:
            label = m.group(1)
            target = m.group(2).strip()

            # Skip empty, anchors, absolute paths, and URLs
            if not target:
                return m.group(0)
            if target.startswith(("#", "/", "http://", "https://", "mailto:")):
                return m.group(0)
            # Skip other URL schemes
            if re.match(r"^[a-zA-Z]+:", target):
                return m.group(0)

            abs_path = (skill_dir / target).resolve()
            return f"[{label}]({abs_path})"

        return self.LINK_RE.sub(replace_link, content)

    def _ensure_discovered(self) -> None:
        """Ensure skills have been discovered before access."""
        if not self._discovered:
            self.discover_skills()


# -----------------------------------------------------------------------------
# Tool Implementation
# -----------------------------------------------------------------------------


class SkillModel(BaseModel):
    """Parameter model for the get_skill tool."""

    skill_name: str


# Module-level cache for merged skills from all directories
_merged_skills: Optional[dict[str, Skill]] = None


def _get_merged_skills() -> dict[str, Skill]:
    """Lazily load and merge skills from project and user directories.

    Priority: project skills override user skills with the same name.

    Returns:
        Dictionary mapping skill names to Skill instances
    """
    global _merged_skills

    if _merged_skills is None:
        _merged_skills = {}

        # Load user skills first (lower priority)
        user_dir = Path.home() / ".koder" / "skills"
        if user_dir.exists():
            user_loader = SkillLoader(user_dir)
            for skill in user_loader.discover_skills():
                _merged_skills[skill.name] = skill

        # Load project skills (higher priority, overrides user)
        project_dir = Path(".koder/skills")
        if project_dir.exists():
            project_loader = SkillLoader(project_dir)
            for skill in project_loader.discover_skills():
                _merged_skills[skill.name] = skill

    return _merged_skills


@function_tool
def get_skill(skill_name: str) -> str:
    """Get the full content for a named skill.

    Use this tool to load detailed skill guidance when you need to perform
    a task that matches an available skill. The skill's full content will
    be returned as a formatted prompt.

    Note: If the skill declares allowed_tools, those tools will be added to
    the current allowed set (union semantics). Loading a skill without
    allowed_tools will clear all restrictions.

    Args:
        skill_name: The name of the skill to load

    Returns:
        The formatted skill prompt, or an error with available skills
    """
    from .skill_context import add_skill_restrictions, clear_restrictions

    if not skill_name:
        return "Invalid skill name: skill_name cannot be empty"

    skills = _get_merged_skills()
    skill = skills.get(skill_name)

    if skill:
        # Handle tool restrictions based on skill configuration
        if skill.allowed_tools:
            # Add this skill's tools to the allowed set (union behavior)
            add_skill_restrictions(skill)
        else:
            # Skill has no restrictions - clear any active restrictions
            clear_restrictions()

        return skill.to_prompt()

    available = sorted(skills.keys())
    if not available:
        return f"Skill '{skill_name}' not found. No skills are currently available."

    return f"Skill '{skill_name}' not found. Available skills: {', '.join(available)}"
