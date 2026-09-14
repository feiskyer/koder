"""Skill tool and loader for progressive disclosure of agent skills."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel

from koder_agent.config import get_config
from koder_agent.harness.execution_context import execution_path, get_execution_cwd
from koder_agent.harness.memory.budget import estimate_text_tokens
from koder_agent.harness.paths import harness_home_dir
from koder_agent.harness.plugins.context import get_plugin_root, normalize_plugin_root
from koder_agent.harness.skills.bundled import get_bundled_skills
from koder_agent.harness.skills.discovery import discover_skills_for_paths

from .compat import function_tool

logger = logging.getLogger(__name__)

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
NAME_PATTERN = re.compile(r"^[a-z0-9]([a-z0-9:.-]*[a-z0-9])?$")
SKILL_ADDITIONAL_DIRS_ENV = "KODER_ADDITIONAL_DIRS"
INLINE_COMMAND_RE = re.compile(r"!\`([^`]+)\`")
# Matches positional argument placeholders: ``$ARGUMENTS[<n>]`` or bare ``$<n>``.
# Using ``\d+`` (not a per-index loop) so multi-digit indices like ``$10`` are
# substituted atomically instead of ``$1`` matching the ``$1`` prefix of ``$10``.
_POSITIONAL_ARG_RE = re.compile(r"\$ARGUMENTS\[(?P<bracket>\d+)\]|\$(?P<bare>\d+)")
SKILL_BUDGET_CONTEXT_PERCENT = 0.01
# ``DEFAULT_TOKEN_BUDGET`` is the token-space equivalent of the previous
# ``DEFAULT_CHAR_BUDGET`` of 8_000 characters. Budgeting is now token-accurate
# (via ``estimate_text_tokens``) rather than a fixed 4-chars-per-token heuristic,
# so both the env override and the ~1% context-window budget are expressed in
# tokens.
DEFAULT_TOKEN_BUDGET = 2_000
MAX_LISTING_DESC_CHARS = 250


@dataclass
class Skill:
    """A loaded skill with its metadata and content."""

    name: str
    description: str
    content: str
    allowed_tools: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None
    skill_path: Optional[Path] = None
    source: str = "project"
    disable_model_invocation: bool = False
    user_invocable: bool = True
    argument_hint: str | None = None
    argument_names: list[str] | None = None
    model: str | None = None
    effort: str | int | None = None
    execution_context: str | None = None
    agent: str | None = None
    hooks: dict[str, Any] | None = None
    paths: list[str] | None = None
    shell: str | None = None
    base_dir: Path | None = None
    plugin_name: str | None = None

    def to_prompt(self) -> str:
        lines: list[str] = [
            f"Skill Name: {self.name}",
            f"Description: {self.description}",
        ]
        if self.argument_hint:
            lines.append(f"Argument hint: {self.argument_hint}")
        if self.allowed_tools is not None:
            lines.append(
                "Allowed tools: "
                + (", ".join(self.allowed_tools) if self.allowed_tools else "(none specified)")
            )
        if self.execution_context:
            lines.append(f"Execution context: {self.execution_context}")
        if self.agent:
            lines.append(f"Agent: {self.agent}")
        if self.model:
            lines.append(f"Model: {self.model}")
        if self.effort is not None:
            lines.append(f"Effort: {self.effort}")
        if self.paths:
            lines.append(f"Paths: {', '.join(self.paths)}")
        if self.metadata:
            lines.append("Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"- {key}: {value}")

        lines.append("")
        lines.append("Skill Content:")
        lines.append(self.content.strip())
        return "\n".join(lines).strip()

    def render_prompt(self, args: list[str], *, session_id: str | None = None) -> str:
        final = self.content
        joined = " ".join(args).strip()

        # Substitute positional placeholders in a single regex pass so multi-digit
        # indices resolve correctly. A naive ascending ``str.replace`` loop turns
        # ``$10`` into ``<value of $1>0``; matching the whole ``\d+`` avoids that.
        def _positional(match: re.Match[str]) -> str:
            index = int(match.group("bracket") or match.group("bare"))
            if 0 <= index < len(args):
                return args[index]
            return match.group(0)

        final = _POSITIONAL_ARG_RE.sub(_positional, final)

        if "$ARGUMENTS" in final:
            final = final.replace("$ARGUMENTS", joined)
        elif joined:
            final = f"{final.rstrip()}\n\nARGUMENTS: {joined}"

        if self.base_dir is not None:
            final = final.replace("${KODER_SKILL_DIR}", str(self.base_dir))
        if session_id is not None:
            final = final.replace("${KODER_SESSION_ID}", session_id)

        # Expand plugin env vars for plugin-sourced skills
        if self.source == "plugin" and self.plugin_name and self.base_dir is not None:
            from koder_agent.harness.plugins.env import expand_plugin_vars

            final = expand_plugin_vars(final, self.plugin_name, self.base_dir)

        final = INLINE_COMMAND_RE.sub(lambda m: self._run_inline_command(m.group(1).strip()), final)
        return final.strip()

    def _run_inline_command(self, command: str) -> str:
        if not command:
            return ""

        # Security gate 0: trust scoping. Inline ``!`cmd`` expansion runs at
        # render time with NO human in the loop, so an untrusted third-party
        # skill must never be able to run a command merely by being rendered.
        # A read-only classifier is not sufficient here: a read-only command
        # like ``cat ~/.ssh/id_rsa`` exfiltrates secrets into the rendered
        # (LLM-visible) prompt, which IS the attack. Only first-party (bundled)
        # skills may execute inline commands; user/project/plugin/additional
        # skills get a placeholder instead. Operators who deliberately trust a
        # local skill can still run it via a normal tool call.
        if not _inline_commands_trusted(self.source):
            return "[blocked: inline command execution is only permitted for built-in skills]"

        # Security gate 1: allow operators to disable inline command execution
        # entirely via env flag (consistent with the codebase's env-flag style).
        # When disabled, never execute -- substitute a clear placeholder.
        if not _inline_commands_enabled():
            return "[inline command execution disabled]"

        # Security gate 1b: command/process substitution smuggles an arbitrary
        # command inside an otherwise read-only line (e.g. ``echo $(rm -rf /)``).
        # The static classifiers below are word/segment based and do not police
        # what runs inside ``$(...)``/backticks, so reject substitution outright
        # for both shells before classification.
        if _contains_command_substitution(command):
            return "[blocked: command substitution is not permitted in inline commands]"

        if self.shell == "powershell":
            # Security gate 2 (PowerShell): ALLOWLIST posture. Inline expansion
            # runs at render time with no human in the loop, so only clearly
            # read-only commands (allowed AND read_only AND not requiring
            # approval) may run; everything else is blocked.
            from koder_agent.harness.permissions.powershell_classifier import (
                classify_powershell_command,
            )

            decision = classify_powershell_command(command)
            if not (decision.allowed and decision.read_only and not decision.requires_approval):
                return f"[blocked: {decision.reason}]"

            argv = ["powershell", "-NoProfile", "-Command", command]
        else:
            # Security gate 2 (bash): ALLOWLIST posture mirroring the PowerShell
            # branch instead of a weak denylist. Inline expansion happens at
            # render time with no human in the loop, so a malicious third-party
            # skill could otherwise gain arbitrary code execution just by having
            # its prompt rendered. Only clearly read-only commands may run: the
            # classifier must report allowed AND read_only AND not
            # requires_approval. Anything that mutates the filesystem, executes
            # arbitrary code (e.g. curl|sh), contains command substitution, or
            # otherwise needs approval is blocked.
            from koder_agent.harness.permissions.shell_classifier import (
                classify_shell_command,
            )

            decision = classify_shell_command(command)
            if not (decision.allowed and decision.read_only and not decision.requires_approval):
                return f"[blocked: {decision.reason}]"

            # Keep the legacy bash analyzer as an ADDITIONAL block on top of the
            # allowlist -- its denylist may catch patterns the classifier permits.
            from koder_agent.core.bash_security import analyze_command

            analysis = analyze_command(command)
            if analysis.blocked:
                return f"[blocked: {analysis.reason}]"

            argv = ["/bin/bash", "-lc", command]

        # A hanging command would block prompt rendering forever, so bound it.
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                cwd=str(get_execution_cwd()),
                check=False,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "[timed out after 30s]"
        output = result.stdout.strip()
        if output:
            return output
        return result.stderr.strip()


SKILL_INLINE_COMMANDS_ENV = "KODER_SKILL_INLINE_COMMANDS"

# Substrings indicating command/process substitution. Applies to both bash
# (``$(...)`` / backticks / ``<(...)`` / ``>(...)``) and PowerShell
# (``$(...)`` / backticks). A read-only classifier cannot reason about the
# arbitrary command hidden inside these, so they are rejected before running.
_COMMAND_SUBSTITUTION_MARKERS = ("$(", "`", "<(", ">(", "${")


def _contains_command_substitution(command: str) -> bool:
    return any(marker in command for marker in _COMMAND_SUBSTITUTION_MARKERS)


# Skill sources whose content is first-party / shipped with koder. Only these
# may run inline ``!`cmd`` expansions at render time. Everything else
# (user/project/plugin/additional) is untrusted third-party content.
_TRUSTED_INLINE_SOURCES = frozenset({"bundled"})


def _inline_commands_trusted(source: str | None) -> bool:
    """Whether a skill's source is trusted enough to run inline commands.

    Inline expansion runs with no human in the loop, so only first-party
    (bundled) skills qualify. An env override lets operators opt a specific
    deployment into trusting all sources, but it must be set explicitly.
    """
    override = os.environ.get("KODER_SKILL_INLINE_TRUST_ALL")
    if override is not None and override.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return source in _TRUSTED_INLINE_SOURCES


def _inline_commands_enabled() -> bool:
    """Whether skill inline ``!`cmd`` expansion may execute commands.

    Honors the ``KODER_SKILL_INLINE_COMMANDS`` env flag. Default is enabled so
    first-party skills keep working, but when set to a falsy value (``0``/
    ``false``/``no``/``off``) inline execution is skipped entirely. Even when
    enabled, bash commands still pass through the security analyzer.
    """
    raw = os.environ.get(SKILL_INLINE_COMMANDS_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _validate_skill_name(name: str, skill_path: Path) -> list[str]:
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
    warnings: list[str] = []
    if len(description) > MAX_DESCRIPTION_LENGTH:
        warnings.append(
            f"Skill description exceeds {MAX_DESCRIPTION_LENGTH} characters "
            f"(length: {len(description)}) in {skill_path}"
        )
    return warnings


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return [part for part in parts if part]
    return [str(value).strip()]


def _skill_default_name(skill_path: Path) -> str:
    if skill_path.name.lower() == "skill.md":
        return skill_path.parent.name
    return skill_path.stem


def _parse_paths(value: Any) -> list[str] | None:
    parsed = _parse_list(value)
    return parsed or None


def _expand_path(value: str | Path) -> Path:
    text = str(value)
    if text.startswith("~/"):
        return (Path.home() / text[2:]).resolve()
    return Path(text).expanduser().resolve()


class SkillLoader:
    """Loader for discovering and parsing skill definitions from SKILL.md files."""

    FRONTMATTER_RE = re.compile(r"^---\s*\n(?P<yaml>.*?)\n---\s*\n?(?P<body>.*)$", re.DOTALL)
    LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def __init__(self, skills_dir: str | Path):
        self.skills_dir = Path(skills_dir).expanduser()
        self._cache: dict[str, Skill] = {}
        self._discovered = False

    def load_skill(
        self,
        skill_path: Path,
        *,
        source: str = "project",
        plugin_name: str | None = None,
    ) -> Optional[Skill]:
        try:
            raw = skill_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read skill file %s: %s", skill_path, exc)
            return None

        text = raw.lstrip("\ufeff")
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
                    logger.warning("Frontmatter in %s must be a mapping", skill_path)
                    return None  # Reject skill with non-mapping frontmatter
            except yaml.YAMLError as exc:
                logger.warning("invalid YAML in %s: %s", skill_path, exc)
                return None  # Reject skill with malformed YAML (fail-closed)
        else:
            logger.warning("no frontmatter found in %s", skill_path)

        resolved_name = str(meta.get("name") or _skill_default_name(skill_path))
        if plugin_name:
            resolved_name = f"{plugin_name}:{resolved_name}"
        description = str(meta.get("description") or "")

        for warning in _validate_skill_name(resolved_name, skill_path):
            logger.warning("%s", warning)
        for warning in _validate_skill_description(description, skill_path):
            logger.warning("%s", warning)

        tools_raw = meta["allowed-tools"] if "allowed-tools" in meta else meta.get("allowed_tools")
        allowed_tools = _parse_list(tools_raw)
        if tools_raw == []:
            allowed_tools = []

        reserved = {
            "name",
            "description",
            "allowed_tools",
            "allowed-tools",
            "argument-hint",
            "arguments",
            "disable-model-invocation",
            "user-invocable",
            "model",
            "effort",
            "context",
            "agent",
            "hooks",
            "paths",
            "shell",
        }
        extra = {k: v for k, v in meta.items() if k not in reserved}
        extra_meta = extra if extra else None

        body = body.lstrip("\n")
        base_dir = skill_path.parent
        body = self._resolve_paths(body, base_dir)
        argument_hint_value = meta.get("argument-hint")
        if isinstance(argument_hint_value, list):
            argument_hint = "[" + " ".join(str(item) for item in argument_hint_value) + "]"
        elif argument_hint_value is not None:
            argument_hint = str(argument_hint_value)
        else:
            argument_hint = None

        return Skill(
            name=resolved_name,
            description=description,
            content=body,
            allowed_tools=allowed_tools,
            metadata=extra_meta,
            skill_path=skill_path,
            source=source,
            disable_model_invocation=_parse_bool(
                meta.get("disable-model-invocation"), default=False
            ),
            user_invocable=_parse_bool(meta.get("user-invocable"), default=True),
            argument_hint=argument_hint,
            argument_names=_parse_list(meta.get("arguments")),
            model=(
                (
                    "inherit"
                    if str(meta.get("model")).strip().lower() == "inherit"
                    else str(meta.get("model")).strip()
                )
                if meta.get("model") is not None and str(meta.get("model")).strip()
                else None
            ),
            effort=meta.get("effort") if meta.get("effort") is not None else None,
            execution_context="fork" if meta.get("context") == "fork" else None,
            agent=(
                str(meta.get("agent")).strip()
                if meta.get("agent") is not None and str(meta.get("agent")).strip()
                else None
            ),
            hooks=meta.get("hooks") if isinstance(meta.get("hooks"), dict) else None,
            paths=_parse_paths(meta.get("paths")),
            shell=(
                str(meta.get("shell")).strip()
                if meta.get("shell") is not None and str(meta.get("shell")).strip()
                else None
            ),
            base_dir=base_dir,
            plugin_name=plugin_name,
        )

    def discover_skills(
        self, *, source: str = "project", plugin_name: str | None = None
    ) -> list[Skill]:
        self._cache.clear()

        if not self.skills_dir.exists():
            logger.debug("Skills directory does not exist: %s", self.skills_dir)
            self._discovered = True
            return []

        # Load SKILL.md files (standard skills format)
        for skill_file in self.skills_dir.rglob("SKILL.md"):
            skill = self.load_skill(skill_file, source=source, plugin_name=plugin_name)
            if not skill:
                continue
            if skill.name in self._cache:
                logger.warning("duplicate skill name '%s' in %s", skill.name, skill_file)
                continue
            self._cache[skill.name] = skill

        # Load plain .md command files (legacy format)
        for md_file in sorted(self.skills_dir.glob("*.md")):
            if md_file.name == "SKILL.md":
                continue
            skill = self.load_skill(md_file, source=source, plugin_name=plugin_name)
            if not skill:
                continue
            if skill.name in self._cache:
                continue
            self._cache[skill.name] = skill

        self._discovered = True
        return list(self._cache.values())

    def get_skill(self, name: str) -> Optional[Skill]:
        self._ensure_discovered()
        return self._cache.get(name)

    def list_skills(self) -> list[str]:
        self._ensure_discovered()
        return sorted(self._cache.keys())

    def get_skills_metadata_prompt(self) -> str:
        self._ensure_discovered()
        return build_skills_metadata_prompt(self._cache)

    def _resolve_paths(self, content: str, skill_dir: Path) -> str:
        def replace_link(m: re.Match[str]) -> str:
            label = m.group(1)
            target = m.group(2).strip()
            if not target:
                return m.group(0)
            if target.startswith(("#", "/", "http://", "https://", "mailto:")):
                return m.group(0)
            if re.match(r"^[a-zA-Z]+:", target):
                return m.group(0)
            abs_path = (skill_dir / target).resolve()
            return f"[{label}]({abs_path})"

        return self.LINK_RE.sub(replace_link, content)

    def _ensure_discovered(self) -> None:
        if not self._discovered:
            self.discover_skills()


def _bounded_find_koder_skills(root: Path, max_depth: int = 3) -> list[Path]:
    """Walk up to *max_depth* levels below *root* looking for .koder/skills dirs.

    This replaces an unbounded ``rglob(".koder/skills")`` which is prohibitively
    slow on large monorepos.
    """
    results: list[Path] = []
    resolved = root.resolve()

    def _walk(current: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = list(current.iterdir())
        except (PermissionError, OSError):
            return
        for entry in entries:
            if not entry.is_dir():
                continue
            if entry.name == ".koder":
                candidate = entry / "skills"
                if candidate.is_dir():
                    results.append(candidate.resolve())
            else:
                # Skip hidden directories and common heavy dirs for speed
                if entry.name.startswith("."):
                    continue
                _walk(entry, depth + 1)

    _walk(resolved, 1)
    return results


def _project_skill_dirs(cwd: Path) -> list[Path]:
    dirs: list[Path] = []
    current = cwd.resolve()
    home = Path.home().resolve()
    while True:
        candidate = current / ".koder" / "skills"
        if candidate.is_dir():
            dirs.append(candidate)
        if current == home or current.parent == current:
            break
        if (current / ".git").exists():
            break
        current = current.parent

    nested = sorted(
        _bounded_find_koder_skills(cwd.resolve(), max_depth=3),
        key=lambda path: (len(path.parts), str(path)),
    )
    for path in nested:
        if path not in dirs:
            dirs.append(path)

    # Dynamically discover new skill directories from file paths
    try:
        known_dirs = {str(d) for d in dirs}
        discovered = discover_skills_for_paths([str(cwd)], known_dirs)
        for new_dir in discovered:
            if new_dir not in dirs:
                dirs.append(new_dir)
    except Exception:
        pass  # Don't fail if discovery fails

    return dirs


@contextmanager
def _plugin_skill_dirs(plugin_root: Path):
    """Return (dir, plugin_name) pairs for plugin skills and commands.

    Both ``skills/`` (SKILL.md inside subdirs) and ``commands/``
    (plain .md files — legacy format) are scanned.
    """
    from koder_agent.harness.plugins.lifecycle import PluginLifecycleService
    from koder_agent.harness.plugins.path_safety import (
        PluginPathError,
        open_plugin_component,
    )

    dirs: list[tuple[Path, str]] = []
    try:
        lifecycle = PluginLifecycleService(plugin_root)
    except (OSError, ValueError):
        yield dirs
        return
    with ExitStack() as stack:
        for manifest, state in lifecycle.installed_plugins():
            if not state.enabled:
                continue
            plugin_dir = lifecycle.resolve_plugin_target(manifest.name)
            for declared, default, field_name in (
                (manifest.skills, "skills", "skills"),
                (manifest.commands, "commands", "commands"),
            ):
                try:
                    component = stack.enter_context(
                        open_plugin_component(
                            plugin_dir,
                            declared,
                            default=default,
                            field_name=field_name,
                            expect="directory",
                        )
                    )
                except PluginPathError:
                    continue
                if component is not None:
                    dirs.append((component, manifest.name))
        yield dirs


def _additional_skill_dirs(additional_dirs: list[Path]) -> list[Path]:
    dirs: list[Path] = []
    for directory in additional_dirs:
        candidate = directory / ".koder" / "skills"
        if candidate.is_dir():
            dirs.append(candidate.resolve())
    return dirs


def _additional_dirs_from_env() -> list[Path]:
    raw = os.environ.get(SKILL_ADDITIONAL_DIRS_ENV, "").strip()
    if not raw:
        return []
    return [Path(part).expanduser().resolve() for part in raw.split(os.pathsep) if part.strip()]


def _get_skill_token_budget(context_window_tokens: int | None = None) -> int:
    """Return the metadata-prompt budget as a token count.

    Budgeting is token-accurate (see ``build_skills_metadata_prompt``). The
    ``SLASH_COMMAND_TOOL_CHAR_BUDGET`` env override is honored as an explicit
    token cap (its historical name is kept for backwards compatibility), and the
    ~1% context-window budget is expressed directly in tokens rather than being
    multiplied by a fixed 4-chars-per-token heuristic.
    """
    raw = os.environ.get("SLASH_COMMAND_TOOL_CHAR_BUDGET")
    if raw and raw.isdigit():
        return int(raw)
    if context_window_tokens:
        return int(context_window_tokens * SKILL_BUDGET_CONTEXT_PERCENT)
    return DEFAULT_TOKEN_BUDGET


def build_skills_metadata_prompt(
    skills: dict[str, Skill],
    *,
    context_window_tokens: int | None = None,
) -> str:
    visible = [
        skill for skill in skills.values() if not skill.disable_model_invocation and not skill.paths
    ]
    if not visible:
        return "No skills are currently available."

    budget = _get_skill_token_budget(context_window_tokens)
    lines = ["Available skills:", ""]
    used = estimate_text_tokens("Available skills:\n\n")

    for skill in sorted(visible, key=lambda s: s.name.lower()):
        desc = (skill.description or "").strip()
        if len(desc) > MAX_LISTING_DESC_CHARS:
            desc = desc[: MAX_LISTING_DESC_CHARS - 1] + "…"
        prefix = f"- {skill.name}: "
        entry = prefix + desc
        entry_tokens = estimate_text_tokens(entry) + 1  # +1 for the joining newline
        if used + entry_tokens > budget:
            # No room for even the bare "- name:" prefix -> stop, but ensure at
            # least one entry is emitted so the prompt is never just a header.
            prefix_tokens = estimate_text_tokens(prefix) + 1
            if used + prefix_tokens > budget:
                if len(lines) == 2:
                    lines.append(prefix.rstrip())
                break
            if len(lines) > 2:
                break
            # Trim the description down to fit the remaining token budget.
            trimmed = _trim_desc_to_tokens(desc, budget - used - prefix_tokens)
            entry = prefix + trimmed
            entry_tokens = estimate_text_tokens(entry) + 1
        lines.append(entry)
        used += entry_tokens
    return "\n".join(lines)


def _trim_desc_to_tokens(desc: str, token_room: int) -> str:
    """Trim *desc* so ``desc + '…'`` fits within *token_room* tokens.

    Falls back to a character-proportional guess then shrinks until the token
    estimate fits, so the result is token-accurate rather than heuristic.
    """
    if token_room <= 0:
        return "…"
    if estimate_text_tokens(desc + "…") <= token_room:
        return desc + "…"
    # Start from a proportional character estimate and shrink to fit.
    approx_chars = max(1, len(desc) * token_room // max(1, estimate_text_tokens(desc)))
    for length in range(min(approx_chars, len(desc)), 0, -1):
        candidate = desc[:length] + "…"
        if estimate_text_tokens(candidate) <= token_room:
            return candidate
    return "…"


def _merge_skill(merged: dict[str, Skill], skill: Skill) -> None:
    """Insert *skill* into *merged* under its name, warning on cross-source shadowing.

    Precedence is unchanged (last writer wins), but when the incoming skill
    overrides an existing entry that came from a DIFFERENT source, a warning is
    emitted recording ``shadowed source -> overriding source`` and the name.
    This surfaces the previously-silent case where, e.g., a user/project/plugin
    skill named ``docx`` clobbers the bundled one. Same-source overrides (e.g.
    two project dirs) stay silent to avoid noise.
    """
    existing = merged.get(skill.name)
    if existing is not None and existing.source != skill.source:
        logger.warning(
            "skill '%s' from source '%s' overrides skill of the same name from source '%s'",
            skill.name,
            skill.source,
            existing.source,
        )
    merged[skill.name] = skill


def discover_merged_skills(
    *,
    cwd: str | Path | None = None,
    user_dir: str | Path | None = None,
    project_dir: str | Path | None = None,
    plugin_root: str | Path | None = None,
    additional_dirs: list[str | Path] | None = None,
) -> dict[str, Skill]:
    current_cwd = (execution_path(cwd) if cwd else get_execution_cwd()).resolve()
    resolved_user_dir = (
        _expand_path(user_dir) if user_dir is not None else (harness_home_dir() / "skills")
    )
    resolved_project_dir = _expand_path(project_dir) if project_dir is not None else None
    resolved_plugin_root = (
        normalize_plugin_root(plugin_root) if plugin_root is not None else get_plugin_root()
    )
    resolved_additional_dirs = (
        [Path(path).expanduser().resolve() for path in additional_dirs]
        if additional_dirs is not None
        else _additional_dirs_from_env()
    )

    merged: dict[str, Skill] = {}

    for _name, skill in get_bundled_skills().items():
        _merge_skill(merged, skill)

    if resolved_user_dir.exists():
        for skill in SkillLoader(resolved_user_dir).discover_skills(source="user"):
            _merge_skill(merged, skill)

    if resolved_project_dir is not None and resolved_project_dir.exists():
        for skill in SkillLoader(resolved_project_dir).discover_skills(source="project"):
            _merge_skill(merged, skill)

    for project_dir in _project_skill_dirs(current_cwd):
        for skill in SkillLoader(project_dir).discover_skills(source="project"):
            _merge_skill(merged, skill)

    for extra_dir in _additional_skill_dirs(resolved_additional_dirs):
        for skill in SkillLoader(extra_dir).discover_skills(source="additional"):
            _merge_skill(merged, skill)

    with _plugin_skill_dirs(resolved_plugin_root) as plugin_skill_dirs:
        for skills_dir, plugin_name in plugin_skill_dirs:
            for skill in SkillLoader(skills_dir).discover_skills(
                source="plugin", plugin_name=plugin_name
            ):
                _merge_skill(merged, skill)

    return merged


class SkillModel(BaseModel):
    skill_name: str


_merged_skills: Optional[dict[str, Skill]] = None
_merged_skills_key: Optional[tuple] = None
_merged_skills_lock = threading.RLock()


def _skill_dir_signature(path: Path) -> tuple[tuple[str, int, int, int, int, int], ...]:
    """Track every discoverable skill's path, identity and filesystem version.

    A directory-wide maximum timestamp misses changes to other files, including
    removal. Match SkillLoader's discovery patterns without reading instruction
    bodies or nested reference files on every cache lookup.
    """
    entries = []
    candidates = set(path.rglob("SKILL.md")) | set(path.glob("*.md"))
    for skill_file in sorted(candidates):
        try:
            version = skill_file.stat()
        except FileNotFoundError:
            continue  # A removed file must disappear from the signature.
        entries.append(
            (
                skill_file.relative_to(path).as_posix(),
                version.st_dev,
                version.st_ino,
                version.st_size,
                version.st_mtime_ns,
                version.st_ctime_ns,
            )
        )
    return tuple(entries)


def _compute_merged_skills_cache_key(
    *,
    cwd: str,
    user_dir: str,
    project_dir: str,
    plugin_root: str,
    additional: tuple[str, ...],
) -> tuple:
    """Build the cache key for ``_get_merged_skills``.

    The key records every skill file in the FULL set of skill directories --
    computed the same way ``discover_merged_skills`` does, including walked-up
    parents, nested monorepo packages, and dynamically discovered directories.
    Per-file signatures detect insertion, removal, rename and edits even when
    an unrelated file retains the newest modification time.
    """
    scanned_dirs: list[Path] = [Path(user_dir), Path(project_dir), Path(plugin_root)]
    scanned_dirs.extend(_project_skill_dirs(Path(cwd)))
    scanned_dirs.extend(Path(p) for p in additional)
    scanned_dirs.extend(_additional_skill_dirs([Path(p) for p in additional]))

    # Record each scanned directory, keyed by resolved path so duplicates
    # collapse and ordering is stable.
    directory_signatures = tuple(
        sorted(
            (str(path.resolve()), _skill_dir_signature(path))
            for path in {d.resolve(): d for d in scanned_dirs}.values()
        )
    )

    return (
        cwd,
        user_dir,
        project_dir,
        plugin_root,
        additional,
        directory_signatures,
    )


def _get_merged_skills() -> dict[str, Skill]:
    global _merged_skills, _merged_skills_key

    # Root/key comparison and publication must be one operation: copied thread
    # contexts can otherwise return a different runtime's global cache value.
    with _merged_skills_lock:
        config = get_config()
        cwd = str(get_execution_cwd().resolve())
        user_dir = str(_expand_path(config.skills.user_skills_dir))
        project_dir = str(_expand_path(config.skills.project_skills_dir))
        plugin_root = str(get_plugin_root())
        additional = tuple(str(path) for path in _additional_dirs_from_env())

        cache_key = _compute_merged_skills_cache_key(
            cwd=cwd,
            user_dir=user_dir,
            project_dir=project_dir,
            plugin_root=plugin_root,
            additional=additional,
        )

        if _merged_skills is None or _merged_skills_key != cache_key:
            _merged_skills = discover_merged_skills(
                cwd=cwd,
                user_dir=user_dir,
                project_dir=project_dir,
                plugin_root=plugin_root,
                additional_dirs=list(additional),
            )
            _merged_skills_key = cache_key

        return _merged_skills


def _apply_skill_restrictions(skill: Skill) -> None:
    """Apply a loaded skill's turn-scoped tool and hook policy.

    Loading a skill only ADDS its restrictions (when it declares
    ``allowed_tools``). Loading an unrestricted skill is a NO-OP for
    restrictions -- it must NOT silently erase the restrictions contributed by a
    previously-loaded restricted skill, otherwise the model could self-clear its
    own sandbox just by loading any benign skill. ``clear_restrictions()``
    remains an explicit API for callers that genuinely need to reset state.
    """
    from .skill_context import activate_skill_policy

    activate_skill_policy(skill)


@function_tool
def get_skill(skill_name: str) -> str:
    """Load the full instructions of a skill by name.

    Skills are listed with their descriptions in the system prompt; call this
    to expand one before performing its task.

    Args:
        skill_name: Exact name of the skill to load (as listed in the system prompt)
    """
    if not skill_name:
        return "Invalid skill name: skill_name cannot be empty"

    skills = _get_merged_skills()
    skill = skills.get(skill_name)

    if skill:
        if skill.disable_model_invocation:
            return (
                f"Skill '{skill_name}' cannot be loaded by the model "
                f"(disable_model_invocation=true). It is user-invocable only."
            )
        _apply_skill_restrictions(skill)
        return skill.to_prompt()

    available = sorted(skills.keys())
    if not available:
        return f"Skill '{skill_name}' not found. No skills are currently available."
    return f"Skill '{skill_name}' not found. Available skills: {', '.join(available)}"
