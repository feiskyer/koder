"""Skill context manager for tracking active skill restrictions.

This module provides async-safe state management for skill-based tool restrictions
using Python's contextvars. When a skill with `allowed_tools` is loaded, only
those tools (plus always-allowed tools) can be used.

The restriction model uses UNION semantics:
- Multiple skills with `allowed_tools` accumulate their allowed tools
- Loading a skill without `allowed_tools` is a NO-OP for restrictions; it does
  NOT clear restrictions contributed by previously-loaded restricted skills.
  (Allowing an unrestricted skill to clear restrictions would let the model
  self-escape its sandbox by loading any benign skill.) Use the explicit
  `clear_restrictions()` API to reset state.

Pattern syntax for allowed_tools:
- "read_file"           - Exact tool name match
- "run_shell:git *"     - Shell commands matching glob pattern
- "run_powershell:Get-*" - PowerShell commands matching glob pattern
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
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Optional

from agents import RunHooks

if TYPE_CHECKING:
    from .skill import Skill

# Context variable to track active skill restrictions (async-safe)
_active_restrictions: ContextVar[Optional["SkillRestrictions"]] = ContextVar(
    "active_skill_restrictions", default=None
)
_active_skill_invocation: ContextVar[object | None] = ContextVar(
    "active_skill_invocation", default=None
)
_active_skill_run_state: ContextVar[Optional["_SkillRunState"]] = ContextVar(
    "active_skill_run_state", default=None
)

# Substrings that indicate command/process substitution. A pattern like
# ``git *`` cannot reason about what runs inside ``$(...)`` / backticks, so any
# command containing these is rejected outright instead of glob-matched.
_SUBSTITUTION_MARKERS = ("$(", "`", "<(", ">(", "${")


def _contains_substitution(command: str) -> bool:
    return any(marker in command for marker in _SUBSTITUTION_MARKERS)


def _command_matches_pattern(command: str, pattern: str) -> bool:
    """Return True only if EVERY chained segment of *command* matches *pattern*.

    A naive ``fnmatch(command, "git *")`` lets ``git status; rm -rf /`` through
    because the whole string still starts with ``git ``. We instead split the
    command into segments on shell operators (``;`` ``&&`` ``||`` ``|`` and
    newlines) using the quote-aware tokenizer, then require the pattern to match
    every segment. Command/process substitution is rejected outright because a
    first-token pattern cannot police what runs inside it.

    ``pattern == "*"`` keeps its "allow anything" meaning (the caller uses it as
    an explicit escape hatch), but still rejects substitution smuggling.
    """
    if _contains_substitution(command):
        return False

    segments = _split_command_segments(command)
    if not segments:
        # No runnable segment (e.g. empty or only operators): match only if the
        # pattern would also match the empty/stripped command string.
        return fnmatch.fnmatch(command.strip(), pattern)

    return all(fnmatch.fnmatch(segment, pattern) for segment in segments)


_SEGMENT_SEPARATORS = {"|", "||", "&&", ";", ";;", "&"}
_OPERATOR_ONLY_CHARS = set(";&|<>")


def _split_command_segments(command: str) -> list[str]:
    """Split a command line into per-segment strings on shell operators.

    Uses a quote-aware ``shlex`` tokenizer (``punctuation_chars=True``) so that
    operators inside quotes -- e.g. the ``|`` in ``grep 'a|b'`` -- are NOT
    treated as segment separators. Segments are reconstructed as space-joined
    tokens for glob matching. Falls back to a conservative regex split (stricter,
    never looser) if the command cannot be tokenized (e.g. unbalanced quotes).

    Newlines separate whole commands at execution time (``shell.py`` passes the
    raw string to ``/bin/sh -c``), but ``shlex`` treats ``\\n`` as ordinary
    whitespace and would merge ``git log\\nrm -rf /`` into a single segment that
    fnmatches ``git *``. So the raw command is split on line boundaries FIRST and
    each physical line is tokenized independently.
    """
    import shlex

    segments: list[str] = []
    for line in command.splitlines():
        if not line.strip():
            continue
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            tokens = list(lexer)
        except ValueError:
            import re

            parts = re.split(r"(?:\|\||&&|[|;&])", line)
            segments.extend(part.strip() for part in parts if part.strip())
            continue

        current: list[str] = []
        for token in tokens:
            if token in _SEGMENT_SEPARATORS or (
                token and all(ch in _OPERATOR_ONLY_CHARS for ch in token)
            ):
                if current:
                    segments.append(" ".join(current))
                    current = []
                continue
            current.append(token)
        if current:
            segments.append(" ".join(current))
    return segments


@dataclass
class SkillRestrictions:
    """Tracks tool restrictions from active skills.

    Uses union semantics: tools from multiple loaded skills are combined.

    Pattern syntax for allowed_tools:
    - "read_file"           - Exact tool name match
    - "run_shell:git *"     - Shell commands matching glob pattern
    - "run_powershell:Get-*" - PowerShell commands matching glob pattern
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
        - Command pattern: "run_powershell:Get-*" matches run_powershell commands

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

            # For shell tools, match against the command argument
            if tool_name in {"run_shell", "run_powershell"} and tool_args:
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
            if not isinstance(command, str):
                return False
            return _command_matches_pattern(command, pattern)
        except (json.JSONDecodeError, TypeError, AttributeError):
            return False

    def _matches_git_command(self, pattern: str, tool_args: str) -> bool:
        """Match a git command against a glob pattern.

        Args:
            pattern: Glob pattern to match (e.g., "status", "commit *", "*")
            tool_args: JSON string containing {"command": "..."}

        Returns:
            True if the git args match the pattern
        """
        try:
            args = json.loads(tool_args)
            if not isinstance(args, dict):
                return False
            git_args = args.get("command", "")
            if not isinstance(git_args, str):
                return False
            # ``git_command`` runs a single ``git <args>`` invocation, but the
            # args string can still smuggle chained commands (``status; rm -rf /``)
            # if consumed by a shell. Reject any segment that does not match the
            # pattern, and reject operators/substitutions the pattern didn't
            # account for -- same defense as the shell matcher below.
            return _command_matches_pattern(git_args, pattern)
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


@dataclass(frozen=True)
class _SkillActivationBatch:
    """Skill activation calls from one Runner model-response batch."""

    activation_call_ids: frozenset[str]

    def blocks(self, tool_name: str, call_id: str | None) -> bool:
        if not self.activation_call_ids:
            return False
        if tool_name == "get_skill" and str(call_id or "") in self.activation_call_ids:
            return False
        # An activation batch must fail closed if the SDK reconstructs a call
        # without preserving the expected call metadata.
        return True


@dataclass
class _SkillRunState:
    """Runner-local activation state inherited by that Runner's tool tasks."""

    response: Any | None = None

    def bind_response(self, response: Any) -> None:
        """Bind the latest response for tool-call policy checks."""
        self.response = response

    def effective_batch(self) -> _SkillActivationBatch | None:
        """Refresh from the final response output and return its tool-call batch."""
        if self.response is None:
            return None
        return self._refresh_batch()

    def _refresh_batch(self) -> _SkillActivationBatch:
        calls = [
            item
            for item in getattr(self.response, "output", [])
            if getattr(item, "type", None) == "function_call"
        ]
        activation_call_ids = frozenset(
            str(getattr(item, "call_id", "") or "")
            for item in calls
            if getattr(item, "name", None) == "get_skill"
        )
        return _SkillActivationBatch(activation_call_ids=activation_call_ids)


def get_active_restrictions() -> Optional[SkillRestrictions]:
    """Get the currently active skill restrictions.

    Returns:
        SkillRestrictions instance if restrictions are active, None otherwise
    """
    return _active_restrictions.get()


def get_skill_activation_block_message(
    tool_name: str,
    call_id: str | None,
) -> str | None:
    """Return a model-visible rejection when activation and work share a batch."""
    state = _active_skill_run_state.get()
    batch = state.effective_batch() if state is not None else None
    if batch is None or not batch.blocks(tool_name, call_id):
        return None
    return (
        f"Tool '{tool_name}' was not executed because this model response also requested "
        "get_skill. Skill activation must complete before sibling tools can run; call this "
        "tool again in the next model step with a new tool call ID."
    )


def clear_restrictions() -> None:
    """Clear any active skill restrictions.

    This is an explicit reset API. It is intentionally NOT called when a skill
    without `allowed_tools` is loaded (see module docstring) -- loading an
    unrestricted skill must not erase another skill's active restrictions.
    """
    _active_restrictions.set(None)


def begin_skill_restriction_scope(skill: "Skill | None" = None) -> Token:
    """Seed a persistent restrictions container at the current (run-loop) scope.

    The scheduler MUST call this before ``Runner.run`` so that skill restrictions
    survive the SDK task boundary. The openai-agents runner executes every tool
    call inside its own ``asyncio.Task``, which runs in a COPY of the context
    captured when the task was created. A ContextVar ``.set()`` performed *inside*
    a tool task (as ``get_skill`` does when it loads a restricted skill) mutates
    only that task's copy and is invisible to sibling / later tool tasks.

    By seeding a mutable ``SkillRestrictions`` container in the parent scope up
    front, ``add_skill_restrictions`` can mutate it *in place* from within any
    child tool task; because contextvar copies share the same object *reference*,
    every other tool task under the same parent observes the mutation. This
    mirrors ``set_tool_permission_context`` / ``set_goal_context``.

    Nested scopes inherit a snapshot of the current restrictions. This is needed
    for manual ``/<skill>`` invocation: the command activates the skill before
    calling ``scheduler.handle()``, and the scheduler then seeds the SDK run-loop
    without discarding that policy. A snapshot keeps additions made by nested
    model-loaded skills local to the nested run while retaining union semantics
    inside it.

    Args:
        skill: Optional manually-invoked skill to activate in the new scope.

    Returns a token for :func:`reset_skill_restriction_scope`.
    """
    current = _active_restrictions.get()
    seeded = (
        SkillRestrictions(
            loaded_skills=list(current.loaded_skills),
            allowed_tools=set(current.allowed_tools),
        )
        if current is not None
        else SkillRestrictions()
    )
    token = _active_restrictions.set(seeded)
    if skill is not None:
        add_skill_restrictions(skill)
    return token


def reset_skill_restriction_scope(token: Token) -> None:
    """Restore the restrictions container captured before the scope began."""
    try:
        _active_restrictions.reset(token)
    except (ValueError, LookupError):
        # Token created in a different context (e.g. across task boundaries);
        # best-effort clear instead.
        _active_restrictions.set(None)


@contextmanager
def skill_restriction_scope(
    skill: "Skill | None" = None,
) -> Iterator[SkillRestrictions]:
    """Backward-compatible alias for the unified skill invocation scope."""
    with skill_invocation_scope(skill) as restrictions:
        yield restrictions


@contextmanager
def skill_invocation_scope(
    skill: "Skill | None" = None,
) -> Iterator[SkillRestrictions]:
    """Own restrictions and dynamic hooks for one logical scheduler turn.

    Nested users join the existing mutable scope instead of creating a
    continuation-local snapshot. This lets manual slash skills, model-loaded
    skills, hidden goal continuations, and forked agents share one policy and
    one cleanup boundary.
    """
    if _active_skill_invocation.get() is not None:
        if skill is not None:
            activate_skill_policy(skill)
        restrictions = _active_restrictions.get()
        assert restrictions is not None
        yield restrictions
        return

    from koder_agent.harness.hooks.runtime import (
        begin_skill_hook_scope,
        reset_skill_hook_scope,
    )

    restriction_token = begin_skill_restriction_scope()
    hook_token = begin_skill_hook_scope()
    invocation_token = _active_skill_invocation.set(object())
    try:
        if skill is not None:
            activate_skill_policy(skill)
        restrictions = _active_restrictions.get()
        assert restrictions is not None
        yield restrictions
    finally:
        try:
            _active_skill_invocation.reset(invocation_token)
        finally:
            try:
                reset_skill_hook_scope(hook_token)
            finally:
                reset_skill_restriction_scope(restriction_token)


class SkillPolicyRunHooks(RunHooks):
    """Run-hook wrapper that establishes batch policy before tool execution."""

    def __init__(self, state: _SkillRunState, wrapped_hooks: RunHooks | None = None):
        self.state = state
        self.wrapped_hooks = wrapped_hooks

    def __getattr__(self, name: str) -> Any:
        if self.wrapped_hooks is None:
            raise AttributeError(name)
        return getattr(self.wrapped_hooks, name)

    async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
        if self.wrapped_hooks is not None:
            await self.wrapped_hooks.on_llm_start(context, agent, system_prompt, input_items)

    async def on_llm_end(self, context, agent, response) -> None:
        if self.wrapped_hooks is not None:
            await self.wrapped_hooks.on_llm_end(context, agent, response)
        # Bind after the wrapped hook because it may mutate or reconstruct
        # response.output. The state also retains the response reference so the
        # guardrail can refresh after concurrently-running agent hooks finish.
        self.state.bind_response(response)

    async def on_agent_start(self, context, agent) -> None:
        if self.wrapped_hooks is not None:
            await self.wrapped_hooks.on_agent_start(context, agent)

    async def on_agent_end(self, context, agent, output) -> None:
        if self.wrapped_hooks is not None:
            await self.wrapped_hooks.on_agent_end(context, agent, output)

    async def on_handoff(self, context, from_agent, to_agent) -> None:
        if self.wrapped_hooks is not None:
            await self.wrapped_hooks.on_handoff(context, from_agent, to_agent)

    async def on_tool_start(self, context, agent, tool) -> None:
        if self.wrapped_hooks is not None:
            await self.wrapped_hooks.on_tool_start(context, agent, tool)

    async def on_tool_end(self, context, agent, tool, result) -> None:
        if self.wrapped_hooks is not None:
            await self.wrapped_hooks.on_tool_end(context, agent, tool, result)


@contextmanager
def skill_run_scope(hooks: RunHooks | None = None) -> Iterator[SkillPolicyRunHooks]:
    """Seed one tool-capable SDK run and wrap its hooks with the batch barrier."""
    with skill_invocation_scope():
        if isinstance(hooks, SkillPolicyRunHooks):
            hooks = hooks.wrapped_hooks
        state = _SkillRunState()
        state_token = _active_skill_run_state.set(state)
        try:
            yield SkillPolicyRunHooks(state, hooks)
        finally:
            _active_skill_run_state.reset(state_token)


def activate_skill_policy(skill: "Skill") -> None:
    """Add one skill's allowed-tool union and hooks to the active turn."""
    from koder_agent.harness.hooks.runtime import register_skill_hooks

    add_skill_restrictions(skill)
    register_skill_hooks(skill.name, skill.hooks, skill.base_dir)


def add_skill_restrictions(skill: "Skill") -> None:
    """Add tool restrictions from a loaded skill.

    Uses union semantics: if restrictions already exist, the skill's allowed
    tools are added to the existing set.

    Mutates the active container IN PLACE (never ``.set()`` a fresh object when a
    scope is already seeded) so a restriction registered from inside ``get_skill``'s
    tool task is visible to every other tool task sharing the parent context. If no
    scope has been seeded (e.g. a direct call outside a scheduler run), fall back to
    creating one so behavior is still correct for standalone callers/tests.

    Args:
        skill: The skill whose restrictions should be added
    """
    if not skill.allowed_tools:
        return

    current = _active_restrictions.get()

    if current is None:
        # No seeded scope (standalone call): create one. Note this ``.set()`` only
        # persists when called outside an isolated tool task; the scheduler seeds a
        # scope via begin_skill_restriction_scope so the in-place path below runs.
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
