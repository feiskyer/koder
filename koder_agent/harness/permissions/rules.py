"""Permission rule parsing and matching primitives."""

from __future__ import annotations

import fnmatch
import os
import shlex
from dataclasses import dataclass

from .shell_segments import parse_shell_segments


@dataclass(frozen=True)
class PermissionRule:
    """Normalized permission rule."""

    kind: str
    value: str


def parse_permission_rule(permission_rule: str) -> PermissionRule:
    """Parse exact, legacy prefix, path-prefix, or wildcard rules."""
    if permission_rule.startswith("Skill(") and permission_rule.endswith(")"):
        permission_rule = permission_rule[len("Skill(") : -1].strip()
    # A trailing "/" (or the platform separator) marks a directory/path-prefix
    # rule, e.g. "/a/b/", so matching happens on path-component boundaries
    # rather than shell-token boundaries. This is how an approved file edit
    # persists a per-directory rule.
    if permission_rule.endswith("/") or permission_rule.endswith(os.sep):
        return PermissionRule(kind="path_prefix", value=permission_rule)
    if permission_rule.endswith(":*"):
        return PermissionRule(kind="prefix", value=permission_rule[:-2])
    if "*" in permission_rule:
        return PermissionRule(kind="wildcard", value=permission_rule)
    return PermissionRule(kind="exact", value=permission_rule)


def _normalize_path_prefix(value: str) -> str:
    """Normalize a directory prefix to a canonical, trailing-separator form."""
    stripped = value.rstrip("/").rstrip(os.sep)
    normalized = os.path.normpath(os.path.expanduser(stripped))
    return normalized.rstrip(os.sep) + os.sep


def match_permission_rule(rule: PermissionRule, command: str) -> bool:
    """Match a shell command or file target against a normalized permission rule."""
    if rule.kind == "exact":
        return command == rule.value
    if rule.kind == "prefix":
        # A prefix rule "npm test" matches the bare command and any command that
        # extends it with a following token (a flag/arg), e.g. "npm test --watch".
        # It must NOT match "npm testfoo" — the next character has to be a space.
        return command == rule.value or command.startswith(f"{rule.value} ")
    if rule.kind == "path_prefix":
        prefix = _normalize_path_prefix(rule.value)
        target = os.path.normpath(os.path.expanduser(command))
        # The directory itself and any descendant path match; a sibling whose
        # name merely shares the prefix string ("/a/bc" vs "/a/b/") must not.
        return (target + os.sep).startswith(prefix)
    if rule.kind == "wildcard":
        pattern = rule.value
        # Preserve legacy behavior where a trailing " *" wildcard also matches bare command.
        if pattern.endswith(" *") and fnmatch.fnmatchcase(command, pattern[:-2]):
            return True
        return fnmatch.fnmatchcase(command, pattern)
    raise ValueError(f"Unknown rule kind: {rule.kind}")


# --- Prefix rule derivation for "always allow" ------------------------------
#
# When a user chooses "always allow" for a tool call, we want the decision to
# generalize sensibly instead of keying on the exact command/target string (a
# path or flag change would otherwise re-prompt — the permission-fatigue bug).
#
# The derivation below is deliberately CONSERVATIVE: it only widens commands it
# can positively prove are safe verbs, and refuses to widen anything
# destructive, privileged, chained, or otherwise ambiguous. When it cannot
# derive a safe prefix it returns ``None`` and the caller should fall back to an
# exact rule.

# Safe base commands whose invocations are widened to a bare-command prefix
# (``<cmd>:*``). These run project build/test tooling that the user must already
# trust to approve at all; widening a flag change is low-risk.
_SAFE_SINGLE_TOKEN_VERBS = frozenset(
    {
        "cargo",
        "go",
        "gradle",
        "make",
        "mvn",
        "poetry",
        "pytest",
        "rake",
        "tox",
        "uv",
    }
)

# Safe ``<cmd> <subcommand>`` pairs that are widened to ``<cmd> <sub>:*`` so a
# following flag (``npm test --watch``) reuses the same rule. Only inspect/test
# oriented subcommands are listed; install/publish/exec style subcommands are
# intentionally omitted so they keep prompting.
_SAFE_TWO_TOKEN_VERBS: dict[str, frozenset[str]] = {
    "npm": frozenset({"test", "run", "ci", "lint", "audit"}),
    "pnpm": frozenset({"test", "run", "lint"}),
    "yarn": frozenset({"test", "run", "lint"}),
    "bun": frozenset({"test", "run"}),
    "cargo": frozenset({"test", "build", "check", "clippy", "fmt", "bench", "doc"}),
    "go": frozenset({"test", "build", "vet", "run"}),
    "git": frozenset({"status", "log", "diff", "show"}),
    "pip": frozenset({"list", "show", "freeze"}),
    "docker": frozenset({"ps", "images", "logs"}),
    "kubectl": frozenset({"get", "describe", "logs"}),
}

# Commands that must NEVER be widened to a prefix, even if reachable through the
# generic single-token path. Deleting/moving/permission commands, privilege
# escalation, and network fetchers stay exact so a later, differently-targeted
# invocation re-prompts.
_NEVER_WIDEN = frozenset(
    {
        "rm",
        "rmdir",
        "mv",
        "cp",
        "dd",
        "mkfs",
        "chmod",
        "chown",
        "chgrp",
        "ln",
        "shred",
        "truncate",
        "tee",
        "sudo",
        "doas",
        "su",
        "kill",
        "killall",
        "pkill",
        "shutdown",
        "reboot",
        "halt",
        "curl",
        "wget",
        "ssh",
        "scp",
        "eval",
        "exec",
        "source",
    }
)


def _shell_prefix_is_safe(tokens: list[str]) -> str | None:
    """Return a widened ``<...>:*`` prefix rule for safe verbs, else ``None``.

    ``tokens`` is a single command segment (already split from any chain). The
    A basename selects the known verb family, but the resulting rule retains
    the actual executable path the user approved.
    """
    if not tokens:
        return None
    base = os.path.basename(tokens[0])
    if not base or base in _NEVER_WIDEN:
        return None
    # A relative/explicit script path ("./deploy.sh") is opaque unless its
    # basename is a recognized safe verb; never widen an arbitrary script path.
    is_known_verb = base in _SAFE_SINGLE_TOKEN_VERBS or base in _SAFE_TWO_TOKEN_VERBS
    if "/" in tokens[0] and not is_known_verb:
        return None

    two_token_subs = _SAFE_TWO_TOKEN_VERBS.get(base)
    if two_token_subs is not None and len(tokens) >= 2 and tokens[1] in two_token_subs:
        return f"{shlex.join(tokens[:2])}:*"

    if base in _SAFE_SINGLE_TOKEN_VERBS:
        return f"{shlex.quote(tokens[0])}:*"

    return None


def _default_shell_segments(command: str) -> list[list[str]]:
    """Use the same literal segment boundaries as permission evaluation."""
    return [list(segment.tokens) for segment in parse_shell_segments(command)]


def derive_shell_prefix_rule(command: str, *, segmenter=None) -> str | None:
    """Derive a conservative prefix rule for an approved shell command.

    Returns a ``<prefix>:*`` rule string when the command is a single, clearly
    safe verb we can widen, else ``None`` (the caller then persists the exact
    command instead).

    Refuses to widen when:
      * the command chains multiple segments (``a && b``) — widening one segment
        would silently authorize the others,
      * the command cannot be parsed,
      * the (only) segment is destructive/privileged/unknown.

    ``segmenter`` is injected only for testing; it defaults to the shared
    quote-aware segmenter.
    """
    if not command or not command.strip():
        return None
    if segmenter is None:
        segmenter = _default_shell_segments
    try:
        segments = [tokens for tokens in segmenter(command) if tokens]
    except ValueError:
        return None
    # Never widen a chain: approving ``ls && rm -rf x`` must stay exact.
    if len(segments) != 1:
        return None
    prefix = _shell_prefix_is_safe(segments[0])
    # Unusual quoting/spacing remains an exact approval rather than publishing
    # a different spelling that may not authorize the original request.
    if prefix is not None and not command.lstrip().startswith(prefix[:-2]):
        return None
    return prefix


def derive_path_prefix_rule(target: str) -> str | None:
    """Derive a per-directory path-prefix rule for an approved file target.

    Approving an edit to ``/proj/src/app.py`` persists a ``/proj/src/`` rule so a
    sibling edit in the same directory (``/proj/src/util.py``) is auto-allowed
    without re-prompting, while a file in a different directory still prompts.

    Returns ``None`` when no directory can be derived (e.g. a bare filename with
    no parent), so the caller falls back to an exact rule.
    """
    if not target or not target.strip():
        return None
    expanded = os.path.expanduser(target)
    parent = os.path.dirname(expanded)
    if not parent:
        return None
    normalized = os.path.normpath(parent)
    return normalized.rstrip(os.sep) + os.sep
