"""Workspace and filesystem policy helpers for sandboxed shell execution."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from koder_agent.harness.permissions.shell_classifier import classify_shell_command

from .policy import SandboxPolicy

WRITE_COMMANDS = {
    "cat",
    "chmod",
    "chown",
    "cp",
    "dd",
    "install",
    "ln",
    "mkdir",
    "mv",
    "rm",
    "rmdir",
    "tee",
    "touch",
    "truncate",
}

# Interpreters whose payload is opaque to the literal-token preflight: the code
# they run (via -c/-e/-E or stdin) can touch protected paths without the path
# ever appearing as a bare token (finding #5).
_INTERPRETER_NAMES = {
    "sh",
    "bash",
    "zsh",
    "dash",
    "ksh",
    "fish",
    "python",
    "python2",
    "python3",
    "perl",
    "ruby",
    "node",
    "deno",
    "bun",
    "php",
    "lua",
    "tclsh",
    "Rscript",
    "awk",
    "gawk",
    "mawk",
}
# Commands that serve as wrappers/prefixes before an interpreter (e.g.
# ``env bash -c ...``, ``sudo python3 -c ...``) — the interpreter check must
# look past these to find the real command word.
_COMMAND_PREFIXES = {"env", "sudo", "nice", "nohup", "command", "exec", "xargs", "time"}
# Flags that introduce an inline code payload for the interpreters above.
_INLINE_CODE_FLAGS = {"-c", "-e", "-E", "--eval", "--exec"}
_HEREDOC_RE = re.compile(r"<<-?\s*[\"']?[A-Za-z_][A-Za-z0-9_]*")


# Leading redirection operators (optionally fd-prefixed, e.g. ``1>``, ``2>>``,
# ``&>``) that shlex leaves glued to a target when there is no separating space:
# ``echo x >.env.local`` tokenizes to the single token ``>.env.local``. Stripping
# the operator recovers the real write target ``.env.local`` so the deny-write /
# protected-path scan sees it (finding: no-space-redirection preflight bypass).
_REDIRECTION_PREFIX_RE = re.compile(r"^\d*(?:&?>>?|<<?|>&|<&|&>>?)")


def _strip_redirection_prefix(token: str) -> str:
    """Return *token* with a leading redirection operator removed.

    ``>.env.local`` -> ``.env.local``; ``1>>out`` -> ``out``; ``&>x`` -> ``x``.
    A bare operator (``>`` with the target in a separate token) collapses to the
    empty string and is filtered out by the caller.
    """
    return _REDIRECTION_PREFIX_RE.sub("", token, count=1)


def _token_paths(command: str) -> tuple[str, ...]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return ()
    paths: list[str] = []
    for token in tokens:
        if not token or token.startswith("-"):
            continue
        # Recover a target glued to a redirection operator (``>.env.local``); a
        # spaced operator (``>`` alone) collapses to "" and is dropped.
        stripped = _strip_redirection_prefix(token)
        if stripped:
            paths.append(stripped)
    return tuple(paths)


# A write redirection to a file: ``>``/``>>`` optionally fd-prefixed (``1>``,
# ``2>>``, ``&>``), NOT to /dev/null. Detected independently of the classifier,
# whose ``WRITE_REDIRECTION_PATTERN`` deliberately excludes a digit before ``>``
# and so misses ``1>.env.local`` (finding: fd-prefixed redirection preflight
# bypass). Any such redirection means the command writes a file, regardless of
# whether the leading word is a known write command (``echo x >f`` writes ``f``).
_WRITE_REDIRECTION_RE = re.compile(r"(?:^|\s|\d)&?>>?(?!\s*/dev/null\b)")


def _has_write_redirection(command: str) -> bool:
    return bool(_WRITE_REDIRECTION_RE.search(command))


def _looks_like_write_command(command: str) -> bool:
    if _has_write_redirection(command):
        return True
    try:
        tokens = shlex.split(command)
    except ValueError:
        return True
    if not tokens:
        return False
    if tokens[0] in WRITE_COMMANDS:
        return True
    return classify_shell_command(command).requires_approval


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _basename(name: str) -> str:
    # Strip a leading path so `python3`, `/usr/bin/python3`, `./python` all match.
    return name.rsplit("/", 1)[-1]


def interpreter_payload_violation(command: str) -> str | None:
    """Flag commands whose real work hides inside an interpreter payload/heredoc.

    The literal-token preflight only sees bare path tokens, so
    ``python3 -c "open('.git/config','w')"`` or a ``cat <<EOF`` heredoc slips
    through untouched (finding #5). The unix-local backend cannot enforce
    protected paths at the kernel level, so rather than let such payloads pass
    the preflight silently we surface a violation requiring approval.
    """

    if _HEREDOC_RE.search(command):
        return "command uses a heredoc; interpreter payload cannot be verified by preflight"

    try:
        tokens = shlex.split(command)
    except ValueError:
        # Unparseable (e.g. unbalanced quotes) — treat conservatively as opaque.
        return "command could not be parsed for preflight; requires approval"

    for index, token in enumerate(tokens):
        if not token or token.startswith("-"):
            continue
        base = _basename(token)
        if base in _INTERPRETER_NAMES:
            rest = tokens[index + 1 :]
            if any(flag in _INLINE_CODE_FLAGS for flag in rest):
                return (
                    f"command runs an inline {base} payload; "
                    "interpreter code cannot be verified by preflight"
                )
            # No inline payload was identified. This heuristic does not inspect
            # script contents or establish that the interpreter invocation is safe.
            break
        if base in _COMMAND_PREFIXES:
            # Look past wrapper commands (env, sudo, etc.) to the next token.
            continue
        # Non-interpreter, non-prefix command word — stop scanning.
        break
    return None


def protected_write_violation(
    command: str,
    *,
    policy: SandboxPolicy,
    repo_root: Path,
) -> str | None:
    """Best-effort exact-path preflight for protected metadata writes.

    The SDK local backend enforces workspace escapes at the OS boundary. Some backends do not
    expose read-only carve-outs for subpaths inside the writable workspace, so Koder adds a
    conservative exact-token preflight before running commands in the sandbox.
    """

    # Interpreter payloads / heredocs hide their real work from the literal-token
    # scan below; in a confined mode we must not let them pass silently (finding
    # #5). Kernel-level enforcement is out of scope — flagging for approval is.
    if policy.mode in {"workspace-write", "read-only"}:
        interpreter_violation = interpreter_payload_violation(command)
        if interpreter_violation is not None:
            return interpreter_violation

    if not _looks_like_write_command(command):
        return None

    protected_roots = policy.protected_path_roots(repo_root)
    for token in _token_paths(command):
        if token in WRITE_COMMANDS or token in {"sh", "bash", "zsh", "python", "python3"}:
            continue
        if ":" in token and not token.startswith("/"):
            continue
        # Glob deny_write patterns (e.g. `.env.*`) were previously dropped from
        # protected roots; match write targets against them by name/path here so
        # `touch .env.local` and `echo x > .env.production` are flagged (finding #3).
        matched_glob = policy.matches_deny_write_glob(token)
        if matched_glob is not None:
            return f"write targets protected path {token} (matches deny_write {matched_glob})"
        raw = Path(token).expanduser()
        if not raw.is_absolute():
            raw = repo_root / raw
        # Check the currently resolved symlink target, including nonexistent
        # trailing components. The filesystem can change after this check;
        # realpath is not a race-free substitute for backend enforcement.
        candidate = Path(os.path.realpath(str(raw)))
        for protected_root in protected_roots:
            if candidate == protected_root or _is_under(candidate, protected_root):
                try:
                    display = candidate.relative_to(repo_root)
                except ValueError:
                    display = candidate
                return f"write targets protected path {display}"
        # Additionally check the resolved path against deny_write globs — a
        # symlink's real target name may match a protected glob even when the
        # symlink name itself does not (symlink-based bypass of glob matching).
        resolved_glob = policy.matches_deny_write_glob(candidate.name)
        if resolved_glob is None and candidate != raw:
            try:
                resolved_rel = str(candidate.relative_to(repo_root))
                resolved_glob = policy.matches_deny_write_glob(resolved_rel)
            except ValueError:
                pass
        if resolved_glob is not None:
            return (
                f"write targets protected path {token} "
                f"(resolves to {candidate.name}, matches deny_write {resolved_glob})"
            )
    return None


def read_only_violation(command: str, *, policy: SandboxPolicy) -> str | None:
    if policy.mode != "read-only":
        return None
    decision = classify_shell_command(command)
    if decision.requires_approval or not decision.allowed:
        return "read-only sandbox mode does not allow mutating shell commands"
    return None
