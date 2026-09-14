"""Static shell command classification for permission decisions."""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass

from .sed_classifier import is_readonly_sed
from .shell_runners import resolve_runner_segment as _resolve_runner_segment
from .shell_segments import has_dynamic_shell_words, parse_shell_segments

READ_ONLY_COMMANDS = {
    "cat",
    "column",
    "cut",
    "diff",
    "echo",
    "file",
    "find",
    "git",
    "grep",
    "head",
    "jq",
    "ls",
    "md5sum",
    "nl",
    "od",
    "paste",
    "pwd",
    "rg",
    "sha1sum",
    "sha256sum",
    "sort",
    "stat",
    "strings",
    "tail",
    "tr",
    "uniq",
    "wc",
    "which",
}

READ_ONLY_GIT_SUBCOMMANDS = {
    "branch",
    "diff",
    "log",
    "reflog",
    "rev-parse",
    "show",
    "status",
}

# Flags that make otherwise read-only git subcommands into write operations
GIT_WRITE_FLAGS: dict[str, set[str]] = {
    "branch": {
        "-d",
        "-D",
        "--delete",
        "-m",
        "-M",
        "--move",
        "-c",
        "-C",
        "--copy",
        "--set-upstream-to",
        "--unset-upstream",
        "--edit-description",
    },
    "stash": {"pop", "drop", "apply", "push", "save", "clear", "create", "store"},
    "tag": {"-d", "--delete", "-f", "--force"},
    "remote": {"add", "remove", "rm", "rename", "set-url", "set-head", "prune"},
    "config": {
        "--unset",
        "--unset-all",
        "--remove-section",
        "--rename-section",
        "--replace-all",
    },
    "notes": {"add", "append", "copy", "edit", "merge", "remove", "prune"},
    "worktree": {"add", "remove", "prune", "move", "repair", "lock", "unlock"},
}

# Sub-subcommands that are definitely read-only for flag-checked subcommands
GIT_READONLY_SUB_SUBCOMMANDS: dict[str, set[str]] = {
    "stash": {"list", "show"},
    "remote": {"show", "get-url"},
    "worktree": {"list"},
    "notes": {"list", "show"},
    "config": {"--get", "--get-all", "--get-regexp", "--list", "-l"},
}

# config flags that only read values; any other form of ``git config`` (a bare
# ``key value`` assignment, ``--global user.email x``, ``core.hooksPath ...``)
# is a write and must require approval.
_GIT_CONFIG_READ_FLAGS = {
    "--get",
    "--get-all",
    "--get-regexp",
    "--get-urlmatch",
    "--list",
    "-l",
}
# config options that take no value and are safe to ignore when deciding whether
# an assignment is present (scope/type selectors, not the operation itself).
_GIT_CONFIG_SCOPE_FLAGS = {
    "--global",
    "--system",
    "--local",
    "--worktree",
    "-z",
    "--null",
    "--name-only",
    "--show-origin",
    "--show-scope",
}
# tag flags that force a write even without a positional tag name.
_GIT_TAG_WRITE_FLAGS = {"-a", "-s", "-m", "-F", "--annotate", "--sign", "--message", "--file"}
# tag flags that indicate read-only listing.
_GIT_TAG_READ_FLAGS = {"-l", "--list", "-n", "--contains", "--points-at", "--sort", "--format"}

# Extended set of always-read-only git subcommands
EXTENDED_READ_ONLY_GIT_SUBCOMMANDS = (
    READ_ONLY_GIT_SUBCOMMANDS
    | GIT_WRITE_FLAGS.keys()
    | {
        "blame",
        "shortlog",
        "describe",
        "ls-files",
        "ls-tree",
        "ls-remote",
        "cat-file",
        "name-rev",
        "for-each-ref",
        "count-objects",
        "fsck",
        "verify-pack",
        "verify-commit",
        "verify-tag",
        "whatchanged",
    }
)

# Branch flags that indicate read-only listing (not positional branch names)
_BRANCH_READ_FLAGS = {"-a", "--all", "-r", "--remotes", "-v", "--verbose", "-vv", "--list"}

_GIT_REFLOG_WRITE_COMMANDS = {"expire", "delete", "drop", "write"}


def _has_effectful_option(
    tokens: list[str],
    options: set[str],
    *,
    short_value_options: frozenset[str] = frozenset(),
    short_optional_value_options: frozenset[str] = frozenset(),
    long_value_options: frozenset[str] = frozenset(),
) -> bool:
    """Recognize effectful options in abbreviated and attached-value forms.

    Some programs accept long-option prefixes and clustered/attached short flags.
    A possible effectful prefix therefore requires approval, even when the
    real parser might reject it as ambiguous. Read-option arguments are not
    short-option clusters; optional values consume only an attached remainder.
    Stop at the option terminator.
    """
    long_options = {option for option in options if option.startswith("--")}
    short_options = {option[1] for option in options if len(option) == 2 and option[0] == "-"}
    skip_argument = False
    for token in tokens:
        if skip_argument:
            skip_argument = False
            continue
        if token == "--":
            break
        if token.startswith("--"):
            name, separator, _value = token.partition("=")
            if any(option.startswith(name) for option in long_options):
                return True
            skip_argument = not separator and name in long_value_options
        elif token.startswith("-"):
            for index, flag in enumerate(token[1:], start=1):
                if flag in short_options:
                    return True
                if flag in short_value_options or flag in short_optional_value_options:
                    skip_argument = flag in short_value_options and index == len(token) - 1
                    break
    return False


def _git_config_is_read_only(rest: list[str]) -> bool:
    """``git config`` is read-only only when it reads a value and sets nothing.

    A get/list flag must be present, and there must be no positional assignment
    (a bare ``key`` lookup after ``--get`` is fine, but ``key value`` or a bare
    ``key value`` with no read flag is a write). This blocks the
    ``git config core.hooksPath ...`` privilege-escalation vector.
    """
    if not rest:
        # Bare ``git config`` opens an editor / is not a pure read; treat as write.
        return False

    has_read_flag = any(token in _GIT_CONFIG_READ_FLAGS for token in rest)
    if not has_read_flag:
        return False

    # Any unset/replace/rename operation is a write even alongside a read flag.
    if _has_effectful_option(rest, GIT_WRITE_FLAGS["config"]):
        return False

    # A read flag plus at most one positional (the key to look up) is read-only;
    # a second positional is a value assignment -> write.
    positionals = [
        token
        for token in rest
        if not token.startswith("-")
        and token not in _GIT_CONFIG_READ_FLAGS
        and token not in _GIT_CONFIG_SCOPE_FLAGS
    ]
    return len(positionals) <= 1


def _git_tag_is_read_only(rest: list[str]) -> bool:
    """``git tag`` is read-only only for listing.

    Bare ``git tag`` lists tags. ``-l``/``--list`` (and other listing flags)
    keep it read-only even with a pattern argument. Any create/delete flag
    (``-a``/``-s``/``-m``/``-F``/``-d``/``--force`` ...) or a bare positional
    tag name with no listing flag is a write.
    """
    if _has_effectful_option(
        rest,
        _GIT_TAG_WRITE_FLAGS | GIT_WRITE_FLAGS["tag"],
        short_optional_value_options=frozenset({"l", "n"}),
    ):
        return False
    has_listing_flag = any(token in _GIT_TAG_READ_FLAGS for token in rest)
    has_positional = any(not token.startswith("-") for token in rest)
    # A positional with no listing flag names a tag to create -> write.
    if has_positional and not has_listing_flag:
        return False
    return True


def is_readonly_git_subcommand(tokens: list[str]) -> bool:
    """Check whether a tokenized git command is read-only.

    Examines flags and sub-subcommands to distinguish read-only invocations
    (e.g. ``git branch -a``) from write operations (e.g. ``git branch -D feat``).
    """
    if len(tokens) < 2 or tokens[0] != "git":
        return False

    subcommand = tokens[1]
    rest = tokens[2:]

    # ``--output[=]FILE`` (git's diff-machinery flag, accepted by log/show/
    # diff/...) writes/truncates an arbitrary path, so any otherwise read-only
    # subcommand carrying it must require approval. A stray ``-o FILE`` is
    # likewise treated as a write, but only for the core read-only subcommands:
    # on extended subcommands such as ``ls-files``, ``-o`` means ``--others``
    # and is a legitimate read-only flag.
    output_options = {"--output", "--ext-diff", "--textconv"}
    if subcommand in READ_ONLY_GIT_SUBCOMMANDS:
        output_options.add("-o")
    if _has_effectful_option(
        rest, output_options, short_value_options=frozenset({"G", "S", "L", "n", "O"})
    ):
        return False

    if subcommand == "reflog" and rest and rest[0] in _GIT_REFLOG_WRITE_COMMANDS:
        return False
    if subcommand == "fsck" and _has_effectful_option(rest, {"--lost-found"}):
        return False

    # Pure read-only subcommands that have no write flags
    if subcommand in READ_ONLY_GIT_SUBCOMMANDS and subcommand not in GIT_WRITE_FLAGS:
        return True

    # Subcommands with per-flag write rules
    if subcommand in GIT_WRITE_FLAGS:
        write_flags = GIT_WRITE_FLAGS[subcommand]

        # ``config`` and ``tag`` need value-assignment analysis, not just a flag
        # scan: a bare assignment such as ``git config key value`` has no write
        # flag yet still mutates state (and can rewrite hooksPath).
        if subcommand == "config":
            return _git_config_is_read_only(rest)
        if subcommand == "tag":
            return _git_tag_is_read_only(rest)

        # Check for known read-only sub-subcommands first
        readonly_subs = GIT_READONLY_SUB_SUBCOMMANDS.get(subcommand, set())
        if rest and rest[0] in readonly_subs:
            return True

        # ``stash``/``remote``/``notes``/``worktree`` are read-only ONLY via an
        # explicit read-only sub-subcommand. A bare ``git stash`` (== stash push)
        # or bare ``git remote`` add-form must not be auto-allowed. ``remote``
        # additionally accepts ``-v``/``--verbose`` as a listing flag.
        if subcommand in {"stash", "remote", "notes", "worktree"}:
            if (
                subcommand == "remote"
                and rest
                and all(token in {"-v", "--verbose"} for token in rest)
            ):
                return True
            return False

        # Check if any write flag is present
        if _has_effectful_option(rest, write_flags):
            return False

        # Special case for "branch": positional args mean branch creation
        if subcommand == "branch":
            for token in rest:
                if not token.startswith("-") and token not in _BRANCH_READ_FLAGS:
                    return False

        return True

    # Extended read-only subcommands (blame, ls-files, etc.)
    if subcommand in EXTENDED_READ_ONLY_GIT_SUBCOMMANDS:
        return True

    return False


WRITE_COMMANDS = {
    "chmod",
    "chown",
    "cp",
    "git",
    "mkdir",
    "mv",
    "rm",
    "rmdir",
    "tee",
    "touch",
}

# Privilege-escalation commands: hard-denied, never routed to an approval prompt.
PRIVILEGED_PREFIXES = {
    "doas",
    "su",
    "sudo",
}

# Pre-compiled regex to catch privilege escalation wrapped in subshell groups.
# shlex treats `(` as a regular character, so `(sudo rm -rf /)` bypasses
# token-level hard-deny unless we check the raw command first.
_SUBSHELL_PRIV_RE = re.compile(
    r"\(\s*(?:" + "|".join(re.escape(p) for p in PRIVILEGED_PREFIXES) + r")\b"
)

# Interpreters and script runners: they execute arbitrary code so they always
# require approval, but they are everyday dev tools (pytest, build scripts,
# node tooling) and must stay approvable rather than hard-denied.
CODE_EXECUTION_PREFIXES = {
    "bash",
    "bun",
    "bunx",
    "deno",
    "eval",
    "exec",
    "fish",
    "lua",
    "node",
    "npm",
    "npx",
    "perl",
    "php",
    "python",
    "python2",
    "python3",
    "ruby",
    "sh",
    "ssh",
    "tsx",
    "yarn",
    "zsh",
}

DANGEROUS_PATTERNS = [
    re.compile(r">\s*/dev/(?!null\b)", re.IGNORECASE),
    re.compile(r"\bdd\s+if=", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:", re.IGNORECASE),
]

# A write redirection to a file: ``>``/``>>``, optionally fd-prefixed (``1>``,
# ``2>>``) — a digit before ``>`` picks the stream but still truncates/appends
# the target file, so it MUST count as a write (mirrors bash_security's
# ``_REDIRECT_RE``). Exemptions: ``/dev/null`` (discard) and fd duplication /
# closing (``2>&1``, ``>&2``, ``2>&-``), which retarget a descriptor rather
# than write a file. Other ``/dev/*`` targets are caught earlier as dangerous.
WRITE_REDIRECTION_PATTERN = re.compile(r"\d*>>?(?!\s*/dev/null\b)(?!&(?:\d|-))")


@dataclass(frozen=True)
class ShellCommandDecision:
    """Static safety classification for a shell command."""

    command: str
    allowed: bool
    read_only: bool
    requires_approval: bool
    destructive: bool
    malformed: bool
    reason: str


# find actions that mutate the filesystem, execute commands, or write to a
# caller-supplied file (the whole ``-f*`` output family truncates its target).
_FIND_MUTATING_FLAGS = {
    "-delete",
    "-exec",
    "-execdir",
    "-fls",
    "-fprint",
    "-fprint0",
    "-fprintf",
    "-ok",
    "-okdir",
}


def _normalize_command_name(token: str) -> str:
    """Return an executable basename for conservative hazard recognition.

    This must not grant readonly trust: ``./ls`` can be unrelated executable
    code. Basename matching may make a denial stricter, never an allow broader.
    """
    if not token:
        return token
    return os.path.basename(token)


def normalize_segment_for_rule(tokens: list[str], *, for_deny: bool = False) -> str | None:
    """Return the effective-command string for a tokenized segment, or ``None``.

    Only known argument-preserving runners generalize an inner allow rule.
    Environment assignments are not discarded: they can change executable
    lookup or loading, and shlex has lost whether an assignment-like name was
    quoted (and thus actually names a program). Full-command rules remain
    available for these calls. Deny-only projection may discard assignments
    and paths, but it must never grant an allow rule.
    """
    if not tokens:
        return None
    inner, resolvable = _resolve_runner_segment(tokens, for_deny=for_deny)
    if not inner or (not for_deny and not resolvable):
        return None
    normalized = shlex.join(inner)
    if not normalized or normalized == shlex.join(tokens):
        # Nothing was stripped: the raw target already equals the normalized one.
        return None
    return normalized


def _sort_is_effectful(tokens: list[str]) -> bool:
    """Output files and caller-selected compression programs are not pure reads."""
    return _has_effectful_option(
        tokens[1:],
        {"-o", "--output", "--compress-program"},
        short_value_options=frozenset({"k", "t", "S", "T"}),
        long_value_options=frozenset(
            {
                "--key",
                "--field-separator",
                "--buffer-size",
                "--temporary-directory",
                "--files0-from",
                "--parallel",
                "--batch-size",
                "--random-source",
            }
        ),
    )


def _is_read_only_segment(tokens: list[str], *, raw_command: str | None = None) -> bool:
    if not tokens:
        return True
    if tokens[0] != _normalize_command_name(tokens[0]):
        return False
    command = _normalize_command_name(tokens[0])
    if command == "git":
        return is_readonly_git_subcommand([command, *tokens[1:]])
    if command == "sed":
        return is_readonly_sed(tokens, raw_command=raw_command)
    if command == "find":
        return not any(token in _FIND_MUTATING_FLAGS for token in tokens)
    if command == "sort":
        return not _sort_is_effectful(tokens)
    if command == "rg":
        return not _has_effectful_option(
            tokens[1:],
            {"--pre", "--hostname-bin"},
            short_value_options=frozenset("efgtTABCmjMrE"),
            long_value_options=frozenset(
                {
                    "--regexp",
                    "--file",
                    "--glob",
                    "--iglob",
                    "--type",
                    "--type-not",
                    "--type-add",
                    "--type-clear",
                    "--pre-glob",
                    "--replace",
                }
            ),
        )
    if command == "file":
        return not _has_effectful_option(
            tokens[1:],
            {"-C", "--compile"},
            short_value_options=frozenset("mfFeP"),
            long_value_options=frozenset(
                {"--magic-file", "--files-from", "--separator", "--exclude", "--parameter"}
            ),
        )
    return command in READ_ONLY_COMMANDS


def _is_write_segment(tokens: list[str], *, raw_command: str | None = None) -> bool:
    if not tokens:
        return False
    if tokens[0] != _normalize_command_name(tokens[0]):
        return False
    command = _normalize_command_name(tokens[0])
    if command == "git":
        return not _is_read_only_segment(tokens)
    if command == "sed":
        return not is_readonly_sed(tokens, raw_command=raw_command)
    if command == "find":
        return any(token in _FIND_MUTATING_FLAGS for token in tokens)
    if command == "sort":
        return _sort_is_effectful(tokens)
    return command in WRITE_COMMANDS


def _is_privileged_segment(tokens: list[str], lowered_command: str) -> bool:
    """Segments that are hard-denied: privilege escalation or destructive deletes."""
    if not tokens:
        return False
    # Normalize argv[0] so ``/usr/bin/sudo`` and ``./sudo`` are caught, not just
    # a bare ``sudo`` token (absolute-path privilege bypass).
    command = _normalize_command_name(tokens[0])

    if command in PRIVILEGED_PREFIXES:
        return True

    if command in {"rm", "rmdir"}:
        # Detect recursive force delete of root regardless of flag style:
        # rm -rf /, rm -r -f /, rm --recursive --force /, rm -Rf /*, etc.
        args = tokens[1:]
        has_recursive = False
        has_force = False
        targets_root = False
        for tok in args:
            if tok.startswith("--"):
                if tok == "--recursive":
                    has_recursive = True
                elif tok == "--force":
                    has_force = True
            elif tok.startswith("-") and len(tok) > 1:
                # Combined short flags: -rf, -fr, -r, -f, -Rf, etc.
                flag_chars = tok[1:]
                if "r" in flag_chars or "R" in flag_chars:
                    has_recursive = True
                if "f" in flag_chars:
                    has_force = True
            else:
                # Non-flag argument: check if it targets root
                if tok == "/" or tok.startswith("/*"):
                    targets_root = True
        if has_recursive and has_force and targets_root:
            return True

    return False


def _is_code_execution_segment(tokens: list[str]) -> bool:
    """Segments that run arbitrary code: allowed, but always need approval."""
    if not tokens:
        return False
    command = _normalize_command_name(tokens[0])

    if command not in CODE_EXECUTION_PREFIXES:
        return False
    if command == "npm" and len(tokens) > 1 and tokens[1] not in {"run", "exec"}:
        return False
    if command in {"yarn", "bun"} and len(tokens) > 1 and tokens[1] not in {"run", "exec"}:
        return False
    return True


def classify_shell_command(command: str) -> ShellCommandDecision:
    """Classify a shell command into read-only, mutating, or blocked states."""
    raw = command.strip(" \t\n")
    if not raw:
        return ShellCommandDecision(
            command=command,
            allowed=False,
            read_only=False,
            requires_approval=False,
            destructive=False,
            malformed=True,
            reason="empty command",
        )

    lowered = raw.lower()
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(raw):
            return ShellCommandDecision(
                command=command,
                allowed=False,
                read_only=False,
                requires_approval=True,
                destructive=True,
                malformed=False,
                reason="dangerous command pattern detected",
            )

    # Guard against subshell-wrapped privilege escalation: (sudo ...) or (rm ...)
    # bypass the token-level hard-deny because shlex treats `(` as a regular char.
    if _SUBSHELL_PRIV_RE.search(raw):
        return ShellCommandDecision(
            command=command,
            allowed=False,
            read_only=False,
            requires_approval=True,
            destructive=True,
            malformed=False,
            reason="privilege escalation inside subshell group",
        )

    try:
        tokenized_segments = [list(segment.tokens) for segment in parse_shell_segments(raw)]
    except ValueError:
        # Unparseable syntax (e.g. unbalanced quotes) is not proof of danger;
        # fall back to manual approval instead of hard-denying the call.
        return ShellCommandDecision(
            command=command,
            allowed=True,
            read_only=False,
            requires_approval=True,
            destructive=False,
            malformed=True,
            reason="command could not be parsed; requires manual approval",
        )

    if not tokenized_segments or not any(tokens for tokens in tokenized_segments):
        return ShellCommandDecision(
            command=command,
            allowed=False,
            read_only=False,
            requires_approval=False,
            destructive=False,
            malformed=True,
            reason="empty command",
        )

    # Only recognized, argument-preserving runners are transparent to reads.
    # Environment edits and opaque/unknown wrapper forms require approval.
    resolved_segments: list[list[str]] = []
    hazard_segments: list[list[str]] = []
    all_resolvable = True
    for tokens in tokenized_segments:
        inner, resolvable = _resolve_runner_segment(tokens)
        if not resolvable:
            all_resolvable = False
        if inner:
            resolved_segments.append(inner)
        hazard, _ = _resolve_runner_segment(tokens, for_deny=True)
        if hazard:
            hazard_segments.append(hazard)

    if any(_is_privileged_segment(tokens, lowered) for tokens in hazard_segments):
        return ShellCommandDecision(
            command=command,
            allowed=False,
            read_only=False,
            requires_approval=True,
            destructive=True,
            malformed=False,
            reason="dangerous command prefix detected",
        )

    if any(_is_code_execution_segment(tokens) for tokens in resolved_segments):
        return ShellCommandDecision(
            command=command,
            allowed=True,
            read_only=False,
            requires_approval=True,
            destructive=False,
            malformed=False,
            reason="command executes arbitrary code; requires approval",
        )

    read_only = all_resolvable and all(
        _is_read_only_segment(tokens, raw_command=raw) for tokens in resolved_segments
    )
    mutates_filesystem = any(
        _is_write_segment(tokens, raw_command=raw) for tokens in resolved_segments
    ) or bool(WRITE_REDIRECTION_PATTERN.search(raw))

    # Command/process substitution can smuggle arbitrary commands into an
    # otherwise read-only line; never auto-allow those.
    has_dynamic_arguments = has_dynamic_shell_words(raw) or "<(" in raw or ">(" in raw

    if read_only and not mutates_filesystem and not has_dynamic_arguments:
        return ShellCommandDecision(
            command=command,
            allowed=True,
            read_only=True,
            requires_approval=False,
            destructive=False,
            malformed=False,
            reason="read-only command",
        )

    return ShellCommandDecision(
        command=command,
        allowed=True,
        read_only=False,
        requires_approval=True,
        destructive=False,
        malformed=False,
        reason="command may mutate filesystem or execute code",
    )
