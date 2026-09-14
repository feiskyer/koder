"""Bounded command-runner projections for shell permission decisions.

Only a known argument-preserving subset is transparent to positive decisions.
Environment edits, stdin-built arguments, output redirection by a wrapper and
unknown options must not borrow the inner command's authorization.
"""

from __future__ import annotations

import os
import re

_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_RUNNERS = frozenset({"command", "env", "nice", "nohup", "setsid", "stdbuf", "timeout", "xargs"})
_OPAQUE_RUNNERS = frozenset({"nohup", "setsid", "xargs"})
_SHELL_RESERVED_WORDS = frozenset(
    {
        "!",
        "{",
        "}",
        "case",
        "do",
        "done",
        "elif",
        "else",
        "esac",
        "fi",
        "for",
        "function",
        "if",
        "in",
        "select",
        "then",
        "time",
        "until",
        "while",
        "coproc",
    }
)
_VALUE_OPTIONS = {
    "command": frozenset(),
    "env": frozenset(),
    "nice": frozenset({"-n", "--adjustment"}),
    "nohup": frozenset(),
    "setsid": frozenset(),
    "stdbuf": frozenset({"-i", "-o", "-e", "--input", "--output", "--error"}),
    "timeout": frozenset({"-k", "--kill-after", "-s", "--signal"}),
    "xargs": frozenset(),
}
_FLAG_OPTIONS = {
    "command": frozenset({"-p"}),
    "timeout": frozenset({"--foreground", "--preserve-status", "--verbose", "-v"}),
}
_DENY_VALUE_OPTIONS = {
    "env": frozenset({"-u", "--unset", "-C", "--chdir", "-P"}),
    "xargs": frozenset(
        {
            "-I",
            "-i",
            "-n",
            "-P",
            "-d",
            "-E",
            "-L",
            "-s",
            "-a",
            "--replace",
            "--max-args",
            "--max-procs",
            "--delimiter",
            "--eof",
            "--max-lines",
            "--max-chars",
            "--arg-file",
        }
    ),
}
_DENY_FLAG_OPTIONS = {
    "env": frozenset({"-i", "--ignore-environment", "-v", "--debug", "-0", "--null"}),
    "setsid": frozenset({"-f", "--fork", "-w", "--wait", "-c", "--ctty"}),
    "xargs": frozenset({"-0", "--null", "-r", "--no-run-if-empty", "-t", "--verbose", "-p"}),
}


def _arguments_start(tokens: list[str], name: str, *, for_deny: bool) -> int | None:
    values = _VALUE_OPTIONS[name]
    flags = _FLAG_OPTIONS.get(name, frozenset())
    if for_deny:
        values |= _DENY_VALUE_OPTIONS.get(name, frozenset())
        flags |= _DENY_FLAG_OPTIONS.get(name, frozenset())
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            return index + 1
        if not token.startswith("-") or token == "-":
            return index
        if token in flags:
            index += 1
        elif token in values:
            if index + 1 >= len(tokens):
                return None
            index += 2
        elif token.startswith("--") and token.partition("=")[0] in values and "=" in token:
            index += 1
        elif any(
            len(option) == 2 and token.startswith(option) and len(token) > 2 for option in values
        ):
            index += 1
        else:
            # Guessing the arity of an unknown option can authorize its operand
            # as the command while the runner actually executes a later word.
            return None
    return index


def resolve_runner_segment(tokens: list[str], *, for_deny: bool = False) -> tuple[list[str], bool]:
    """Project known runners; a false result can never grant automatic approval.

    Positive projection preserves paths and all environment-assignment words.
    Deny projection may additionally recognize basenames and discard assignments
    before a shell command or after env. It cannot grant permission.
    """
    if for_deny:
        tokens = list(tokens)
        while tokens and _ASSIGNMENT.match(tokens[0]):
            tokens.pop(0)
    for _ in range(8):
        if not tokens:
            return [], False
        name = os.path.basename(tokens[0]) if for_deny else tokens[0]
        if not for_deny and (_ASSIGNMENT.match(name) or name in _SHELL_RESERVED_WORDS):
            # A wrapper executes this word literally. Serializing it as a bare
            # assignment/keyword could borrow a different shell expression's rule.
            return tokens, False
        if name not in _RUNNERS:
            return [name, *tokens[1:]], True
        if not for_deny and name in _OPAQUE_RUNNERS:
            return tokens, False
        index = _arguments_start(tokens, name, for_deny=for_deny)
        if index is None:
            return tokens, False
        if name == "env":
            while index < len(tokens) and _ASSIGNMENT.match(tokens[index]):
                if not for_deny:
                    return tokens, False
                index += 1
        if name == "timeout":
            if index >= len(tokens) or tokens[index].startswith("-"):
                return tokens, False
            index += 1  # The duration is an operand, not the child command.
        tokens = tokens[index:]
    return tokens, False
