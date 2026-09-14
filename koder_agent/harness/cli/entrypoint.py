"""Typed runtime requests for the new harness runtime."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from koder_agent.harness.runtime import HarnessRuntime


@dataclass(frozen=True)
class RuntimeRequest:
    argv: list[str]
    mode: str
    help_text: str | None = None
    first_arg: str | None = None


SUBCOMMANDS = {
    "auth",
    "mcp",
    "config",
    "agents",
    "plugin",
    "plugins",
    "doctor",
    "review",
    "completion",
    "upgrade",
}


def _option_action(
    token: str, options: dict[str, argparse.Action]
) -> tuple[argparse.Action | None, bool]:
    """Resolve a declared option and whether its value is attached to the token."""
    option, separator, _value = token.partition("=")
    if option in options:
        return options[option], bool(separator)
    if token.startswith("--"):
        matches = [action for name, action in options.items() if name.startswith(option)]
        return (matches[0], bool(separator)) if len(matches) == 1 else (None, False)
    # argparse accepts both attached values (-sname) and short flag clusters.
    for index, letter in enumerate(token[1:], start=1):
        action = options.get(f"-{letter}")
        if action is None:
            return None, False
        if action.nargs != 0:
            return action, index < len(token) - 1
    return None, False


def detect_first_arg(argv: list[str]) -> str | None:
    from koder_agent.cli import _build_cli_parser

    # The real parser is the single source of option arity. A second hand-kept
    # flag list previously treated --bare as valued and missed --image/plugin-dir.
    options = {
        option: action
        for action in _build_cli_parser(None)._actions
        for option in action.option_strings
    }
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--":
            # Explicit positional text must not be reinterpreted as a subcommand.
            return None
        if not token.startswith("-") or token == "-":
            return token
        action, attached = _option_action(token, options)
        if action is None:
            index += 1
            continue
        if action.dest == "print_prompt":
            return None
        if attached:
            index += 1
            continue
        if action.nargs is None:
            index += 2
            continue
        if action.nargs == "?":
            if index + 1 < len(argv) and not argv[index + 1].startswith("-"):
                index += 2
            else:
                index += 1
            continue
        index += 1
    return None


def build_runtime_request(argv: list[str]) -> RuntimeRequest:
    if not argv:
        return RuntimeRequest(argv=[], mode="interactive", first_arg=None)
    first = argv[0]
    if first in {"-h", "--help"}:
        return RuntimeRequest(argv=argv, mode="help", first_arg=None)
    if first in {"-V", "-v", "--version"}:
        return RuntimeRequest(argv=argv, mode="version", first_arg=None)

    first_arg = detect_first_arg(argv)
    if first_arg == "auth":
        return RuntimeRequest(argv=argv, mode="auth_passthrough", first_arg=first_arg)
    if first_arg in SUBCOMMANDS:
        return RuntimeRequest(argv=argv, mode="subcommand", first_arg=first_arg)
    return RuntimeRequest(argv=argv, mode="prompt", first_arg=first_arg)


async def run_harness_runtime(request: RuntimeRequest) -> int:
    runtime = HarnessRuntime(request=request)
    return await runtime.run()
