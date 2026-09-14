"""Conservative read-only recognition for literal sed invocations.

This is not a sed interpreter. Unknown syntax falls back to approval rather
than treating every program without an in-place flag as a read.
Executable provenance and other shell-command families remain separate checks.
"""

from .shell_segments import has_dynamic_shell_words

_READ_SHORT_OPTIONS = frozenset("nErzsub")
_READ_LONG_OPTIONS = frozenset(
    {
        "--silent",
        "--quiet",
        "--regexp-extended",
        "--posix",
        "--null-data",
        "--separate",
        "--unbuffered",
        "--binary",
        "--debug",
        "--sandbox",
    }
)
_LONG_OPTIONS = _READ_LONG_OPTIONS | {"--expression", "--file", "--in-place"}
_READ_COMMANDS = frozenset("pPnNdDhHgGxzlqQ=")
_SUBSTITUTE_FLAGS = frozenset("gpIiMm0123456789")
_MAX_SCRIPT_CHARS = 65536


def _inline_script(arguments: list[str]) -> str | None:
    scripts = []
    operands = []
    options_ended = False
    index = 0
    while index < len(arguments):
        token = arguments[index]
        index += 1
        if token == "--" and not options_ended:
            options_ended = True
            continue
        if options_ended or not token.startswith("-") or token == "-":
            operands.append(token)
            continue
        if operands:
            # BSD and GNU disagree about options after operands. Never infer a
            # harmless program using only one platform's argument permutation.
            return None
        if token.startswith("--"):
            name, separator, value = token.partition("=")
            matches = [option for option in _LONG_OPTIONS if option.startswith(name)]
            if len(matches) != 1:
                return None
            option = matches[0]
            if option in {"--file", "--in-place"}:
                return None
            if option == "--expression":
                if not separator:
                    if index >= len(arguments):
                        return None
                    value = arguments[index]
                    index += 1
                scripts.append(value)
            elif separator:
                return None
            continue
        for offset, flag in enumerate(token[1:], start=1):
            if flag == "e":
                value = token[offset + 1 :]
                if not value:
                    if index >= len(arguments):
                        return None
                    value = arguments[index]
                    index += 1
                scripts.append(value)
                break
            if flag not in _READ_SHORT_OPTIONS:
                return None
    if not scripts:
        if not operands:
            return None
        scripts.append(operands[0])
    return "\n".join(scripts)


def _delimited_end(script: str, start: int) -> int | None:
    delimiter = script[start]
    if delimiter in "\\\r\n":
        return None
    index = start + 1
    while index < len(script):
        char = script[index]
        if char == "\\":
            index += 2
        elif char == delimiter:
            return index + 1
        elif char in "\r\n":
            return None
        else:
            index += 1
    return None


def _address_end(script: str, start: int) -> int | None:
    if start >= len(script):
        return start
    if script[start] in "0123456789":
        index = start
        while index < len(script) and script[index] in "0123456789":
            index += 1
        return index
    if script[start] == "$":
        return start + 1
    if script[start] == "/":
        return _delimited_end(script, start)
    if script[start] == "\\" and start + 1 < len(script):
        return _delimited_end(script, start + 1)
    return start


def _skip_blanks(script: str, index: int) -> int:
    while index < len(script) and script[index] in " \t":
        index += 1
    return index


def _script_is_readonly(script: str) -> bool:
    if len(script) > _MAX_SCRIPT_CHARS or "\0" in script:
        return False
    index = 0
    groups = 0
    while index < len(script):
        if script[index] in " \t;\r\n":
            index += 1
            continue
        if script[index] == "#":
            newline = script.find("\n", index)
            index = len(script) if newline < 0 else newline
            continue
        if script[index] == "}":
            if not groups:
                return False
            groups -= 1
            index += 1
            continue
        address_end = _address_end(script, index)
        if address_end is None:
            return False
        has_address = address_end != index
        index = _skip_blanks(script, address_end)
        if index < len(script) and script[index] == ",":
            if not has_address:
                return False
            start = _skip_blanks(script, index + 1)
            end = _address_end(script, start)
            if end is None or end == start:
                return False
            index = _skip_blanks(script, end)
        if index < len(script) and script[index] == "!":
            index = _skip_blanks(script, index + 1)
        if index >= len(script):
            return False
        operation = script[index]
        index += 1
        if operation == "{":
            groups += 1
            if groups > 64:
                return False
        elif operation in _READ_COMMANDS:
            if operation in "qQl":
                index = _skip_blanks(script, index)
                while index < len(script) and script[index] in "0123456789":
                    index += 1
        elif operation in {"s", "y"} and index < len(script):
            first_end = _delimited_end(script, index)
            if first_end is None:
                return False
            second_end = _delimited_end(script, first_end - 1)
            if second_end is None:
                return False
            index = second_end
            if operation == "s":
                while index < len(script) and script[index] not in ";}\r\n":
                    if script[index] not in _SUBSTITUTE_FLAGS and script[index] not in " \t":
                        return False
                    index += 1
        else:
            # Includes w/W, e, s///w, s///e and unsupported text/control forms.
            return False
    return groups == 0


def is_readonly_sed(tokens: list[str], *, raw_command: str | None) -> bool:
    """Require a literal, bounded, recognized read-only program and arguments."""
    if raw_command is None or has_dynamic_shell_words(raw_command):
        return False
    arguments = tokens[1:]
    if arguments in (["--help"], ["--version"]):
        return True
    script = _inline_script(arguments)
    return script is not None and _script_is_readonly(script)
