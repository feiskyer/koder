"""Literal shell segments shared by classification and permission rules.

This is a conservative simple-command parser, not a shell evaluator. Preserve
source spelling as well as decoded words: quotes can change an assignment-like
word into an executable name. Unsupported grouping/multiline syntax fails closed.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

_SEPARATORS = {"|", "||", "&&", ";", ";;", "&", "|&"}
_REDIRECTIONS = {">", ">>", "<", "<<", "<<<", ">&", "<&", "&>", "&>>", ">|", "<>"}
_OPERATOR_CHARS = frozenset(";&|<>()")


def has_dynamic_shell_words(command: str) -> bool:
    """Detect active expansions without treating literal regex anchors as variables."""
    quote = None
    index = 0
    while index < len(command):
        char = command[index]
        if quote == "'":
            if char == "'":
                quote = None
            index += 1
            continue
        if char == "\\":
            index += 2
            continue
        if char == '"':
            quote = None if quote == '"' else '"'
        elif char == "'" and quote is None:
            quote = "'"
        elif char == "`":
            return True
        elif char == "$" and index + 1 < len(command):
            following = command[index + 1]
            if following.isalnum() or following in "_{(['\"*@#?-$!":
                return True
        elif quote is None and char in "*?[]{}":
            return True
        elif quote is None and char == "#" and (index == 0 or command[index - 1] in " \t\n;&|()"):
            newline = command.find("\n", index)
            index = len(command) if newline < 0 else newline
            continue
        index += 1
    return quote is not None


@dataclass(frozen=True)
class ShellSegment:
    raw: str
    tokens: tuple[str, ...]


def parse_shell_segments(command: str) -> list[ShellSegment]:
    """Keep literal segment spelling, discard only real redirection operands."""
    if any(ord(char) < 32 and char not in "\t\n" for char in command):
        raise ValueError("unsupported shell control character")
    segments: list[ShellSegment] = []
    for line in command.split("\n"):
        words: list[str] = []
        word_start: int | None = None
        segment_start = 0
        pending_redirect = False
        quote: str | None = None

        def finish_word(end: int) -> str | None:
            nonlocal word_start, pending_redirect
            if word_start is None:
                return None
            spelling = line[word_start:end]
            decoded = shlex.split(spelling, comments=False, posix=True)
            if len(decoded) != 1:
                raise ValueError("unsupported shell word")
            word_start = None
            if pending_redirect:
                pending_redirect = False
                return None
            words.append(decoded[0])
            return spelling

        def finish_segment(end: int) -> None:
            if pending_redirect:
                raise ValueError("missing redirection operand")
            if words:
                segments.append(ShellSegment(line[segment_start:end].strip(" \t"), tuple(words)))
                words.clear()

        index = 0
        while index < len(line):
            char = line[index]
            if quote is not None:
                if char == quote:
                    quote = None
                elif char == "\\" and quote == '"':
                    index += 1
                index += 1
                continue
            if char in "'\"\\":
                if word_start is None:
                    word_start = index
                if char == "\\":
                    index += 2
                else:
                    quote = char
                    index += 1
                continue
            if char in " \t":
                finish_word(index)
                index += 1
                continue
            if char == "#" and word_start is None:
                # A # inside a word is literal; shlex's default commenter rule
                # would truncate an executable such as ls#local to trusted ls.
                break
            if char in _OPERATOR_CHARS:
                preceding_word = finish_word(index)
                end = index + 1
                while end < len(line) and line[end] in _OPERATOR_CHARS:
                    end += 1
                operator = line[index:end]
                if operator in _SEPARATORS:
                    finish_segment(index)
                    segment_start = end
                elif operator in _REDIRECTIONS:
                    if pending_redirect:
                        raise ValueError("missing redirection operand")
                    # Only adjacent, unquoted ASCII digits are an IO number.
                    # A quoted "2" is a command/argument, not descriptor syntax.
                    if preceding_word and preceding_word.isascii() and preceding_word.isdecimal():
                        words.pop()
                    pending_redirect = True
                else:
                    raise ValueError("unsupported shell operator")
                index = end
                continue
            if word_start is None:
                word_start = index
            index += 1
        if quote is not None:
            raise ValueError("unclosed shell quote")
        finish_word(min(index, len(line)))
        finish_segment(min(index, len(line)))
    return segments
