"""Standalone permission-relay primitives, not a runtime approval consumer.

These helpers do not authenticate a sender. Any integrating caller must own
admission and provenance; the current channel runtime does not wire this relay.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 25-char alphabet: a-z without 'l' (avoids confusion with 1/I)
ID_ALPHABET = "abcdefghijkmnopqrstuvwxyz"

# Regex for parsing permission replies from channel users
PERMISSION_REPLY_RE = re.compile(r"^\s*(y|yes|n|no)\s+([a-km-z]{5})\s*$", re.IGNORECASE)

# Profanity substrings to avoid in generated IDs
ID_AVOID_SUBSTRINGS = [
    "fuck",
    "shit",
    "cunt",
    "cock",
    "dick",
    "twat",
    "piss",
    "crap",
    "bitch",
    "whore",
    "ass",
    "tit",
    "cum",
    "fag",
    "dyke",
    "nig",
    "kike",
    "rape",
    "nazi",
    "damn",
    "poo",
    "pee",
    "wank",
    "anus",
]

# ---------------------------------------------------------------------------
# FNV-1a hashing
# ---------------------------------------------------------------------------

_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193
_MASK32 = 0xFFFFFFFF


def _fnv1a_hash(input_str: str) -> int:
    """Compute FNV-1a hash of *input_str* as a uint32."""
    h = _FNV_OFFSET
    for ch in input_str:
        h ^= ord(ch)
        h = (h * _FNV_PRIME) & _MASK32
    return h


def _hash_to_id(input_str: str) -> str:
    """Hash *input_str* with FNV-1a and encode as a 5-char base-25 ID."""
    h = _fnv1a_hash(input_str)
    result = []
    for _ in range(5):
        result.append(ID_ALPHABET[h % 25])
        h //= 25
    return "".join(result)


def short_request_id(tool_use_id: str) -> str:
    """Generate a short 5-letter request ID for permission relay.

    Retries with salted input up to 10 times if the candidate contains
    a blocklisted substring.
    """
    candidate = _hash_to_id(tool_use_id)
    for salt in range(10):
        if not any(bad in candidate for bad in ID_AVOID_SUBSTRINGS):
            return candidate
        candidate = _hash_to_id(f"{tool_use_id}:{salt}")
    return candidate


# ---------------------------------------------------------------------------
# Preview truncation
# ---------------------------------------------------------------------------


def truncate_for_preview(input_data: Any, max_len: int = 200) -> str:
    """JSON-serialize *input_data* and truncate to *max_len* chars."""
    try:
        text = json.dumps(input_data, ensure_ascii=False)
    except (TypeError, ValueError):
        return "(unserializable)"
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\u2026"


# ---------------------------------------------------------------------------
# Permission callbacks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelPermissionResponse:
    """A resolved permission verdict from a channel."""

    behavior: Literal["allow", "deny"]
    from_server: str


@dataclass(frozen=True)
class _PendingPermission:
    handler: Callable[[ChannelPermissionResponse], None]


class ChannelPermissionCallbacks:
    """Track pending permission relay requests and resolve them.

    Each pending request is keyed by a lowercased ``request_id``.
    """

    def __init__(self) -> None:
        self._pending: dict[str, _PendingPermission] = {}

    def on_response(
        self,
        request_id: str,
        handler: Callable[[ChannelPermissionResponse], None],
    ) -> Callable[[], None]:
        """Register *handler* for *request_id*.  Returns an unsubscribe function."""
        key = request_id.lower()
        if key in self._pending:
            raise ValueError("Permission request ID is already pending")
        pending = _PendingPermission(handler)
        self._pending[key] = pending

        def unsubscribe() -> None:
            if self._pending.get(key) is pending:
                self._pending.pop(key, None)

        return unsubscribe

    def resolve(
        self,
        request_id: str,
        behavior: str,
        from_server: str,
    ) -> bool:
        """Resolve *request_id* with *behavior*.

        Returns ``True`` if a pending handler was found and called,
        ``False`` if no handler was registered for that ID.  The entry
        is deleted **before** the handler is called to prevent re-entrancy.
        """
        if behavior not in ("allow", "deny"):
            return False
        key = request_id.lower()
        pending = self._pending.pop(key, None)
        if pending is None:
            return False
        pending.handler(
            ChannelPermissionResponse(
                behavior="allow" if behavior == "allow" else "deny", from_server=from_server
            )
        )
        return True

    @property
    def pending_count(self) -> int:
        return len(self._pending)


def create_channel_permission_callbacks() -> ChannelPermissionCallbacks:
    """Factory for a fresh set of permission callbacks."""
    return ChannelPermissionCallbacks()
