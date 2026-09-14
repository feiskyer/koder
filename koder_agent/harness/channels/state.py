"""Runtime-owned channel policy, with a legacy default for direct callers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .types import ChannelEntry

if TYPE_CHECKING:
    from .inbox import ChannelInbox


@dataclass
class ChannelState:
    """A shared policy object for one runtime and its reader tasks."""

    allowed_channels: tuple[ChannelEntry, ...] = ()
    has_dev_channels: bool = False
    inbox: ChannelInbox | None = None


_default_state = ChannelState()
_current_state: ContextVar[ChannelState | None] = ContextVar("koder_channel_state", default=None)


def get_channel_state() -> ChannelState:
    """Return the current owner, retaining shared revocation within that owner."""
    state = _current_state.get()
    return state if state is not None else _default_state


@contextmanager
def channel_state_scope() -> Iterator[ChannelState]:
    """Isolate one runtime's policy and revoke it before restoring its caller."""
    state = ChannelState()
    token = _current_state.set(state)
    try:
        yield state
    finally:
        state.allowed_channels = ()
        state.has_dev_channels = False
        state.inbox = None
        _current_state.reset(token)


def get_allowed_channels() -> list[ChannelEntry]:
    """Return the list of channel entries enabled for this session."""
    return list(get_channel_state().allowed_channels)


def set_allowed_channels(entries: list[ChannelEntry]) -> None:
    """Set the channel entries enabled for this session."""
    get_channel_state().allowed_channels = tuple(entries)


def get_has_dev_channels() -> bool:
    """Return whether development channels are loaded."""
    return get_channel_state().has_dev_channels


def set_has_dev_channels(value: bool) -> None:
    """Set whether development channels are loaded."""
    get_channel_state().has_dev_channels = value


def reset_channel_state() -> None:
    """Reset all channel state (for testing)."""
    state = get_channel_state()
    state.allowed_channels = ()
    state.has_dev_channels = False
    state.inbox = None
