"""Channel gate logic — determines whether an MCP server registers as a channel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional

from .state import get_allowed_channels
from .types import ChannelEntry, ChannelEntryPlugin, ChannelEntryServer


@dataclass(frozen=True)
class ChannelGateResult:
    """Result of the channel gate check."""

    action: Literal["register", "skip"]
    kind: Optional[str] = None
    reason: Optional[str] = None


@dataclass(frozen=True)
class PluginChannelOrigin:
    """Discovery provenance, never inferred from a public MCP server name."""

    name: str
    marketplace: str | None = None


def find_channel_entry(
    server_name: str,
    channels: Sequence[ChannelEntry],
    *,
    plugin_origin: PluginChannelOrigin | None = None,
) -> Optional[ChannelEntry]:
    """Find the channel entry matching *server_name* in the enabled list.

    - Server-kind: exact match on the full ``server_name``.
    - Plugin-kind: require the actual discovered plugin and recorded marketplace.
      A server name that looks like ``plugin:name`` is not provenance.
    """
    for entry in channels:
        if isinstance(entry, ChannelEntryServer):
            if server_name == entry.name:
                return entry
        elif isinstance(entry, ChannelEntryPlugin):
            if (
                plugin_origin is not None
                and plugin_origin.name == entry.name
                and plugin_origin.marketplace == entry.marketplace
            ):
                return entry
    return None


def gate_channel_server(
    server_name: str,
    capabilities: Any = None,
    *,
    plugin_origin: PluginChannelOrigin | None = None,
    channels: Sequence[ChannelEntry] | None = None,
) -> ChannelGateResult:
    """Decide whether *server_name* should register as a channel.

    Simplified 2-step gate that checks capability and session membership:

    1. **Capability** — server must declare
       ``capabilities.experimental['claude/channel']``.
    2. **Session** — server must be in the ``--channels`` list for this
       session (matched via :func:`find_channel_entry`).
    """
    # Step 1: Capability check
    experimental = None
    if capabilities is not None:
        experimental = getattr(capabilities, "experimental", None)
        if experimental is None and isinstance(capabilities, dict):
            experimental = capabilities.get("experimental")

    if experimental is None:
        return ChannelGateResult(
            action="skip",
            kind="capability",
            reason="Server does not declare experimental capabilities",
        )

    has_channel = False
    if isinstance(experimental, dict):
        has_channel = "claude/channel" in experimental
    else:
        has_channel = hasattr(experimental, "claude/channel") or (
            isinstance(experimental, dict) and "claude/channel" in experimental
        )

    if not has_channel:
        return ChannelGateResult(
            action="skip",
            kind="capability",
            reason="Server does not declare claude/channel capability",
        )

    # Step 2: Session check
    entry = find_channel_entry(
        server_name,
        get_allowed_channels() if channels is None else channels,
        plugin_origin=plugin_origin,
    )
    if entry is None:
        return ChannelGateResult(
            action="skip",
            kind="session",
            reason=f"Server '{server_name}' is not in the --channels list",
        )

    return ChannelGateResult(action="register")
