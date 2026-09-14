"""Channel notification handling and message wrapping."""

from __future__ import annotations

import logging
import re
from html import escape
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHANNEL_TAG = "channel"
CHANNEL_NOTIFICATION_METHOD = "notifications/claude/channel"
CHANNEL_PERMISSION_METHOD = "notifications/claude/channel/permission"
CHANNEL_PERMISSION_REQUEST_METHOD = "notifications/claude/channel/permission_request"

SAFE_META_KEY = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------


def escape_xml_attr(s: str) -> str:
    """Escape a string for use in an XML attribute value."""
    return (
        s.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def wrap_channel_message(
    server_name: str,
    content: str,
    meta: Optional[dict[str, str]] = None,
) -> str:
    """Wrap a channel message in a ``<channel>`` XML tag.

    Produces::

        <channel source="server_name" key="val">
        content
        </channel>

    Meta keys are filtered to safe identifiers (excluding the reserved source);
    values and message content are XML-escaped so untrusted input cannot forge
    another channel envelope.
    """
    attrs = f' source="{escape_xml_attr(server_name)}"'
    if meta:
        for key, val in meta.items():
            if key != "source" and SAFE_META_KEY.fullmatch(key):
                attrs += f' {key}="{escape_xml_attr(str(val))}"'
    return f"<{CHANNEL_TAG}{attrs}>\n{escape(content, quote=False)}\n</{CHANNEL_TAG}>"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_channel_message_params(params: dict[str, Any]) -> tuple[str, Optional[dict[str, str]]]:
    """Validate inbound channel message notification params.

    Returns ``(content, meta)`` or raises :class:`ValueError`.
    """
    content = params.get("content")
    if not isinstance(content, str):
        raise ValueError("Channel message params must include a string 'content' field")
    meta = params.get("meta")
    if meta is not None and not isinstance(meta, dict):
        raise ValueError("Channel message params 'meta' must be a dict or absent")
    return content, meta


def validate_channel_permission_params(params: dict[str, Any]) -> tuple[str, str]:
    """Validate inbound channel permission notification params.

    Returns ``(request_id, behavior)`` or raises :class:`ValueError`.
    """
    request_id = params.get("request_id")
    if not isinstance(request_id, str):
        raise ValueError("Channel permission params must include a string 'request_id'")
    behavior = params.get("behavior")
    if behavior not in ("allow", "deny"):
        raise ValueError("Channel permission params 'behavior' must be 'allow' or 'deny'")
    return request_id, behavior


# ---------------------------------------------------------------------------
# Notification router
# ---------------------------------------------------------------------------

ChannelMessageCallback = Callable[[str, str, Optional[dict[str, str]]], Awaitable[None]]
ChannelPermissionCallback = Callable[[str, str, str], Awaitable[None]]
UnregisterCallback = Callable[[], None]


class ChannelNotificationRouter:
    """Routes inbound channel notifications to registered callbacks.

    Callbacks receive ``(server_name, content, meta)`` for messages
    and ``(server_name, request_id, behavior)`` for permissions.
    """

    def __init__(self) -> None:
        self._message_callbacks: dict[int, ChannelMessageCallback] = {}
        self._permission_callbacks: dict[int, ChannelPermissionCallback] = {}
        self._next_callback_id = 0

    def on_channel_message(self, callback: ChannelMessageCallback) -> UnregisterCallback:
        """Register a message callback and return an idempotent unregister handle."""
        return self._register(self._message_callbacks, callback)

    def on_channel_permission(self, callback: ChannelPermissionCallback) -> UnregisterCallback:
        """Register a permission callback and return an idempotent unregister handle."""
        return self._register(self._permission_callbacks, callback)

    def _register(self, callbacks: dict[int, Any], callback: Any) -> UnregisterCallback:
        self._next_callback_id += 1
        callback_id = self._next_callback_id
        callbacks[callback_id] = callback

        def unregister() -> None:
            callbacks.pop(callback_id, None)

        return unregister

    async def handle_channel_message(
        self,
        server_name: str,
        content: str,
        meta: Optional[dict[str, str]] = None,
    ) -> None:
        """Dispatch a channel message to all registered callbacks."""
        for callback_id in list(self._message_callbacks):
            cb = self._message_callbacks.get(callback_id)
            if cb is None:
                continue
            try:
                await cb(server_name, content, meta)
            except Exception as exc:
                logger.error("Error in channel message callback: %s", exc)

    async def handle_channel_permission(
        self,
        server_name: str,
        request_id: str,
        behavior: str,
    ) -> None:
        """Dispatch a channel permission verdict to all registered callbacks."""
        for callback_id in list(self._permission_callbacks):
            cb = self._permission_callbacks.get(callback_id)
            if cb is None:
                continue
            try:
                await cb(server_name, request_id, behavior)
            except Exception as exc:
                logger.error("Error in channel permission callback: %s", exc)

    async def dispatch_raw_notification(
        self,
        server_name: str,
        method: str,
        params: dict[str, Any],
    ) -> bool:
        """Dispatch a raw JSON-RPC notification if it is channel-related.

        Returns ``True`` if the notification was handled, ``False`` otherwise.
        """
        if method == CHANNEL_NOTIFICATION_METHOD:
            try:
                content, meta = validate_channel_message_params(params)
                await self.handle_channel_message(server_name, content, meta)
            except ValueError as exc:
                logger.warning("Invalid channel message from '%s': %s", server_name, exc)
            return True

        if method == CHANNEL_PERMISSION_METHOD:
            try:
                request_id, behavior = validate_channel_permission_params(params)
                await self.handle_channel_permission(server_name, request_id, behavior)
            except ValueError as exc:
                logger.warning("Invalid channel permission from '%s': %s", server_name, exc)
            return True

        return False
