"""Read-stream interceptor for channel notifications.

The MCP Python SDK validates incoming notifications against a closed
``ServerNotification`` union.  ``notifications/claude/channel`` is not in
that union, so the SDK drops it with a warning.  This module provides a
thin wrapper around the read stream that intercepts channel notification
JSON-RPC messages *before* they reach the SDK's validation layer,
dispatches them to a callback, and swallows them so the SDK never sees
them.  All non-channel messages pass through unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCNotification

from .admission import ChannelAdmission
from .notification import CHANNEL_NOTIFICATION_METHOD, CHANNEL_PERMISSION_METHOD

logger = logging.getLogger(__name__)

ChannelNotificationCallback = Callable[[str, str, dict[str, Any]], Awaitable[None]]


class ChannelInterceptingStream:
    """Wraps an MCP read stream to intercept channel notifications.

    Fully implements the ``anyio.abc.ObjectReceiveStream`` protocol
    required by ``ClientSession``.
    """

    def __init__(
        self,
        inner: Any,
        on_notification: ChannelNotificationCallback,
        server_name: str = "",
    ) -> None:
        self._inner = inner
        self._on_notification = (
            on_notification.for_stream(self)
            if isinstance(on_notification, ChannelAdmission)
            else on_notification
        )
        self._server_name = server_name
        self.bound_session: Any = None

    def bind_session(self, session: Any) -> None:
        """Bind the actual negotiated session without inspecting SDK internals."""
        self.bound_session = session

    async def receive(self) -> Any:
        """Receive the next message, intercepting channel notifications."""
        while True:
            message = await self._inner.receive()

            if isinstance(message, Exception):
                return message
            if not isinstance(message, SessionMessage):
                return message

            # MCP 1 wraps the union in a RootModel; MCP 2 delivers the concrete
            # JSON-RPC notification/response/error model directly.
            root = getattr(message.message, "root", message.message)
            if not isinstance(root, JSONRPCNotification):
                return message

            method = root.method
            if method not in (CHANNEL_NOTIFICATION_METHOD, CHANNEL_PERMISSION_METHOD):
                return message

            # Channel notification — extract params, dispatch, swallow
            params = {}
            if root.params is not None:
                if isinstance(root.params, dict):
                    params = root.params
                else:
                    try:
                        params = (
                            root.params.model_dump()
                            if hasattr(root.params, "model_dump")
                            else dict(root.params)
                        )
                    except Exception:
                        params = {}

            logger.info(
                "Channel notification from '%s': %s",
                self._server_name,
                method,
            )

            try:
                await self._on_notification(self._server_name, method, params)
            except Exception as exc:
                logger.error("Channel notification callback error: %s", exc)

            # Loop to get the next message (swallow this one)

    def __aiter__(self):
        return self

    async def __anext__(self):
        import anyio

        try:
            return await self.receive()
        except (anyio.ClosedResourceError, anyio.EndOfStream):
            raise StopAsyncIteration  # noqa: B904

    async def aclose(self):
        """Close the underlying stream."""
        if hasattr(self._inner, "aclose"):
            await self._inner.aclose()

    async def __aenter__(self):
        if hasattr(self._inner, "__aenter__"):
            await self._inner.__aenter__()
        return self

    async def __aexit__(self, *args):
        if hasattr(self._inner, "__aexit__"):
            await self._inner.__aexit__(*args)

    # Delegate any other attributes to the inner stream
    # (e.g., _closed, _state that anyio checks)
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)
