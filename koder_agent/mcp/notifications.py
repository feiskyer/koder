"""MCP notification handling for list_changed and channel events."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from koder_agent.harness.channels.notification import ChannelNotificationRouter

logger = logging.getLogger(__name__)


class MCPNotificationHandler:
    """Handles MCP server notifications for capability changes and channels."""

    def __init__(self):
        self._refresh_callbacks = []
        self._channel_router: Optional[ChannelNotificationRouter] = None

    def set_channel_router(self, router: ChannelNotificationRouter) -> None:
        """Attach a channel notification router."""
        self._channel_router = router

    @property
    def channel_router(self) -> Optional[ChannelNotificationRouter]:
        return self._channel_router

    async def dispatch_notification(
        self, server_name: str, method: str, params: dict[str, Any]
    ) -> bool:
        """Try to dispatch a raw notification via the channel router.

        Returns True if handled, False otherwise.
        """
        if self._channel_router is not None:
            return await self._channel_router.dispatch_raw_notification(server_name, method, params)
        return False

    def on_refresh(self, callback):
        """Register a callback to be called when capabilities change."""
        self._refresh_callbacks.append(callback)

    async def handle_tools_list_changed(self, server_name: str) -> None:
        """Handle tools/list_changed notification from a server."""
        logger.info(f"MCP server '{server_name}' sent tools/list_changed notification")
        for callback in self._refresh_callbacks:
            try:
                await callback("tools", server_name)
            except Exception as exc:
                logger.error(f"Error in refresh callback: {exc}")

    async def handle_resources_list_changed(self, server_name: str) -> None:
        """Handle resources/list_changed notification from a server."""
        logger.info(f"MCP server '{server_name}' sent resources/list_changed notification")
        for callback in self._refresh_callbacks:
            try:
                await callback("resources", server_name)
            except Exception as exc:
                logger.error(f"Error in refresh callback: {exc}")

    async def handle_prompts_list_changed(self, server_name: str) -> None:
        """Handle prompts/list_changed notification from a server."""
        logger.info(f"MCP server '{server_name}' sent prompts/list_changed notification")
        for callback in self._refresh_callbacks:
            try:
                await callback("prompts", server_name)
            except Exception as exc:
                logger.error(f"Error in refresh callback: {exc}")


# Global notification handler
_handler = MCPNotificationHandler()
_current_handler: ContextVar[MCPNotificationHandler | None] = ContextVar(
    "koder_mcp_notification_handler", default=None
)


@contextmanager
def notification_handler_scope() -> Iterator[MCPNotificationHandler]:
    """Keep runtime registrations separate, including nested runtime calls."""
    handler = MCPNotificationHandler()
    token = _current_handler.set(handler)
    try:
        yield handler
    finally:
        _current_handler.reset(token)


def get_notification_handler() -> MCPNotificationHandler:
    handler = _current_handler.get()
    return handler if handler is not None else _handler
