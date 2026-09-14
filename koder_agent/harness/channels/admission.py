"""Connection-scoped delivery checks for session-enabled MCP channels."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from .gate import PluginChannelOrigin, gate_channel_server
from .notification import ChannelNotificationRouter
from .state import get_channel_state


class ChannelAdmission:
    """Reject events until the owner publishes an initialized, allowed server.

    Admission is checked again for every event. Pre-publication events are
    dropped, not buffered. The receive-stream identity binds each event to the
    current SDK session, excluding retired and not-yet-published connections.
    """

    def __init__(
        self,
        server_name: str,
        router: ChannelNotificationRouter,
        owner_active: Callable[[], bool],
        plugin_origin: PluginChannelOrigin | None = None,
    ) -> None:
        self._name = server_name
        self._router = router
        self._owner_active = owner_active
        self._plugin_origin = plugin_origin
        self._channel_state = get_channel_state()
        self._server: Any = None

    def bind(self, server: Any) -> None:
        """Publish a server only after its resource owner has adopted it."""
        self._server = server

    def for_stream(self, stream: Any) -> Callable[[str, str, dict[str, Any]], Awaitable[None]]:
        async def dispatch(server_name: str, method: str, params: dict[str, Any]) -> None:
            await self._dispatch(stream, server_name, method, params)

        return dispatch

    async def __call__(self, server_name: str, method: str, params: dict[str, Any]) -> None:
        await self._dispatch(None, server_name, method, params)

    async def _dispatch(
        self, stream: Any, server_name: str, method: str, params: dict[str, Any]
    ) -> None:
        server = self._server
        if server is None or server_name != self._name or not self._owner_active():
            return
        get_session = getattr(server, "_koder_raw_session", None)
        session = get_session() if callable(get_session) else getattr(server, "session", None)
        if session is None:
            return
        # Koder owns this binding. MCP 2 moved read streams into a dispatcher;
        # reaching into a pinned SDK's _read_stream silently drops valid events.
        if stream is not None and getattr(stream, "bound_session", None) is not session:
            return
        result = getattr(server, "server_initialize_result", None)
        capabilities = getattr(result, "capabilities", result)
        if (
            gate_channel_server(
                self._name,
                capabilities,
                plugin_origin=self._plugin_origin,
                channels=self._channel_state.allowed_channels,
            ).action
            != "register"
        ):
            return
        await self._router.dispatch_raw_notification(self._name, method, params)
