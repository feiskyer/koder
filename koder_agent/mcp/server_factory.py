"""Factory for creating MCP server instances."""

from __future__ import annotations

import asyncio
import functools
import json as _json
import logging
import os
import signal
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, List, Optional

from agents.mcp import (
    MCPServer,
    MCPServerSse,
    MCPServerSseParams,
    MCPServerStdio,
    MCPServerStdioParams,
    MCPServerStreamableHttp,
    MCPServerStreamableHttpParams,
    create_static_tool_filter,
)
from mcp.client.session import ClientSession, ElicitationFnT

from .connection import MCPConnection
from .lifecycle import cleanup_mcp_servers
from .limits import get_timeout_seconds
from .reconnection import (
    LiveMCPServer,
    ReconnectionConfig,
    ReconnectionManager,
    retain_orphaned_retirements,
)
from .server_config import MCPServerConfig, MCPServerScope, MCPServerType

logger = logging.getLogger(__name__)

# Type for the channel notification callback
ChannelNotifCallbackT = Callable[[str, str, dict[str, Any]], Awaitable[None]]


def _get_elicitation_callback() -> ElicitationFnT:
    """Return the global elicitation handler (lazy import to avoid cycles)."""
    from .elicitation import get_elicitation_handler

    return get_elicitation_handler()


class _ElicitationMixin:
    """Wire elicitation before negotiation and retain same-task scope ownership."""

    @asynccontextmanager
    async def _session_streams(self):
        async with self.create_streams() as transport:
            read, write, *rest = transport
            self._get_session_id = rest[0] if rest and callable(rest[0]) else None
            callback = getattr(self, "_channel_callback", None)
            if callback is not None:
                from koder_agent.harness.channels.interceptor import ChannelInterceptingStream

                read = ChannelInterceptingStream(
                    read,
                    on_notification=callback,
                    server_name=self._channel_server_name,
                )
                self._channel_read = read
            try:
                yield read, write
            finally:
                if callback is not None:
                    read.bind_session(None)
                    self._channel_read = None

    def _bind_channel_session(self, session):
        stream = getattr(self, "_channel_read", None)
        if stream is not None:
            stream.bind_session(session)

    @asynccontextmanager
    async def _connected_session(self):
        import mcp

        timeout = self.client_session_timeout_seconds or None
        client_class = getattr(mcp, "Client", None)
        if client_class is not None:
            # MCP 2 owns negotiation in its public Client API and expects
            # numeric seconds. Do not copy the MCP 1 ClientSession constructor.
            async with client_class(
                self._session_streams(),
                mode="auto",
                cache=None,
                read_timeout_seconds=timeout,
                elicitation_callback=_get_elicitation_callback(),
                message_handler=self.message_handler,
            ) as client:
                self._bind_channel_session(client.session)
                result = client.session.initialize_result
                if result is None:
                    # Modern discovery has capabilities but no legacy InitializeResult.
                    result = SimpleNamespace(capabilities=client.server_capabilities)
                yield client.session, result
        else:
            # Compatibility with installs that still use the supported MCP 1 API.
            async with AsyncExitStack() as stack:
                read, write = await stack.enter_async_context(self._session_streams())
                session = await stack.enter_async_context(
                    ClientSession(
                        read,
                        write,
                        read_timeout_seconds=timedelta(seconds=timeout) if timeout else None,
                        elicitation_callback=_get_elicitation_callback(),
                        message_handler=self.message_handler,
                    )
                )
                result = await session.initialize()
                self._bind_channel_session(session)
                yield session, result

    async def connect(self) -> None:  # type: ignore[override]
        connection = getattr(self, "_koder_connection", None)
        if connection is None:
            self._koder_connection = connection = MCPConnection(self._connected_session)
        try:
            self.session, self.server_initialize_result = await connection.connect()
        except Exception:
            logger.error("Error connecting MCP server with elicitation", exc_info=True)
            raise

    async def cleanup(self) -> None:
        connection = getattr(self, "_koder_connection", None)
        if connection is not None:
            await connection.close()
            self.session = None
        else:
            await super().cleanup()


class ChannelAwareMCPServerStdio(_ElicitationMixin, MCPServerStdio):
    """Owned MCP connection intercepting channel notifications before validation."""

    def __init__(
        self,
        *args: Any,
        channel_callback: ChannelNotifCallbackT | None = None,
        channel_server_name: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._channel_callback = channel_callback
        self._channel_server_name = channel_server_name


class ElicitationAwareStdio(_ElicitationMixin, MCPServerStdio):
    """MCPServerStdio with elicitation support."""


class ElicitationAwareSse(_ElicitationMixin, MCPServerSse):
    """MCPServerSse with elicitation support."""


class ElicitationAwareHttp(_ElicitationMixin, MCPServerStreamableHttp):
    """MCPServerStreamableHttp with elicitation support."""


HEADERS_HELPER_TIMEOUT_S = 10.0


async def _stop_headers_helper_process(proc: asyncio.subprocess.Process) -> None:
    """Kill and drain a headers-helper subprocess without leaking its transport."""
    if proc.returncode is None:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:  # pragma: no cover - exercised on Windows
                proc.kill()
        except ProcessLookupError:
            pass
    try:
        await proc.communicate()
    except (ProcessLookupError, RuntimeError):
        pass


async def _settle_headers_helper_process(proc: asyncio.subprocess.Process) -> None:
    task = asyncio.create_task(_stop_headers_helper_process(proc))
    caller_cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            caller_cancelled = True
    task.result()
    if caller_cancelled:
        raise asyncio.CancelledError


async def _resolve_headers_helper(
    helper_cmd: str,
    *,
    cwd: str | None = None,
    argv: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Run a headersHelper shell command and return the parsed JSON headers.

    The command must write a JSON object of string key-value pairs to stdout.
    User-scoped helpers retain shell compatibility. Project-scoped helpers pass
    a reviewed *argv* and explicit environment, and therefore never invoke a
    shell. On any error the result is an empty dict so the connection attempt
    can still proceed with static headers only.
    """
    proc: asyncio.subprocess.Process | None = None
    try:
        if argv is not None:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
                start_new_session=os.name == "posix",
            )
        else:
            proc = await asyncio.create_subprocess_shell(
                helper_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                start_new_session=os.name == "posix",
            )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=HEADERS_HELPER_TIMEOUT_S
        )
        if proc.returncode != 0:
            logger.warning(
                "headersHelper returned exit code %d: %s",
                proc.returncode,
                stderr.decode(errors="replace").strip()[:200],
            )
            return {}
        parsed = _json.loads(stdout.decode())
        if not isinstance(parsed, dict):
            logger.warning("headersHelper output is not a JSON object")
            return {}
        return {str(k): str(v) for k, v in parsed.items()}
    except asyncio.TimeoutError:
        if proc is not None:
            await _settle_headers_helper_process(proc)
        logger.warning("headersHelper timed out after %.0fs", HEADERS_HELPER_TIMEOUT_S)
        return {}
    except asyncio.CancelledError:
        if proc is not None:
            await _settle_headers_helper_process(proc)
        raise
    except (_json.JSONDecodeError, Exception) as exc:
        logger.warning("headersHelper failed: %s", exc)
        return {}


async def _build_effective_headers(
    config: MCPServerConfig, *, trusted: bool = True
) -> dict[str, str]:
    """Merge static headers, OAuth headers, and dynamic headersHelper output.

    Layer order (later overrides earlier):
    1. Static ``config.headers``
    2. OAuth ``Authorization`` header (if ``config.oauth`` is set)
    3. Dynamic ``headersHelper`` output

    ``headersHelper`` runs a shell command read straight from configuration. For
    an untrusted (unapproved project-scoped) server that is a command-injection
    vector, so the helper is skipped entirely unless *trusted* is True.
    """
    headers = dict(config.headers or {})

    if config.oauth and config.url:
        from .oauth import resolve_oauth_headers

        try:
            oauth_headers = await resolve_oauth_headers(config.name, config.url, config.oauth)
            headers.update(oauth_headers)
        except Exception as exc:
            logger.warning("OAuth flow failed for '%s': %s", config.name, exc)

    if not config.headers_helper:
        return headers
    if not trusted:
        logger.warning(
            "Skipping headersHelper for untrusted MCP server '%s': "
            "project-scoped auth helpers only run after project approval.",
            config.name,
        )
        return headers

    dynamic = await _resolve_headers_helper(
        config.headers_helper,
        cwd=config.execution_cwd,
        argv=config.headers_helper_argv,
        env=(dict(config.env_vars or {}) if config.headers_helper_argv else None),
    )
    headers.update(dynamic)
    return headers


def _install_output_truncation(server: MCPServer, server_name: str) -> None:
    """Wrap ``server.call_tool`` so oversized results are truncated.

    MCP tool calls go through the SDK's ``MCPServer.call_tool`` and never touch
    koder's ``function_tool`` truncation, so the ``MAX_MCP_OUTPUT_TOKENS`` cap
    would otherwise be dead. We wrap the bound method with a thin coroutine that
    post-processes the ``CallToolResult`` via :func:`truncate_call_tool_result`.

    Idempotent: a server already wrapped (``_koder_output_capped``) is skipped.
    """
    from .limits import truncate_call_tool_result

    original = getattr(server, "call_tool", None)
    if original is None or getattr(server, "_koder_output_capped", False):
        return

    @functools.wraps(original)
    async def _capped_call_tool(*args: Any, **kwargs: Any) -> Any:
        result = await original(*args, **kwargs)
        try:
            return truncate_call_tool_result(result, server_name)
        except Exception:  # never let capping break a real tool call
            logger.debug("MCP output truncation failed for '%s'", server_name, exc_info=True)
            return result

    try:
        server.call_tool = _capped_call_tool  # type: ignore[assignment]
        server._koder_output_capped = True  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        # Some SDK server objects may not allow attribute assignment; skip.
        logger.debug("Could not install MCP output cap on '%s'", server_name)


def _install_cleanup_guard(server: MCPServer) -> None:
    """Make concrete-server cleanup retryable and safe for concurrent callers."""
    original = getattr(server, "cleanup", None)
    if original is None or getattr(server, "_koder_cleanup_guarded", False):
        return

    cleanup_lock = asyncio.Lock()
    cleaned = False
    cleanup_task: asyncio.Task[Any] | None = None

    async def _run_cleanup(*args: Any, **kwargs: Any) -> Any:
        nonlocal cleaned, cleanup_task
        task = asyncio.current_task()
        try:
            result = await original(*args, **kwargs)
        except BaseException:
            async with cleanup_lock:
                if cleanup_task is task:
                    cleanup_task = None
            raise
        else:
            async with cleanup_lock:
                if cleanup_task is task:
                    cleaned = True
                    cleanup_task = None
            return result

    @functools.wraps(original)
    async def _cleanup_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal cleanup_task
        async with cleanup_lock:
            if cleaned:
                return None
            if cleanup_task is None:
                cleanup_task = asyncio.create_task(_run_cleanup(*args, **kwargs))
            task = cleanup_task
        return await asyncio.shield(task)

    try:
        server.cleanup = _cleanup_once  # type: ignore[method-assign]
        server._koder_cleanup_guarded = True  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        logger.debug("Could not install MCP cleanup guard on '%s'", getattr(server, "name", ""))


def _install_project_authorization_guard(
    server: MCPServer,
    config: MCPServerConfig,
    *,
    validator: Any = None,
) -> None:
    """Install the centralized project authorization/session boundary."""
    if config.scope != MCPServerScope.PROJECT:
        return

    from .runtime_authorization import attach_project_authorization_validator

    try:
        attach_project_authorization_validator(server, config, validator=validator)
    except (AttributeError, TypeError):
        logger.error(
            "Could not install project authorization boundary on '%s'; refusing server",
            config.name,
        )
        raise RuntimeError(
            f"Project MCP server '{config.name}' cannot be guarded at runtime"
        ) from None


class MCPServerFactory:
    """Factory for creating MCP server instances from configurations."""

    @staticmethod
    async def _build_concrete_server(
        config: MCPServerConfig,
        channel_callback: ChannelNotifCallbackT | None = None,
        *,
        trusted: bool = True,
    ) -> MCPServer:
        """Build one unguarded concrete transport with retryable cleanup."""
        tool_filter = None
        if config.allowed_tools or config.blocked_tools:
            tool_filter = create_static_tool_filter(
                allowed_tool_names=config.allowed_tools,
                blocked_tool_names=config.blocked_tools,
            )

        if config.transport_type == MCPServerType.STDIO:
            server: MCPServer = MCPServerFactory._create_stdio_server(
                config,
                tool_filter,
                channel_callback=channel_callback,
            )
        elif config.transport_type == MCPServerType.SSE:
            server = await MCPServerFactory._create_sse_server(
                config,
                tool_filter,
                trusted=trusted,
            )
        elif config.transport_type == MCPServerType.HTTP:
            server = await MCPServerFactory._create_http_server(
                config,
                tool_filter,
                trusted=trusted,
            )
        else:
            raise ValueError(f"Unsupported transport type: {config.transport_type}")

        _install_output_truncation(server, config.name)
        _install_cleanup_guard(server)
        return server

    @staticmethod
    async def create_server(
        config: MCPServerConfig,
        channel_callback: ChannelNotifCallbackT | None = None,
        *,
        trusted: bool = True,
    ) -> MCPServer:
        """Create a standalone MCP server instance from configuration.

        If *channel_callback* is provided and the transport is stdio,
        a :class:`ChannelAwareMCPServerStdio` is used instead of the
        standard ``MCPServerStdio`` so that ``notifications/claude/channel``
        messages are intercepted before the SDK drops them.

        For SSE/HTTP servers with a ``headersHelper``, the helper command
        is executed and its JSON output is merged into the connection headers —
        but only when *trusted* is True (unapproved project servers do not run
        their auth helper).
        """

        async def _prepare_server(authorization_validator: Any = None) -> MCPServer:
            server = await MCPServerFactory._build_concrete_server(
                config,
                channel_callback,
                trusted=trusted,
            )
            if authorization_validator is not None:
                _install_project_authorization_guard(
                    server,
                    config,
                    validator=authorization_validator,
                )
            return server

        try:
            if config.scope == MCPServerScope.PROJECT:
                from .runtime_authorization import ProjectServerAuthorizationValidator

                # The authorization owner exists before tool-filter creation,
                # OAuth, headersHelper, transport construction, or any other
                # fallible connection preparation. The whole preparation is an
                # admitted operation, and server callbacks still run lock-free.
                validator = ProjectServerAuthorizationValidator(config)
                try:
                    return await validator.run_authorized(_prepare_server, validator)
                except BaseException:
                    if validator.server is not None:
                        await validator.disable()
                    raise

            return await _prepare_server()

        except Exception as e:
            logger.error(f"Failed to create MCP server '{config.name}': {e}")
            raise

    @staticmethod
    def _create_stdio_server(
        config: MCPServerConfig,
        tool_filter: Any = None,
        channel_callback: ChannelNotifCallbackT | None = None,
    ) -> MCPServerStdio:
        """Create a stdio MCP server."""
        if not config.command:
            raise ValueError("stdio server requires a command")

        params = MCPServerStdioParams(
            command=config.command,
            args=config.args or [],
            env=config.env_vars or {},
            cwd=config.execution_cwd,
        )

        if channel_callback is not None:
            return ChannelAwareMCPServerStdio(
                params=params,
                client_session_timeout_seconds=get_timeout_seconds(),
                tool_filter=tool_filter,
                cache_tools_list=config.cache_tools_list,
                channel_callback=channel_callback,
                channel_server_name=config.name,
            )

        return ElicitationAwareStdio(
            params=params,
            client_session_timeout_seconds=get_timeout_seconds(),
            tool_filter=tool_filter,
            cache_tools_list=config.cache_tools_list,
        )

    @staticmethod
    async def _create_sse_server(
        config: MCPServerConfig, tool_filter, *, trusted: bool = True
    ) -> MCPServerSse:
        """Create an SSE MCP server."""
        if not config.url:
            raise ValueError("SSE server requires a URL")

        effective_headers = await _build_effective_headers(config, trusted=trusted)

        params = MCPServerSseParams(
            url=config.url,
            headers=effective_headers,
            timeout=get_timeout_seconds(),
        )

        return ElicitationAwareSse(
            params=params,
            tool_filter=tool_filter,
            cache_tools_list=config.cache_tools_list,
        )

    @staticmethod
    async def _create_http_server(
        config: MCPServerConfig, tool_filter, *, trusted: bool = True
    ) -> MCPServerStreamableHttp:
        """Create an HTTP MCP server."""
        if not config.url:
            raise ValueError("HTTP server requires a URL")

        effective_headers = await _build_effective_headers(config, trusted=trusted)

        params = MCPServerStreamableHttpParams(
            url=config.url,
            headers=effective_headers,
            timeout=get_timeout_seconds(),
        )

        return ElicitationAwareHttp(
            params=params,
            tool_filter=tool_filter,
            cache_tools_list=config.cache_tools_list,
        )

    @staticmethod
    async def create_servers_from_configs(
        configs: List[MCPServerConfig],
        *,
        existing_servers: List[MCPServer] | None = None,
    ) -> List[MCPServer]:
        """Create stable, authorization-aware handles for agent-specific configs."""
        from .prompts import normalize_mcp_name

        runtime_names: dict[str, str] = {}
        for server in existing_servers or []:
            name = str(getattr(server, "name", ""))
            normalized = normalize_mcp_name(name).lower()
            if normalized:
                runtime_names[normalized] = name
        for config in configs:
            normalized = normalize_mcp_name(config.name).lower()
            existing_name = runtime_names.get(normalized)
            if not normalized:
                raise ValueError(
                    f"MCP server name '{config.name}' normalizes to an empty runtime name"
                )
            if existing_name is not None:
                raise ValueError(
                    f"MCP server name collision for runtime name '{normalized}': "
                    f"'{config.name}' conflicts with '{existing_name}'"
                )
            runtime_names[normalized] = config.name

        servers: list[MCPServer] = []
        try:
            for config in configs:
                try:
                    server, _manager = await MCPServerFactory.create_and_connect_with_retry(config)
                    servers.append(server)
                    logger.info(f"Created MCP server '{config.name}' ({config.transport_type})")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error(f"Failed to create MCP server '{config.name}': {exc}")
                    continue
        except BaseException:
            await cleanup_mcp_servers(
                servers,
                logger=logger,
                propagate_cancellation=False,
            )
            raise

        return servers

    @staticmethod
    async def create_and_connect_with_retry(
        config: MCPServerConfig,
        channel_callback: Any = None,
        reconnection_config: Optional[ReconnectionConfig] = None,
        *,
        trusted: bool = True,
    ) -> tuple[MCPServer, ReconnectionManager]:
        """Create and connect an MCP server with reconnection support.

        Returns:
            A tuple of (server, reconnection_manager) where the manager can be
            used for future reconnection attempts. The manager retains the
            factory closure so callers can trigger a fresh connect later.
        """
        reconnection_mgr = ReconnectionManager(reconnection_config)
        retirement_owner = reconnection_mgr._retirement_owner
        authorization_validator = None
        if config.scope == MCPServerScope.PROJECT:
            from .runtime_authorization import ProjectServerAuthorizationValidator

            authorization_validator = ProjectServerAuthorizationValidator(config)

        async def _retire_candidate(server: MCPServer) -> None:
            retirement = retirement_owner.retire(server)
            if retirement is None:
                return
            try:
                await asyncio.shield(retirement)
            except asyncio.CancelledError:
                retain_orphaned_retirements(retirement_owner)
                raise
            except Exception:
                logger.warning(
                    "Failed to clean up MCP connection candidate '%s'",
                    config.name,
                    exc_info=True,
                )
            except BaseException:
                retain_orphaned_retirements(retirement_owner)
                raise

        async def connect_fn() -> MCPServer:
            if retirement_owner.pending_count:
                await retirement_owner.drain(max_attempts=1)
            server: MCPServer | None = None
            try:
                if authorization_validator is not None:
                    # The stable validator owns the surrounding admission. The
                    # concrete candidate itself stays unwrapped so retirement can
                    # retry cleanup independently of authorization state.
                    server = await MCPServerFactory._build_concrete_server(
                        config,
                        channel_callback,
                        trusted=trusted,
                    )
                else:
                    # Keep the public factory seam for non-project providers and
                    # existing transport retry tests.
                    server = await MCPServerFactory.create_server(
                        config,
                        channel_callback,
                        trusted=trusted,
                    )
                _install_cleanup_guard(server)
                retirement_owner.hold(server)
                await server.connect()
            except BaseException:
                if server is not None:
                    await _retire_candidate(server)
                raise
            return server

        # Try initial connection with retry
        live_server: LiveMCPServer | None = None
        success = False

        async def _connect_initial_handle() -> LiveMCPServer:
            candidate = await connect_fn()
            handle = LiveMCPServer(
                config.name,
                candidate,
                retirement_owner=retirement_owner,
            )
            try:
                if authorization_validator is not None:
                    _install_project_authorization_guard(
                        handle,
                        config,
                        validator=authorization_validator,
                    )
            except BaseException:
                try:
                    await handle.cleanup()
                except BaseException:
                    retain_orphaned_retirements(retirement_owner)
                raise
            return handle

        async def retry_connect() -> None:
            nonlocal live_server
            if authorization_validator is not None:
                live_server = await authorization_validator.run_authorized(_connect_initial_handle)
            else:
                live_server = await _connect_initial_handle()

        try:
            success = await reconnection_mgr.reconnect_with_backoff(retry_connect)
        except BaseException:
            retain_orphaned_retirements(retirement_owner)
            raise

        if not success or live_server is None:
            try:
                await retirement_owner.drain()
            except asyncio.CancelledError:
                retain_orphaned_retirements(retirement_owner)
                raise
            except Exception:
                retain_orphaned_retirements(retirement_owner)
            except BaseException:
                retain_orphaned_retirements(retirement_owner)
                raise
            raise ConnectionError(
                f"Failed to connect to MCP server '{config.name}' after "
                f"{reconnection_mgr.config.max_attempts} attempts"
            )

        # Retain the stable handle plus the raw candidate factory. Reconnect wraps
        # construction, connect, and publication in the same long-lived validator.
        reconnection_mgr.bind(config=config, server=live_server, connect_fn=connect_fn)

        return live_server, reconnection_mgr

    @staticmethod
    def validate_config(config: MCPServerConfig) -> Optional[str]:
        """Validate an MCP server configuration."""
        try:
            if config.transport_type == MCPServerType.STDIO:
                if not config.command:
                    return "stdio servers must have a command"
            elif config.transport_type in [MCPServerType.SSE, MCPServerType.HTTP]:
                if not config.url:
                    return f"{config.transport_type} servers must have a URL"
                if not config.url.startswith(("http://", "https://")):
                    return "URL must start with http:// or https://"
            else:
                return f"Unsupported transport type: {config.transport_type}"

            # Validate tool lists don't overlap
            if config.allowed_tools and config.blocked_tools:
                overlap = set(config.allowed_tools) & set(config.blocked_tools)
                if overlap:
                    return f"Tools cannot be in both allowed and blocked lists: {list(overlap)}"

            return None
        except Exception as e:
            return f"Configuration validation error: {e}"
