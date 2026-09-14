"""MCP (Model Context Protocol) support for Koder."""

import asyncio
import atexit
import concurrent.futures
import logging
import threading
import weakref
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

from rich.console import Console

from koder_agent.harness.channels.admission import ChannelAdmission
from koder_agent.harness.channels.gate import (
    PluginChannelOrigin,
    find_channel_entry,
    gate_channel_server,
)
from koder_agent.harness.channels.notification import ChannelNotificationRouter
from koder_agent.harness.channels.state import get_allowed_channels
from koder_agent.harness.channels.types import ChannelEntryPlugin
from koder_agent.harness.execution_context import get_execution_cwd
from koder_agent.harness.plugins.context import get_plugin_root

try:  # pragma: no cover - depends on optional SDK extras at import time
    from agents.mcp import MCPServer
except ImportError:  # pragma: no cover
    MCPServer = Any

from .lifecycle import cleanup_mcp_servers as _cleanup_legacy_mcp_servers
from .prompts import (
    MCPPrompt,
    MCPPromptRegistry,
    execute_prompt,
    get_prompt_registry,
    normalize_mcp_name,
)
from .reconnection import (
    LiveMCPServer,
    ReconnectionConfig,
    ReconnectionManager,
    drain_orphaned_retirements,
)
from .server_config import MCPServerConfig, MCPServerScope, MCPServerType
from .server_manager import MCPServerManager

try:  # pragma: no cover - optional transport dependencies may be absent
    from .server_factory import MCPServerFactory
except ImportError:  # pragma: no cover
    MCPServerFactory = None  # type: ignore[assignment]

# Dedicated stderr console so MCP-connection warnings reach the user even when
# the root logger is pinned high elsewhere.
_console = Console(stderr=True)
_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _MCPServerCleanupOutcome:
    cancelled: bool = False
    error: BaseException | None = None


class _MCPServerOwnerState:
    """Finalizer-safe state shared by one transferable MCP owner."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.servers: list[MCPServer] = []
        self.runtime_resources: ExitStack | None = None
        self.reconnection_managers: dict[str, ReconnectionManager] = {}
        self.prompt_registry = MCPPromptRegistry()
        self.accepting_adoptions = True
        self.closed = False
        self.orphaned = False
        self.cleanup_future: concurrent.futures.Future[_MCPServerCleanupOutcome] | None = None

    def adopt(
        self,
        server: MCPServer,
        *,
        server_name: str,
        reconnection_manager: ReconnectionManager | None = None,
        runtime_resources: ExitStack | None = None,
    ) -> None:
        with self.lock:
            if not self.accepting_adoptions:
                raise RuntimeError("Cannot adopt an MCP server into a closed owner")
            if runtime_resources is not None and self.runtime_resources is not None:
                raise RuntimeError("MCP runtime resources already adopted")
            if runtime_resources is not None:
                self.runtime_resources = runtime_resources
            self.servers.append(server)
            if reconnection_manager is not None:
                self.reconnection_managers[server_name] = reconnection_manager

    def begin_cleanup(
        self,
    ) -> tuple[concurrent.futures.Future[_MCPServerCleanupOutcome], bool] | None:
        with self.lock:
            if self.closed:
                return None
            self.accepting_adoptions = False
            if self.cleanup_future is not None:
                return self.cleanup_future, False
            cleanup_future: concurrent.futures.Future[_MCPServerCleanupOutcome] = (
                concurrent.futures.Future()
            )
            self.cleanup_future = cleanup_future
            return cleanup_future, True

    def pending_servers(self) -> list[MCPServer]:
        with self.lock:
            return list(self.servers)

    def mark_server_cleaned(self, server: MCPServer) -> None:
        with self.lock:
            for index, pending in enumerate(self.servers):
                if pending is server:
                    del self.servers[index]
                    break

    def resources_if_servers_cleaned(self) -> tuple[bool, ExitStack | None]:
        with self.lock:
            if self.servers:
                return False, None
            return True, self.runtime_resources

    def finish_cleanup(
        self,
        cleanup_future: concurrent.futures.Future[_MCPServerCleanupOutcome],
        *,
        resources_closed: bool,
    ) -> bool:
        with self.lock:
            if resources_closed and not self.servers:
                self.runtime_resources = None
            if not self.servers and self.runtime_resources is None:
                self.reconnection_managers.clear()
                self.prompt_registry.clear()
                self.closed = True
            if self.cleanup_future is cleanup_future:
                self.cleanup_future = None
            return self.closed

    def abandon(self) -> bool:
        with self.lock:
            if self.closed or self.orphaned:
                return False
            self.accepting_adoptions = False
            self.orphaned = True
            return True

    def is_closed(self) -> bool:
        with self.lock:
            return self.closed

    def is_accepting(self) -> bool:
        with self.lock:
            return self.accepting_adoptions


async def _run_mcp_owner_cleanup(
    state: _MCPServerOwnerState,
    cleanup_future: concurrent.futures.Future[_MCPServerCleanupOutcome],
    *,
    timeout: float = 3.0,
) -> None:
    cancelled = False
    error: BaseException | None = None
    resources_closed = False
    try:
        for server in state.pending_servers():
            cleanup = getattr(server, "cleanup", None)
            if cleanup is None:
                state.mark_server_cleaned(server)
                continue
            try:
                await asyncio.wait_for(cleanup(), timeout=timeout)
            except asyncio.CancelledError:
                cancelled = True
                _logger.debug(
                    "MCP server cleanup was cancelled for %s; retaining it for retry",
                    getattr(server, "name", ""),
                )
            except asyncio.TimeoutError:
                _logger.debug(
                    "Timed out cleaning up MCP server %s; retaining it for retry",
                    getattr(server, "name", ""),
                )
            except Exception as exc:
                _logger.debug(
                    "Failed to clean up MCP server %s; retaining it for retry: %s",
                    getattr(server, "name", ""),
                    exc,
                    exc_info=True,
                )
            else:
                state.mark_server_cleaned(server)

        servers_cleaned, resources = state.resources_if_servers_cleaned()
        if servers_cleaned:
            if resources is None:
                resources_closed = True
            else:
                try:
                    resources.close()
                except BaseException as exc:
                    if isinstance(exc, asyncio.CancelledError):
                        cancelled = True
                    else:
                        _logger.debug(
                            "Failed to close MCP runtime resources; retaining them for retry",
                            exc_info=True,
                        )
                else:
                    resources_closed = True
    except BaseException as exc:  # pragma: no cover - defensive state-machine guard
        if isinstance(exc, asyncio.CancelledError):
            cancelled = True
        else:
            error = exc
            _logger.debug("Unexpected MCP owner cleanup failure", exc_info=True)
    finally:
        state.finish_cleanup(cleanup_future, resources_closed=resources_closed)
        if not cleanup_future.done():
            cleanup_future.set_result(_MCPServerCleanupOutcome(cancelled=cancelled, error=error))


async def _await_mcp_owner_cleanup(
    state: _MCPServerOwnerState,
) -> tuple[bool, _MCPServerCleanupOutcome, asyncio.CancelledError | None]:
    operation = state.begin_cleanup()
    if operation is None:
        return False, _MCPServerCleanupOutcome(), None

    cleanup_future, should_start = operation
    if should_start:
        try:
            cleanup_task = asyncio.create_task(_run_mcp_owner_cleanup(state, cleanup_future))
        except BaseException as exc:  # pragma: no cover - create_task is normally infallible
            state.finish_cleanup(cleanup_future, resources_closed=False)
            if not cleanup_future.done():
                cleanup_future.set_result(_MCPServerCleanupOutcome(error=exc))
        else:
            cleanup_task.add_done_callback(
                lambda completed: completed.exception() if not completed.cancelled() else None
            )

    caller_cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            outcome = await asyncio.shield(asyncio.wrap_future(cleanup_future))
            break
        except asyncio.CancelledError as exc:
            if caller_cancellation is None:
                caller_cancellation = exc
    return True, outcome, caller_cancellation


_orphaned_owner_lock = threading.Lock()
_orphaned_owner_states: list[_MCPServerOwnerState] = []
_ATEXIT_ORPHAN_DRAIN_ATTEMPTS = 3


def _remove_orphaned_owner(state: _MCPServerOwnerState) -> None:
    with _orphaned_owner_lock:
        _orphaned_owner_states[:] = [
            queued for queued in _orphaned_owner_states if queued is not state
        ]


async def drain_orphaned_mcp_owners() -> None:
    """Retry durable owners while propagating only cancellation of this caller."""
    with _orphaned_owner_lock:
        states = list(_orphaned_owner_states)

    caller_cancellation: asyncio.CancelledError | None = None
    cleanup_cancelled = False
    first_error: BaseException | None = None
    for state in states:
        _attempted, outcome, state_caller_cancellation = await _await_mcp_owner_cleanup(state)
        if state.is_closed():
            _remove_orphaned_owner(state)
        if caller_cancellation is None and state_caller_cancellation is not None:
            caller_cancellation = state_caller_cancellation
        cleanup_cancelled = cleanup_cancelled or outcome.cancelled
        if first_error is None and outcome.error is not None:
            first_error = outcome.error

    if caller_cancellation is not None:
        raise caller_cancellation
    if first_error is not None:
        raise first_error
    if cleanup_cancelled:
        _logger.debug(
            "MCP orphan cleanup reported cancellation; retaining unfinished owners for retry"
        )


def _finalize_mcp_owner(state: _MCPServerOwnerState) -> None:
    if not state.abandon():
        return
    with _orphaned_owner_lock:
        _orphaned_owner_states.append(state)


def _drain_orphaned_mcp_owners_at_exit() -> None:
    for attempt in range(1, _ATEXIT_ORPHAN_DRAIN_ATTEMPTS + 1):
        with _orphaned_owner_lock:
            if not _orphaned_owner_states:
                return
        try:
            asyncio.run(drain_orphaned_mcp_owners())
        except BaseException:
            _logger.debug(
                "Process-exit MCP orphan cleanup attempt %d was incomplete",
                attempt,
                exc_info=True,
            )

    with _orphaned_owner_lock:
        remaining = len(_orphaned_owner_states)
    if remaining:
        _logger.debug(
            "Process-exit MCP orphan cleanup left %d owner(s) retryable after %d attempts",
            remaining,
            _ATEXIT_ORPHAN_DRAIN_ATTEMPTS,
        )


atexit.register(_drain_orphaned_mcp_owners_at_exit)


class MCPServerSet(list[MCPServer]):
    """One transferable owner for servers, prompts, reconnect state, and snapshots."""

    def __init__(
        self,
        servers: Iterable[MCPServer] = (),
        *,
        runtime_resources: ExitStack | None = None,
        reconnection_managers: dict[str, ReconnectionManager] | None = None,
    ) -> None:
        initial_servers = list(servers)
        super().__init__(initial_servers)
        self._owner_state = _MCPServerOwnerState()
        self.reconnection_managers = self._owner_state.reconnection_managers
        self.prompt_registry = self._owner_state.prompt_registry
        for server in initial_servers:
            self._owner_state.adopt(
                server,
                server_name=str(getattr(server, "name", "")),
            )
        if runtime_resources is not None:
            self._owner_state.runtime_resources = runtime_resources
        self.reconnection_managers.update(reconnection_managers or {})
        self._owner_finalizer = weakref.finalize(
            self,
            _finalize_mcp_owner,
            self._owner_state,
        )

    def adopt_server(
        self,
        server: MCPServer,
        *,
        server_name: str,
        reconnection_manager: ReconnectionManager | None = None,
        runtime_resources: ExitStack | None = None,
    ) -> None:
        """Adopt a newly connected server before any later await can fail."""
        self._owner_state.adopt(
            server,
            server_name=server_name,
            reconnection_manager=reconnection_manager,
            runtime_resources=runtime_resources,
        )
        super().append(server)

    async def aclose(self, *, propagate_cancellation: bool = True) -> bool:
        """Close this owner through a shared, cancellation-safe retry operation."""
        try:
            attempted, outcome, caller_cancellation = await _await_mcp_owner_cleanup(
                self._owner_state
            )
        finally:
            if self._owner_state.is_closed() and self._owner_finalizer.alive:
                self._owner_finalizer.detach()
        if outcome.error is not None:
            raise outcome.error
        if propagate_cancellation and caller_cancellation is not None:
            raise caller_cancellation
        if propagate_cancellation and outcome.cancelled:
            raise asyncio.CancelledError
        return attempted


async def close_mcp_servers(
    servers: Iterable[MCPServer] | None,
    *,
    propagate_cancellation: bool = True,
) -> bool:
    """Close a server owner exactly once, with a plain-list compatibility path."""
    if servers is None:
        return False
    close = getattr(servers, "aclose", None)
    if callable(close):
        return bool(await close(propagate_cancellation=propagate_cancellation))
    await _cleanup_legacy_mcp_servers(
        list(servers),
        logger=_logger,
        propagate_cancellation=propagate_cancellation,
    )
    return True


def detach_mcp_server_owner(holder: object | None) -> Iterable[MCPServer] | None:
    """Detach one agent's MCP owner before cleanup so repeats are harmless."""
    if holder is None:
        return None
    owner = getattr(holder, "_koder_mcp_servers", None)
    if owner is None:
        owner = getattr(holder, "mcp_servers", None)
    try:
        setattr(holder, "_koder_mcp_servers", None)
    except Exception:
        pass
    try:
        setattr(holder, "mcp_servers", [])
    except Exception:
        pass
    return owner


def get_reconnection_managers(
    servers: Iterable[MCPServer] | None = None,
) -> "dict[str, ReconnectionManager]":
    """Return reconnection managers owned by one server/session set."""
    managers = getattr(servers, "reconnection_managers", None)
    return managers if isinstance(managers, dict) else {}


async def reconnect_unhealthy_servers(
    servers: Iterable[MCPServer] | None = None,
) -> "dict[str, bool]":
    """Reconnect any retained MCP servers whose connection looks unhealthy.

    Returns a mapping of server name -> healthy(after) for every managed server.
    This is the runtime reconnect entry point; wiring it into the scheduler's
    lifecycle is a cross-file concern handled outside this package.
    """
    results: dict[str, bool] = {}
    for name, mgr in get_reconnection_managers(servers).items():
        try:
            results[name] = await mgr.reconnect_if_needed()
        except Exception:  # pragma: no cover - defensive
            results[name] = False
    return results


class _PluginMCPConfigs(list[MCPServerConfig]):
    """Retain plugin origins separately from user-controlled MCP server fields."""

    def __init__(self) -> None:
        super().__init__()
        self.channel_origins: dict[int, PluginChannelOrigin] = {}


@contextmanager
def _load_plugin_mcp_configs():
    """Yield plugin MCP configs while their private plugin snapshots exist."""
    import json

    from koder_agent.harness.plugins.env import expand_plugin_vars, plugin_env_vars
    from koder_agent.harness.plugins.lifecycle import PluginLifecycleService
    from koder_agent.harness.plugins.marketplace import MarketplaceStore
    from koder_agent.harness.plugins.path_safety import (
        PluginPathError,
        open_plugin_component,
        snapshot_plugin_tree,
    )

    with ExitStack() as snapshots:
        configs = _PluginMCPConfigs()
        try:
            plugin_root = get_plugin_root()
            lifecycle = PluginLifecycleService(plugin_root)
            marketplaces = MarketplaceStore.default()
            for manifest, state in lifecycle.installed_plugins():
                if not state.enabled:
                    continue
                try:
                    plugin_dir = snapshots.enter_context(
                        snapshot_plugin_tree(lifecycle.resolve_plugin_target(manifest.name))
                    )
                except PluginPathError:
                    continue
                mcp_source_path = plugin_dir.joinpath(
                    *((manifest.mcp_servers or ".mcp.json").split("/"))
                )
                try:
                    with open_plugin_component(
                        plugin_dir,
                        manifest.mcp_servers,
                        default=".mcp.json",
                        field_name="mcpServers",
                        expect="file",
                    ) as mcp_json_path:
                        if mcp_json_path is None:
                            continue
                        raw = json.loads(mcp_json_path.read_text("utf-8"))
                except (PluginPathError, OSError, ValueError) as exc:
                    _logger.warning(
                        "Failed to read .mcp.json from plugin '%s': %s",
                        manifest.name,
                        exc,
                    )
                    continue
                try:
                    servers = raw.get("mcpServers", {})
                    env_vars = plugin_env_vars(manifest.name, plugin_dir)

                    for server_name, server_def in servers.items():
                        command = server_def.get("command", "")
                        args = server_def.get("args", [])

                        command = expand_plugin_vars(command, manifest.name, plugin_dir)
                        args = [expand_plugin_vars(a, manifest.name, plugin_dir) for a in args]

                        server_env = dict(server_def.get("env", {}))
                        server_env.update(env_vars)

                        koder_channels_dir = Path.home() / ".koder" / "channels" / server_name
                        state_dir_key = f"{server_name.upper()}_STATE_DIR"
                        if state_dir_key not in server_env:
                            server_env[state_dir_key] = str(koder_channels_dir)

                        transport_type = server_def.get("type", "stdio")
                        config = MCPServerConfig(
                            name=server_name,
                            transport_type=MCPServerType(transport_type),
                            command=command,
                            args=args,
                            env_vars=server_env,
                            url=server_def.get("url"),
                            headers=server_def.get("headers"),
                            cache_tools_list=server_def.get("cacheToolsList", False),
                            source_path=str(mcp_source_path),
                        )
                        configs.append(config)
                        installation = state.origin
                        configs.channel_origins[id(config)] = PluginChannelOrigin(
                            name=manifest.name,
                            marketplace=(
                                installation.marketplace
                                if installation is not None
                                and marketplaces.matches_origin(installation)
                                else None
                            ),
                        )
                        _logger.info(
                            "Loaded MCP server '%s' from plugin '%s'",
                            server_name,
                            manifest.name,
                        )
                except Exception as exc:
                    _logger.warning(
                        "Failed to load .mcp.json from plugin '%s': %s",
                        manifest.name,
                        exc,
                    )
        except Exception as exc:
            _logger.debug("Plugin MCP loading skipped: %s", exc)

        yield configs


def _mcp_config_source(config: MCPServerConfig) -> str:
    return str(config.source_path or "runtime configuration")


def _validate_mcp_server_identities(configs: Iterable[MCPServerConfig]) -> None:
    """Reject raw or normalized public server-name collisions before connection."""
    by_raw_name: dict[str, MCPServerConfig] = {}
    by_public_identity: dict[str, MCPServerConfig] = {}

    for config in configs:
        name = config.name
        previous = by_raw_name.get(name)
        if previous is not None:
            raise ValueError(
                f"Duplicate public MCP server identity '{name}': server name '{name}' "
                f"from {_mcp_config_source(previous)} conflicts exactly with "
                f"{_mcp_config_source(config)}"
            )

        public_identity = normalize_mcp_name(name).lower()
        if not public_identity:
            raise ValueError(
                f"MCP server name '{name}' from {_mcp_config_source(config)} normalizes to an "
                "empty public identity; choose a name containing letters, numbers, '_' or '-'."
            )
        previous = by_public_identity.get(public_identity)
        if previous is not None:
            raise ValueError(
                f"Duplicate public MCP server identity '{public_identity}': server names "
                f"'{previous.name}' from {_mcp_config_source(previous)} and '{name}' from "
                f"{_mcp_config_source(config)} normalize and case-fold to the same public name"
            )

        by_raw_name[name] = config
        by_public_identity[public_identity] = config


async def load_mcp_servers(
    extra_configs: Iterable[MCPServerConfig] | None = None,
) -> MCPServerSet:
    """Load and create MCP server instances from configuration."""
    try:
        await drain_orphaned_mcp_owners()
    except asyncio.CancelledError:
        raise
    except Exception:
        _logger.debug("Failed to retry abandoned MCP owner cleanup", exc_info=True)
    try:
        await drain_orphaned_retirements(max_attempts=1)
    except asyncio.CancelledError:
        raise
    except Exception:
        _logger.debug("Failed to retry orphaned MCP transport cleanup", exc_info=True)

    plugin_snapshots = ExitStack()
    plugin_resources_adopted = False
    owner = MCPServerSet()
    try:
        manager = MCPServerManager()
        configs = list(await manager.list_servers(cwd=str(get_execution_cwd())))

        # Also load MCP servers from enabled plugins
        plugin_config_source = _load_plugin_mcp_configs()
        if hasattr(plugin_config_source, "__enter__"):
            plugin_configs = plugin_snapshots.enter_context(plugin_config_source)
        else:
            plugin_configs = list(plugin_config_source)
        plugin_config_ids = {id(config) for config in plugin_configs}
        plugin_origins = getattr(plugin_configs, "channel_origins", {})
        if plugin_configs:
            configs.extend(plugin_configs)
        if extra_configs:
            configs.extend(extra_configs)

        _validate_mcp_server_identities(configs)

        if not configs:
            return owner

        # Interception is installed before initialization, but each callback
        # stays closed until its initialized server is adopted below.
        channel_admissions: dict[str, ChannelAdmission] = {}
        allowed = get_allowed_channels()
        if allowed:
            from .notifications import get_notification_handler

            handler = get_notification_handler()
            router = handler.channel_router or ChannelNotificationRouter()
            if handler.channel_router is None:
                handler.set_channel_router(router)
            for config in configs:
                origin = plugin_origins.get(id(config))
                if find_channel_entry(config.name, allowed, plugin_origin=origin) is not None:
                    channel_admissions[config.name] = ChannelAdmission(
                        config.name, router, owner._owner_state.is_accepting, origin
                    )
                elif origin is not None and origin.marketplace is None:
                    if any(
                        isinstance(entry, ChannelEntryPlugin) and entry.name == origin.name
                        for entry in allowed
                    ):
                        _console.print(
                            f"[yellow]Channel '{config.name}' skipped: plugin '{origin.name}' "
                            "has no matching verified marketplace provenance. Reinstall it "
                            "from the registered marketplace, or explicitly enable "
                            f"this installed server, use --channels server:{config.name}.[/yellow]"
                        )

        # Create server instances — channel-aware for those in --channels
        if MCPServerFactory is None:
            raise RuntimeError("MCP transport dependencies are unavailable")

        _logger.debug(
            "Channel server names: %s, callback set: %s",
            set(channel_admissions),
            bool(channel_admissions),
        )
        # Configure reconnection with retry
        reconnection_config = ReconnectionConfig(max_attempts=3, initial_delay=1.0, max_delay=10.0)
        connected: list[tuple[MCPServerConfig, MCPServer]] = []
        for config in configs:
            try:
                # SECURITY GATE: PROJECT-scoped servers come from an in-repo
                # .mcp.json and can run arbitrary commands / auth helpers. Only
                # connect them when the project is explicitly approved. Undecided
                # or rejected => skip (safe headless default). User/local scopes
                # (the user's own config) are unaffected.
                is_project = config.scope == MCPServerScope.PROJECT
                source_path = config.source_path
                if is_project and not manager.revalidate_project_config(config):
                    approval_message = (
                        f"Approval required for project MCP server '{config.name}' from "
                        f"{source_path or 'an unknown project source'}. Review and approve "
                        "that source's current configuration to enable it."
                    )
                    _logger.warning(approval_message)
                    _console.print(f"[yellow]⚠ {approval_message}[/yellow]")
                    continue

                cb = channel_admissions.get(config.name)
                _logger.debug(
                    "Creating server '%s': channel_callback=%s",
                    config.name,
                    cb is not None,
                )
                # A project server only reaches here when the gate above passed,
                # so every server at this point is trusted. The trusted flag also
                # gates the headersHelper inside the factory. Use
                # create_and_connect_with_retry for resilient connection and
                # retain the reconnection manager for runtime reconnects.
                server, reconnection_mgr = await MCPServerFactory.create_and_connect_with_retry(
                    config,
                    channel_callback=cb,
                    reconnection_config=reconnection_config,
                    trusted=True,
                )
                _logger.debug(
                    "Server '%s' class: %s",
                    config.name,
                    type(server).__name__,
                )
                runtime_resources = None
                if id(config) in plugin_config_ids and not plugin_resources_adopted:
                    runtime_resources = plugin_snapshots.pop_all()
                try:
                    owner.adopt_server(
                        server,
                        server_name=config.name,
                        reconnection_manager=reconnection_mgr,
                        runtime_resources=runtime_resources,
                    )
                except BaseException:
                    if runtime_resources is not None:
                        runtime_resources.close()
                    reconnection_mgr.retire_unpublished()
                    raise
                if runtime_resources is not None:
                    plugin_resources_adopted = True
                connected.append((config, server))
                if cb is not None:
                    cb.bind(server)
                    # Verify capability after connection
                    caps = getattr(server, "server_initialize_result", None)
                    if caps is not None:
                        caps = getattr(caps, "capabilities", caps)
                    result = gate_channel_server(
                        config.name, caps, plugin_origin=plugin_origins.get(id(config))
                    )
                    if result.action == "register":
                        _logger.info("Channel registered: '%s'", config.name)
                    else:
                        _logger.debug(
                            "Channel skipped for '%s': %s (%s)",
                            config.name,
                            result.kind,
                            result.reason,
                        )
                _logger.info(f"Created MCP server '{config.name}' ({config.transport_type})")
            except Exception as e:
                _logger.error(f"Failed to create MCP server '{config.name}': {e}")
                # Surface the failure to the user — the root logger is pinned
                # high elsewhere, so a bare _logger.error() would be invisible.
                _console.print(f"[yellow]⚠ MCP server '{config.name}' unavailable: {e}[/yellow]")
                continue

        # Try to discover prompts from connected servers
        # Iterate the (config, server) pairs we actually connected, rather than
        # zip(configs, servers): a skipped server would misalign the zip and
        # attribute prompts to the wrong config after the first failure.
        registry = owner.prompt_registry
        for config, server in connected:
            try:
                await _discover_prompts(config.name, server, registry)
            except Exception:
                pass  # Prompt discovery is best-effort

        return owner

    except BaseException as exc:
        try:
            await owner.aclose(propagate_cancellation=False)
        except BaseException:
            _logger.debug("Failed to close MCP owner after load failure", exc_info=True)
        if isinstance(exc, Exception):
            raise RuntimeError(f"Failed to load MCP servers: {exc}") from exc
        raise
    finally:
        plugin_snapshots.close()


async def _discover_prompts(
    server_name: str,
    server: MCPServer,
    registry: MCPPromptRegistry,
) -> None:
    """Try to discover prompts from an MCP server."""
    try:
        from .runtime_authorization import call_authorized_server_method

        result = await call_authorized_server_method(server, "list_prompts")
        for prompt_info in getattr(result, "prompts", []):
            prompt = MCPPrompt(
                server_name=server_name,
                prompt_name=prompt_info.name,
                description=getattr(prompt_info, "description", "") or "",
                arguments=[
                    {
                        "name": arg.name,
                        "required": getattr(arg, "required", False),
                    }
                    for arg in (getattr(prompt_info, "arguments", None) or [])
                ],
            )
            registry.register(prompt)
    except Exception:
        pass  # Server may not support prompts


async def discover_mcp_resources(
    servers: List[MCPServer],
) -> List[tuple[str, str]]:
    """Discover resources from connected MCP servers.

    Returns a list of ``(display_uri, description)`` tuples suitable for
    the ``AtMentionCompleter.update_mcp_resources()`` method.  The
    *display_uri* uses the ``server:protocol://path`` format so users can
    reference them as ``@server:protocol://path`` in prompts.
    """
    results: List[tuple[str, str]] = []

    for server in servers:
        server_name = getattr(server, "name", None)
        if not server_name:
            continue
        try:
            from .runtime_authorization import call_authorized_server_method

            response = await call_authorized_server_method(server, "list_resources")
            for resource in getattr(response, "resources", []):
                uri = str(resource.uri)
                description = getattr(resource, "description", "") or getattr(resource, "name", "")
                display_uri = f"{server_name}:{uri}"
                results.append((display_uri, description))
        except Exception as exc:
            _logger.debug(
                "Resource discovery skipped for server '%s': %s",
                server_name,
                exc,
            )

    return results


__all__ = [
    "load_mcp_servers",
    "close_mcp_servers",
    "drain_orphaned_mcp_owners",
    "detach_mcp_server_owner",
    "discover_mcp_resources",
    "get_reconnection_managers",
    "reconnect_unhealthy_servers",
    "MCPPrompt",
    "MCPPromptRegistry",
    "MCPServerConfig",
    "MCPServerSet",
    "MCPServerType",
    "MCPServerManager",
    "MCPServerFactory",
    "LiveMCPServer",
    "execute_prompt",
    "get_prompt_registry",
    "normalize_mcp_name",
]
