"""MCP server reconnection with exponential backoff."""

from __future__ import annotations

import asyncio
import inspect
import logging
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .iterators import closing_async_iterator

logger = logging.getLogger(__name__)

# Upstream defaults
INITIAL_DELAY = 1.0  # seconds
MAX_DELAY = 30.0  # seconds
MAX_ATTEMPTS = 5
_CLOSED_SERVER_MARKER = "_koder_live_mcp_closed"
_MAX_FALLBACK_CLOSED_SERVERS = 256
_MAX_RETIREMENT_DRAIN_ATTEMPTS = 3


@dataclass
class _RetirementRecord:
    server: Any
    wait_until_idle: Callable[[], Awaitable[None]] | None = None
    retire_requested: bool = False


class RetirementOwner:
    """Durably own concrete servers until their cleanup succeeds.

    Ownership changes are synchronous so a candidate is retained before any
    cancellable transition-lock wait. Failed or cancelled cleanup attempts keep
    their strong reference and can be retried by the next reconnect or shutdown.
    Successful cleanup is remembered without retaining the concrete server.
    """

    def __init__(self) -> None:
        self._records: dict[int, _RetirementRecord] = {}
        self._tasks: dict[int, asyncio.Task[None]] = {}
        self._closed_server_refs: dict[int, weakref.ReferenceType[Any]] = {}
        self._fallback_closed_servers: deque[Any] = deque()

    @property
    def pending_count(self) -> int:
        return sum(record.retire_requested for record in self._records.values())

    @property
    def held_count(self) -> int:
        return len(self._records)

    def _is_closed(self, server: Any) -> bool:
        try:
            if getattr(server, _CLOSED_SERVER_MARKER, False):
                return True
        except (AttributeError, TypeError):
            pass

        key = id(server)
        server_ref = self._closed_server_refs.get(key)
        if server_ref is not None:
            referenced = server_ref()
            if referenced is server:
                return True
            self._closed_server_refs.pop(key, None)
        return any(closed_server is server for closed_server in self._fallback_closed_servers)

    def _mark_closed(self, server: Any) -> None:
        try:
            setattr(server, _CLOSED_SERVER_MARKER, True)
            return
        except (AttributeError, TypeError):
            pass

        key = id(server)
        owner_ref = weakref.ref(self)

        def _discard_closed_ref(server_ref: weakref.ReferenceType[Any]) -> None:
            owner = owner_ref()
            if owner is not None and owner._closed_server_refs.get(key) is server_ref:
                owner._closed_server_refs.pop(key, None)

        try:
            self._closed_server_refs[key] = weakref.ref(server, _discard_closed_ref)
            return
        except TypeError:
            # Objects that reject attributes and weak references need bounded
            # strong identity evidence. A raw id is insufficient because the
            # allocator may reuse it for a fresh transport after collection.
            if any(closed_server is server for closed_server in self._fallback_closed_servers):
                return
            if len(self._fallback_closed_servers) >= _MAX_FALLBACK_CLOSED_SERVERS:
                self._fallback_closed_servers.popleft()
            self._fallback_closed_servers.append(server)

    def transfer_retirements_to(self, target: RetirementOwner) -> int:
        """Move settled retirement records to another durable owner.

        In-flight tasks stay with their source owner until a later drain. Every
        moved record remains strongly owned and keeps its exactly-once state.
        """
        if target is self:
            return 0

        transferred = 0
        for key, record in list(self._records.items()):
            if key in self._tasks:
                continue
            if target._is_closed(record.server):
                self._records.pop(key, None)
                transferred += 1
                continue

            target_record = target._records.get(key)
            if target_record is None or target_record.server is not record.server:
                target._records[key] = record
            else:
                if target_record.wait_until_idle is None:
                    target_record.wait_until_idle = record.wait_until_idle
                target_record.retire_requested = (
                    target_record.retire_requested or record.retire_requested
                )
            self._records.pop(key, None)
            transferred += 1
        return transferred

    def hold(
        self,
        server: Any,
        *,
        wait_until_idle: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Synchronously retain *server* before the caller can be cancelled."""
        if self._is_closed(server):
            return
        key = id(server)
        record = self._records.get(key)
        if record is None or record.server is not server:
            self._records[key] = _RetirementRecord(server, wait_until_idle)
            return
        if wait_until_idle is not None:
            record.wait_until_idle = wait_until_idle

    def adopt(self, server: Any) -> None:
        """Transfer a held candidate into a live current-server slot."""
        key = id(server)
        record = self._records.get(key)
        if record is None or record.server is not server:
            return
        if record.retire_requested or key in self._tasks:
            raise RuntimeError("Cannot adopt an MCP server after retirement started")
        self._records.pop(key, None)

    def owns_or_closed(self, server: Any) -> bool:
        record = self._records.get(id(server))
        return (record is not None and record.server is server) or self._is_closed(server)

    async def _retirement_attempt(self, record: _RetirementRecord) -> None:
        server = record.server
        key = id(server)
        task = asyncio.current_task()
        try:
            if record.wait_until_idle is not None:
                await record.wait_until_idle()
                # Once the transport is idle, do not retain a closure that may
                # keep its former LiveMCPServer handle reachable across retries.
                record.wait_until_idle = None
            cleanup = getattr(server, "cleanup", None)
            if cleanup is not None:
                await cleanup()
        except BaseException:
            if self._tasks.get(key) is task:
                self._tasks.pop(key, None)
            raise
        else:
            if self._tasks.get(key) is task:
                self._tasks.pop(key, None)
                current = self._records.get(key)
                if current is record:
                    self._records.pop(key, None)
                    self._mark_closed(server)

    def retire(
        self,
        server: Any,
        *,
        wait_until_idle: Callable[[], Awaitable[None]] | None = None,
    ) -> asyncio.Task[None] | None:
        """Retain *server* and start one cleanup attempt."""
        if self._is_closed(server):
            return None
        self.hold(server, wait_until_idle=wait_until_idle)
        key = id(server)
        record = self._records[key]
        record.retire_requested = True
        task = self._tasks.get(key)
        if task is None:
            task = asyncio.create_task(self._retirement_attempt(record))
            self._tasks[key] = task
        return task

    def _ensure_pending_tasks(self) -> list[asyncio.Task[None]]:
        tasks: list[asyncio.Task[None]] = []
        for record in list(self._records.values()):
            if not record.retire_requested:
                continue
            task = self.retire(record.server, wait_until_idle=record.wait_until_idle)
            if task is not None:
                tasks.append(task)
        return tasks

    async def drain(self, *, max_attempts: int = _MAX_RETIREMENT_DRAIN_ATTEMPTS) -> None:
        """Retry pending cleanup a bounded number of times.

        Every round attempts all retained retirements. Resources that still fail
        remain owned for a later reconnect or shutdown; successfully cleaned
        resources are removed and never cleaned a second time.
        """
        last_failure: BaseException | None = None
        for _ in range(max(1, max_attempts)):
            tasks = self._ensure_pending_tasks()
            if not tasks:
                return
            results = await asyncio.gather(
                *(asyncio.shield(task) for task in tasks),
                return_exceptions=True,
            )
            cancellation = next(
                (result for result in results if isinstance(result, asyncio.CancelledError)),
                None,
            )
            if cancellation is not None:
                # ``return_exceptions=True`` must never demote cancellation behind
                # an earlier ordinary cleanup failure in the same drain round.
                # The records remain owned and retryable because failed attempts
                # remove only their task entry, not their retirement record.
                raise cancellation
            failures = [result for result in results if isinstance(result, BaseException)]
            if not failures:
                if self.pending_count == 0:
                    return
                continue
            last_failure = failures[0]
            if self.pending_count == 0:
                return

        if last_failure is not None:
            raise last_failure
        if self.pending_count:
            raise RuntimeError("MCP server cleanup did not complete")


_ORPHANED_RETIREMENT_OWNER = RetirementOwner()
_ORPHANED_RETIREMENT_SOURCES: set[RetirementOwner] = set()


def retain_orphaned_retirements(owner: RetirementOwner) -> None:
    """Transfer failed cleanup into process-durable orphan ownership."""
    if owner is _ORPHANED_RETIREMENT_OWNER:
        return
    owner.transfer_retirements_to(_ORPHANED_RETIREMENT_OWNER)
    if owner.held_count:
        # Only an in-flight retirement task can prevent immediate transfer.
        # Keep that source owner until a later drain settles and consolidates it.
        _ORPHANED_RETIREMENT_SOURCES.add(owner)


def get_orphaned_retirement_counts() -> tuple[int, int]:
    """Return ``(held transports, durable owner roots)`` for diagnostics."""
    held_count = _ORPHANED_RETIREMENT_OWNER.held_count + sum(
        owner.held_count for owner in _ORPHANED_RETIREMENT_SOURCES
    )
    owner_count = len(_ORPHANED_RETIREMENT_SOURCES)
    if _ORPHANED_RETIREMENT_OWNER.held_count:
        owner_count += 1
    return held_count, owner_count


async def drain_orphaned_retirements(*, max_attempts: int = _MAX_RETIREMENT_DRAIN_ATTEMPTS) -> None:
    """Retry resources whose original manager or handle may no longer exist."""
    first_failure: BaseException | None = None
    for owner in list(_ORPHANED_RETIREMENT_SOURCES):
        owner.transfer_retirements_to(_ORPHANED_RETIREMENT_OWNER)
        try:
            if owner.held_count:
                await owner.drain(max_attempts=max_attempts)
        except asyncio.CancelledError:
            owner.transfer_retirements_to(_ORPHANED_RETIREMENT_OWNER)
            if owner.held_count:
                _ORPHANED_RETIREMENT_SOURCES.add(owner)
            else:
                _ORPHANED_RETIREMENT_SOURCES.discard(owner)
            raise
        except BaseException as exc:
            if first_failure is None:
                first_failure = exc
        owner.transfer_retirements_to(_ORPHANED_RETIREMENT_OWNER)
        if owner.held_count == 0:
            _ORPHANED_RETIREMENT_SOURCES.discard(owner)

    try:
        await _ORPHANED_RETIREMENT_OWNER.drain(max_attempts=max_attempts)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        if first_failure is None:
            first_failure = exc
    if first_failure is not None:
        raise first_failure


@dataclass
class ReconnectionConfig:
    """Configuration for reconnection behavior."""

    initial_delay: float = INITIAL_DELAY
    max_delay: float = MAX_DELAY
    max_attempts: int = MAX_ATTEMPTS


class LiveMCPServer:
    """Stable MCP server identity whose live connection can be replaced.

    FunctionTools and runtime services retain this handle, never a concrete
    transport instance. Calls lease the current server so a reconnect can make
    the replacement visible immediately while deferring cleanup of the retired
    server until its in-flight calls finish.
    """

    def __init__(
        self,
        name: str,
        server: Any,
        *,
        retirement_owner: RetirementOwner | None = None,
    ):
        self.name = name
        self._retirement_owner = retirement_owner or RetirementOwner()
        self._retirement_owner.adopt(server)
        self._current = server
        self._state_lock = asyncio.Lock()
        self._transition_lock = asyncio.Lock()
        self._active_calls: dict[int, int] = {}
        self._idle_events: dict[int, asyncio.Event] = {}
        self._leased_session_proxy: Any | None = None
        # Keep the diagnostic attributes used by regression tests while the
        # ownership implementation lives in the shared retirement owner.
        self._retirement_tasks = self._retirement_owner._tasks
        self._closed_server_refs = self._retirement_owner._closed_server_refs
        self._fallback_closed_servers = self._retirement_owner._fallback_closed_servers

    @property
    def current(self) -> Any:
        """Return this stable public handle while a concrete transport is live."""
        return self if self._current is not None else None

    def _koder_current_transport(self) -> Any:
        """Return the concrete transport for reconnection internals and tests only."""
        return self._current

    @property
    def _retiring_servers(self) -> dict[int, Any]:
        """Expose retained concrete servers for diagnostics and regressions."""
        return {key: record.server for key, record in self._retirement_owner._records.items()}

    @property
    def session(self) -> Any:
        """Expose a stable leased proxy, with authorization for project servers."""
        from .runtime_authorization import get_project_authorization_validator

        if self._koder_raw_session() is None:
            return None
        validator = get_project_authorization_validator(self)
        if validator is not None:
            return validator.authorized_session()
        if self._leased_session_proxy is None:
            from .runtime_authorization import LeasedMCPClientSession

            self._leased_session_proxy = LeasedMCPClientSession(self)
        return self._leased_session_proxy

    def _koder_raw_session(self) -> Any:
        """Return the current concrete session for health and validator refresh."""
        server = self._current
        return None if server is None else getattr(server, "session", None)

    @property
    def cached_tools(self) -> Any:
        """Return a detached view of the current SDK tool cache."""
        server = self._current
        cached = None if server is None else getattr(server, "cached_tools", None)
        return None if cached is None else list(cached)

    @property
    def use_structured_content(self) -> bool:
        """Preserve the SDK's structured-content setting without exposing transport state."""
        server = self._current
        return bool(server is not None and getattr(server, "use_structured_content", False))

    @property
    def server_initialize_result(self) -> Any:
        """Expose only the protocol initialization result used for capability checks."""
        server = self._current
        return None if server is None else getattr(server, "server_initialize_result", None)

    @property
    def tool_meta_resolver(self) -> Any:
        """Return a stable guarded wrapper for an optional SDK metadata resolver."""
        server = self._current
        resolver = None if server is None else getattr(server, "tool_meta_resolver", None)
        return None if resolver is None else self._resolve_tool_meta

    async def _resolve_tool_meta(self, *args: Any, **kwargs: Any) -> Any:
        return await self._invoke_public_server_path(("tool_meta_resolver",), *args, **kwargs)

    @property
    def custom_data_extractor(self) -> Any:
        """Return a stable guarded wrapper for an optional SDK custom-data extractor."""
        server = self._current
        extractor = None if server is None else getattr(server, "custom_data_extractor", None)
        return None if extractor is None else self._extract_custom_data

    async def _extract_custom_data(self, *args: Any, **kwargs: Any) -> Any:
        return await self._invoke_public_server_path(("custom_data_extractor",), *args, **kwargs)

    async def _acquire_current(self) -> Any:
        async with self._state_lock:
            server = self._current
            if server is None:
                raise RuntimeError(f"MCP server '{self.name}' is closed")
            key = id(server)
            self._active_calls[key] = self._active_calls.get(key, 0) + 1
            return server

    async def _release(self, server: Any) -> None:
        async with self._state_lock:
            key = id(server)
            remaining = self._active_calls.get(key, 0) - 1
            if remaining > 0:
                self._active_calls[key] = remaining
                return
            self._active_calls.pop(key, None)
            event = self._idle_events.pop(key, None)
            if event is not None:
                event.set()

    async def _invoke(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        server = await self._acquire_current()
        try:
            method = getattr(server, method_name)
            return await method(*args, **kwargs)
        finally:
            await self._release(server)

    @staticmethod
    def _resolve_server_callable(
        server: Any, attribute_path: tuple[str, ...]
    ) -> Callable[..., Any]:
        if not attribute_path:
            raise AttributeError("MCP server attribute path cannot be empty")
        attribute = server
        for name in attribute_path:
            attribute = getattr(attribute, name)
        if not callable(attribute):
            raise TypeError(f"MCP server attribute '{'.'.join(attribute_path)}' is not callable")
        return attribute

    async def _invoke_server_path(
        self,
        attribute_path: tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        server = await self._acquire_current()
        try:
            method = self._resolve_server_callable(server, attribute_path)
            result = method(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            if hasattr(result, "__aiter__"):
                close = getattr(result, "aclose", None)
                if close is not None:
                    close_result = close()
                    if inspect.isawaitable(close_result):
                        await close_result
                raise TypeError(
                    f"MCP server method '{'.'.join(attribute_path)}' returns an async iterator; "
                    "consume it with 'async for'"
                )
            return result
        finally:
            await self._release(server)

    async def _iterate_server_path(
        self,
        attribute_path: tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        server = await self._acquire_current()
        try:
            method = self._resolve_server_callable(server, attribute_path)
            result = method(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            if not hasattr(result, "__aiter__"):
                raise TypeError(
                    f"MCP server method '{'.'.join(attribute_path)}' does not return an async "
                    "iterator"
                )
            async with closing_async_iterator(result) as iterator:
                async for item in iterator:
                    yield item
        finally:
            await self._release(server)

    async def _invoke_public_server_path(
        self,
        attribute_path: tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        from .runtime_authorization import get_project_authorization_validator

        validator = get_project_authorization_validator(self)
        if validator is not None:
            return await validator.run_authorized(
                self._invoke_server_path,
                attribute_path,
                *args,
                **kwargs,
            )
        return await self._invoke_server_path(attribute_path, *args, **kwargs)

    async def _iterate_public_server_path(
        self,
        attribute_path: tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        from .runtime_authorization import get_project_authorization_validator

        validator = get_project_authorization_validator(self)
        if validator is not None:
            source = validator._run_authorized_iterator(
                self._iterate_server_path,
                attribute_path,
                *args,
                **kwargs,
            )
        else:
            source = self._iterate_server_path(attribute_path, *args, **kwargs)
        async with closing_async_iterator(source) as iterator:
            async for item in iterator:
                yield item

    @staticmethod
    def _resolve_session_callable(
        server: Any, attribute_path: tuple[str, ...]
    ) -> Callable[..., Any]:
        if not attribute_path:
            raise AttributeError("MCP session attribute path cannot be empty")
        attribute = getattr(server, "session", None)
        if attribute is None:
            raise RuntimeError("MCP server has no active client session")
        for name in attribute_path:
            attribute = getattr(attribute, name)
        if not callable(attribute):
            raise TypeError(f"MCP session attribute '{'.'.join(attribute_path)}' is not callable")
        return attribute

    async def _invoke_session(
        self,
        attribute_path: str | tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Invoke a session capability while pinning its concrete transport."""
        path = (attribute_path,) if isinstance(attribute_path, str) else attribute_path
        server = await self._acquire_current()
        try:
            method = self._resolve_session_callable(server, path)
            result = method(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            if hasattr(result, "__aiter__"):
                close = getattr(result, "aclose", None)
                if close is not None:
                    close_result = close()
                    if inspect.isawaitable(close_result):
                        await close_result
                raise TypeError(
                    f"MCP session method '{'.'.join(path)}' returns an async iterator; "
                    "consume it with 'async for'"
                )
            return result
        finally:
            await self._release(server)

    async def _iterate_session(
        self,
        attribute_path: str | tuple[str, ...],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Iterate a session capability while pinning its concrete transport."""
        path = (attribute_path,) if isinstance(attribute_path, str) else attribute_path
        server = await self._acquire_current()
        try:
            method = self._resolve_session_callable(server, path)
            result = method(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            if not hasattr(result, "__aiter__"):
                raise TypeError(
                    f"MCP session method '{'.'.join(path)}' does not return an async iterator"
                )
            async with closing_async_iterator(result) as iterator:
                async for item in iterator:
                    yield item
        finally:
            await self._release(server)

    async def connect(self, *args: Any, **kwargs: Any) -> Any:
        return await self._invoke("connect", *args, **kwargs)

    async def list_tools(self, *args: Any, **kwargs: Any) -> Any:
        return await self._invoke("list_tools", *args, **kwargs)

    async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        return await self._invoke("call_tool", *args, **kwargs)

    async def list_prompts(self, *args: Any, **kwargs: Any) -> Any:
        return await self._invoke("list_prompts", *args, **kwargs)

    async def get_prompt(self, *args: Any, **kwargs: Any) -> Any:
        return await self._invoke("get_prompt", *args, **kwargs)

    async def list_resources(self, *args: Any, **kwargs: Any) -> Any:
        return await self._invoke("list_resources", *args, **kwargs)

    async def list_resource_templates(self, *args: Any, **kwargs: Any) -> Any:
        return await self._invoke("list_resource_templates", *args, **kwargs)

    async def read_resource(self, *args: Any, **kwargs: Any) -> Any:
        return await self._invoke("read_resource", *args, **kwargs)

    async def _wait_until_idle(self, server: Any) -> None:
        async with self._state_lock:
            if self._active_calls.get(id(server), 0) == 0:
                return
            event = self._idle_events.setdefault(id(server), asyncio.Event())
        await event.wait()

    async def _owns_or_closed(self, server: Any) -> bool:
        """Whether this handle adopted *server* or already completed its cleanup."""
        async with self._state_lock:
            return self._current is server or self._retirement_owner.owns_or_closed(server)

    async def replace(self, server: Any) -> None:
        """Atomically publish *server*, then retire and close its predecessor."""
        self._retirement_owner.hold(server)
        adopted = False
        retirement: asyncio.Task[None] | None = None
        try:
            async with self._transition_lock:
                async with self._state_lock:
                    previous = self._current
                    if previous is server:
                        self._retirement_owner.adopt(server)
                        return
                    if previous is None:
                        raise RuntimeError(f"MCP server '{self.name}' is closed")
                    self._current = server
                    self._retirement_owner.adopt(server)
                    adopted = True
                    retirement = self._retirement_owner.retire(
                        previous,
                        wait_until_idle=lambda: self._wait_until_idle(previous),
                    )
        except BaseException:
            if not adopted:
                self._retirement_owner.retire(server)
            raise

        if retirement is not None:
            try:
                await asyncio.shield(retirement)
            except Exception:
                logger.warning(
                    "Failed to clean up replaced MCP server '%s'",
                    self.name,
                    exc_info=True,
                )

    async def cleanup(self) -> None:
        """Retire every concrete server, retaining failed attempts for retry."""
        async with self._transition_lock:
            async with self._state_lock:
                current = self._current
                if current is not None:
                    self._current = None
                    self._retirement_owner.retire(
                        current,
                        wait_until_idle=lambda: self._wait_until_idle(current),
                    )

        await self._retirement_owner.drain()

    def _get_failure_error_function(self, failure_error_function: Any) -> Any:
        server = self._current
        if server is None:
            return failure_error_function
        resolver = getattr(server, "_get_failure_error_function", None)
        return failure_error_function if resolver is None else resolver(failure_error_function)

    def _get_needs_approval_for_tool(self, tool: Any, agent: Any) -> Any:
        server = self._current
        if server is None:
            return False
        resolver = getattr(server, "_get_needs_approval_for_tool", None)
        return False if resolver is None else resolver(tool, agent)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        server = self._current
        if server is None:
            raise AttributeError(name)
        try:
            inspect.getattr_static(server, name)
        except AttributeError as exc:
            raise AttributeError(name) from exc

        from .runtime_authorization import _MCPClientSessionPath

        return _MCPClientSessionPath(
            self._invoke_public_server_path,
            self._iterate_public_server_path,
            (name,),
        )


class ReconnectionManager:
    """Manages reconnection attempts with exponential backoff.

    Usage:
        mgr = ReconnectionManager(ReconnectionConfig())
        success = await mgr.reconnect_with_backoff(connect_function)
    """

    def __init__(
        self,
        config: ReconnectionConfig | None = None,
        *,
        retirement_owner: RetirementOwner | None = None,
    ):
        self.config = config or ReconnectionConfig()
        self.attempt_count = 0
        # Retained after a successful initial connect so a later health check
        # can rebuild the connection without re-deriving trust/channel wiring.
        self.config_ref: Any = None
        self.server: Any = None
        self._connect_fn: Callable[[], Awaitable] | None = None
        self._reconnect_lock = asyncio.Lock()
        self._retirement_owner = retirement_owner or RetirementOwner()
        self._authorization_disabled = False

    def bind(
        self,
        *,
        config: Any = None,
        server: Any = None,
        connect_fn: Callable[[], Awaitable] | None = None,
    ) -> None:
        """Retain the server, its config, and the factory closure.

        This lets callers hold onto the manager and later trigger a reconnect
        for a dropped server. Without this the manager was discarded and no
        runtime reconnection was possible.
        """
        if config is not None:
            self.config_ref = config
        if server is not None:
            self.server = server
            if isinstance(server, LiveMCPServer):
                if (
                    self._retirement_owner.held_count
                    and self._retirement_owner is not server._retirement_owner
                ):
                    raise RuntimeError(
                        "Cannot rebind an MCP manager with retained cleanup ownership"
                    )
                self._retirement_owner = server._retirement_owner
        if connect_fn is not None:
            self._connect_fn = connect_fn

    def retire_unpublished(self) -> None:
        """Synchronously transfer an unpublished server to durable cleanup.

        Publication races are detected after a factory await. This method makes
        the concrete transport retirement-owned before the caller can encounter
        another cancellation point, so dropping the unpublished manager cannot
        drop the final strong transport reference.
        """
        server = self.server
        try:
            if isinstance(server, LiveMCPServer):
                current = server._koder_current_transport()
                if current is not None:
                    self._retirement_owner.retire(
                        current,
                        wait_until_idle=lambda: server._wait_until_idle(current),
                    )
            elif server is not None:
                self._retirement_owner.retire(server)
        finally:
            self._clear_binding()
            retain_orphaned_retirements(self._retirement_owner)

    def _clear_binding(self) -> None:
        self.server = None
        self.config_ref = None
        self._connect_fn = None

    async def _disable_current_server(self) -> None:
        """Disable the stable handle without discarding manager ownership."""
        server = self.server
        self._authorization_disabled = True
        if server is None:
            return
        from .runtime_authorization import get_project_authorization_validator

        validator = get_project_authorization_validator(server)
        if validator is not None:
            await validator.disable()
            return
        if isinstance(server, LiveMCPServer):
            await server.cleanup()
            return
        self._retirement_owner.retire(server)
        await self._retirement_owner.drain()

    async def validate_runtime_authorization(self) -> bool:
        """Revalidate project provenance before any healthy-server early return."""
        config = self.config_ref
        if config is None:
            return True
        if self._authorization_disabled:
            return False
        from .server_config import MCPServerScope

        if config.scope != MCPServerScope.PROJECT:
            return True
        from .runtime_authorization import get_project_authorization_validator

        validator = (
            get_project_authorization_validator(self.server) if self.server is not None else None
        )
        if validator is not None:
            authorized = await validator.validate()
        else:
            try:
                from .server_manager import MCPServerManager

                authorized = MCPServerManager().revalidate_project_config(config)
            except Exception:
                authorized = False
        if not authorized:
            if validator is None:
                await self._disable_current_server()
            else:
                self._authorization_disabled = True
            logger.warning(
                "Project MCP server '%s' disabled because approval or execution identity changed",
                getattr(config, "name", "unknown"),
            )
            return False
        self._authorization_disabled = False
        return True

    def _server_is_healthy(self) -> bool:
        """Best-effort liveness check for the retained server.

        A connected SDK ``MCPServer`` exposes a non-None ``session``; we treat a
        missing/None session as "needs reconnect". Servers that do not expose a
        ``session`` attribute are assumed healthy (nothing to probe).
        """
        server = self.server
        if server is None:
            return False
        if isinstance(server, LiveMCPServer):
            return server._koder_raw_session() is not None
        if not hasattr(server, "session"):
            return True
        return getattr(server, "session", None) is not None

    async def reconnect_if_needed(self) -> bool:
        """Reconnect the retained server if it looks unhealthy.

        Returns True if the server is healthy (already, or after a successful
        reconnect), False if it is unhealthy and reconnection failed or no
        factory closure was bound.
        """
        async with self._reconnect_lock:
            if not await self.validate_runtime_authorization():
                return False
            if self._retirement_owner.pending_count:
                try:
                    await self._retirement_owner.drain(max_attempts=1)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning(
                        "Failed to retry retained MCP cleanup for '%s'",
                        getattr(self.server, "name", self.config_ref),
                        exc_info=True,
                    )
            if self._server_is_healthy():
                return True
            if self._connect_fn is None:
                logger.debug("reconnect_if_needed: no connect_fn bound; cannot reconnect")
                return False

            connect_fn = self._connect_fn

            async def _do_connect() -> None:
                if not await self.validate_runtime_authorization():
                    from .runtime_authorization import MCPAuthorizationError

                    raise MCPAuthorizationError(
                        f"Project MCP server '{getattr(self.config_ref, 'name', 'unknown')}' "
                        "is no longer authorized"
                    )
                if self._retirement_owner.pending_count:
                    await self._retirement_owner.drain(max_attempts=1)

                async def _connect_and_publish() -> None:
                    candidate = await connect_fn()
                    self._retirement_owner.hold(candidate)
                    server = self.server
                    if isinstance(server, LiveMCPServer):
                        try:
                            await server.replace(candidate)
                            from .runtime_authorization import (
                                get_project_authorization_validator,
                            )

                            validator = get_project_authorization_validator(server)
                            if validator is not None:
                                validator.authorized_session()
                        except BaseException:
                            if not await server._owns_or_closed(candidate):
                                self._retirement_owner.retire(candidate)
                            raise
                    else:
                        self.server = candidate
                        self._retirement_owner.adopt(candidate)

                from .runtime_authorization import get_project_authorization_validator

                stable_validator = (
                    get_project_authorization_validator(self.server)
                    if self.server is not None
                    else None
                )
                if stable_validator is not None:
                    await stable_validator.run_authorized(_connect_and_publish)
                else:
                    await _connect_and_publish()

            return await self.reconnect_with_backoff(_do_connect)

    async def cleanup(self) -> None:
        """Close the live server and drain every retained candidate."""
        async with self._reconnect_lock:
            server = self.server
            clear_binding = True
            try:
                if isinstance(server, LiveMCPServer):
                    await server.cleanup()
                elif server is not None:
                    self._retirement_owner.retire(server)
                await self._retirement_owner.drain()
            except asyncio.CancelledError:
                # If cancellation won before LiveMCPServer detached its current
                # transport, keep the binding so the registry remains its durable
                # owner. Otherwise the retirement owner already owns everything.
                clear_binding = (
                    not isinstance(server, LiveMCPServer)
                    or server._koder_current_transport() is None
                )
                raise
            finally:
                if clear_binding:
                    self._clear_binding()

    def get_next_delay(self) -> float:
        """Calculate next delay with exponential backoff."""
        delay = self.config.initial_delay * (2**self.attempt_count)
        self.attempt_count += 1
        return min(delay, self.config.max_delay)

    def should_retry(self) -> bool:
        """Check if more attempts are available."""
        return self.attempt_count < self.config.max_attempts

    def reset(self) -> None:
        """Reset attempt counter after successful connection."""
        self.attempt_count = 0

    async def reconnect_with_backoff(
        self,
        connect_fn: Callable[[], Awaitable],
    ) -> bool:
        """Attempt reconnection with exponential backoff.

        Args:
            connect_fn: Async function that attempts to establish connection.
                       Should raise on failure, return on success.

        Returns:
            True if reconnection succeeded, False if all attempts exhausted.
        """
        self.reset()

        while self.should_retry():
            delay = self.get_next_delay()
            try:
                await connect_fn()
                logger.info(
                    "MCP server reconnected successfully after %d attempts", self.attempt_count
                )
                return True
            except PermissionError as e:
                logger.warning("MCP reconnection blocked by current approval state: %s", e)
                return False
            except Exception as e:
                logger.warning(
                    "MCP reconnection attempt %d/%d failed: %s (retry in %.1fs)",
                    self.attempt_count,
                    self.config.max_attempts,
                    e,
                    delay,
                )
                if self.should_retry():
                    await asyncio.sleep(delay)

        logger.error(
            "MCP server reconnection failed after %d attempts",
            self.config.max_attempts,
        )
        return False
