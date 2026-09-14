"""Linearizable runtime authorization for project MCP servers."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import logging
from collections.abc import AsyncIterator, Coroutine, Iterator
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Iterable, TypeVar

from .iterators import close_async_iterator, closing_async_iterator
from .server_config import MCPServerConfig, MCPServerScope

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
_ActiveAdmission = tuple[Any, asyncio.Task[Any] | None]
_SessionPath = tuple[str, ...]
_SessionInvoke = Callable[..., Awaitable[Any]]
_SessionIterate = Callable[..., AsyncIterator[Any]]
_ACTIVE_ADMISSION: contextvars.ContextVar[_ActiveAdmission | None] = contextvars.ContextVar(
    "koder_mcp_active_authorization_admission", default=None
)


def _direct_instance_attr(instance: Any, name: str, default: Any = None) -> Any:
    """Read an instance attribute without triggering a forwarding ``__getattr__``."""
    try:
        namespace = object.__getattribute__(instance, "__dict__")
    except (AttributeError, TypeError):
        return default
    return namespace.get(name, default)


class MCPAuthorizationError(PermissionError):
    """Raised before contacting a project MCP server whose trust is stale."""


def _format_session_path(attribute_path: _SessionPath) -> str:
    return ".".join(attribute_path)


def _resolve_session_attribute(session: Any, attribute_path: _SessionPath) -> Any:
    if session is None:
        raise RuntimeError("MCP server has no active client session")
    if not attribute_path:
        raise AttributeError("MCP session attribute path cannot be empty")
    attribute = session
    for name in attribute_path:
        attribute = getattr(attribute, name)
    return attribute


async def _invoke_session_attribute(
    session: Any,
    attribute_path: _SessionPath,
    *args: Any,
    **kwargs: Any,
) -> Any:
    attribute = _resolve_session_attribute(session, attribute_path)
    if not callable(attribute):
        raise TypeError(
            f"MCP session attribute '{_format_session_path(attribute_path)}' is not callable"
        )
    result = attribute(*args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    if hasattr(result, "__aiter__"):
        close = getattr(result, "aclose", None)
        if close is not None:
            close_result = close()
            if inspect.isawaitable(close_result):
                await close_result
        raise TypeError(
            f"MCP session method '{_format_session_path(attribute_path)}' returns an async "
            "iterator; consume it with 'async for'"
        )
    return result


async def _iterate_session_attribute(
    session: Any,
    attribute_path: _SessionPath,
    *args: Any,
    **kwargs: Any,
) -> AsyncIterator[Any]:
    attribute = _resolve_session_attribute(session, attribute_path)
    if not callable(attribute):
        raise TypeError(
            f"MCP session attribute '{_format_session_path(attribute_path)}' is not callable"
        )
    result = attribute(*args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    if not hasattr(result, "__aiter__"):
        raise TypeError(
            f"MCP session method '{_format_session_path(attribute_path)}' does not return "
            "an async iterator"
        )
    async with closing_async_iterator(result) as iterator:
        async for item in iterator:
            yield item


class _MCPClientSessionInvocation(Coroutine[Any, Any, Any]):
    """A lazy session call usable as either a coroutine or async iterator."""

    __slots__ = (
        "__args",
        "__coroutine",
        "__invoke",
        "__iterate",
        "__iteration_started",
        "__kwargs",
        "__path",
    )

    def __init__(
        self,
        invoke: _SessionInvoke,
        iterate: _SessionIterate,
        attribute_path: _SessionPath,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self.__invoke = invoke
        self.__iterate = iterate
        self.__path = attribute_path
        self.__args = args
        self.__kwargs = kwargs
        self.__coroutine: Coroutine[Any, Any, Any] | None = None
        self.__iteration_started = False

    def __ensure_coroutine(self) -> Coroutine[Any, Any, Any]:
        if self.__iteration_started:
            raise RuntimeError("MCP session invocation is already being iterated")
        if self.__coroutine is None:
            self.__coroutine = self.__invoke(
                self.__path,
                *self.__args,
                **self.__kwargs,
            )
        return self.__coroutine

    def __await__(self):
        return self.__ensure_coroutine().__await__()

    def send(self, value: Any) -> Any:
        return self.__ensure_coroutine().send(value)

    def throw(self, *args: Any) -> Any:
        return self.__ensure_coroutine().throw(*args)

    def close(self) -> None:
        if self.__coroutine is not None:
            self.__coroutine.close()

    def __aiter__(self) -> AsyncIterator[Any]:
        if self.__coroutine is not None:
            raise RuntimeError("MCP session invocation is already being awaited")
        if self.__iteration_started:
            raise RuntimeError("MCP session invocation is already being iterated")
        self.__iteration_started = True
        return self.__iterate_call()

    async def __iterate_call(self) -> AsyncIterator[Any]:
        source = self.__iterate(
            self.__path,
            *self.__args,
            **self.__kwargs,
        )
        async with closing_async_iterator(source) as iterator:
            async for item in iterator:
                yield item


class _MCPClientSessionPath:
    """Lazy public attribute path that never returns a concrete session object."""

    __slots__ = ("__invoke", "__iterate", "__path")

    def __init__(
        self,
        invoke: _SessionInvoke,
        iterate: _SessionIterate,
        attribute_path: _SessionPath = (),
    ) -> None:
        self.__invoke = invoke
        self.__iterate = iterate
        self.__path = attribute_path

    def __getattr__(self, name: str) -> _MCPClientSessionPath:
        if name == "raw_session" or name.startswith("_"):
            raise AttributeError(name)
        return _MCPClientSessionPath(
            self.__invoke,
            self.__iterate,
            (*self.__path, name),
        )

    def __call__(self, *args: Any, **kwargs: Any) -> _MCPClientSessionInvocation:
        if not self.__path:
            raise TypeError("The MCP client session proxy is not directly callable")
        return _MCPClientSessionInvocation(
            self.__invoke,
            self.__iterate,
            self.__path,
            args,
            kwargs,
        )


class AuthorizedMCPClientSession(_MCPClientSessionPath):
    """Proxy every reachable client-session call through admission and leasing."""

    def __init__(self, validator: ProjectServerAuthorizationValidator):
        super().__init__(
            validator._invoke_authorized_session_path,
            validator._iterate_authorized_session_path,
        )


class LeasedMCPClientSession(_MCPClientSessionPath):
    """Proxy non-project session calls through the stable transport lease."""

    def __init__(self, server: Any):
        super().__init__(server._invoke_session, server._iterate_session)


class ProjectServerAuthorizationValidator:
    """Own one project server's validation, admission, revocation, and cleanup.

    The linearization point for a server-backed operation is the final
    revalidation in :meth:`_admit`. That check runs while both the in-process
    state lock and the same cross-process approval-file lock used by reset are
    held. A reset that acquires the approval lock first prevents admission; an
    operation admitted first is tracked as in flight and may finish. Revocation
    prevents later admissions and cleanup runs exactly once after admitted work
    drains. No server/helper callback runs while either lock is held.
    """

    def __init__(self, config: MCPServerConfig, server: Any | None = None):
        self.config = config
        self.server: Any | None = None
        self.disabled = False
        self._state_lock = asyncio.Lock()
        self._in_flight = 0
        self._cleanup_task: asyncio.Task[None] | None = None
        self._cleanup_complete = asyncio.Event()
        self._state_changed = asyncio.Event()
        self._original_cleanup: Callable[[], Any] | None = None
        self._raw_session: Any | None = None
        self._authorized_session_proxy = AuthorizedMCPClientSession(self)
        self._wrapped_server_methods: dict[str, Callable[..., Any]] = {}
        if server is not None:
            self.bind_server(server)

    def _koder_raw_session(self) -> Any | None:
        """Return the concrete SDK session for internal refresh and invocation only."""
        return self._raw_session

    @property
    def in_flight(self) -> int:
        """Return the number of admitted outer operations (primarily for diagnostics)."""
        return self._in_flight

    def _authorization_error(self) -> MCPAuthorizationError:
        return MCPAuthorizationError(
            f"Project MCP server '{self.config.name}' is unavailable because approval was "
            "reset or the reviewed execution identity changed due to source/executable drift"
        )

    def _is_active_operation(self) -> bool:
        active = _ACTIVE_ADMISSION.get()
        return active is not None and active == (self, asyncio.current_task())

    def _has_active_admission_context(self) -> bool:
        """Return whether this task descends from an admitted operation."""
        active = _ACTIVE_ADMISSION.get()
        return active is not None and active[0] is self

    @contextmanager
    def _operation_context(self) -> Iterator[None]:
        """Grant nested-call admission only while server code is executing."""
        token = _ACTIVE_ADMISSION.set((self, asyncio.current_task()))
        try:
            yield
        finally:
            _ACTIVE_ADMISSION.reset(token)

    async def _invoke_authorized_session_path(
        self,
        attribute_path: _SessionPath,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        server = self.server
        invoke_session = (
            getattr(server, "_invoke_session", None)
            if server is not None and callable(getattr(type(server), "_invoke_session", None))
            else None
        )
        if invoke_session is not None:
            return await self.run_authorized(
                invoke_session,
                attribute_path,
                *args,
                **kwargs,
            )

        async def _invoke_current_session() -> Any:
            return await _invoke_session_attribute(
                self._koder_raw_session(),
                attribute_path,
                *args,
                **kwargs,
            )

        return await self.run_authorized(_invoke_current_session)

    async def _iterate_authorized_session_path(
        self,
        attribute_path: _SessionPath,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        server = self.server
        iterate_session = (
            getattr(server, "_iterate_session", None)
            if server is not None and callable(getattr(type(server), "_iterate_session", None))
            else None
        )
        if iterate_session is not None:
            source = self._run_authorized_iterator(
                iterate_session,
                attribute_path,
                *args,
                **kwargs,
            )
        else:

            def _iterate_current_session() -> AsyncIterator[Any]:
                return _iterate_session_attribute(
                    self._koder_raw_session(),
                    attribute_path,
                    *args,
                    **kwargs,
                )

            source = self._run_authorized_iterator(_iterate_current_session)
        async with closing_async_iterator(source) as iterator:
            async for item in iterator:
                yield item

    def _revalidate(self, *, approval_lock_held: bool = False) -> bool:
        if self.config.scope != MCPServerScope.PROJECT:
            return True
        try:
            from .server_manager import MCPServerManager

            return MCPServerManager().revalidate_project_config(
                self.config,
                approval_lock_held=approval_lock_held,
            )
        except Exception:
            logger.debug(
                "Project MCP authorization revalidation failed for '%s'",
                self.config.name,
                exc_info=True,
            )
            return False

    async def _after_preflight_validation(self) -> None:
        """Deterministic test seam between advisory validation and admission.

        Production does not pause here. Race tests override this method to
        prove that reset or identity drift after a successful preflight still
        wins before any server/helper callback is admitted.
        """

    def _notify_state_locked(self) -> None:
        event = self._state_changed
        self._state_changed = asyncio.Event()
        event.set()

    def _start_cleanup_locked(self) -> asyncio.Task[None] | None:
        if self._cleanup_complete.is_set() or self._in_flight or self._cleanup_task is not None:
            return self._cleanup_task
        cleanup = self._original_cleanup
        if cleanup is None:
            self._raw_session = None
            self._cleanup_complete.set()
            self._notify_state_locked()
            return None
        start_cleanup = asyncio.Event()
        self._cleanup_task = asyncio.create_task(self._run_cleanup_attempt(cleanup, start_cleanup))
        # Even under an eager task factory the task can only wait on this gate;
        # the real cleanup callback starts on the next loop turn, after the
        # validator state lock has been released.
        asyncio.get_running_loop().call_soon(start_cleanup.set)
        return self._cleanup_task

    async def _run_cleanup_attempt(
        self,
        cleanup: Callable[[], Any],
        start_cleanup: asyncio.Event,
    ) -> None:
        await start_cleanup.wait()
        task = asyncio.current_task()
        try:
            result = cleanup()
            if inspect.isawaitable(result):
                await result
        except BaseException:
            async with self._state_lock:
                if self._cleanup_task is task:
                    self._cleanup_task = None
                self._notify_state_locked()
            raise
        else:
            async with self._state_lock:
                if self._cleanup_task is task:
                    self._cleanup_task = None
                self._raw_session = None
                self._cleanup_complete.set()
                self._notify_state_locked()

    async def _wait_for_cleanup(self) -> None:
        while True:
            async with self._state_lock:
                if self._cleanup_complete.is_set():
                    return
                task = self._start_cleanup_locked()
                if self._cleanup_complete.is_set():
                    return
                state_changed = self._state_changed
            if task is not None:
                await asyncio.shield(task)
                continue
            await state_changed.wait()

    async def _revoke(
        self,
        *,
        wait_for_cleanup: bool = True,
        wait_for_in_flight: bool = True,
        propagate_cleanup_failure: bool = True,
    ) -> None:
        async with self._state_lock:
            self.disabled = True
            self._start_cleanup_locked()
            self._notify_state_locked()
            # A child task inherits context but not bypass authority. It must
            # still be denied after revocation, while avoiding a cleanup wait
            # that would deadlock if its admitted parent is awaiting the child.
            active_operation = self._has_active_admission_context()
            if self._in_flight and not wait_for_in_flight:
                wait_for_cleanup = False

        if not wait_for_cleanup or active_operation:
            return
        try:
            await self._wait_for_cleanup()
        except asyncio.CancelledError:
            raise
        except Exception:
            if propagate_cleanup_failure:
                raise
            logger.debug(
                "Cleanup failed while disabling project MCP server '%s'",
                self.config.name,
                exc_info=True,
            )

    async def _finish_operation(self) -> None:
        async with self._state_lock:
            if self._in_flight <= 0:  # pragma: no cover - defensive invariant
                raise RuntimeError("MCP authorization operation accounting underflow")
            self._in_flight -= 1
            task = self._start_cleanup_locked() if self.disabled else None
            self._notify_state_locked()
        if task is not None:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                raise
            except Exception:
                # The stable handle and retirement owner remain reachable. A
                # later explicit cleanup or runtime shutdown retries the same
                # concrete transport rather than marking cleanup complete.
                logger.debug(
                    "Cleanup failed after project MCP operation drain for '%s'",
                    self.config.name,
                    exc_info=True,
                )

    async def _admit(self) -> None:
        """Atomically revalidate approval and record one in-flight operation."""
        authorized = False

        async with self._state_lock:
            if not self.disabled:
                from .project_approvals import _approvals_lock

                # Reset writers use this same cross-process lock. Keeping it
                # until _in_flight is incremented makes reset-vs-admission
                # ordering explicit without invoking server code under a lock.
                with _approvals_lock():
                    authorized = self._revalidate(approval_lock_held=True)
                    if authorized:
                        self._in_flight += 1
                    else:
                        self.disabled = True
                        self._start_cleanup_locked()
                    self._notify_state_locked()

        if not authorized:
            await self._revoke(wait_for_in_flight=False, propagate_cleanup_failure=False)
            raise self._authorization_error()

    async def disable(self) -> None:
        """Revoke future admissions and clean up once admitted work drains."""
        await self._revoke()

    async def cleanup(self) -> None:
        """Idempotent lifecycle cleanup using the revocation boundary."""
        await self.disable()

    async def validate(
        self,
        *,
        propagate_cleanup_failure: bool = True,
        wait_for_in_flight: bool = True,
    ) -> bool:
        """Observe current trust and revoke the handle if it is stale."""
        async with self._state_lock:
            disabled = self.disabled

        if disabled:
            await self._revoke(
                propagate_cleanup_failure=propagate_cleanup_failure,
                wait_for_in_flight=wait_for_in_flight,
            )
            return False

        if self._revalidate():
            return True

        await self._revoke(
            propagate_cleanup_failure=propagate_cleanup_failure,
            wait_for_in_flight=wait_for_in_flight,
        )
        logger.warning(
            "Project MCP server '%s' disabled because approval or execution identity changed",
            self.config.name,
        )
        return False

    async def require_authorized(self) -> None:
        """Raise when current project authorization is stale."""
        if await self.validate(propagate_cleanup_failure=False):
            return
        raise self._authorization_error()

    async def run_authorized(
        self,
        operation: Callable[..., _T | Awaitable[_T]],
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        """Admit and track one server-backed operation before invoking it."""
        if self._is_active_operation():
            result = operation(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result

        # Rejected work must not wait for other admitted work to drain: the
        # caller may itself own a paused iterator that it can only close after
        # this call returns. Explicit validate/cleanup still await the drain.
        if not await self.validate(propagate_cleanup_failure=False, wait_for_in_flight=False):
            raise self._authorization_error()
        await self._after_preflight_validation()
        await self._admit()

        try:
            with self._operation_context():
                result = operation(*args, **kwargs)
                return await result if inspect.isawaitable(result) else result
        finally:
            await asyncio.shield(self._finish_operation())

    async def _run_authorized_iterator(
        self,
        operation: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Keep admission for the stream, without granting it to its consumer."""
        owns_admission = not self._is_active_operation()
        if owns_admission:
            if not await self.validate(propagate_cleanup_failure=False, wait_for_in_flight=False):
                raise self._authorization_error()
            await self._after_preflight_validation()
            await self._admit()

        iterator: AsyncIterator[Any] | None = None
        try:
            with self._operation_context():
                result = operation(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                if not hasattr(result, "__aiter__"):
                    raise TypeError("MCP session operation does not return an async iterator")
                iterator = aiter(result)
            while True:
                # Never retain a ContextVar token across yield. Besides
                # leaking bypass authority, aclose may run in another task.
                with self._operation_context():
                    try:
                        item = await anext(iterator)
                    except StopAsyncIteration:
                        return
                yield item
        finally:
            try:
                if iterator is not None:
                    with self._operation_context():
                        await close_async_iterator(iterator)
            finally:
                if owns_admission:
                    await asyncio.shield(self._finish_operation())

    def bind_server(self, server: Any) -> None:
        """Bind the guarded identity to a raw server or a stable live handle.

        The separate MCP runtime branch supplies the stable ``LiveMCPServer``
        reconnect identity in the combined integration. This layer deliberately
        only requires a stable object exposing server methods and ``session``;
        it does not duplicate that runtime implementation.
        """
        if self.server is not None and self.server is not server:
            raise RuntimeError("MCP authorization validator cannot change server identity")
        self.server = server
        cleanup = getattr(server, "cleanup", None)
        is_our_cleanup = (
            getattr(cleanup, "__self__", None) is self
            and getattr(cleanup, "__func__", None) is ProjectServerAuthorizationValidator.cleanup
        )
        if cleanup is not None and not is_our_cleanup:
            self._original_cleanup = cleanup
        self._refresh_authorized_session()

    def _refresh_authorized_session(self) -> Any | None:
        server = self.server
        if server is None:
            self._raw_session = None
            return None
        raw_session_getter = (
            getattr(server, "_koder_raw_session", None)
            if callable(getattr(type(server), "_koder_raw_session", None))
            else None
        )
        session = (
            raw_session_getter()
            if raw_session_getter is not None
            else getattr(server, "session", None)
        )
        if session is self._authorized_session_proxy:
            return self._authorized_session_proxy if self._raw_session is not None else None
        self._raw_session = session
        if session is None:
            return None
        try:
            setattr(server, "session", self._authorized_session_proxy)
        except (AttributeError, TypeError):
            # Stable runtime handles may expose a read-only session property;
            # centralized callers still receive the generic proxy below.
            pass
        return self._authorized_session_proxy

    def authorized_session(self) -> Any | None:
        """Return a proxy for the current live client session."""
        return self._refresh_authorized_session()

    def install(self) -> None:
        """Install the single boundary on server, session, and cleanup methods."""
        server = self.server
        if server is None:
            raise RuntimeError("Cannot install MCP authorization without a server handle")
        if _direct_instance_attr(
            server,
            "_koder_project_authorization_boundary_installed",
            False,
        ):
            self.authorized_session()
            return

        for method_name in (
            "connect",
            "list_tools",
            "call_tool",
            "list_prompts",
            "get_prompt",
            "list_resources",
            "list_resource_templates",
            "read_resource",
        ):
            try:
                inspect.getattr_static(server, method_name)
            except AttributeError:
                continue
            original = getattr(server, method_name)
            self._wrapped_server_methods[method_name] = original

            async def _authorized_server_call(
                *args: Any,
                __original: Callable[..., Any] = original,
                __method_name: str = method_name,
                **kwargs: Any,
            ) -> Any:
                result = await self.run_authorized(__original, *args, **kwargs)
                if __method_name == "connect":
                    self.authorized_session()
                return result

            setattr(server, method_name, _authorized_server_call)

        if self._original_cleanup is not None:
            setattr(server, "cleanup", self.cleanup)

        self.authorized_session()
        setattr(server, "_koder_project_authorization_boundary_installed", True)


def get_project_authorization_validator(
    server: Any,
) -> ProjectServerAuthorizationValidator | None:
    """Return only a validator attached directly to *server*."""
    validator = _direct_instance_attr(server, "_koder_project_authorization_validator")
    return validator if isinstance(validator, ProjectServerAuthorizationValidator) else None


def attach_project_authorization_validator(
    server: Any,
    config: MCPServerConfig,
    *,
    validator: ProjectServerAuthorizationValidator | None = None,
) -> ProjectServerAuthorizationValidator | None:
    """Attach and install the authorization boundary for one project server."""
    if config.scope != MCPServerScope.PROJECT:
        return None
    existing = get_project_authorization_validator(server)
    if existing is not None:
        existing.install()
        return existing
    validator = validator or ProjectServerAuthorizationValidator(config)
    validator.bind_server(server)
    setattr(server, "_koder_project_authorization_validator", validator)
    validator.install()
    return validator


def get_authorized_session(server: Any) -> Any | None:
    """Return a server session that admits every server-backed operation."""
    validator = get_project_authorization_validator(server)
    if validator is not None:
        return validator.authorized_session()
    return getattr(server, "session", None)


async def call_authorized_session(
    server: Any,
    method_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Invoke any current or future client-session method through admission."""
    session = get_authorized_session(server)
    if session is None:
        validator = get_project_authorization_validator(server)
        if validator is not None:
            await validator.require_authorized()
        raise RuntimeError("MCP server has no active client session")
    method = getattr(session, method_name)
    result = method(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


async def call_authorized_server_method(
    server: Any,
    method_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Prefer a stable server method while preserving authorization admission.

    ``LiveMCPServer`` methods pin the selected concrete transport for the whole
    call. Servers without an explicit method continue to use the generic
    authorized ``ClientSession`` proxy.
    """
    method = (
        getattr(server, method_name, None)
        if callable(getattr(type(server), method_name, None))
        else None
    )
    if method is None:
        return await call_authorized_session(server, method_name, *args, **kwargs)

    validator = get_project_authorization_validator(server)
    if validator is not None:
        return await validator.run_authorized(method, *args, **kwargs)

    result = method(*args, **kwargs)
    return await result if inspect.isawaitable(result) else result


async def validate_project_server_authorizations(servers: Iterable[Any]) -> dict[str, bool]:
    """Revalidate all attached project servers at a turn boundary."""
    results: dict[str, bool] = {}
    for server in servers:
        validator = get_project_authorization_validator(server)
        if validator is None:
            continue
        results[validator.config.name] = await validator.validate()
    return results
