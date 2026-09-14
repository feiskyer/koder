"""Runtime-owned agent service binding for SDK tools and nested agent runs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    from .service import AgentService

_PROVIDER: ContextVar[Callable[[], AgentService] | None] = ContextVar(
    "koder_agent_service_provider", default=None
)
_DIRECT_SERVICES: WeakKeyDictionary[asyncio.AbstractEventLoop, dict[Path, AgentService]] = (
    WeakKeyDictionary()
)


class AgentSessionReader(Protocol):
    async def get_items(self) -> list[dict]: ...


@dataclass
class _SessionBinding:
    session: AgentSessionReader | None


_SESSION: ContextVar[_SessionBinding | None] = ContextVar("koder_agent_session", default=None)


def set_agent_session(session: AgentSessionReader | None) -> Token:
    """Bind the actual session rather than reconstructing its storage location."""
    return _SESSION.set(_SessionBinding(session))


def reset_agent_session(token: Token) -> None:
    binding = _SESSION.get()
    if binding is not None:
        # Tasks that inherited this binding must not reuse a retired turn.
        binding.session = None
    _SESSION.reset(token)


@contextmanager
def agent_session_scope(session: AgentSessionReader | None) -> Iterator[None]:
    token = set_agent_session(session)
    try:
        yield
    finally:
        reset_agent_session(token)


def get_runtime_agent_session() -> AgentSessionReader | None:
    binding = _SESSION.get()
    return binding.session if binding is not None else None


def has_agent_session_scope() -> bool:
    """Distinguish unscoped helpers from a detached or retired managed binding."""
    return _SESSION.get() is not None


def set_agent_service_provider(provider: Callable[[], AgentService]) -> Token:
    """Bind a lazy owner; merely starting a turn does not load agent history."""
    return _PROVIDER.set(provider)


def reset_agent_service_provider(token: Token) -> None:
    _PROVIDER.reset(token)


@contextmanager
def agent_service_scope(service: AgentService) -> Iterator[AgentService]:
    token = set_agent_service_provider(lambda: service)
    try:
        yield service
    finally:
        reset_agent_service_provider(token)


def get_bound_agent_service() -> AgentService | None:
    """Resolve a bound owner without creating an unscoped fallback service."""
    provider = _PROVIDER.get()
    return provider() if provider is not None else None


def get_runtime_agent_service() -> AgentService:
    provider = _PROVIDER.get()
    if provider is not None:
        return provider()

    # Standalone async callers / a single-client MCP loop still have continuity.
    # Separate loops and profiles do not accidentally share tasks or aliases.
    from koder_agent.tools.permission_context import get_tool_permission_context

    from .service import AgentService

    loop = asyncio.get_running_loop()
    root = (Path.home() / ".koder" / "agents").resolve()
    services = _DIRECT_SERVICES.setdefault(loop, {})
    service = services.get(root)
    if service is None or service.is_closed:
        permission = get_tool_permission_context()
        service = AgentService(
            output_root=root,
            permission_service=permission.permission_service if permission else None,
            retain_completed_tasks=False,
        )
        services[root] = service
    return service
