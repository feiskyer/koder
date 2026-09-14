"""Regression tests for stable MCP server identity across reconnects."""

from __future__ import annotations

import asyncio
import gc
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from mcp.types import Tool as MCPTool

import koder_agent.mcp as mcp_pkg
import koder_agent.mcp.reconnection as reconnection_module
from koder_agent.agentic.agent import _build_prefixed_mcp_tools
from koder_agent.config import reset_config_manager
from koder_agent.config.manager import ConfigManager
from koder_agent.mcp import (
    MCPServerSet,
    close_mcp_servers,
    discover_mcp_resources,
    drain_orphaned_mcp_owners,
)
from koder_agent.mcp.prompts import MCPPrompt, execute_prompt
from koder_agent.mcp.reconnection import (
    LiveMCPServer,
    ReconnectionConfig,
    ReconnectionManager,
    RetirementOwner,
    drain_orphaned_retirements,
    get_orphaned_retirement_counts,
    retain_orphaned_retirements,
)
from koder_agent.mcp.server_config import MCPServerConfig, MCPServerScope, MCPServerType
from koder_agent.mcp.server_factory import _install_cleanup_guard

_SHORT_SHA1_COLLISION_TOOLS = ("a-.+!&&b", "a-::-%/b")


class _Session:
    def __init__(self, label: str):
        self.label = label

    async def list_resources(self):
        return SimpleNamespace(
            resources=[SimpleNamespace(uri=f"memory://{self.label}", description=self.label)]
        )

    async def get_prompt(self, name, arguments=None):
        return SimpleNamespace(
            messages=[SimpleNamespace(role="user", content=SimpleNamespace(text=self.label))],
            description=name,
        )


class _Server:
    def __init__(self, name: str, label: str):
        self.name = name
        self.label = label
        self.session = _Session(label)
        self.cleanup_count = 0
        self.calls = []
        self.use_structured_content = False

    async def list_tools(self, *args, **kwargs):
        return [
            MCPTool(
                name="echo",
                description="echo",
                inputSchema={"type": "object", "properties": {}},
            )
        ]

    async def call_tool(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.label

    async def list_resources(self, *args, **kwargs):
        return await self.session.list_resources(*args, **kwargs)

    async def get_prompt(self, *args, **kwargs):
        return await self.session.get_prompt(*args, **kwargs)

    async def cleanup(self):
        self.cleanup_count += 1

    def _get_failure_error_function(self, failure_error_function):
        return failure_error_function

    def _get_needs_approval_for_tool(self, tool, agent):
        return False


def test_live_server_public_attributes_never_expose_concrete_transport():
    server = _Server("srv", "old")
    server.experimental = SimpleNamespace(send_ping=AsyncMock())
    handle = LiveMCPServer("srv", server)

    public_attributes = {name for name in dir(handle) if not name.startswith("_")}
    assert public_attributes == {
        "cached_tools",
        "call_tool",
        "cleanup",
        "connect",
        "current",
        "custom_data_extractor",
        "get_prompt",
        "list_prompts",
        "list_resource_templates",
        "list_resources",
        "list_tools",
        "name",
        "read_resource",
        "replace",
        "server_initialize_result",
        "session",
        "tool_meta_resolver",
        "use_structured_content",
    }
    for name in public_attributes:
        value = getattr(handle, name)
        assert value is not server
        assert value is not server.session
        assert getattr(value, "__self__", None) is not server

    assert handle.current is handle
    assert handle.current.current is handle
    assert handle.current.session is handle.session
    assert handle.label is not server.label
    assert handle.experimental is not server.experimental


@pytest.mark.asyncio
@pytest.mark.parametrize("public_path", [("send_ping",), ("experimental", "send_ping")])
async def test_non_project_generic_public_calls_remain_leased_and_compatible(public_path):
    old = _Server("srv", "old")
    old.send_ping = AsyncMock(return_value="old")
    old.experimental = SimpleNamespace(send_ping=AsyncMock(return_value="old"))
    new = _Server("srv", "new")
    new.send_ping = AsyncMock(return_value="new")
    new.experimental = SimpleNamespace(send_ping=AsyncMock(return_value="new"))
    handle = LiveMCPServer("srv", old)
    retained_call = handle
    for attribute_name in public_path:
        retained_call = getattr(retained_call, attribute_name)

    assert await retained_call() == "old"
    await handle.replace(new)
    assert await retained_call() == "new"

    await handle.cleanup()
    assert old.cleanup_count == 1
    assert new.cleanup_count == 1


def _stdio_config(
    name: str,
    command: str,
    *,
    source_path: str,
    scope: MCPServerScope | None = None,
) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        transport_type=MCPServerType.STDIO,
        command=command,
        args=["server.py"],
        scope=scope,
        source_path=source_path,
    )


def _write_runtime_config(home: Path, data: dict) -> Path:
    config_path = home / ".koder" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return config_path


def _configured_list_payload(
    scope: MCPServerScope,
    project: Path,
    entries: list[dict],
) -> dict:
    if scope == MCPServerScope.USER:
        return {"mcp_servers": entries}
    return {
        "mcp_local_projects": [
            {
                "project_root": str(project),
                "servers": entries,
            }
        ]
    }


@pytest.mark.asyncio
async def test_old_function_tool_closure_calls_reconnected_server(monkeypatch):
    from mcp.types import CallToolResult, TextContent

    class SDKServer(_Server):
        async def call_tool(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return CallToolResult(content=[TextContent(type="text", text=self.label)])

    old = SDKServer("srv", "old")
    new = SDKServer("srv", "new")
    handle = LiveMCPServer("srv", old)
    tools = await _build_prefixed_mcp_tools([handle], [])
    tool = tools[0]
    manager = ReconnectionManager(ReconnectionConfig(initial_delay=0, max_delay=0, max_attempts=1))
    manager.bind(server=handle, connect_fn=AsyncMock(return_value=new))
    old.session = None

    assert await manager.reconnect_if_needed() is True
    result = await tool.on_invoke_tool(None, "{}")

    assert "new" in str(result)
    assert old.calls == []
    assert len(new.calls) == 1
    assert old.cleanup_count == 1


@pytest.mark.asyncio
async def test_reconnect_list_tools_order_keeps_public_names_stable():
    class NamingServer(_Server):
        def __init__(self, name: str, label: str, tool_names: tuple[str, ...]):
            super().__init__(name, label)
            self.tool_names = tool_names

        async def list_tools(self, *args, **kwargs):
            return [
                MCPTool(
                    name=tool_name,
                    description=tool_name,
                    inputSchema={"type": "object", "properties": {}},
                )
                for tool_name in self.tool_names
            ]

    def name_mapping(tools):
        return {tool.description: tool.name for tool in tools}

    old = NamingServer("srv", "old", _SHORT_SHA1_COLLISION_TOOLS)
    new = NamingServer("srv", "new", tuple(reversed(_SHORT_SHA1_COLLISION_TOOLS)))
    handle = LiveMCPServer("srv", old)
    before = name_mapping(await _build_prefixed_mcp_tools([handle], []))
    manager = ReconnectionManager(ReconnectionConfig(initial_delay=0, max_delay=0, max_attempts=1))
    manager.bind(server=handle, connect_fn=AsyncMock(return_value=new))
    old.session = None

    assert await manager.reconnect_if_needed() is True
    after = name_mapping(await _build_prefixed_mcp_tools([handle], []))

    assert manager.server is handle
    assert handle._koder_current_transport() is new
    assert before == after
    assert len(set(after.values())) == 2
    assert all(len(name) <= 64 for name in after.values())


@pytest.mark.asyncio
async def test_reconnect_duplicate_tool_names_keep_metadata_identity_when_order_reverses():
    class DuplicateNamingServer(_Server):
        def __init__(self, name: str, label: str, definitions: tuple[tuple[str, str], ...]):
            super().__init__(name, label)
            self.definitions = definitions

        async def list_tools(self, *args, **kwargs):
            return [
                MCPTool(
                    name="search",
                    description=description,
                    inputSchema={
                        "type": "object",
                        "properties": {"query": {"type": query_type}},
                    },
                )
                for description, query_type in self.definitions
            ]

    definitions = (("text search", "string"), ("numeric search", "integer"))
    old = DuplicateNamingServer("srv", "old", definitions)
    new = DuplicateNamingServer("srv", "new", tuple(reversed(definitions)))
    handle = LiveMCPServer("srv", old)

    before_tools = await _build_prefixed_mcp_tools([handle], [])
    before = {tool.description: tool.name for tool in before_tools}
    manager = ReconnectionManager(ReconnectionConfig(initial_delay=0, max_delay=0, max_attempts=1))
    manager.bind(server=handle, connect_fn=AsyncMock(return_value=new))
    old.session = None

    assert await manager.reconnect_if_needed() is True
    after_tools = await _build_prefixed_mcp_tools([handle], [])
    after = {tool.description: tool.name for tool in after_tools}

    assert before == after
    assert len(set(after.values())) == 2


@pytest.mark.asyncio
async def test_resources_prompts_and_final_cleanup_use_current_server():
    old = _Server("srv", "old")
    new = _Server("srv", "new")
    handle = LiveMCPServer("srv", old)
    await handle.replace(new)

    resources = await discover_mcp_resources([handle])
    prompt = MCPPrompt(server_name="srv", prompt_name="status")
    prompt_result = await execute_prompt(prompt, [handle], [])
    await handle.cleanup()

    assert resources == [("srv:memory://new", "new")]
    assert prompt_result.messages == [{"role": "user", "content": "new"}]
    assert old.cleanup_count == 1
    assert new.cleanup_count == 1


@pytest.mark.asyncio
async def test_concurrent_call_finishes_before_replaced_server_closes():
    started = asyncio.Event()
    finish = asyncio.Event()

    class BlockingServer(_Server):
        async def call_tool(self, *args, **kwargs):
            started.set()
            await finish.wait()
            return self.label

    old = BlockingServer("srv", "old")
    new = _Server("srv", "new")
    handle = LiveMCPServer("srv", old)

    old_call = asyncio.create_task(handle.call_tool("echo", {}))
    await started.wait()
    swap = asyncio.create_task(handle.replace(new))
    await asyncio.sleep(0)

    assert await handle.call_tool("echo", {}) == "new"
    assert old.cleanup_count == 0
    finish.set()
    assert await old_call == "old"
    await swap
    assert old.cleanup_count == 1


@pytest.mark.asyncio
async def test_public_session_proxy_pins_nested_call_before_replaced_server_closes():
    started = asyncio.Event()
    finish = asyncio.Event()

    async def blocking_ping():
        started.set()
        await finish.wait()
        return "old"

    old = _Server("srv", "old")
    old.session = SimpleNamespace(
        experimental=SimpleNamespace(send_ping=blocking_ping),
    )
    new = _Server("srv", "new")
    new.session = SimpleNamespace(
        experimental=SimpleNamespace(send_ping=AsyncMock(return_value="new")),
    )
    handle = LiveMCPServer("srv", old)
    public_session = handle.session
    retained_ping = public_session.experimental.send_ping

    with pytest.raises(AttributeError, match="raw_session"):
        _ = public_session.raw_session

    old_call = asyncio.create_task(retained_ping())
    await started.wait()
    swap = asyncio.create_task(handle.replace(new))
    for _ in range(100):
        if handle._koder_current_transport() is new:
            break
        await asyncio.sleep(0)

    assert handle._koder_current_transport() is new
    assert not swap.done()
    assert old.cleanup_count == 0
    assert await retained_ping() == "new"

    finish.set()
    assert await old_call == "old"
    await swap
    assert old.cleanup_count == 1

    await handle.cleanup()
    assert new.cleanup_count == 1


@pytest.mark.asyncio
async def test_cancelled_reconnect_waiting_for_transition_lock_cleans_candidate_once():
    old = _Server("srv", "old")
    old.session = None

    class FlakyCandidate(_Server):
        async def cleanup(self):
            self.cleanup_count += 1
            if self.cleanup_count == 1:
                raise RuntimeError("first cleanup failed")

    candidate = FlakyCandidate("srv", "candidate")
    candidate_connected = asyncio.Event()

    async def connect():
        candidate_connected.set()
        return candidate

    handle = LiveMCPServer("srv", old)
    manager = ReconnectionManager(ReconnectionConfig(initial_delay=0, max_delay=0, max_attempts=1))
    manager.bind(server=handle, connect_fn=connect)

    await handle._transition_lock.acquire()
    reconnect = asyncio.create_task(manager.reconnect_if_needed())
    try:
        await candidate_connected.wait()
        await asyncio.sleep(0)
        assert not reconnect.done()

        reconnect.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reconnect
    finally:
        handle._transition_lock.release()

    assert handle._koder_current_transport() is old
    assert old.cleanup_count == 0
    assert candidate.cleanup_count == 1
    assert handle._retiring_servers[id(candidate)] is candidate

    await manager.cleanup()
    assert candidate.cleanup_count == 2
    assert old.cleanup_count == 1
    assert not handle._retiring_servers

    await manager.cleanup()
    assert candidate.cleanup_count == 2
    assert old.cleanup_count == 1


@pytest.mark.asyncio
async def test_thousand_replacements_release_retired_servers_after_exactly_once_cleanup():
    closed_labels: list[str] = []

    class ReclaimableServer(_Server):
        async def cleanup(self):
            await super().cleanup()
            closed_labels.append(self.label)

    current = ReclaimableServer("srv", "server-0")
    handle = LiveMCPServer("srv", current)
    retired_refs = []

    for index in range(1, 1001):
        retired_refs.append(weakref.ref(current))
        current = ReclaimableServer("srv", f"server-{index}")
        await handle.replace(current)

    await asyncio.sleep(0)
    gc.collect()

    assert len(closed_labels) == 1000
    assert len(set(closed_labels)) == 1000
    assert all(server_ref() is None for server_ref in retired_refs)
    assert not handle._retiring_servers
    assert not handle._retirement_tasks

    await handle.cleanup()
    assert len(closed_labels) == 1001


@pytest.mark.asyncio
async def test_cancelled_cleanup_during_active_call_can_be_rejoined():
    call_started = asyncio.Event()
    finish_call = asyncio.Event()

    class BlockingServer(_Server):
        async def call_tool(self, *args, **kwargs):
            call_started.set()
            await finish_call.wait()
            return self.label

    server = BlockingServer("srv", "old")
    handle = LiveMCPServer("srv", server)
    active_call = asyncio.create_task(handle.call_tool("echo", {}))
    await call_started.wait()

    first_cleanup = asyncio.create_task(handle.cleanup())
    await asyncio.sleep(0)
    assert handle._koder_current_transport() is None
    assert server.cleanup_count == 0

    first_cleanup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_cleanup

    retry_cleanup = asyncio.create_task(handle.cleanup())
    await asyncio.sleep(0)
    assert not retry_cleanup.done()

    finish_call.set()
    assert await active_call == "old"
    await retry_cleanup
    await handle.cleanup()
    assert server.cleanup_count == 1


@pytest.mark.asyncio
async def test_concurrent_live_cleanup_callers_share_transport_cleanup():
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()

    class BlockingCleanupServer(_Server):
        async def cleanup(self):
            self.cleanup_count += 1
            cleanup_started.set()
            await finish_cleanup.wait()

    server = BlockingCleanupServer("srv", "old")
    handle = LiveMCPServer("srv", server)
    first_cleanup = asyncio.create_task(handle.cleanup())
    await cleanup_started.wait()
    second_cleanup = asyncio.create_task(handle.cleanup())
    await asyncio.sleep(0)

    assert server.cleanup_count == 1
    assert not first_cleanup.done()
    assert not second_cleanup.done()

    first_cleanup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_cleanup

    finish_cleanup.set()
    await second_cleanup
    await handle.cleanup()
    assert server.cleanup_count == 1


@pytest.mark.asyncio
async def test_live_cleanup_drains_first_failure_then_retry_success():
    class FlakyCleanupServer(_Server):
        async def cleanup(self):
            self.cleanup_count += 1
            if self.cleanup_count == 1:
                raise RuntimeError("cleanup failed")

    server = FlakyCleanupServer("srv", "old")
    handle = LiveMCPServer("srv", server)

    await handle.cleanup()
    assert handle._koder_current_transport() is None
    assert server.cleanup_count == 2

    await handle.cleanup()
    await handle.cleanup()
    assert server.cleanup_count == 2


@pytest.mark.asyncio
async def test_retirement_drain_prioritizes_cancellation_over_other_failures():
    class FailsOnceServer(_Server):
        def __init__(self, label: str, failure: BaseException):
            super().__init__("srv", label)
            self.failure = failure

        async def cleanup(self):
            self.cleanup_count += 1
            if self.cleanup_count == 1:
                raise self.failure

    ordinary_failure = FailsOnceServer("ordinary", RuntimeError("cleanup failed"))
    cancelled_failure = FailsOnceServer("cancelled", asyncio.CancelledError())
    owner = RetirementOwner()
    owner.retire(ordinary_failure)
    owner.retire(cancelled_failure)

    with pytest.raises(asyncio.CancelledError):
        await owner.drain(max_attempts=1)

    assert ordinary_failure.cleanup_count == 1
    assert cancelled_failure.cleanup_count == 1
    assert owner.pending_count == 2

    await owner.drain(max_attempts=1)
    assert ordinary_failure.cleanup_count == 2
    assert cancelled_failure.cleanup_count == 2
    assert owner.held_count == 0


@pytest.mark.asyncio
async def test_repeated_cleanup_failures_remain_owned_until_later_success():
    class RepeatedlyFailingServer(_Server):
        def __init__(self, name: str, label: str):
            super().__init__(name, label)
            self.allow_cleanup = False

        async def cleanup(self):
            self.cleanup_count += 1
            if not self.allow_cleanup:
                raise RuntimeError("cleanup still failing")

    server = RepeatedlyFailingServer("srv", "old")
    server_ref = weakref.ref(server)
    handle = LiveMCPServer("srv", server)

    with pytest.raises(RuntimeError, match="cleanup still failing"):
        await handle.cleanup()

    assert server.cleanup_count == 3
    assert handle._retiring_servers[id(server)] is server
    del server
    gc.collect()
    retained = server_ref()
    assert retained is not None

    retained.allow_cleanup = True
    await handle.cleanup()
    assert retained.cleanup_count == 4
    assert not handle._retiring_servers

    await handle.cleanup()
    assert retained.cleanup_count == 4


@pytest.mark.asyncio
async def test_failed_replacement_retains_candidate_until_handle_shutdown():
    class FlakyCandidate(_Server):
        async def cleanup(self):
            self.cleanup_count += 1
            if self.cleanup_count == 1:
                raise RuntimeError("cleanup failed")

    old = _Server("srv", "old")
    handle = LiveMCPServer("srv", old)
    await handle.cleanup()
    candidate = FlakyCandidate("srv", "candidate")

    with pytest.raises(RuntimeError, match="closed"):
        await handle.replace(candidate)
    await asyncio.sleep(0)

    assert handle._retiring_servers[id(candidate)] is candidate
    assert candidate.cleanup_count == 1

    await handle.cleanup()
    assert candidate.cleanup_count == 2
    assert not handle._retiring_servers


@pytest.mark.asyncio
async def test_manager_shutdown_drains_every_retained_candidate():
    class FlakyCandidate(_Server):
        async def cleanup(self):
            self.cleanup_count += 1
            if self.cleanup_count == 1:
                raise RuntimeError(f"{self.label} cleanup failed")

    owner = RetirementOwner()
    current = _Server("srv", "current")
    handle = LiveMCPServer("srv", current, retirement_owner=owner)
    manager = ReconnectionManager(
        ReconnectionConfig(initial_delay=0, max_delay=0, max_attempts=1),
        retirement_owner=owner,
    )
    manager.bind(server=handle)
    candidates = [FlakyCandidate("srv", f"candidate-{index}") for index in range(3)]
    first_attempts = []
    for candidate in candidates:
        first_attempts.append(owner.retire(candidate))

    await asyncio.gather(
        *(task for task in first_attempts if task is not None),
        return_exceptions=True,
    )
    assert owner.pending_count == 3

    await manager.cleanup()

    assert current.cleanup_count == 1
    assert [candidate.cleanup_count for candidate in candidates] == [2, 2, 2]
    assert owner.held_count == 0


@pytest.mark.asyncio
async def test_successful_non_weakrefable_cleanup_identity_cache_is_bounded():
    class SlottedServer:
        __slots__ = ("cleanup_count",)

        def __init__(self):
            self.cleanup_count = 0

        async def cleanup(self):
            self.cleanup_count += 1

    owner = RetirementOwner()
    servers = [SlottedServer() for _ in range(1000)]

    for server in servers:
        retirement = owner.retire(server)
        assert retirement is not None
        await retirement

    assert all(server.cleanup_count == 1 for server in servers)
    assert owner.held_count == 0
    assert len(owner._fallback_closed_servers) == 256
    assert all(server in servers for server in owner._fallback_closed_servers)


@pytest.mark.asyncio
async def test_non_weakrefable_fresh_transport_with_reused_id_is_not_closed():
    class SlottedServer:
        __slots__ = ("cleanup_count",)

        def __init__(self):
            self.cleanup_count = 0

        async def cleanup(self):
            self.cleanup_count += 1

    owner = RetirementOwner()
    reused_id = 0xC105ED

    with patch.object(reconnection_module, "id", return_value=reused_id, create=True):
        previous = SlottedServer()
        retirement = owner.retire(previous)
        assert retirement is not None
        await retirement
        assert previous.cleanup_count == 1

        # Drop the caller's prior reference and deterministically simulate the
        # allocator assigning the same object id to a fresh transport.
        del previous
        gc.collect()
        fresh = SlottedServer()

        owner.hold(fresh)
        assert owner.held_count == 1
        assert owner._records[reused_id].server is fresh

        retirement = owner.retire(fresh)
        assert retirement is not None
        await retirement
        assert fresh.cleanup_count == 1
        assert owner.retire(fresh) is None
        assert fresh.cleanup_count == 1
        assert owner.held_count == 0
        assert len(owner._fallback_closed_servers) == 2


@pytest.mark.asyncio
async def test_cleanup_guard_shares_in_flight_attempt_across_cancelled_waiter():
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()

    class BlockingCleanupServer:
        def __init__(self):
            self.cleanup_count = 0

        async def cleanup(self):
            self.cleanup_count += 1
            cleanup_started.set()
            await finish_cleanup.wait()

    server = BlockingCleanupServer()
    _install_cleanup_guard(server)
    first_cleanup = asyncio.create_task(server.cleanup())
    await cleanup_started.wait()
    second_cleanup = asyncio.create_task(server.cleanup())
    await asyncio.sleep(0)

    first_cleanup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_cleanup
    assert server.cleanup_count == 1
    assert not second_cleanup.done()

    finish_cleanup.set()
    await second_cleanup
    await server.cleanup()
    assert server.cleanup_count == 1


@pytest.mark.asyncio
async def test_cleanup_guard_retries_cancelled_transport_attempt():
    class CancelledCleanupServer:
        def __init__(self):
            self.cleanup_count = 0

        async def cleanup(self):
            self.cleanup_count += 1
            if self.cleanup_count == 1:
                raise asyncio.CancelledError

    server = CancelledCleanupServer()
    _install_cleanup_guard(server)

    with pytest.raises(asyncio.CancelledError):
        await server.cleanup()
    await server.cleanup()
    await server.cleanup()
    assert server.cleanup_count == 2


@pytest.mark.asyncio
async def test_cleanup_guard_retries_failed_transport_attempt():
    class FailedCleanupServer:
        def __init__(self):
            self.cleanup_count = 0

        async def cleanup(self):
            self.cleanup_count += 1
            if self.cleanup_count == 1:
                raise RuntimeError("cleanup failed")

    server = FailedCleanupServer()
    _install_cleanup_guard(server)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await server.cleanup()
    await server.cleanup()
    await server.cleanup()
    assert server.cleanup_count == 2


@pytest.mark.asyncio
async def test_create_servers_cleanup_cancellation_stops_before_next_connection(monkeypatch):
    await drain_orphaned_retirements()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    created_names = []

    class FailedCandidate(_Server):
        async def connect(self):
            raise ConnectionError("connect failed")

        async def cleanup(self):
            self.cleanup_count += 1
            cleanup_started.set()
            await finish_cleanup.wait()

    candidate = FailedCandidate("first", "first")
    configs = [
        _stdio_config(
            "first",
            "python-first",
            source_path="configured.json",
            scope=MCPServerScope.USER,
        ),
        _stdio_config(
            "second",
            "python-second",
            source_path="configured.json",
            scope=MCPServerScope.USER,
        ),
    ]

    async def fake_create(config, *_args, **_kwargs):
        created_names.append(config.name)
        if config.name == "second":
            raise AssertionError("second connection started after cancellation")
        return candidate

    monkeypatch.setattr(mcp_pkg.MCPServerFactory, "create_server", fake_create)
    task = asyncio.create_task(mcp_pkg.MCPServerFactory.create_servers_from_configs(configs))
    await cleanup_started.wait()
    task.cancel()
    finish_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert created_names == ["first"]
    await reconnection_module.drain_orphaned_retirements()
    assert candidate.cleanup_count == 1
    assert get_orphaned_retirement_counts() == (0, 0)


@pytest.mark.asyncio
async def test_retry_factory_cleanup_cancellation_stops_before_new_candidate(monkeypatch):
    await drain_orphaned_retirements()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    create_count = 0

    class FailedCandidate(_Server):
        async def connect(self):
            raise ConnectionError("connect failed")

        async def cleanup(self):
            self.cleanup_count += 1
            cleanup_started.set()
            await finish_cleanup.wait()

    candidate = FailedCandidate("srv", "candidate")

    async def fake_create(*_args, **_kwargs):
        nonlocal create_count
        create_count += 1
        if create_count > 1:
            raise AssertionError("new candidate created after cancellation")
        return candidate

    monkeypatch.setattr(mcp_pkg.MCPServerFactory, "create_server", fake_create)
    config = _stdio_config(
        "srv",
        "python",
        source_path="configured.json",
        scope=MCPServerScope.USER,
    )
    task = asyncio.create_task(
        mcp_pkg.MCPServerFactory.create_and_connect_with_retry(
            config,
            reconnection_config=ReconnectionConfig(
                initial_delay=0,
                max_delay=0,
                max_attempts=3,
            ),
        )
    )
    await cleanup_started.wait()
    task.cancel()
    finish_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert create_count == 1
    await reconnection_module.drain_orphaned_retirements()
    assert candidate.cleanup_count == 1
    assert get_orphaned_retirement_counts() == (0, 0)


@pytest.mark.asyncio
async def test_retry_factory_final_drain_cancellation_is_not_replaced_by_connection_error(
    monkeypatch,
):
    await drain_orphaned_retirements()
    final_drain_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    create_count = 0

    class FailedCandidate(_Server):
        async def connect(self):
            raise ConnectionError("connect failed")

        async def cleanup(self):
            self.cleanup_count += 1
            if self.cleanup_count == 1:
                raise RuntimeError("first cleanup failed")
            final_drain_started.set()
            await finish_cleanup.wait()

    candidate = FailedCandidate("srv", "candidate")

    async def fake_create(*_args, **_kwargs):
        nonlocal create_count
        create_count += 1
        return candidate

    monkeypatch.setattr(mcp_pkg.MCPServerFactory, "create_server", fake_create)
    config = _stdio_config(
        "srv",
        "python",
        source_path="configured.json",
        scope=MCPServerScope.USER,
    )
    task = asyncio.create_task(
        mcp_pkg.MCPServerFactory.create_and_connect_with_retry(
            config,
            reconnection_config=ReconnectionConfig(
                initial_delay=0,
                max_delay=0,
                max_attempts=1,
            ),
        )
    )
    await final_drain_started.wait()
    task.cancel()
    finish_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert create_count == 1
    await reconnection_module.drain_orphaned_retirements()
    assert candidate.cleanup_count == 2
    assert get_orphaned_retirement_counts() == (0, 0)


@pytest.mark.asyncio
async def test_failed_reconnect_preserves_old_handle_and_cleans_candidate():
    old = _Server("srv", "old")
    old.session = None
    candidate = _Server("srv", "candidate")

    async def connect():
        try:
            raise ConnectionError("failed after allocation")
        finally:
            await candidate.cleanup()

    handle = LiveMCPServer("srv", old)
    manager = ReconnectionManager(ReconnectionConfig(initial_delay=0, max_delay=0, max_attempts=1))
    manager.bind(server=handle, connect_fn=connect)

    assert await manager.reconnect_if_needed() is False
    assert handle._koder_current_transport() is old
    assert candidate.cleanup_count == 1
    assert old.cleanup_count == 0


@pytest.mark.asyncio
async def test_scheduler_cleanup_closes_current_server_once():
    from koder_agent.core.scheduler import AgentScheduler

    old = _Server("srv", "old")
    new = _Server("srv", "new")
    handle = LiveMCPServer("srv", old)
    await handle.replace(new)
    owner = MCPServerSet([handle])

    with (
        patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
        patch("koder_agent.core.scheduler.get_display_hooks"),
        patch("koder_agent.core.scheduler.ApprovalHooks"),
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession"),
    ):
        scheduler = AgentScheduler(session_id="test")
        scheduler._mcp_servers = owner
        scheduler.dev_agent = AsyncMock()
        await scheduler.reset_agent()
        await scheduler.reset_agent()

    assert old.cleanup_count == 1
    assert new.cleanup_count == 1


@pytest.mark.asyncio
async def test_scheduler_reset_keeps_failed_owner_cleanup_retryable_and_isolated():
    from koder_agent.core.scheduler import AgentScheduler

    class FailingServer(_Server):
        def __init__(self):
            super().__init__("srv", "scheduler-failing")
            self.allow_cleanup = False

        async def cleanup(self):
            self.cleanup_count += 1
            if not self.allow_cleanup:
                raise RuntimeError("cleanup still failing")

    transport = FailingServer()
    retirement_owner = RetirementOwner()
    handle = LiveMCPServer("srv", transport, retirement_owner=retirement_owner)
    manager = ReconnectionManager(retirement_owner=retirement_owner)
    manager.bind(server=handle)
    server_owner = MCPServerSet([handle], reconnection_managers={"srv": manager})

    other_transport = _Server("other", "other")
    other_handle = LiveMCPServer("other", other_transport)
    other_owner = MCPServerSet([other_handle])
    with (
        patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
        patch("koder_agent.core.scheduler.get_display_hooks"),
        patch("koder_agent.core.scheduler.ApprovalHooks"),
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession"),
    ):
        scheduler = AgentScheduler(session_id="test-failed-cleanup")
        scheduler._mcp_servers = server_owner
        scheduler.dev_agent = AsyncMock()
        await scheduler.reset_agent()

    assert transport.cleanup_count == 3
    assert scheduler._mcp_servers == []
    assert mcp_pkg.get_reconnection_managers(server_owner) == {"srv": manager}
    assert other_handle._koder_current_transport() is other_transport
    assert other_transport.cleanup_count == 0
    assert get_orphaned_retirement_counts() == (0, 0)

    transport.allow_cleanup = True
    await close_mcp_servers(server_owner)
    assert transport.cleanup_count == 4
    assert mcp_pkg.get_reconnection_managers(server_owner) == {}
    assert get_orphaned_retirement_counts() == (0, 0)

    await close_mcp_servers(other_owner)
    assert other_transport.cleanup_count == 1


def _live_runtime_parts(name: str, label: str):
    transport = _Server(name, label)
    retirement_owner = RetirementOwner()
    handle = LiveMCPServer(name, transport, retirement_owner=retirement_owner)
    manager = ReconnectionManager(retirement_owner=retirement_owner)
    manager.bind(server=handle)
    return handle, manager, transport


def _owned_runtime(name: str, label: str):
    handle, manager, transport = _live_runtime_parts(name, label)
    owner = MCPServerSet([handle], reconnection_managers={name: manager})
    return owner, handle, manager, transport


@pytest.mark.asyncio
async def test_abandoned_owner_finalizer_retries_failed_cleanup():
    await drain_orphaned_mcp_owners()

    class FailingServer(_Server):
        def __init__(self):
            super().__init__("srv", "abandoned")
            self.allow_cleanup = False

        async def cleanup(self):
            self.cleanup_count += 1
            if not self.allow_cleanup:
                raise RuntimeError("cleanup still failing")

    transport = FailingServer()
    retirement_owner = RetirementOwner()
    handle = LiveMCPServer("srv", transport, retirement_owner=retirement_owner)
    owner = MCPServerSet([handle])

    await close_mcp_servers(owner)
    assert transport.cleanup_count == 3
    assert handle._koder_current_transport() is None

    owner_ref = weakref.ref(owner)
    del owner
    gc.collect()
    assert owner_ref() is None

    transport.allow_cleanup = True
    await drain_orphaned_mcp_owners()
    assert transport.cleanup_count == 4
    assert get_orphaned_retirement_counts() == (0, 0)


@pytest.mark.asyncio
async def test_transport_orphan_retry_is_explicit_and_owner_isolated():
    await drain_orphaned_retirements()

    class FailingServer(_Server):
        def __init__(self):
            super().__init__("orphan", "orphan")
            self.allow_cleanup = False

        async def cleanup(self):
            self.cleanup_count += 1
            if not self.allow_cleanup:
                raise RuntimeError("cleanup still failing")

    orphan = FailingServer()
    retirement_owner = RetirementOwner()
    retirement = retirement_owner.retire(orphan)
    assert retirement is not None
    with pytest.raises(RuntimeError, match="cleanup still failing"):
        await retirement
    retain_orphaned_retirements(retirement_owner)

    other = _Server("other", "other")
    other_handle = LiveMCPServer("other", other)
    await other_handle.cleanup()

    assert other.cleanup_count == 1
    assert orphan.cleanup_count == 1
    assert get_orphaned_retirement_counts() == (1, 1)

    orphan.allow_cleanup = True
    await drain_orphaned_retirements()
    assert orphan.cleanup_count == 2
    assert get_orphaned_retirement_counts() == (0, 0)


@pytest.mark.asyncio
async def test_concurrent_same_name_loads_create_independent_owners(monkeypatch):
    config = _stdio_config(
        "shared.name",
        "python",
        source_path="configured.json",
        scope=MCPServerScope.USER,
    )
    started = 0
    both_started = asyncio.Event()
    release = asyncio.Event()
    created = []

    async def create_server(config, **_kwargs):
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await release.wait()
        runtime = _live_runtime_parts(config.name, f"owner-{started}")
        created.append(runtime)
        handle, manager, _transport = runtime
        return handle, manager

    monkeypatch.setattr(
        mcp_pkg.MCPServerManager,
        "list_servers",
        AsyncMock(return_value=[config]),
    )
    monkeypatch.setattr(mcp_pkg, "_load_plugin_mcp_configs", lambda: [])
    monkeypatch.setattr(
        "koder_agent.harness.channels.state.get_allowed_channels",
        lambda: [],
    )
    monkeypatch.setattr(
        mcp_pkg.MCPServerFactory,
        "create_and_connect_with_retry",
        create_server,
    )

    first_load = asyncio.create_task(mcp_pkg.load_mcp_servers())
    second_load = asyncio.create_task(mcp_pkg.load_mcp_servers())
    await asyncio.wait_for(both_started.wait(), timeout=1)
    release.set()
    first_owner, second_owner = await asyncio.gather(first_load, second_load)

    assert first_owner is not second_owner
    assert len(first_owner) == len(second_owner) == 1
    first_manager = mcp_pkg.get_reconnection_managers(first_owner)["shared.name"]
    second_manager = mcp_pkg.get_reconnection_managers(second_owner)["shared.name"]
    assert first_manager is not second_manager
    assert first_owner[0] is first_manager.server
    assert second_owner[0] is second_manager.server

    first_transport = first_owner[0]._koder_current_transport()
    second_transport = second_owner[0]._koder_current_transport()
    await close_mcp_servers(first_owner)
    assert first_transport.cleanup_count == 1
    assert second_transport.cleanup_count == 0
    assert second_owner[0]._koder_current_transport() is second_transport

    await close_mcp_servers(second_owner)
    assert second_transport.cleanup_count == 1


@pytest.mark.asyncio
async def test_cancelled_partial_load_closes_only_its_owner(monkeypatch, cancellation_observer):
    observe, cancellations = cancellation_observer
    other_owner, other_handle, other_manager, other_transport = _owned_runtime(
        "shared.name", "other-owner"
    )
    configs = [
        _stdio_config(
            "first",
            "python-first",
            source_path="configured.json",
            scope=MCPServerScope.USER,
        ),
        _stdio_config(
            "second",
            "python-second",
            source_path="configured.json",
            scope=MCPServerScope.USER,
        ),
    ]
    second_started = asyncio.Event()
    first_runtime = None

    async def create_server(config, **_kwargs):
        nonlocal first_runtime
        if config.name == "second":
            second_started.set()
            await asyncio.Event().wait()
        first_runtime = _live_runtime_parts(config.name, "partial-owner")
        handle, manager, _transport = first_runtime
        return handle, manager

    monkeypatch.setattr(
        mcp_pkg.MCPServerManager,
        "list_servers",
        AsyncMock(return_value=configs),
    )
    monkeypatch.setattr(mcp_pkg, "_load_plugin_mcp_configs", lambda: [])
    monkeypatch.setattr(
        "koder_agent.harness.channels.state.get_allowed_channels",
        lambda: [],
    )
    monkeypatch.setattr(
        mcp_pkg.MCPServerFactory,
        "create_and_connect_with_retry",
        create_server,
    )

    load_task = asyncio.create_task(observe(mcp_pkg.load_mcp_servers()))
    await asyncio.wait_for(second_started.wait(), timeout=1)
    load_task.cancel("partial-owner-cancel")
    with pytest.raises(asyncio.CancelledError):
        await load_task

    assert [error.args for error in cancellations] == [("partial-owner-cancel",)]
    assert first_runtime is not None
    first_transport = first_runtime[2]
    assert first_transport.cleanup_count == 1
    assert other_handle._koder_current_transport() is other_transport
    assert other_transport.cleanup_count == 0
    assert mcp_pkg.get_reconnection_managers(other_owner) == {"shared.name": other_manager}

    await close_mcp_servers(other_owner)
    assert other_transport.cleanup_count == 1


@pytest.mark.parametrize(
    ("layout", "first_name", "second_name", "public_identity"),
    [
        ("configured/configured", "dup", "dup", "dup"),
        ("configured/plugin", "dup", "dup", "dup"),
        ("plugin/plugin", "dup.plugin", "dup_plugin", "dup_plugin"),
    ],
)
@pytest.mark.asyncio
async def test_duplicate_preflight_preserves_unrelated_owner(
    monkeypatch,
    layout,
    first_name,
    second_name,
    public_identity,
):
    old_owner, old_handle, old_manager, old_transport = _owned_runtime("old", "old")

    first = _stdio_config(
        first_name,
        "python-first",
        source_path="configured-one.json" if layout != "plugin/plugin" else "plugin-one/.mcp.json",
        scope=MCPServerScope.USER if layout != "plugin/plugin" else None,
    )
    second = _stdio_config(
        second_name,
        "python-second",
        source_path=(
            "configured-two.json" if layout == "configured/configured" else "plugin-two/.mcp.json"
        ),
        scope=MCPServerScope.USER if layout == "configured/configured" else None,
    )
    if layout == "configured/configured":
        configured, plugins = [first, second], []
    elif layout == "configured/plugin":
        configured, plugins = [first], [second]
    else:
        configured, plugins = [], [first, second]

    factory = AsyncMock()
    monkeypatch.setattr(
        mcp_pkg.MCPServerManager,
        "list_servers",
        AsyncMock(return_value=configured),
    )
    monkeypatch.setattr(mcp_pkg, "_load_plugin_mcp_configs", lambda: plugins)
    monkeypatch.setattr(mcp_pkg.MCPServerFactory, "create_and_connect_with_retry", factory)

    with pytest.raises(
        RuntimeError,
        match=f"Duplicate public MCP server identity '{public_identity}'",
    ):
        await mcp_pkg.load_mcp_servers()

    assert factory.await_count == 0
    assert mcp_pkg.get_reconnection_managers(old_owner) == {"old": old_manager}
    assert old_handle._koder_current_transport() is old_transport
    assert old_transport.cleanup_count == 0

    await close_mcp_servers(old_owner)
    assert old_transport.cleanup_count == 1


@pytest.mark.asyncio
async def test_empty_normalized_identity_fails_before_connection_and_preserves_other_owner(
    monkeypatch,
):
    old_owner, old_handle, old_manager, old_transport = _owned_runtime("old", "old")
    invalid = _stdio_config(
        "...",
        "python",
        source_path="configured.json",
        scope=MCPServerScope.USER,
    )
    factory = AsyncMock()
    monkeypatch.setattr(
        mcp_pkg.MCPServerManager,
        "list_servers",
        AsyncMock(return_value=[invalid]),
    )
    monkeypatch.setattr(mcp_pkg, "_load_plugin_mcp_configs", lambda: [])
    monkeypatch.setattr(mcp_pkg.MCPServerFactory, "create_and_connect_with_retry", factory)

    with pytest.raises(RuntimeError, match="normalizes to an empty public identity"):
        await mcp_pkg.load_mcp_servers()

    assert factory.await_count == 0
    assert mcp_pkg.get_reconnection_managers(old_owner) == {"old": old_manager}
    assert old_handle._koder_current_transport() is old_transport
    assert old_transport.cleanup_count == 0

    await close_mcp_servers(old_owner)
    assert old_transport.cleanup_count == 1


@pytest.mark.asyncio
async def test_adoption_failure_retires_unpublished_handle(monkeypatch):
    config = _stdio_config(
        "adoption",
        "python",
        source_path="configured.json",
        scope=MCPServerScope.USER,
    )
    retirement_owner = RetirementOwner()
    transport = _Server(config.name, "adoption")
    handle = LiveMCPServer(config.name, transport, retirement_owner=retirement_owner)
    manager = ReconnectionManager(retirement_owner=retirement_owner)
    manager.bind(config=config, server=handle)

    monkeypatch.setattr(
        mcp_pkg.MCPServerManager,
        "list_servers",
        AsyncMock(return_value=[config]),
    )
    monkeypatch.setattr(mcp_pkg, "_load_plugin_mcp_configs", lambda: [])
    monkeypatch.setattr(
        "koder_agent.harness.channels.state.get_allowed_channels",
        lambda: [],
    )
    monkeypatch.setattr(
        mcp_pkg.MCPServerFactory,
        "create_and_connect_with_retry",
        AsyncMock(return_value=(handle, manager)),
    )
    monkeypatch.setattr(
        MCPServerSet,
        "adopt_server",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("adoption failed")),
    )

    loaded = await mcp_pkg.load_mcp_servers()
    assert loaded == []
    assert manager.server is None

    await asyncio.sleep(0)
    await drain_orphaned_retirements()
    assert transport.cleanup_count == 1
    await close_mcp_servers(loaded)


@pytest.mark.parametrize(
    "scope",
    [MCPServerScope.USER, MCPServerScope.LOCAL],
    ids=["user-list", "local-list"],
)
@pytest.mark.parametrize(
    ("names", "expected_error"),
    [
        (("dup", "dup"), "Duplicate MCP server name 'dup'"),
        (("dup.name", "dup_name"), "Duplicate public MCP server identity 'dup_name'"),
    ],
    ids=["exact-duplicate", "normalized-collision"],
)
@pytest.mark.asyncio
async def test_real_manager_configured_list_collisions_fail_before_connection(
    monkeypatch,
    tmp_path,
    scope,
    names,
    expected_error,
):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = tmp_path / ".koder" / "config.yaml"
    monkeypatch.setattr(ConfigManager, "DEFAULT_CONFIG_PATH", config_path)
    reset_config_manager()

    entries = [
        {
            "name": names[0],
            "transport_type": "stdio",
            "command": "python-first",
            "args": ["server.py"],
        },
        {
            "name": names[1],
            "transport_type": "stdio",
            "command": "python-second",
            "args": ["server.py"],
        },
    ]
    _write_runtime_config(
        tmp_path,
        _configured_list_payload(scope, project, entries),
    )

    old_owner, old_handle, old_manager, old_transport = _owned_runtime("old", "old")
    factory = AsyncMock()
    monkeypatch.setattr(mcp_pkg, "_load_plugin_mcp_configs", lambda: [])
    monkeypatch.setattr(
        mcp_pkg.MCPServerFactory,
        "create_and_connect_with_retry",
        factory,
    )

    try:
        with pytest.raises(RuntimeError, match=expected_error):
            await mcp_pkg.load_mcp_servers()

        assert factory.await_count == 0
        assert mcp_pkg.get_reconnection_managers(old_owner) == {"old": old_manager}
        assert old_handle._koder_current_transport() is old_transport
        assert old_transport.cleanup_count == 0
    finally:
        await close_mcp_servers(old_owner)
        reset_config_manager()

    assert old_transport.cleanup_count == 1
