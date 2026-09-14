from __future__ import annotations

import asyncio
import gc
import shutil
import subprocess
import sys
import weakref
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import koder_agent.mcp as mcp_pkg
from koder_agent.core.scheduler import AgentScheduler
from koder_agent.harness.plugins.context import get_plugin_root
from koder_agent.mcp.server_config import MCPServerConfig, MCPServerType


class _SnapshotLoads:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.paths: list[Path] = []
        self.close_counts: list[int] = []

    @contextmanager
    def configs(self):
        index = len(self.paths)
        snapshot = self.root / f"snapshot-{index}"
        snapshot.mkdir()
        command = snapshot / "server.py"
        command.write_text("# private plugin copy\n", encoding="utf-8")
        self.paths.append(snapshot)
        self.close_counts.append(0)
        try:
            yield [
                MCPServerConfig(
                    name=f"plugin-{index}",
                    transport_type=MCPServerType.STDIO,
                    command=str(command),
                    source_path=str(snapshot / ".mcp.json"),
                )
            ]
        finally:
            self.close_counts[index] += 1
            shutil.rmtree(snapshot, ignore_errors=True)


class _Server:
    def __init__(self, config: MCPServerConfig) -> None:
        self.name = config.name
        self.session = None
        self.snapshot = Path(config.command).parent
        self.cleanup_count = 0

    async def cleanup(self) -> None:
        self.cleanup_count += 1


class _PromptSession:
    def __init__(self, prompt_name: str, *, wait_event: asyncio.Event | None = None) -> None:
        self.prompt_name = prompt_name
        self.wait_event = wait_event
        self.started = asyncio.Event()

    async def list_prompts(self):
        self.started.set()
        if self.wait_event is not None:
            await self.wait_event.wait()
        prompt = SimpleNamespace(name=self.prompt_name, description=self.prompt_name, arguments=[])
        return SimpleNamespace(prompts=[prompt])


class _AdvertisingSession(_PromptSession):
    def __init__(self, label: str, events: list[str]) -> None:
        super().__init__(f"{label}_prompt")
        self.label = label
        self.events = events

    async def list_prompts(self):
        self.events.append(f"prompt:{self.label}")
        return await super().list_prompts()

    async def list_resources(self):
        self.events.append(f"resource:{self.label}")
        resource = SimpleNamespace(
            uri=f"demo://{self.label}",
            description=self.label,
            name=self.label,
        )
        return SimpleNamespace(resources=[resource])


@contextmanager
def _two_plugin_configs(root: Path, close_counts: list[int]):
    snapshot = root / "two-server-snapshot"
    snapshot.mkdir()
    close_counts.append(0)
    try:
        yield [
            MCPServerConfig(
                name=name,
                transport_type=MCPServerType.STDIO,
                command=str(snapshot / f"{name}.py"),
                source_path=str(snapshot / ".mcp.json"),
            )
            for name in ("first", "second")
        ]
    finally:
        close_counts[0] += 1
        shutil.rmtree(snapshot, ignore_errors=True)


@contextmanager
def _empty_configs():
    yield []


@contextmanager
def _static_plugin_configs(configs: list[MCPServerConfig], close_events: list[str]):
    try:
        yield configs
    finally:
        close_events.append("closed")


@pytest.fixture
def snapshot_loads(tmp_path, monkeypatch):
    loads = _SnapshotLoads(tmp_path)

    async def no_configured_servers(self, cwd=None):
        return []

    async def create_server(config, **kwargs):
        return _Server(config), SimpleNamespace(reconnect_if_needed=None)

    monkeypatch.setattr(mcp_pkg, "_load_plugin_mcp_configs", loads.configs)
    monkeypatch.setattr(mcp_pkg.MCPServerManager, "list_servers", no_configured_servers)
    monkeypatch.setattr(
        mcp_pkg.MCPServerFactory,
        "create_and_connect_with_retry",
        create_server,
    )
    return loads


@pytest.mark.asyncio
async def test_sequential_overlapping_loads_keep_each_snapshot_alive(snapshot_loads):
    first = await mcp_pkg.load_mcp_servers()
    second = await mcp_pkg.load_mcp_servers()

    assert first[0].snapshot.is_dir()
    assert second[0].snapshot.is_dir()
    assert first[0].snapshot != second[0].snapshot

    await mcp_pkg.close_mcp_servers(first)
    assert not first[0].snapshot.exists()
    assert second[0].snapshot.is_dir()

    await mcp_pkg.close_mcp_servers(second)
    assert snapshot_loads.close_counts == [1, 1]


@pytest.mark.asyncio
async def test_concurrent_overlapping_loads_keep_each_snapshot_alive(snapshot_loads, monkeypatch):
    both_creating = asyncio.Event()
    create_count = 0

    async def create_server(config, **kwargs):
        nonlocal create_count
        create_count += 1
        if create_count == 2:
            both_creating.set()
        await asyncio.wait_for(both_creating.wait(), timeout=1)
        return _Server(config), SimpleNamespace(reconnect_if_needed=None)

    monkeypatch.setattr(
        mcp_pkg.MCPServerFactory,
        "create_and_connect_with_retry",
        create_server,
    )

    first, second = await asyncio.gather(
        mcp_pkg.load_mcp_servers(),
        mcp_pkg.load_mcp_servers(),
    )

    assert first[0].snapshot.is_dir()
    assert second[0].snapshot.is_dir()
    assert first[0].snapshot != second[0].snapshot

    await mcp_pkg.close_mcp_servers(second)
    assert first[0].snapshot.is_dir()
    await mcp_pkg.close_mcp_servers(first)
    assert snapshot_loads.close_counts == [1, 1]


@pytest.mark.asyncio
async def test_real_load_rejects_exact_configured_plugin_identity_before_advertisement(
    monkeypatch,
):
    configured = MCPServerConfig(
        name="shared",
        transport_type=MCPServerType.STDIO,
        command="configured.py",
        source_path="configured.json",
    )
    plugin = MCPServerConfig(
        name="shared",
        transport_type=MCPServerType.STDIO,
        command="plugin.py",
        source_path="plugin/.mcp.json",
    )
    connection_events: list[str] = []
    advertisement_events: list[str] = []
    snapshot_events: list[str] = []

    async def configured_servers(self, cwd=None):
        return [configured]

    async def create_server(config, **kwargs):
        label = Path(config.command).stem
        connection_events.append(label)
        server = _Server(config)
        server.session = _AdvertisingSession(label, advertisement_events)
        return server, SimpleNamespace(reconnect_if_needed=None)

    monkeypatch.setattr(mcp_pkg.MCPServerManager, "list_servers", configured_servers)
    monkeypatch.setattr(
        mcp_pkg,
        "_load_plugin_mcp_configs",
        lambda: _static_plugin_configs([plugin], snapshot_events),
    )
    monkeypatch.setattr(
        mcp_pkg.MCPServerFactory,
        "create_and_connect_with_retry",
        create_server,
    )

    with pytest.raises(RuntimeError, match="Duplicate public MCP server identity 'shared'"):
        loaded = await mcp_pkg.load_mcp_servers()
        await mcp_pkg.discover_mcp_resources(loaded)
        await loaded.aclose()

    assert connection_events == []
    assert advertisement_events == []
    assert snapshot_events == ["closed"]


@pytest.mark.asyncio
async def test_real_load_rejects_normalized_configured_plugin_identity_before_connection(
    monkeypatch,
):
    configured = MCPServerConfig(
        name="alpha.beta",
        transport_type=MCPServerType.STDIO,
        command="configured.py",
        source_path="configured.json",
    )
    plugin = MCPServerConfig(
        name="alpha_beta",
        transport_type=MCPServerType.STDIO,
        command="plugin.py",
        source_path="plugin/.mcp.json",
    )
    connection_events: list[str] = []
    snapshot_events: list[str] = []

    async def configured_servers(self, cwd=None):
        return [configured]

    async def create_server(config, **kwargs):
        connection_events.append(config.name)
        return _Server(config), SimpleNamespace(reconnect_if_needed=None)

    monkeypatch.setattr(mcp_pkg.MCPServerManager, "list_servers", configured_servers)
    monkeypatch.setattr(
        mcp_pkg,
        "_load_plugin_mcp_configs",
        lambda: _static_plugin_configs([plugin], snapshot_events),
    )
    monkeypatch.setattr(
        mcp_pkg.MCPServerFactory,
        "create_and_connect_with_retry",
        create_server,
    )

    with pytest.raises(RuntimeError, match="Duplicate public MCP server identity 'alpha_beta'"):
        await mcp_pkg.load_mcp_servers()

    assert connection_events == []
    assert snapshot_events == ["closed"]


@pytest.mark.asyncio
async def test_real_load_rejects_case_only_configured_plugin_identity_before_advertisement(
    monkeypatch,
):
    configured = MCPServerConfig(
        name="Foo",
        transport_type=MCPServerType.STDIO,
        command="configured.py",
        source_path="configured.json",
    )
    plugin = MCPServerConfig(
        name="foo",
        transport_type=MCPServerType.STDIO,
        command="plugin.py",
        source_path="plugin/.mcp.json",
    )
    connection_events: list[str] = []
    advertisement_events: list[str] = []
    snapshot_events: list[str] = []

    async def configured_servers(self, cwd=None):
        return [configured]

    async def create_server(config, **kwargs):
        label = Path(config.command).stem
        connection_events.append(label)
        server = _Server(config)
        server.session = _AdvertisingSession(label, advertisement_events)
        return server, SimpleNamespace(reconnect_if_needed=None)

    monkeypatch.setattr(mcp_pkg.MCPServerManager, "list_servers", configured_servers)
    monkeypatch.setattr(
        mcp_pkg,
        "_load_plugin_mcp_configs",
        lambda: _static_plugin_configs([plugin], snapshot_events),
    )
    monkeypatch.setattr(
        mcp_pkg.MCPServerFactory,
        "create_and_connect_with_retry",
        create_server,
    )

    with pytest.raises(RuntimeError, match="Duplicate public MCP server identity 'foo'"):
        loaded = await mcp_pkg.load_mcp_servers()
        await mcp_pkg.discover_mcp_resources(loaded)
        await loaded.aclose()

    assert connection_events == []
    assert advertisement_events == []
    assert snapshot_events == ["closed"]


@pytest.mark.asyncio
async def test_snapshot_cleanup_is_exactly_once_and_abandoned_sets_are_finalized(
    snapshot_loads,
):
    owned = await mcp_pkg.load_mcp_servers()
    snapshot = owned[0].snapshot

    await mcp_pkg.close_mcp_servers(owned)
    await mcp_pkg.close_mcp_servers(owned)
    assert not snapshot.exists()
    assert snapshot_loads.close_counts == [1]

    abandoned = await mcp_pkg.load_mcp_servers()
    abandoned_snapshot = abandoned[0].snapshot
    abandoned_ref = weakref.ref(abandoned)
    abandoned_server = abandoned[0]
    del abandoned
    gc.collect()
    await mcp_pkg.drain_orphaned_mcp_owners()

    assert abandoned_ref() is None
    assert not abandoned_snapshot.exists()
    assert abandoned_server.cleanup_count == 1
    assert snapshot_loads.close_counts == [1, 1]


@pytest.mark.asyncio
async def test_cancelled_server_cleanup_retains_owner_state_and_retries_later():
    resource_closes = 0

    class CancelOnceServer:
        name = "cancel-once"

        def __init__(self):
            self.cleanup_count = 0

        async def cleanup(self):
            self.cleanup_count += 1
            if self.cleanup_count == 1:
                raise asyncio.CancelledError

    class LaterServer:
        name = "later"

        def __init__(self):
            self.cleanup_count = 0

        async def cleanup(self):
            self.cleanup_count += 1

    def close_resource():
        nonlocal resource_closes
        resource_closes += 1

    resources = ExitStack()
    resources.callback(close_resource)
    first = CancelOnceServer()
    second = LaterServer()
    owner = mcp_pkg.MCPServerSet([first, second], runtime_resources=resources)
    owner.prompt_registry.register(
        mcp_pkg.MCPPrompt(server_name=first.name, prompt_name="retained")
    )

    with pytest.raises(asyncio.CancelledError):
        await owner.aclose()

    assert first.cleanup_count == 1
    assert second.cleanup_count == 1
    assert resource_closes == 0
    assert len(owner.prompt_registry.list_prompts()) == 1

    assert await owner.aclose() is True
    assert first.cleanup_count == 2
    assert second.cleanup_count == 1
    assert resource_closes == 1
    assert owner.prompt_registry.list_prompts() == []
    assert await owner.aclose() is False


@pytest.mark.asyncio
async def test_concurrent_owner_close_callers_share_one_cleanup_operation():
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    class SlowServer:
        name = "shared"

        def __init__(self):
            self.cleanup_count = 0

        async def cleanup(self):
            self.cleanup_count += 1
            cleanup_started.set()
            await cleanup_release.wait()

    server = SlowServer()
    owner = mcp_pkg.MCPServerSet([server])
    first = asyncio.create_task(owner.aclose())
    await cleanup_started.wait()
    second = asyncio.create_task(owner.aclose())
    cleanup_release.set()

    assert await asyncio.gather(first, second) == [True, True]
    assert server.cleanup_count == 1
    assert await owner.aclose() is False


def test_abandoned_owner_survives_asyncio_run_boundary_and_repeated_drains():
    cleanup_events: list[str] = []

    class SlowServer:
        name = "slow-abandoned"

        async def cleanup(self):
            cleanup_events.append("started")
            await asyncio.sleep(0.01)
            cleanup_events.append("finished")

    async def abandon_owner():
        owner = mcp_pkg.MCPServerSet([SlowServer()])
        owner_ref = weakref.ref(owner)
        del owner
        gc.collect()
        assert owner_ref() is None

    asyncio.run(abandon_owner())
    assert cleanup_events == []

    asyncio.run(mcp_pkg.drain_orphaned_mcp_owners())
    asyncio.run(mcp_pkg.drain_orphaned_mcp_owners())

    assert cleanup_events == ["started", "finished"]


def test_cancelled_orphan_cleanup_retries_across_asyncio_run_boundaries():
    cleanup_calls: list[str] = []

    class CancelOnceServer:
        name = "cancelled-orphan"

        async def cleanup(self):
            cleanup_calls.append("called")
            if len(cleanup_calls) == 1:
                raise asyncio.CancelledError

    async def abandon_owner():
        owner = mcp_pkg.MCPServerSet([CancelOnceServer()])
        del owner
        gc.collect()

    asyncio.run(abandon_owner())
    asyncio.run(mcp_pkg.drain_orphaned_mcp_owners())

    asyncio.run(mcp_pkg.drain_orphaned_mcp_owners())
    asyncio.run(mcp_pkg.drain_orphaned_mcp_owners())

    assert cleanup_calls == ["called", "called"]


@pytest.mark.parametrize("failure_kind", ["cancel", "error"])
def test_process_exit_retries_transient_orphan_cleanup_after_loop_shutdown(
    tmp_path, failure_kind, python_child_environment
):
    marker = tmp_path / f"atexit-{failure_kind}.txt"
    script = """
import asyncio
import gc
import sys
from pathlib import Path

from koder_agent.mcp import MCPServerSet

marker = Path(sys.argv[1])
failure_kind = sys.argv[2]


def record(event):
    with marker.open("a", encoding="utf-8") as handle:
        handle.write(f"{event}\\n")


class TransientServer:
    name = "transient"

    def __init__(self):
        self.cleanup_count = 0

    async def cleanup(self):
        self.cleanup_count += 1
        record(f"transient-{self.cleanup_count}")
        if self.cleanup_count == 1:
            if failure_kind == "cancel":
                raise asyncio.CancelledError
            raise RuntimeError("transient cleanup failure")
        await asyncio.sleep(0.01)


class LaterServer:
    name = "later"

    async def cleanup(self):
        record("later")
        await asyncio.sleep(0.01)


async def abandon_before_loop_shutdown():
    owner = MCPServerSet([TransientServer(), LaterServer()])
    del owner
    gc.collect()
    assert not marker.exists()


asyncio.run(abandon_before_loop_shutdown())
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(marker), failure_kind],
        cwd=tmp_path,
        env=python_child_environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker.read_text(encoding="utf-8").splitlines() == [
        "transient-1",
        "later",
        "transient-2",
    ]


@pytest.mark.asyncio
async def test_load_drains_abandoned_owner_before_creating_new_owner(monkeypatch):
    cleanup_count = 0

    class Server:
        name = "load-drained"

        async def cleanup(self):
            nonlocal cleanup_count
            cleanup_count += 1

    async def no_configured_servers(self, cwd=None):
        return []

    abandoned = mcp_pkg.MCPServerSet([Server()])
    del abandoned
    gc.collect()
    monkeypatch.setattr(mcp_pkg.MCPServerManager, "list_servers", no_configured_servers)
    monkeypatch.setattr(mcp_pkg, "_load_plugin_mcp_configs", lambda: _empty_configs())

    loaded = await mcp_pkg.load_mcp_servers()

    assert cleanup_count == 1
    assert loaded == []
    await loaded.aclose()


@pytest.mark.asyncio
async def test_construction_failure_does_not_leak_snapshot(snapshot_loads, monkeypatch):
    async def fail_construction(config, **kwargs):
        raise RuntimeError("construction failed")

    monkeypatch.setattr(
        mcp_pkg.MCPServerFactory,
        "create_and_connect_with_retry",
        fail_construction,
    )

    servers = await mcp_pkg.load_mcp_servers()

    assert servers == []
    assert not snapshot_loads.paths[0].exists()
    assert snapshot_loads.close_counts == [1]


@pytest.mark.asyncio
async def test_scheduler_cleanup_is_isolated_and_order_independent(snapshot_loads):
    first = await mcp_pkg.load_mcp_servers()
    second = await mcp_pkg.load_mcp_servers()
    first_scheduler = AgentScheduler.__new__(AgentScheduler)
    first_scheduler._mcp_servers = first
    first_scheduler.dev_agent = object()
    first_scheduler._agent_initialized = True
    second_scheduler = AgentScheduler.__new__(AgentScheduler)
    second_scheduler._mcp_servers = second
    second_scheduler.dev_agent = object()
    second_scheduler._agent_initialized = True

    await second_scheduler.reset_agent()
    await second_scheduler.reset_agent()
    assert first[0].snapshot.is_dir()
    assert not second[0].snapshot.exists()
    assert second[0].cleanup_count == 1
    assert snapshot_loads.close_counts == [0, 1]

    await first_scheduler.reset_agent()
    assert not first[0].snapshot.exists()
    assert first[0].cleanup_count == 1
    assert snapshot_loads.close_counts == [1, 1]


@pytest.mark.asyncio
async def test_scheduler_retains_exact_returned_server_owner(monkeypatch):
    owner = mcp_pkg.MCPServerSet([object()])
    agent = SimpleNamespace(
        mcp_servers=[],
        _koder_mcp_servers=owner,
        model=SimpleNamespace(context_window=128_000),
        model_settings=SimpleNamespace(max_tokens=4_096),
    )

    async def create_agent(*args, **kwargs):
        return agent

    scheduler = AgentScheduler.__new__(AgentScheduler)
    scheduler._migration_done = True
    scheduler._agent_initialized = False
    scheduler.agent_definition = None
    scheduler.instructions_override = None
    scheduler.instructions_append = None
    scheduler.tools = []
    scheduler.plugin_root = get_plugin_root()
    scheduler._mcp_servers = []
    monkeypatch.setattr("koder_agent.core.scheduler.create_dev_agent", create_agent)
    monkeypatch.setattr("koder_agent.core.scheduler.get_model_name", lambda: "gpt-4o")

    await scheduler._ensure_agent_initialized()

    assert scheduler._mcp_servers is owner


@pytest.mark.asyncio
async def test_scheduler_partial_initialization_closes_unattached_agent_owner(monkeypatch):
    class Server:
        name = "scheduler-owned"

        def __init__(self):
            self.cleanup_count = 0

        async def cleanup(self):
            self.cleanup_count += 1

    server = Server()
    owner = mcp_pkg.MCPServerSet([server])
    agent = SimpleNamespace(
        mcp_servers=[],
        _koder_mcp_servers=owner,
        model=SimpleNamespace(context_window=None),
        model_settings=SimpleNamespace(max_tokens=None),
    )

    async def create_agent(*args, **kwargs):
        return agent

    scheduler = AgentScheduler.__new__(AgentScheduler)
    scheduler._migration_done = True
    scheduler._agent_initialized = False
    scheduler.agent_definition = None
    scheduler.instructions_override = None
    scheduler.instructions_append = None
    scheduler.tools = []
    scheduler.plugin_root = get_plugin_root()
    scheduler._mcp_servers = []
    monkeypatch.setattr("koder_agent.core.scheduler.create_dev_agent", create_agent)
    monkeypatch.setattr("koder_agent.core.scheduler.get_model_name", lambda: "gpt-4o")
    monkeypatch.setattr(
        "koder_agent.core.scheduler.get_configured_context_window",
        lambda _model: (_ for _ in ()).throw(RuntimeError("context failed")),
    )

    with pytest.raises(RuntimeError, match="context failed"):
        await scheduler._ensure_agent_initialized()

    assert server.cleanup_count == 1
    assert agent._koder_mcp_servers is None
    assert scheduler._mcp_servers == []


@pytest.mark.asyncio
async def test_cancellation_during_later_connection_closes_adopted_server_and_snapshot(
    tmp_path, monkeypatch
):
    close_counts: list[int] = []
    second_started = asyncio.Event()
    never = asyncio.Event()
    first_server = None

    async def no_configured_servers(self, cwd=None):
        return []

    async def create_server(config, **kwargs):
        nonlocal first_server
        if config.name == "second":
            second_started.set()
            await never.wait()
        server = _Server(config)
        if config.name == "first":
            first_server = server
        return server, SimpleNamespace(reconnect_if_needed=None)

    monkeypatch.setattr(
        mcp_pkg,
        "_load_plugin_mcp_configs",
        lambda: _two_plugin_configs(tmp_path, close_counts),
    )
    monkeypatch.setattr(mcp_pkg.MCPServerManager, "list_servers", no_configured_servers)
    monkeypatch.setattr(
        mcp_pkg.MCPServerFactory,
        "create_and_connect_with_retry",
        create_server,
    )

    task = asyncio.create_task(mcp_pkg.load_mcp_servers())
    await second_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert first_server is not None
    assert first_server.cleanup_count == 1
    assert close_counts == [1]
    assert not (tmp_path / "two-server-snapshot").exists()


@pytest.mark.asyncio
async def test_cancellation_during_prompt_discovery_closes_owner(tmp_path, monkeypatch):
    loads = _SnapshotLoads(tmp_path)
    prompt_release = asyncio.Event()
    server = None

    async def no_configured_servers(self, cwd=None):
        return []

    async def create_server(config, **kwargs):
        nonlocal server
        server = _Server(config)
        server.session = _PromptSession("blocked", wait_event=prompt_release)
        return server, SimpleNamespace(reconnect_if_needed=None)

    monkeypatch.setattr(mcp_pkg, "_load_plugin_mcp_configs", loads.configs)
    monkeypatch.setattr(mcp_pkg.MCPServerManager, "list_servers", no_configured_servers)
    monkeypatch.setattr(
        mcp_pkg.MCPServerFactory,
        "create_and_connect_with_retry",
        create_server,
    )

    task = asyncio.create_task(mcp_pkg.load_mcp_servers())
    while server is None:
        await asyncio.sleep(0)
    await server.session.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert server.cleanup_count == 1
    assert loads.close_counts == [1]


@pytest.mark.asyncio
async def test_owner_close_finishes_once_when_caller_is_cancelled():
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    class SlowServer:
        name = "slow"

        def __init__(self):
            self.cleanup_count = 0

        async def cleanup(self):
            self.cleanup_count += 1
            cleanup_started.set()
            await cleanup_release.wait()

    server = SlowServer()
    owner = mcp_pkg.MCPServerSet([server])
    close_task = asyncio.create_task(mcp_pkg.close_mcp_servers(owner))
    await cleanup_started.wait()
    close_task.cancel()
    cleanup_release.set()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert server.cleanup_count == 1
    assert await mcp_pkg.close_mcp_servers(owner) is False


@pytest.mark.asyncio
async def test_live_owner_prompts_coexist_and_cleanup_is_owner_scoped(snapshot_loads, monkeypatch):
    load_index = 0

    async def create_server(config, **kwargs):
        nonlocal load_index
        server = _Server(config)
        server.session = _PromptSession(f"prompt-{load_index}")
        load_index += 1
        return server, SimpleNamespace(reconnect_if_needed=None)

    monkeypatch.setattr(
        mcp_pkg.MCPServerFactory,
        "create_and_connect_with_retry",
        create_server,
    )
    first = await mcp_pkg.load_mcp_servers()
    second = await mcp_pkg.load_mcp_servers()

    assert [prompt.prompt_name for prompt in first.prompt_registry.list_prompts()] == ["prompt-0"]
    assert [prompt.prompt_name for prompt in second.prompt_registry.list_prompts()] == ["prompt-1"]

    await mcp_pkg.close_mcp_servers(second)
    assert second.prompt_registry.list_prompts() == []
    assert [prompt.prompt_name for prompt in first.prompt_registry.list_prompts()] == ["prompt-0"]
    await mcp_pkg.close_mcp_servers(first)
