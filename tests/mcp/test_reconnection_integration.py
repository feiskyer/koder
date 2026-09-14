"""Integration tests for MCP reconnection in server manager."""

import asyncio
import gc
import weakref
from unittest.mock import MagicMock, patch

import pytest

import koder_agent.mcp as mcp_module
from koder_agent.mcp.reconnection import (
    ReconnectionConfig,
    ReconnectionManager,
    drain_orphaned_retirements,
)
from koder_agent.mcp.server_config import MCPServerConfig, MCPServerScope, MCPServerType
from koder_agent.mcp.server_factory import MCPServerFactory


class MockMCPServer:
    """Mock MCP server for testing."""

    def __init__(self, name: str, should_fail: int = 0):
        self.name = name
        self._connect_count = 0
        self._should_fail = should_fail
        self.session = MagicMock()
        self.server_initialize_result = MagicMock()
        self.cleanup_count = 0

    async def connect(self):
        """Mock connect that can simulate failures."""
        self._connect_count += 1
        if self._connect_count <= self._should_fail:
            raise ConnectionError(f"Mock connection failure {self._connect_count}")

    async def cleanup(self):
        """Mock cleanup."""
        self.cleanup_count += 1


def _config(name: str) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        transport_type=MCPServerType.STDIO,
        command="echo",
        args=[name],
        env_vars={},
        scope=MCPServerScope.USER,
        source_path="/tmp/test",
    )


@pytest.mark.asyncio
async def test_extra_server_factory_cleans_partial_connections_on_failure():
    class FatalFactoryFailure(BaseException):
        pass

    class Server:
        def __init__(self, name, *, failure=None):
            self.name = name
            self.failure = failure
            self.cleanup_calls = 0

        async def connect(self):
            if self.failure is not None:
                raise self.failure

        async def cleanup(self):
            self.cleanup_calls += 1

    first = Server("first")
    second = Server("second", failure=FatalFactoryFailure("fatal connect"))
    servers = iter([first, second])

    async def create_server(*_args, **_kwargs):
        return next(servers)

    with patch.object(MCPServerFactory, "create_server", create_server):
        with pytest.raises(FatalFactoryFailure, match="fatal connect"):
            await MCPServerFactory.create_servers_from_configs(
                [_config("first"), _config("second")]
            )

    assert first.cleanup_calls == 1
    assert second.cleanup_calls == 1


@pytest.mark.asyncio
async def test_extra_server_factory_cleans_partial_connections_on_cancellation():
    second_started = asyncio.Event()

    class Server:
        def __init__(self, name, *, blocks=False):
            self.name = name
            self.blocks = blocks
            self.cleanup_calls = 0

        async def connect(self):
            if self.blocks:
                second_started.set()
                await asyncio.Event().wait()

        async def cleanup(self):
            self.cleanup_calls += 1

    first = Server("first")
    second = Server("second", blocks=True)
    servers = iter([first, second])

    async def create_server(*_args, **_kwargs):
        return next(servers)

    with patch.object(MCPServerFactory, "create_server", create_server):
        task = asyncio.create_task(
            MCPServerFactory.create_servers_from_configs([_config("first"), _config("second")])
        )
        await second_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert first.cleanup_calls == 1
    assert second.cleanup_calls == 1


@pytest.mark.asyncio
async def test_extra_server_factory_cleans_failed_current_server_once_when_cleanup_cancelled():
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()

    class Server:
        def __init__(self, name, *, connect_error=None, blocks_cleanup=False):
            self.name = name
            self.connect_error = connect_error
            self.blocks_cleanup = blocks_cleanup
            self.cleanup_calls = 0

        async def connect(self):
            if self.connect_error is not None:
                raise self.connect_error

        async def cleanup(self):
            self.cleanup_calls += 1
            if self.blocks_cleanup:
                cleanup_started.set()
                await allow_cleanup.wait()

    first = Server("first")
    second = Server(
        "second",
        connect_error=RuntimeError("connect failed"),
        blocks_cleanup=True,
    )
    servers = iter([first, second])

    async def create_server(*_args, **_kwargs):
        return next(servers)

    with patch.object(MCPServerFactory, "create_server", create_server):
        task = asyncio.create_task(
            MCPServerFactory.create_servers_from_configs([_config("first"), _config("second")])
        )
        await cleanup_started.wait()
        task.cancel()
        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert first.cleanup_calls == 1
    assert second.cleanup_calls == 1


@pytest.mark.asyncio
async def test_reconnection_manager_created_per_server():
    """ReconnectionManager should be created per server."""
    # Create a reconnection manager for a server
    reconnection_mgr = ReconnectionManager(
        ReconnectionConfig(initial_delay=0.01, max_delay=0.05, max_attempts=3)
    )

    assert reconnection_mgr.attempt_count == 0
    assert reconnection_mgr.should_retry()


@pytest.mark.asyncio
async def test_successful_reconnection():
    """Reconnection should succeed after transient failures."""
    mock_server = MockMCPServer("test-server", should_fail=2)

    async def connect_fn():
        await mock_server.connect()

    reconnection_mgr = ReconnectionManager(
        ReconnectionConfig(initial_delay=0.01, max_delay=0.05, max_attempts=5)
    )

    success = await reconnection_mgr.reconnect_with_backoff(connect_fn)
    assert success
    assert mock_server._connect_count == 3  # Failed 2, succeeded on 3rd


@pytest.mark.asyncio
async def test_reconnection_resets_failure_counter():
    """Successful reconnection should reset the failure counter."""
    mock_server = MockMCPServer("test-server", should_fail=1)

    async def connect_fn():
        await mock_server.connect()

    reconnection_mgr = ReconnectionManager(
        ReconnectionConfig(initial_delay=0.01, max_delay=0.05, max_attempts=5)
    )

    # First reconnection succeeds after 1 failure
    success = await reconnection_mgr.reconnect_with_backoff(connect_fn)
    assert success
    assert reconnection_mgr.attempt_count == 2

    # Reset should clear the counter
    reconnection_mgr.reset()
    assert reconnection_mgr.attempt_count == 0


@pytest.mark.asyncio
async def test_max_failures_marks_server_unavailable():
    """After max failures, reconnection should give up."""
    mock_server = MockMCPServer("test-server", should_fail=999)

    async def connect_fn():
        await mock_server.connect()

    reconnection_mgr = ReconnectionManager(
        ReconnectionConfig(initial_delay=0.01, max_delay=0.05, max_attempts=3)
    )

    success = await reconnection_mgr.reconnect_with_backoff(connect_fn)
    assert not success
    assert mock_server._connect_count == 3  # Tried max_attempts times


@pytest.mark.asyncio
async def test_server_factory_with_retry_success():
    """ServerFactory.create_and_connect_with_retry should succeed with transient errors."""
    config = MCPServerConfig(
        name="test-server",
        transport_type=MCPServerType.STDIO,
        command="echo",
        args=["test"],
        env_vars={},
        scope=MCPServerScope.USER,
        source_path="/tmp/test",
    )

    call_count = 0
    mock_server = MockMCPServer("test-server")

    async def mock_create_server(cfg, channel_callback=None, *, trusted=True):
        return mock_server

    # Patch server creation to fail first 2 times
    async def failing_create(cfg, channel_callback=None, *, trusted=True):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ConnectionError("Transient failure")
        return await mock_create_server(cfg, channel_callback)

    with patch.object(MCPServerFactory, "create_server", failing_create):
        reconnection_config = ReconnectionConfig(initial_delay=0.01, max_delay=0.05, max_attempts=5)

        server, reconnection_mgr = await MCPServerFactory.create_and_connect_with_retry(
            config, reconnection_config=reconnection_config
        )

        assert server is not None
        assert server.name == "test-server"
        assert reconnection_mgr.attempt_count == 3  # 2 failures + 1 success


@pytest.mark.asyncio
async def test_server_factory_with_retry_max_failures():
    """ServerFactory.create_and_connect_with_retry should raise after max attempts."""
    config = MCPServerConfig(
        name="test-server",
        transport_type=MCPServerType.STDIO,
        command="echo",
        args=["test"],
        env_vars={},
        scope=MCPServerScope.USER,
        source_path="/tmp/test",
    )

    async def always_fail(cfg, channel_callback=None, *, trusted=True):
        raise ConnectionError("Always fails")

    with patch.object(MCPServerFactory, "create_server", always_fail):
        reconnection_config = ReconnectionConfig(initial_delay=0.01, max_delay=0.05, max_attempts=3)

        with pytest.raises(ConnectionError, match="after 3 attempts"):
            await MCPServerFactory.create_and_connect_with_retry(
                config, reconnection_config=reconnection_config
            )


@pytest.mark.asyncio
async def test_server_factory_returns_reconnection_manager():
    """ServerFactory should return a reconnection manager for future use."""
    config = MCPServerConfig(
        name="test-server",
        transport_type=MCPServerType.STDIO,
        command="echo",
        args=["test"],
        env_vars={},
        scope=MCPServerScope.USER,
        source_path="/tmp/test",
    )

    mock_server = MockMCPServer("test-server")

    async def mock_create_server(cfg, channel_callback=None, *, trusted=True):
        return mock_server

    with patch.object(MCPServerFactory, "create_server", mock_create_server):
        reconnection_config = ReconnectionConfig(initial_delay=0.01, max_attempts=3)

        server, reconnection_mgr = await MCPServerFactory.create_and_connect_with_retry(
            config, reconnection_config=reconnection_config
        )

        assert reconnection_mgr is not None
        assert isinstance(reconnection_mgr, ReconnectionManager)
        assert reconnection_mgr.config.initial_delay == 0.01
        assert reconnection_mgr.config.max_attempts == 3


@pytest.mark.asyncio
async def test_server_factory_cleans_partial_server_when_connect_is_cancelled():
    config = MCPServerConfig(
        name="cancelled-server",
        transport_type=MCPServerType.STDIO,
        command="echo",
        args=["test"],
        env_vars={},
        scope=MCPServerScope.USER,
        source_path="/tmp/test",
    )
    connecting = asyncio.Event()
    never = asyncio.Event()

    class BlockingServer(MockMCPServer):
        def __init__(self):
            super().__init__("cancelled-server")
            self.cleaned = False

        async def connect(self):
            connecting.set()
            await never.wait()

        async def cleanup(self):
            self.cleaned = True

    server = BlockingServer()

    async def mock_create_server(cfg, channel_callback=None, *, trusted=True):
        return server

    with patch.object(MCPServerFactory, "create_server", mock_create_server):
        construction = asyncio.create_task(
            MCPServerFactory.create_and_connect_with_retry(
                config,
                reconnection_config=ReconnectionConfig(initial_delay=0.01, max_attempts=1),
            )
        )
        await asyncio.wait_for(connecting.wait(), timeout=1)
        construction.cancel()
        with pytest.raises(asyncio.CancelledError):
            await construction

    assert server.cleaned is True


@pytest.mark.asyncio
async def test_extra_mcp_factory_cleans_completed_servers_when_cancelled():
    configs = [
        MCPServerConfig(
            name=name,
            transport_type=MCPServerType.STDIO,
            command="echo",
            args=["test"],
            env_vars={},
            scope=MCPServerScope.USER,
            source_path="/tmp/test",
        )
        for name in ("ready", "blocked")
    ]
    blocking = asyncio.Event()
    never = asyncio.Event()

    class OwnedServer(MockMCPServer):
        def __init__(self, name):
            super().__init__(name)
            self.cleaned = False

        async def connect(self):
            if self.name == "blocked":
                blocking.set()
                await never.wait()

        async def cleanup(self):
            self.cleaned = True

    servers = {name: OwnedServer(name) for name in ("ready", "blocked")}

    async def mock_create_server(config, *_args, **_kwargs):
        return servers[config.name]

    with patch.object(MCPServerFactory, "create_server", mock_create_server):
        construction = asyncio.create_task(MCPServerFactory.create_servers_from_configs(configs))
        await asyncio.wait_for(blocking.wait(), timeout=1)
        construction.cancel()
        with pytest.raises(asyncio.CancelledError):
            await construction

    assert servers["ready"].cleaned is True
    assert servers["blocked"].cleaned is True


@pytest.mark.asyncio
async def test_load_mcp_servers_preserves_initial_cancel_and_clears_managers(
    monkeypatch, cancellation_observer
):
    observe, cancellations = cancellation_observer
    config = MCPServerConfig(
        name="stale-manager",
        transport_type=MCPServerType.STDIO,
        command="echo",
        args=["test"],
        env_vars={},
        scope=MCPServerScope.USER,
        source_path="/tmp/test",
    )
    prompt_discovery_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_calls = 0
    cleanup_completed = 0

    class Session:
        async def list_prompts(self):
            prompt_discovery_started.set()
            await asyncio.Event().wait()

    class OwnedServer(MockMCPServer):
        def __init__(self):
            super().__init__(config.name)
            self.session = Session()

        async def cleanup(self):
            nonlocal cleanup_calls, cleanup_completed
            cleanup_calls += 1
            cleanup_started.set()
            await cleanup_release.wait()
            cleanup_completed += 1

    class FakeManager:
        async def list_servers(self, cwd=None):
            return [config]

    server = OwnedServer()
    reconnection_manager = object()

    class FakeFactory:
        @staticmethod
        async def create_and_connect_with_retry(*args, **kwargs):
            return server, reconnection_manager

    owners = []
    owner_type = mcp_module.MCPServerSet

    def tracked_owner(*args, **kwargs):
        owner = owner_type(*args, **kwargs)
        owners.append(owner)
        return owner

    monkeypatch.setattr(mcp_module, "MCPServerManager", FakeManager)
    monkeypatch.setattr(mcp_module, "MCPServerFactory", FakeFactory)
    monkeypatch.setattr(mcp_module, "MCPServerSet", tracked_owner)
    monkeypatch.setattr(mcp_module, "_load_plugin_mcp_configs", lambda: [])
    monkeypatch.setattr("koder_agent.harness.channels.state.get_allowed_channels", lambda: [])

    load_task = asyncio.create_task(observe(mcp_module.load_mcp_servers()))
    try:
        await asyncio.wait_for(prompt_discovery_started.wait(), timeout=1)
        assert len(owners) == 1
        managers = mcp_module.get_reconnection_managers(owners[0])
        assert list(managers) == ["stale-manager"]

        load_task.cancel("initial-load-cancel")
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        load_task.cancel("repeat-load-cancel")
        await asyncio.sleep(0)
        cleanup_release.set()

        with pytest.raises(asyncio.CancelledError):
            await load_task
        manager_keys_after = list(managers)
    finally:
        cleanup_release.set()
        if not load_task.done():
            load_task.cancel()
            await asyncio.gather(load_task, return_exceptions=True)

    assert cleanup_calls == 1
    assert cleanup_completed == 1
    assert [error.args for error in cancellations] == [("initial-load-cancel",)]
    assert manager_keys_after == []


@pytest.mark.asyncio
async def test_extra_mcp_factory_transfers_failed_server_before_cleanup_cancellation(
    cancellation_observer,
):
    observe, cancellations = cancellation_observer
    config = MCPServerConfig(
        name="failed-server",
        transport_type=MCPServerType.STDIO,
        command="echo",
        args=["test"],
        env_vars={},
        scope=MCPServerScope.USER,
        source_path="/tmp/test",
    )
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_calls = 0
    cleanup_completed = 0

    class FailedServer(MockMCPServer):
        async def connect(self):
            raise RuntimeError("connection failed")

        async def cleanup(self):
            nonlocal cleanup_calls, cleanup_completed
            cleanup_calls += 1
            cleanup_started.set()
            await cleanup_release.wait()
            cleanup_completed += 1

    server = FailedServer(config.name)

    async def mock_create_server(_config, *_args, **_kwargs):
        return server

    with patch.object(MCPServerFactory, "create_server", mock_create_server):
        construction = asyncio.create_task(
            observe(MCPServerFactory.create_servers_from_configs([config]))
        )
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        construction.cancel("cancel-during-error-cleanup")
        await asyncio.sleep(0)
        cleanup_release.set()

        with pytest.raises(asyncio.CancelledError):
            await construction

    await drain_orphaned_retirements()
    assert cleanup_calls == 1
    assert cleanup_completed == 1
    assert [error.args for error in cancellations] == [("cancel-during-error-cleanup",)]


@pytest.mark.asyncio
async def test_failed_runtime_reconnect_keeps_old_server_and_cleans_candidate():
    config = MCPServerConfig(
        name="test-server",
        transport_type=MCPServerType.STDIO,
        command="echo",
        args=["test"],
        env_vars={},
        scope=MCPServerScope.USER,
        source_path="/tmp/test",
    )
    old = MockMCPServer("test-server")
    candidate = MockMCPServer("test-server", should_fail=999)
    created = iter([old, candidate])

    async def mock_create_server(cfg, channel_callback=None, *, trusted=True):
        return next(created)

    with patch.object(MCPServerFactory, "create_server", mock_create_server):
        handle, manager = await MCPServerFactory.create_and_connect_with_retry(
            config,
            reconnection_config=ReconnectionConfig(
                initial_delay=0,
                max_delay=0,
                max_attempts=1,
            ),
        )
        old.session = None

        assert await manager.reconnect_if_needed() is False
        assert handle._koder_current_transport() is old
        assert old.cleanup_count == 0
        assert candidate.cleanup_count == 1


@pytest.mark.asyncio
async def test_connection_construction_failure_retries_candidate_cleanup_before_raising():
    config = MCPServerConfig(
        name="test-server",
        transport_type=MCPServerType.STDIO,
        command="echo",
        args=["test"],
        env_vars={},
        scope=MCPServerScope.USER,
        source_path="/tmp/test",
    )

    class FailedConstructionServer(MockMCPServer):
        async def connect(self):
            raise ConnectionError("connect failed after construction")

        async def cleanup(self):
            self.cleanup_count += 1
            if self.cleanup_count == 1:
                raise RuntimeError("first cleanup failed")

    candidate = FailedConstructionServer("test-server")

    async def mock_create_server(cfg, channel_callback=None, *, trusted=True):
        return candidate

    with (
        patch.object(MCPServerFactory, "create_server", mock_create_server),
        pytest.raises(ConnectionError, match="Failed to connect"),
    ):
        await MCPServerFactory.create_and_connect_with_retry(
            config,
            reconnection_config=ReconnectionConfig(
                initial_delay=0,
                max_delay=0,
                max_attempts=1,
            ),
        )

    assert candidate.cleanup_count == 2


@pytest.mark.asyncio
async def test_repeated_construction_cleanup_failure_stays_owned_until_explicit_orphan_drain():
    config = MCPServerConfig(
        name="test-server",
        transport_type=MCPServerType.STDIO,
        command="echo",
        args=["test"],
        env_vars={},
        scope=MCPServerScope.USER,
        source_path="/tmp/test",
    )

    class RepeatedlyFailingConstructionServer(MockMCPServer):
        def __init__(self, name):
            super().__init__(name)
            self.allow_cleanup = False

        async def connect(self):
            raise ConnectionError("connect failed after construction")

        async def cleanup(self):
            self.cleanup_count += 1
            if not self.allow_cleanup:
                raise RuntimeError("cleanup still failing")

    candidate = RepeatedlyFailingConstructionServer("test-server")
    candidate_ref = weakref.ref(candidate)
    created = [candidate]

    async def mock_create_server(cfg, channel_callback=None, *, trusted=True):
        return created.pop()

    with (
        patch.object(MCPServerFactory, "create_server", mock_create_server),
        pytest.raises(ConnectionError, match="Failed to connect"),
    ):
        await MCPServerFactory.create_and_connect_with_retry(
            config,
            reconnection_config=ReconnectionConfig(
                initial_delay=0,
                max_delay=0,
                max_attempts=1,
            ),
        )

    assert candidate.cleanup_count == 4
    del candidate
    gc.collect()
    retained = candidate_ref()
    assert retained is not None

    retained.allow_cleanup = True
    await drain_orphaned_retirements()

    assert retained.cleanup_count == 5
    await drain_orphaned_retirements()
    assert retained.cleanup_count == 5
