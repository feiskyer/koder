"""Integration tests for agent team runtime features."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from koder_agent.harness.agents.definitions import AgentDefinition
from koder_agent.harness.agents.service import AgentService
from koder_agent.harness.agents.teams.in_process import InProcessTeammateRunner
from koder_agent.harness.agents.teams.permission_bridge import (
    PermissionBridge,
    PermissionRequest,
    PermissionResponse,
)
from koder_agent.harness.agents.teams.service import TeamService
from koder_agent.tools.agent import _agent_tool_impl
from koder_agent.tools.fork_agent import build_fork_context


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test data."""
    return tmp_path


@pytest.fixture
def agent_service(temp_dir):
    """Create an agent service for testing."""
    return AgentService.for_test(root=temp_dir)


@pytest.fixture
def team_service(temp_dir):
    """Create a team service for testing."""
    return TeamService.for_test(root=temp_dir)


@pytest.fixture
def agent_definition():
    """Basic agent definition for testing."""
    return AgentDefinition(
        agent_type="test-agent",
        when_to_use="Test agent for integration tests",
        system_prompt="You are a test agent",
        source="built-in",
        tools=["read_file", "write_file"],
        model="gpt-4o-mini",
    )


class TestForkContext:
    """Test fork context integration."""

    def test_build_fork_context_with_messages(self):
        """Test that fork context is built from parent messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        fork_ctx = build_fork_context(messages)
        assert fork_ctx.system_prompt == "You are a helpful assistant"
        assert len(fork_ctx.conversation_messages) == 2

    def test_build_fork_context_empty(self):
        """Test fork context with no messages."""
        fork_ctx = build_fork_context([])
        assert fork_ctx.system_prompt is None
        assert len(fork_ctx.conversation_messages) == 0

    def test_fork_context_to_messages(self):
        """Test converting fork context back to messages."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "User message"},
        ]
        fork_ctx = build_fork_context(messages)
        reconstructed = fork_ctx.to_messages()
        assert len(reconstructed) == 2
        assert reconstructed[0]["role"] == "system"
        assert reconstructed[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_agent_tool_with_fork_context(
        self, agent_definition, agent_service, tmp_path, monkeypatch
    ):
        """A real parent snapshot reaches the scoped service's child execution."""
        import json

        from koder_agent.core.session import EnhancedSQLiteSession
        from koder_agent.harness.agents.runtime_context import (
            agent_service_scope,
            agent_session_scope,
        )

        history = [
            {"role": "system", "content": "Test system"},
            {"role": "user", "content": "Test user"},
        ]
        received = []

        async def execute(**kwargs):
            received.extend(kwargs["seed_items"])
            return "Test result"

        monkeypatch.setattr("koder_agent.harness.agents.service._execute_agent_run", execute)
        parent = EnhancedSQLiteSession("fork-parent", db_path=str(tmp_path / "parent.db"))
        try:
            await parent.add_items(history)
            with patch(
                "koder_agent.harness.agents.definitions.get_agent_definitions"
            ) as mock_get_defs:
                mock_defs = MagicMock()
                mock_defs.active_agents = [agent_definition]
                mock_get_defs.return_value = mock_defs
                with agent_service_scope(agent_service), agent_session_scope(parent):
                    result = json.loads(
                        await _agent_tool_impl(
                            description="Test task",
                            prompt="Do something",
                            subagent_type="test-agent",
                            context="fork",
                        )
                    )
            assert result["status"] == "completed"
            assert result["result"] == "Test result"
            assert received == history
        finally:
            await agent_service.aclose()
            parent.close()

    @pytest.mark.asyncio
    async def test_agent_tool_without_fork_context(self, agent_definition):
        """Test that agent_tool works without fork context."""
        with (
            patch("koder_agent.harness.agents.definitions.get_agent_definitions") as mock_get_defs,
            patch("koder_agent.harness.agents.service.AgentService") as mock_service_class,
        ):
            # Setup mocks
            mock_defs = MagicMock()
            mock_defs.active_agents = [agent_definition]
            mock_get_defs.return_value = mock_defs

            mock_service = AsyncMock()
            mock_service.run_sync = AsyncMock(return_value="Test result")
            mock_service_class.return_value = mock_service

            # Call agent_tool without fork context
            await _agent_tool_impl(
                description="Test task",
                prompt="Do something",
                subagent_type="test-agent",
            )

            # Verify seed_items is None
            assert mock_service.run_sync.called
            call_kwargs = mock_service.run_sync.call_args.kwargs
            # seed_items should be None or not passed
            assert call_kwargs.get("seed_items") is None


class TestPermissionBridge:
    """Test permission bridge integration."""

    @pytest.mark.asyncio
    async def test_permission_bridge_creation(self):
        """Test creating a permission bridge with handler."""

        async def handler(req: PermissionRequest) -> PermissionResponse:
            return PermissionResponse(
                request_id=req.request_id, approved=True, reason="Test approved"
            )

        bridge = PermissionBridge(handler=handler)
        assert bridge is not None

    @pytest.mark.asyncio
    async def test_permission_bridge_request(self):
        """Test making a permission request through the bridge."""

        async def handler(req: PermissionRequest) -> PermissionResponse:
            return PermissionResponse(request_id=req.request_id, approved=True, reason="Approved")

        bridge = PermissionBridge(handler=handler)
        response = await bridge.request_permission(
            worker_name="worker-1",
            tool_name="run_shell",
            arguments={"command": "echo test"},
            reason="test command",
        )
        assert response.approved is True
        assert response.reason == "Approved"

    @pytest.mark.asyncio
    async def test_in_process_runner_with_permission_bridge(
        self, agent_service, team_service, agent_definition
    ):
        """Test that InProcessTeammateRunner accepts a permission bridge."""

        async def handler(req: PermissionRequest) -> PermissionResponse:
            return PermissionResponse(request_id=req.request_id, approved=True)

        bridge = PermissionBridge(handler=handler)
        runner = InProcessTeammateRunner(
            agent_service=agent_service,
            team_service=team_service,
            permission_bridge=bridge,
        )
        assert runner._permission_bridge == bridge

    @pytest.mark.asyncio
    async def test_in_process_runner_without_permission_bridge(self, agent_service, team_service):
        """Test that InProcessTeammateRunner works without a permission bridge."""
        runner = InProcessTeammateRunner(
            agent_service=agent_service,
            team_service=team_service,
        )
        assert runner._permission_bridge is None


class TestTmuxBackend:
    """Test tmux backend integration."""

    def test_create_backend_tmux(self):
        """Test creating a tmux backend."""
        from koder_agent.harness.agents.teams.runtime import create_backend

        with (
            patch("koder_agent.harness.agents.teams.runtime.is_tmux_available") as mock_available,
            patch("koder_agent.harness.agents.teams.runtime.TmuxBackend") as mock_backend_class,
        ):
            mock_available.return_value = True
            mock_backend = MagicMock()
            mock_backend_class.return_value = mock_backend

            backend = create_backend("tmux", "test-team")
            assert backend is not None
            assert mock_backend_class.called

    def test_create_backend_tmux_unavailable(self):
        """Test creating a tmux backend when tmux is not available."""
        from koder_agent.harness.agents.teams.runtime import create_backend

        with patch("koder_agent.harness.agents.teams.runtime.is_tmux_available") as mock_available:
            mock_available.return_value = False

            with pytest.raises(RuntimeError, match="tmux is not available"):
                create_backend("tmux", "test-team")

    def test_create_backend_in_process(self):
        """Test that create_backend returns None for in-process mode."""
        from koder_agent.harness.agents.teams.runtime import create_backend

        backend = create_backend("in-process", "test-team")
        assert backend is None

    def test_create_backend_auto(self):
        """Test that create_backend returns None for auto mode."""
        from koder_agent.harness.agents.teams.runtime import create_backend

        backend = create_backend("auto", "test-team")
        assert backend is None


class TestBackwardCompatibility:
    """Test backward compatibility."""

    @pytest.mark.asyncio
    async def test_agent_tool_without_context_parameter(self, agent_definition):
        """Test that agent_tool works without the context parameter (backward compat)."""
        with (
            patch("koder_agent.harness.agents.definitions.get_agent_definitions") as mock_get_defs,
            patch("koder_agent.harness.agents.service.AgentService") as mock_service_class,
        ):
            mock_defs = MagicMock()
            mock_defs.active_agents = [agent_definition]
            mock_get_defs.return_value = mock_defs

            mock_service = AsyncMock()
            mock_service.run_sync = AsyncMock(return_value="Test result")
            mock_service_class.return_value = mock_service

            # Call without context parameter
            await _agent_tool_impl(
                description="Test task",
                prompt="Do something",
                subagent_type="test-agent",
            )

            # Should still work
            assert mock_service.run_sync.called
            call_kwargs = mock_service.run_sync.call_args.kwargs
            assert call_kwargs.get("seed_items") is None
