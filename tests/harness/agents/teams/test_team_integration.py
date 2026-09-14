"""Integration tests for tmux backend and permission bridge wired into team system."""

import asyncio
from unittest.mock import MagicMock

import pytest

from koder_agent.harness.agents.teams.in_process import InProcessTeammateRunner
from koder_agent.harness.agents.teams.permission_bridge import (
    PermissionBridge,
    PermissionRequest,
    PermissionResponse,
)
from koder_agent.harness.agents.teams.runtime import (
    resolve_teammate_execution_mode,
    resolve_teammate_mode,
)
from koder_agent.harness.agents.teams.service import TeamService
from koder_agent.harness.agents.teams.tmux_backend import TmuxBackend, is_tmux_available


class TestTeammateModeTmuxResolution:
    """Test that resolve_teammate_mode handles 'tmux' correctly."""

    def test_resolve_tmux_mode_returns_tmux(self):
        """When cli_mode is 'tmux', resolve_teammate_mode should return 'tmux'."""
        mode = resolve_teammate_mode(cli_mode="tmux")
        assert mode == "tmux"

    def test_resolve_tmux_mode_from_config(self):
        """When config.teammate_mode is 'tmux', resolve_teammate_mode should return 'tmux'."""
        mock_config = MagicMock()
        mock_config.load.return_value.harness.teammate_mode = "tmux"
        mode = resolve_teammate_mode(config_service=mock_config)
        assert mode == "tmux"


class TestTmuxBackendIntegration:
    """Test that TmuxBackend can be created and used for team management."""

    def test_tmux_backend_can_be_instantiated(self):
        """TmuxBackend should be instantiable with a team name."""
        backend = TmuxBackend(session_name="test-team")
        assert backend.session_name == "test-team"

    @pytest.mark.skipif(not is_tmux_available(), reason="tmux not available")
    def test_tmux_backend_basic_lifecycle(self):
        """TmuxBackend should support basic spawn and cleanup operations."""
        backend = TmuxBackend(session_name="koder-test-integration")
        # Just verify the object can be created and cleaned up
        # Actual spawning requires a running tmux session which we handle in E2E
        try:
            backend.cleanup()
        except Exception:
            pass  # cleanup may fail if session doesn't exist


class TestPermissionBridgeIntegration:
    """Test that PermissionBridge can be wired into InProcessTeammateRunner."""

    @pytest.mark.asyncio
    async def test_permission_bridge_handler_is_called(self):
        """When a tool needs approval, the permission bridge handler should be invoked."""
        handler_called = asyncio.Event()
        request_captured = None

        async def mock_handler(req: PermissionRequest) -> PermissionResponse:
            nonlocal request_captured
            request_captured = req
            handler_called.set()
            return PermissionResponse(
                request_id=req.request_id,
                approved=True,
                reason="Auto-approved in test",
            )

        bridge = PermissionBridge(handler=mock_handler)

        # Simulate a permission request
        response = await bridge.request_permission(
            worker_name="test-worker",
            tool_name="run_shell",
            arguments={"command": "echo hi"},
            reason="needs approval",
        )

        assert handler_called.is_set()
        assert request_captured is not None
        assert request_captured.worker_name == "test-worker"
        assert request_captured.tool_name == "run_shell"
        assert response.approved is True

    def test_in_process_runner_accepts_permission_bridge(self):
        """InProcessTeammateRunner should accept an optional permission_bridge parameter."""
        mock_agent_service = MagicMock()
        mock_team_service = MagicMock()

        async def mock_handler(req: PermissionRequest) -> PermissionResponse:
            return PermissionResponse(req.request_id, approved=True)

        bridge = PermissionBridge(handler=mock_handler)

        # This should not raise - we're testing that the parameter is accepted
        runner = InProcessTeammateRunner(
            agent_service=mock_agent_service,
            team_service=mock_team_service,
            permission_bridge=bridge,
        )
        assert runner is not None
        assert runner._permission_bridge is bridge


class TestBackwardCompatibility:
    """Ensure backward compatibility when backend/bridge are not provided."""

    def test_default_mode_still_works(self):
        """Default mode resolution should still return 'auto'."""
        mode = resolve_teammate_mode()
        assert mode == "auto"

    def test_in_process_mode_still_works(self):
        """In-process mode should still be valid."""
        mode = resolve_teammate_mode(cli_mode="in-process")
        assert mode == "in-process"

    def test_auto_executes_as_in_process_mode(self):
        """Auto mode should use the live in-process team runner by default."""
        assert resolve_teammate_execution_mode("auto") == "in-process"
        assert resolve_teammate_execution_mode("in-process") == "in-process"
        assert resolve_teammate_execution_mode("tmux") == "tmux"

    def test_team_service_works_without_backend(self, tmp_path):
        """TeamService should work without explicit backend parameter."""
        service = TeamService.for_test(root=tmp_path)
        team_id = service.create_team("test-team")
        assert team_id is not None
        service.delete_team(team_id)


class TestCreateBackendFunction:
    """Test the create_backend factory function."""

    @pytest.mark.skipif(not is_tmux_available(), reason="tmux not available")
    def test_create_backend_for_tmux_mode(self):
        """create_backend should return TmuxBackend when mode is 'tmux'."""
        from koder_agent.harness.agents.teams.runtime import create_backend

        backend = create_backend(mode="tmux", team_name="test-team")
        assert isinstance(backend, TmuxBackend)
        assert backend.session_name == "koder-test-team"

    def test_create_backend_for_auto_mode(self):
        """create_backend should return None for 'auto' and 'in-process' modes."""
        from koder_agent.harness.agents.teams.runtime import create_backend

        backend_auto = create_backend(mode="auto", team_name="test-team")
        assert backend_auto is None

        backend_in_process = create_backend(mode="in-process", team_name="test-team")
        assert backend_in_process is None
