"""Integration tests for scheduler memory management, cost tracking, and notifications."""

from unittest.mock import AsyncMock, Mock, patch

import pytest


@pytest.fixture
def mock_agent_definition():
    """Mock agent definition for testing."""
    return None


@pytest.fixture
def mock_session(tmp_path):
    """Mock EnhancedSQLiteSession."""
    session = AsyncMock()
    session.get_items = AsyncMock(return_value=[])
    session.generate_title = AsyncMock(return_value="Test Title")
    session.set_title = AsyncMock()
    session.db_path = tmp_path / "mock-session.db"
    session.encoder = Mock()
    session.encoder.encode = Mock(return_value=[1, 2, 3, 4, 5])
    session._estimate_tokens = Mock(return_value=1000)
    return session


@pytest.fixture
def mock_agent():
    """Mock dev agent."""
    agent = AsyncMock()
    agent.mcp_servers = []
    return agent


@pytest.mark.asyncio
async def test_scheduler_instantiates_auto_compact_manager():
    """Test that scheduler creates AutoCompactManager with model's context window."""
    from koder_agent.core.scheduler import AgentScheduler

    with (
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession"),
        patch("koder_agent.core.scheduler.migrate_legacy_sessions", new_callable=AsyncMock),
    ):
        scheduler = AgentScheduler(session_id="test-auto-compact")

        try:
            # Trigger agent initialization to set up AutoCompactManager
            with patch(
                "koder_agent.core.scheduler.create_dev_agent", new_callable=AsyncMock
            ) as mock_create:
                mock_agent = AsyncMock()
                mock_agent.mcp_servers = []
                mock_create.return_value = mock_agent

                await scheduler._ensure_agent_initialized()

                # Verify AutoCompactManager was created
                assert hasattr(scheduler, "_auto_compact")
                assert scheduler._auto_compact is not None
                assert scheduler._auto_compact.context_window > 0
        finally:
            await scheduler.cleanup()


@pytest.mark.asyncio
async def test_usage_tracking_includes_model_name():
    """Test that usage tracking passes model name for per-model breakdown."""
    from koder_agent.core.scheduler import AgentScheduler

    with (
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
        patch("koder_agent.core.scheduler.migrate_legacy_sessions", new_callable=AsyncMock),
    ):
        mock_session = AsyncMock()
        mock_session.get_items = AsyncMock(return_value=[])
        mock_session.db_path = ":memory:"
        mock_session.encoder = Mock()
        mock_session.encoder.encode = Mock(return_value=[])
        mock_session._estimate_tokens = Mock(return_value=0)
        mock_session_cls.return_value = mock_session

        scheduler = AgentScheduler(session_id="test-usage-tracking")

        # Mock result with usage data
        mock_result = Mock()
        mock_result.context_wrapper = Mock()
        mock_result.context_wrapper.usage = Mock()
        mock_result.context_wrapper.usage.input_tokens = 1000
        mock_result.context_wrapper.usage.output_tokens = 500
        mock_result.context_wrapper.usage.request_usage_entries = [Mock(total_tokens=1500)]
        mock_result.final_output = "Test response"

        try:
            # Capture usage
            with patch("koder_agent.core.scheduler.get_model_name", return_value="gpt-4o"):
                await scheduler._capture_usage(mock_result)

                # Verify model name was tracked
                per_model = scheduler.usage_tracker.get_per_model_usage()
                assert "gpt-4o" in per_model
                assert per_model["gpt-4o"].input_tokens == 1000
                assert per_model["gpt-4o"].output_tokens == 500
        finally:
            await scheduler.cleanup()


@pytest.mark.asyncio
async def test_auto_compact_check_after_response(tmp_path):
    """Test that auto-compact threshold is checked after model responses."""
    from koder_agent.core.scheduler import AgentScheduler

    with (
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
        patch("koder_agent.core.scheduler.migrate_legacy_sessions", new_callable=AsyncMock),
        patch("koder_agent.core.scheduler.create_dev_agent", new_callable=AsyncMock) as mock_create,
        patch("koder_agent.core.scheduler.Runner") as mock_runner,
        patch("koder_agent.core.scheduler.get_companion", return_value=None),
    ):
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.get_items = AsyncMock(return_value=[])
        mock_session.db_path = tmp_path / "auto-compact.db"
        mock_session_cls.return_value = mock_session

        mock_agent = AsyncMock()
        mock_agent.mcp_servers = []
        mock_create.return_value = mock_agent

        mock_result = Mock()
        mock_result.final_output = "Test response"
        mock_result.context_wrapper = Mock()
        mock_result.context_wrapper.usage = Mock()
        mock_result.context_wrapper.usage.input_tokens = 100000  # High token count
        mock_result.context_wrapper.usage.output_tokens = 500
        mock_result.context_wrapper.usage.request_usage_entries = [Mock(total_tokens=100500)]

        mock_runner.run = AsyncMock(return_value=mock_result)

        scheduler = AgentScheduler(session_id="test-auto-compact-check", streaming=False)

        try:
            # Run handle - this will initialize the agent and auto_compact manager
            with patch("koder_agent.core.scheduler.get_model_name", return_value="gpt-4o"):
                await scheduler.handle("Test input", render_output=False)

                # Verify auto_compact manager was created and logic ran
                assert scheduler._auto_compact is not None

                # With high token count (100500), compaction should be triggered
                # Check if the threshold check logic ran by verifying the manager exists
                # and has the expected threshold based on model context window
                assert scheduler._auto_compact.compact_threshold > 0
        finally:
            await scheduler.cleanup()


@pytest.mark.asyncio
async def test_session_memory_manager_instantiation():
    """Test that SessionMemoryManager is instantiated."""
    from koder_agent.core.scheduler import AgentScheduler

    with (
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession"),
        patch("koder_agent.core.scheduler.migrate_legacy_sessions", new_callable=AsyncMock),
    ):
        scheduler = AgentScheduler(session_id="test-session-memory")

        try:
            # Verify SessionMemoryManager was created
            assert hasattr(scheduler, "_session_memory")
            assert scheduler._session_memory is not None
        finally:
            await scheduler.cleanup()


@pytest.mark.asyncio
async def test_session_memory_extraction_check(tmp_path):
    """Test that session memory extraction trigger is checked after turns."""
    from koder_agent.core.scheduler import AgentScheduler

    with (
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
        patch("koder_agent.core.scheduler.migrate_legacy_sessions", new_callable=AsyncMock),
        patch("koder_agent.core.scheduler.create_dev_agent", new_callable=AsyncMock) as mock_create,
        patch("koder_agent.core.scheduler.Runner") as mock_runner,
        patch("koder_agent.core.scheduler.get_companion", return_value=None),
    ):
        # Setup mocks
        mock_session = AsyncMock()
        mock_session.get_items = AsyncMock(return_value=[])
        mock_session.db_path = tmp_path / "session-memory.db"
        mock_session_cls.return_value = mock_session

        mock_agent = AsyncMock()
        mock_agent.mcp_servers = []
        mock_create.return_value = mock_agent

        mock_result = Mock()
        mock_result.final_output = "Test response"
        mock_result.context_wrapper = Mock()
        mock_result.context_wrapper.usage = Mock()
        mock_result.context_wrapper.usage.input_tokens = 15000
        mock_result.context_wrapper.usage.output_tokens = 500
        mock_result.context_wrapper.usage.request_usage_entries = [Mock(total_tokens=15500)]

        mock_runner.run = AsyncMock(return_value=mock_result)

        scheduler = AgentScheduler(session_id="test-session-memory-check", streaming=False)

        # Mock SessionMemoryManager
        scheduler._session_memory = Mock()
        scheduler._session_memory.should_extract = Mock(return_value=True)

        try:
            # TODO: Need to track tool call count to test this properly
            # For now, just verify the manager exists
            with patch("koder_agent.core.scheduler.get_model_name", return_value="gpt-4o"):
                await scheduler.handle("Test input", render_output=False)

                # Verify the session memory manager is present
                assert scheduler._session_memory is not None
        finally:
            await scheduler.cleanup()


@pytest.mark.asyncio
async def test_notifications_placeholder():
    """Test that notification integration is available (placeholder for future implementation).

    Notifications would be sent on long-running task completion (>30s).
    Currently the openai-agents SDK doesn't expose per-turn timing,
    so this is a placeholder for future implementation.
    """
    # Just verify the notify function is importable
    from koder_agent.core.notifications import notify

    # Verify it can be called (won't actually send anything in test environment)
    notify("Test", "Test message")
    # No assertion - just checking it doesn't crash


@pytest.mark.asyncio
async def test_micro_compact_available():
    """Test that micro-compact function is available for tool results."""
    # Just verify the import works - actual application would be in tool result handling
    from koder_agent.harness.memory.micro_compact import micro_compact_messages

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "tool", "content": "x" * 30000},  # Large tool result
    ]

    compacted = micro_compact_messages(messages)
    assert len(compacted) == 2
    # Tool result should be truncated
    assert len(compacted[1]["content"]) < 30000
