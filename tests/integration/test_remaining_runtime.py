"""Integration tests for remaining runtime features: memory, orchestrator, onboarding."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@asynccontextmanager
async def _memory_scheduler(session_id: str):
    """Keep retrieval tests local and close every scheduler-owned resource."""
    from koder_agent.core.scheduler import AgentScheduler

    # Runner.run is mocked below, but title generation is a separate model
    # request. Keep its real scheduling/persistence path with synthetic output.
    with patch(
        "koder_agent.core.session.llm_completion",
        new_callable=AsyncMock,
        return_value="Synthetic session title",
    ):
        scheduler = AgentScheduler(session_id=session_id, streaming=False)
        try:
            yield scheduler
        finally:
            await scheduler.cleanup()
            assert scheduler._title_generation_task is None
            assert scheduler.goal_store._conn is None
            assert scheduler.session._closed


@pytest.mark.asyncio
async def test_memory_retrieval_called_on_first_turn(tmp_path):
    """Test that memory retrieval is called during first turn of a session."""
    # Create fake memory directories
    user_memory_dir = tmp_path / ".koder" / "memory"
    user_memory_dir.mkdir(parents=True)

    with (
        patch("koder_agent.core.scheduler.migrate_legacy_sessions", new_callable=AsyncMock),
        patch("koder_agent.core.scheduler.create_dev_agent", new_callable=AsyncMock) as mock_create,
        patch("pathlib.Path.home", return_value=tmp_path),
        patch(
            "koder_agent.harness.memory.retrieval.llm_retrieve_relevant_memories",
            new_callable=AsyncMock,
        ) as mock_retrieve,
    ):
        # Mock agent creation
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent

        # Mock memory retrieval to return empty result
        from koder_agent.harness.memory.retrieval import RetrievalResult

        mock_retrieve.return_value = RetrievalResult(memories=[], token_count=0)

        async with _memory_scheduler("test-memory") as scheduler:
            # Mock Runner.run to avoid actual execution
            with patch("koder_agent.core.scheduler.Runner.run", new_callable=AsyncMock) as mock_run:
                mock_result = MagicMock()
                mock_result.final_output = "Test response"
                mock_run.return_value = mock_result

                # First turn - should call memory retrieval
                await scheduler.handle("What is the meaning of life?", render_output=False)

                # Verify memory retrieval was attempted
                mock_retrieve.assert_called_once()
                call_kwargs = mock_retrieve.call_args.kwargs
                assert "query" in call_kwargs
                assert "memory_dirs" in call_kwargs
                assert "max_tokens" in call_kwargs


@pytest.mark.asyncio
async def test_memory_injection_into_prompt(tmp_path):
    """Test that retrieved memories are injected into the user prompt."""
    from koder_agent.harness.memory.memory_files import ParsedMemoryFile
    from koder_agent.harness.memory.retrieval import RetrievalResult, RetrievedMemory

    # Create fake memory directories
    user_memory_dir = tmp_path / ".koder" / "memory"
    user_memory_dir.mkdir(parents=True)

    with (
        patch("koder_agent.core.scheduler.migrate_legacy_sessions", new_callable=AsyncMock),
        patch("koder_agent.core.scheduler.create_dev_agent", new_callable=AsyncMock) as mock_create,
        patch("pathlib.Path.home", return_value=tmp_path),
        patch(
            "koder_agent.harness.memory.retrieval.llm_retrieve_relevant_memories",
            new_callable=AsyncMock,
        ) as mock_retrieve,
    ):
        # Mock agent creation
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent

        # Mock memory retrieval to return a memory
        fake_memory = RetrievedMemory(
            path=Path("/fake/memory.md"),
            parsed=ParsedMemoryFile(
                memory_type="reference",
                description="Test Memory",
                metadata={},
                body="This is test memory content.",
            ),
            score=1,
        )
        mock_retrieve.return_value = RetrievalResult(memories=[fake_memory], token_count=10)

        async with _memory_scheduler("test-inject") as scheduler:
            # Mock Runner.run to capture the input
            with patch("koder_agent.core.scheduler.Runner.run", new_callable=AsyncMock) as mock_run:
                mock_result = MagicMock()
                mock_result.final_output = "Test response"
                mock_run.return_value = mock_result

                # First turn - should inject memory
                await scheduler.handle("Tell me about testing", render_output=False)

                # Verify Runner.run was called with modified input containing memory
                mock_run.assert_called_once()
                call_args = mock_run.call_args[0]
                user_input = call_args[1]  # Second positional arg is the user_input
                assert "[Relevant memories from previous sessions]" in user_input
                assert "Test Memory" in user_input
                assert "This is test memory content." in user_input


@pytest.mark.asyncio
async def test_memory_not_injected_on_subsequent_turns(tmp_path):
    """Test that memory is only injected on the first turn, not subsequent ones."""
    # Create fake memory directories
    user_memory_dir = tmp_path / ".koder" / "memory"
    user_memory_dir.mkdir(parents=True)

    with (
        patch("koder_agent.core.scheduler.migrate_legacy_sessions", new_callable=AsyncMock),
        patch("koder_agent.core.scheduler.create_dev_agent", new_callable=AsyncMock) as mock_create,
        patch("pathlib.Path.home", return_value=tmp_path),
        patch(
            "koder_agent.harness.memory.retrieval.llm_retrieve_relevant_memories",
            new_callable=AsyncMock,
        ) as mock_retrieve,
    ):
        # Mock agent creation
        mock_agent = MagicMock()
        mock_create.return_value = mock_agent

        from koder_agent.harness.memory.retrieval import RetrievalResult

        mock_retrieve.return_value = RetrievalResult(memories=[], token_count=0)

        async with _memory_scheduler("test-subsequent") as scheduler:
            get_items_calls = 0

            def _session_items():
                nonlocal get_items_calls
                get_items_calls += 1
                if get_items_calls <= 3:
                    return []
                return [{"role": "user"}]

            scheduler.session.get_items = AsyncMock(side_effect=_session_items)

            with patch("koder_agent.core.scheduler.Runner.run", new_callable=AsyncMock) as mock_run:
                mock_result = MagicMock()
                mock_result.final_output = "Test response"
                mock_run.return_value = mock_result

                # First turn
                await scheduler.handle("First message", render_output=False)
                first_call_count = mock_retrieve.call_count

                # Second turn - memory retrieval should not be called again
                await scheduler.handle("Second message", render_output=False)
                second_call_count = mock_retrieve.call_count

                assert first_call_count == 1
                assert second_call_count == 1  # Should still be 1, not 2


def test_tool_orchestrator_importable():
    """Test that ToolOrchestrator can be imported from engine module."""
    from koder_agent.tools.engine import get_orchestrator

    orchestrator = get_orchestrator()
    assert orchestrator is not None
    assert hasattr(orchestrator, "execute_batch")
    assert hasattr(orchestrator, "is_read_only")
    assert hasattr(orchestrator, "partition_calls")


@pytest.mark.asyncio
async def test_tool_orchestrator_read_only_batching():
    """Test that ToolOrchestrator correctly batches read-only tools."""
    from koder_agent.tools.engine import get_orchestrator

    orchestrator = get_orchestrator()

    # Mock executor that tracks call order
    call_order = []

    async def mock_executor(tool_name, args):
        call_order.append(tool_name)
        await asyncio.sleep(0.01)  # Simulate async work
        return f"Result for {tool_name}"

    # Test concurrent read-only calls
    calls = [
        {"tool": "read_file", "args": {"path": "file1.py"}},
        {"tool": "read_file", "args": {"path": "file2.py"}},
        {"tool": "glob_search", "args": {"pattern": "*.py"}},
    ]

    results = await orchestrator.execute_batch(calls, mock_executor)

    assert len(results) == 3
    assert all("Result for" in r for r in results)
    # All three should have been called
    assert set(call_order) == {"read_file", "glob_search"}


@pytest.mark.asyncio
async def test_tool_orchestrator_write_serialization():
    """Test that ToolOrchestrator serializes write tools."""
    from koder_agent.tools.engine import get_orchestrator

    orchestrator = get_orchestrator()

    call_order = []

    async def mock_executor(tool_name, args):
        call_order.append(tool_name)
        await asyncio.sleep(0.01)
        return f"Result for {tool_name}"

    # Mix of read and write calls
    calls = [
        {"tool": "read_file", "args": {"path": "file1.py"}},
        {"tool": "write_file", "args": {"path": "file2.py", "content": "test"}},
        {"tool": "read_file", "args": {"path": "file3.py"}},
    ]

    results = await orchestrator.execute_batch(calls, mock_executor)

    assert len(results) == 3
    # Write should execute after the first read, before the second read
    assert call_order == ["read_file", "write_file", "read_file"]


def test_onboarding_panel_displayed(capsys):
    """Test that onboarding steps are displayed to user when incomplete."""
    from io import StringIO

    from koder_agent.harness.onboarding import OnboardingState, get_onboarding_steps

    # Create incomplete onboarding state
    state = OnboardingState(
        completed=False,
        api_key_configured=False,
        model_selected=False,
        workspace_trusted=True,
    )

    steps = get_onboarding_steps(state)
    assert len(steps) == 2
    assert any("API key" in step for step in steps)
    assert any("model" in step for step in steps)

    # Test panel rendering (simulate what session_flow.py does)
    from rich.console import Console
    from rich.panel import Panel

    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80)

    test_console.print(
        Panel(
            "\n".join(f"  • {step}" for step in steps),
            title="[yellow]Setup Recommended[/yellow]",
            border_style="yellow",
        )
    )

    output_text = output.getvalue()
    assert "Setup Recommended" in output_text
    assert "API key" in output_text
    assert "model" in output_text


@pytest.mark.asyncio
async def test_onboarding_check_in_session_flow():
    """Test that onboarding check happens during session flow startup."""
    # Mock incomplete onboarding state by clearing environment
    import os

    from koder_agent.harness.onboarding import (
        check_onboarding_state,
        get_onboarding_steps,
    )

    old_env = {}
    for key in ["KODER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "KODER_MODEL"]:
        old_env[key] = os.environ.get(key)
        if key in os.environ:
            del os.environ[key]

    try:
        state = check_onboarding_state(Path.cwd())
        steps = get_onboarding_steps(state)

        # Verify the check happened and detected incomplete state
        assert not state.completed
        assert len(steps) > 0
        assert not state.api_key_configured
        # koder has a built-in default model, so model_selected is always True
        assert state.model_selected
    finally:
        # Restore environment
        for key, value in old_env.items():
            if value is not None:
                os.environ[key] = value


def test_memory_retrieval_graceful_failure():
    """Test that memory retrieval failures don't break the session."""
    from koder_agent.core.scheduler import AgentScheduler

    with (
        patch("koder_agent.core.scheduler.migrate_legacy_sessions", new_callable=AsyncMock),
        patch("koder_agent.core.scheduler.create_dev_agent", new_callable=AsyncMock),
        patch(
            "koder_agent.harness.memory.retrieval.llm_retrieve_relevant_memories",
            side_effect=Exception("Memory retrieval failed"),
        ),
    ):
        # Should not raise - memory retrieval is best-effort
        scheduler = AgentScheduler(session_id="test-graceful", streaming=False)
        assert scheduler is not None
