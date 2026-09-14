"""Scheduler model-call preflight behavior."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from koder_agent.core.turn_cancellation import current_turn_cancellation_scope
from koder_agent.harness.memory.auto_compact import AutoCompactManager


@asynccontextmanager
async def _scheduler_with_history(history):
    with (
        patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
        patch("koder_agent.core.scheduler.get_display_hooks"),
        patch("koder_agent.core.scheduler.ApprovalHooks"),
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as session_cls,
        patch("koder_agent.core.scheduler.get_companion", return_value=None),
    ):
        session = AsyncMock()
        session.session_id = "preflight"
        session.db_path = ":memory:"
        session.get_items = AsyncMock(return_value=history)
        session._estimate_tokens = MagicMock(
            side_effect=lambda items: sum(len(str(item.get("content", ""))) for item in items)
        )
        session_cls.return_value = session

        from koder_agent.core.scheduler import AgentScheduler

        scheduler = AgentScheduler(session_id="preflight")
        scheduler.dev_agent = SimpleNamespace(instructions="system", tools=[])
        scheduler._agent_initialized = True
        scheduler._migration_done = True
        scheduler._auto_compact = AutoCompactManager(context_window=100, max_output_tokens=20)
        scheduler._capture_usage = AsyncMock()
        scheduler._refresh_magic_docs_after_turn = AsyncMock()
        scheduler._repair_unreplayable_session_items = AsyncMock()
        scheduler._finish_goal_turn = AsyncMock()
        scheduler._estimate_instruction_context_tokens = MagicMock(return_value=10)
        scheduler._estimate_tool_schema_tokens = MagicMock(return_value=10)
        scheduler._estimate_run_input_tokens = MagicMock(return_value=10)
        try:
            yield scheduler, session
        finally:
            await scheduler.cleanup()


@pytest.mark.asyncio
async def test_scheduler_compacts_history_before_provider_call():
    events = []
    history = [{"role": "user", "content": "h" * 60}]

    async with _scheduler_with_history(history) as (scheduler, session):

        async def compact_once():
            events.append("compact")
            session.get_items.return_value = [{"role": "user", "content": "h" * 10}]

        scheduler._run_auto_compact = AsyncMock(side_effect=compact_once)

        class Result:
            final_output = "ok"
            context_wrapper = None

        async def run_provider(*_args, **_kwargs):
            events.append("provider")
            return Result()

        with patch("koder_agent.core.scheduler.Runner.run", side_effect=run_provider):
            response = await scheduler.handle("current", render_output=False)

    assert response == "ok"
    assert events == ["compact", "provider"]
    scheduler._run_auto_compact.assert_awaited_once()


@pytest.mark.asyncio
async def test_scheduler_rejects_impossible_input_without_provider_or_compaction():
    async with _scheduler_with_history([]) as (scheduler, _session):
        scheduler._estimate_instruction_context_tokens.return_value = 15
        scheduler._estimate_tool_schema_tokens.return_value = 15
        scheduler._estimate_run_input_tokens.return_value = 60
        scheduler._run_auto_compact = AsyncMock()

        provider = AsyncMock()
        with patch("koder_agent.core.scheduler.Runner.run", provider):
            response = await scheduler.handle("oversized", render_output=False)

    provider.assert_not_awaited()
    scheduler._run_auto_compact.assert_not_awaited()
    assert "cannot fit" in response.lower()
    assert "context window=100" in response
    assert "instructions=15" in response
    assert "tools=15" in response
    assert "current input=60" in response
    assert "response reserve=20" in response


@pytest.mark.asyncio
async def test_scheduler_static_estimate_includes_tool_schema_overhead():
    async with _scheduler_with_history([]) as (scheduler, _session):
        del scheduler.__dict__["_estimate_instruction_context_tokens"]
        del scheduler.__dict__["_estimate_tool_schema_tokens"]
        scheduler._static_context_tokens_cache = None
        scheduler._encode_token_count = MagicMock(side_effect=lambda text: len(text))
        scheduler.dev_agent = SimpleNamespace(
            instructions="system instructions",
            tools=[
                SimpleNamespace(
                    name="read_file",
                    description="Read a file",
                    params_json_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                )
            ],
        )

        instructions = scheduler._estimate_instruction_context_tokens()
        tools = scheduler._estimate_tool_schema_tokens()

    assert instructions == len("system instructions")
    assert tools > len("read_file") + len("Read a file")


@pytest.mark.asyncio
async def test_scheduler_counts_responses_function_call_output():
    history = [
        {"type": "function_call", "call_id": "call-1", "name": "read", "arguments": "{}"},
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": "x" * 100_000,
        },
    ]
    async with _scheduler_with_history(history) as (scheduler, _session):
        estimated = await scheduler._estimate_session_tokens()

    assert estimated > 10_000


@pytest.mark.asyncio
async def test_scheduler_counts_function_call_arguments():
    history = [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "write_file",
            "arguments": "x" * 100_000,
        },
        {"type": "function_call_output", "call_id": "call-1", "output": "ok"},
    ]
    async with _scheduler_with_history(history) as (scheduler, _session):
        estimated = await scheduler._estimate_session_tokens()

    assert estimated > 10_000


@pytest.mark.asyncio
async def test_cancellation_scope_precedes_retrieval_preflight_and_compaction():
    class StreamingUI:
        callback = None

        def set_cancel_callback(self, callback):
            self.callback = callback

    ui = StreamingUI()
    async with _scheduler_with_history([]) as (scheduler, _session):

        async def verify_scope(*_args, **_kwargs):
            scope = current_turn_cancellation_scope()
            assert scope is not None
            assert ui.callback is not None
            ui.callback()
            scope.raise_if_cancelled()

        scheduler._run_turn_unlocked = AsyncMock(side_effect=verify_scope)
        response = await scheduler._handle_unlocked(
            "hello",
            render_output=False,
            streaming_ui=ui,
        )

    assert response.startswith("Operation cancelled")
    assert ui.callback is None
