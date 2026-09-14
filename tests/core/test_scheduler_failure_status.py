"""Pre-provider rejection is a failed turn, not a completed goal step."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from koder_agent.harness.memory.budget import ContextPreflightError, ContextPreflightEstimate
from tests.core.test_scheduler_turn_lifecycle import _make_scheduler


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_json", [False, True])
@pytest.mark.parametrize("phase", ["initial_input", "history_preflight"])
async def test_context_rejection_marks_turn_and_goal_failed(monkeypatch, stream_json, phase):
    scheduler = _make_scheduler()
    scheduler.dev_agent = object()
    scheduler._ensure_agent_initialized = AsyncMock()
    scheduler._reconnect_unhealthy_mcp_servers = AsyncMock()
    scheduler._repair_unreplayable_session_items = AsyncMock()
    scheduler._load_memory_context = AsyncMock(return_value="")
    blocked = ContextPreflightEstimate(context_window=100, response_reserve=20, input_tokens=200)
    allowed = ContextPreflightEstimate(context_window=100, response_reserve=20, input_tokens=1)
    scheduler._estimate_main_call_preflight = AsyncMock(
        return_value=blocked if phase == "initial_input" else allowed
    )
    scheduler._preflight_main_model_call = AsyncMock(side_effect=ContextPreflightError(blocked))
    runner = Mock(side_effect=AssertionError("a rejected request must not reach the model"))
    monkeypatch.setattr("koder_agent.core.scheduler.Runner.run", runner)
    monkeypatch.setattr("koder_agent.core.scheduler.Runner.run_streamed", runner)

    try:
        events = []
        if stream_json:
            response = await scheduler.handle_stream_json("oversized input", on_event=events.append)
            assert events[-1]["type"] == "error"
        else:
            response = await scheduler.handle("oversized input", render_output=False)

        assert "cannot fit" in response
        assert scheduler._last_turn_errored is True
        assert scheduler._last_turn_cancelled is False
        assert scheduler.goal_runtime.on_turn_end.await_args.kwargs == {
            "error": True,
            "cancelled": False,
        }
        runner.assert_not_called()
    finally:
        await scheduler.goal_store.close()


@pytest.mark.asyncio
async def test_missing_agent_marks_failure_without_printing_in_quiet_mode(capsys):
    scheduler = _make_scheduler()
    scheduler._ensure_agent_initialized = AsyncMock()
    scheduler._reconnect_unhealthy_mcp_servers = AsyncMock()
    try:
        response = await scheduler.handle("request", render_output=False)
        assert response == "Agent not initialized"
        assert scheduler._last_turn_errored is True
        assert scheduler.goal_runtime.on_turn_end.await_args.kwargs["error"] is True
        assert not capsys.readouterr().out
    finally:
        await scheduler.goal_store.close()


@pytest.mark.asyncio
async def test_reported_cost_limit_marks_turn_failed(monkeypatch):
    scheduler = _make_scheduler()
    scheduler.dev_agent = object()
    scheduler._ensure_agent_initialized = AsyncMock()
    scheduler._reconnect_unhealthy_mcp_servers = AsyncMock()
    scheduler._repair_unreplayable_session_items = AsyncMock()
    scheduler.session.get_items = AsyncMock(return_value=[{"role": "user", "content": "prior"}])
    allowed = ContextPreflightEstimate(context_window=100, response_reserve=20, input_tokens=1)
    scheduler._estimate_main_call_preflight = AsyncMock(return_value=allowed)
    scheduler._preflight_main_model_call = AsyncMock(return_value=allowed)
    scheduler._capture_usage = AsyncMock()
    scheduler._check_session_cost_limit = lambda: "session cost limit reached"
    scheduler._runtime_config_service = SimpleNamespace(load=lambda: SimpleNamespace())
    monkeypatch.setattr("koder_agent.core.scheduler.get_companion", lambda _config: None)
    monkeypatch.setattr(
        "koder_agent.core.scheduler.Runner.run",
        AsyncMock(return_value=SimpleNamespace(final_output="finished")),
    )
    try:
        response = await scheduler.handle("request", render_output=False)
        assert response == "session cost limit reached"
        assert scheduler._last_turn_errored is True
        assert scheduler.goal_runtime.on_turn_end.await_args.kwargs["error"] is True
    finally:
        await scheduler.goal_store.close()
