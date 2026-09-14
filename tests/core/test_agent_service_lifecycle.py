"""Scheduler and app cleanup must own the service shared with model tools."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from koder_agent.core.scheduler import AgentScheduler, _GoalTurnLifecycle
from koder_agent.harness.agents.definitions import AgentDefinition
from koder_agent.harness.agents.runtime_context import get_runtime_agent_service
from koder_agent.harness.agents.service import AgentService
from koder_agent.harness.session_flow import _SchedulerBuilder, _SessionCleanupOwner


def make_scheduler(session_id, service=None):
    with (
        patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
        patch("koder_agent.core.scheduler.get_display_hooks"),
        patch("koder_agent.core.scheduler.ApprovalHooks"),
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as session_class,
    ):
        session = AsyncMock()
        session.session_id = session_id
        session.db_path = ":memory:"
        session.get_items.return_value = []
        session_class.return_value = session
        scheduler = AgentScheduler(session_id=session_id, agent_service=service)
    scheduler.goal_runtime = AsyncMock()
    scheduler.goal_runtime.next_continuation_prompt.return_value = None
    return scheduler


@pytest.fixture
def profile(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "koder_agent.harness.agents.service._redacted_model_config_snapshot", lambda _: {}
    )
    return tmp_path


@pytest.mark.asyncio
async def test_scheduler_binds_service_lazily_and_closes_it(profile):
    scheduler = make_scheduler("owned")
    assert scheduler._agent_service is None
    started, closed = asyncio.Event(), asyncio.Event()

    async def execute(**_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    async with _GoalTurnLifecycle(scheduler):
        service = get_runtime_agent_service()
        assert service is scheduler.get_agent_service()
        await service.launch_background(
            agent_definition=AgentDefinition("worker", "tests", "synthetic", "built-in"),
            prompt="run",
            description="owned",
            cwd=profile,
            executor=execute,
        )
        await asyncio.wait_for(started.wait(), timeout=3)
    await scheduler.cleanup()
    assert closed.is_set()
    assert service.is_closed


@pytest.mark.asyncio
async def test_shared_app_service_survives_scheduler_retirement_and_closes_after_runner(profile):
    service = AgentService.for_test(profile)
    first, second = make_scheduler("first", service), make_scheduler("second", service)
    async with _GoalTurnLifecycle(first):
        assert get_runtime_agent_service() is service
    await first.cleanup()
    assert not service.is_closed
    async with _GoalTurnLifecycle(second):
        assert get_runtime_agent_service() is service

    order = []
    state = SimpleNamespace(cleanup=AsyncMock(), session_id="app")
    owner = _SessionCleanupOwner(state, bare_mode=False, previous_simple=None)
    owner._dispatch_session_end = Mock()
    owner._run_auto_dream = AsyncMock()
    owner.agent_service = service
    close_service = service.aclose

    async def close():
        order.append("service")
        await close_service()

    async def close_runner():
        order.append("runner")

    service.aclose = close
    owner.teammate_runner = SimpleNamespace(aclose=close_runner)
    await owner.finish()
    assert order == ["runner", "service"]
    assert service.is_closed
    await second.cleanup()


def test_scheduler_builder_passes_the_shared_app_owner(profile):
    shared = object()

    class Scheduler:
        def __init__(self, session_id, agent_service=None):
            self.agent_service = agent_service

    builder = _SchedulerBuilder(
        scheduler_type=Scheduler,
        streaming=False,
        agent_definition=None,
        instructions_override=None,
        instructions_append=None,
        permission_service=None,
        approver=None,
        agent_definitions=None,
        agent_service=shared,
    )
    assert builder.build("one").agent_service is shared
    assert builder.build("two").agent_service is shared
