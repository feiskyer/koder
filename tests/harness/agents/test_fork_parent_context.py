"""Forks must copy the actual active parent, or fail before launching a child."""

import asyncio
import json

import pytest
from agents.tool_context import ToolContext

from koder_agent.core.scheduler import AgentScheduler, _GoalTurnLifecycle
from koder_agent.core.session import EnhancedSQLiteSession
from koder_agent.harness.agents.runtime_context import agent_service_scope
from koder_agent.harness.agents.service import AgentService
from koder_agent.tools.agent import _agent_tool_impl, agent_tool


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True], ids=["sync", "background"])
async def test_fork_inherits_active_parent_database(tmp_path, monkeypatch, background):
    seen = []

    async def execute(**kwargs):
        seen.append(kwargs["seed_items"])
        return "synthetic child result"

    monkeypatch.setattr("koder_agent.harness.agents.service._execute_agent_run", execute)
    monkeypatch.setattr(
        "koder_agent.harness.agents.service._redacted_model_config_snapshot", lambda _: {}
    )
    scheduler = AgentScheduler(session_id="same-id", streaming=False)
    service = AgentService.for_test(tmp_path)
    scheduler._agent_service = service
    await scheduler.session.add_items([{"role": "user", "content": "wrong default database"}])
    scheduler.session.close()
    scheduler.session = EnhancedSQLiteSession("same-id", str(tmp_path / "actual-parent.db"))
    history = [
        {"role": "user", "content": "actual parent conversation"},
        {"type": "function_call", "call_id": "completed", "name": "read_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "completed", "output": "parent file content"},
        {"type": "function_call", "call_id": "pending", "name": "read_file", "arguments": "{}"},
    ]
    await scheduler.session.add_items(history)
    try:
        async with _GoalTurnLifecycle(scheduler):
            arguments = json.dumps(
                {
                    "description": "Fork parent",
                    "prompt": "Continue the analysis",
                    "context": "fork",
                    "run_in_background": background,
                }
            )
            tool_context = ToolContext(
                context=None,
                tool_name="agent_tool",
                tool_call_id="fork-parent-call",
                tool_arguments=arguments,
            )
            result = json.loads(await agent_tool.on_invoke_tool(tool_context, arguments))
            if background:
                assert result["status"] == "async_launched"
                assert (await service.wait(result["agent_id"])).state == "completed"
            else:
                assert result["status"] == "completed"
        assert seen == [history[:3]]
        assert await scheduler.session.get_items() == history
    finally:
        await scheduler.cleanup()
        await service.aclose()


@pytest.mark.asyncio
async def test_fork_without_active_parent_does_not_launch(tmp_path, monkeypatch):
    seen = []

    async def execute(**kwargs):
        seen.append(kwargs)
        return "must not execute"

    monkeypatch.setattr("koder_agent.harness.agents.service._execute_agent_run", execute)
    service = AgentService.for_test(tmp_path)
    try:
        with agent_service_scope(service):
            result = json.loads(
                await _agent_tool_impl(
                    description="Missing parent", prompt="Do work", context="fork"
                )
            )
        assert result["status"] == "error"
        assert "parent" in result["error"].lower()
        assert seen == []
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_fork_read_failure_does_not_fall_back_to_empty_context(tmp_path, monkeypatch):
    seen = []

    async def execute(**kwargs):
        seen.append(kwargs)
        return "must not execute"

    async def failed_read(*_args, **_kwargs):
        raise OSError("synthetic parent read failure")

    monkeypatch.setattr("koder_agent.harness.agents.service._execute_agent_run", execute)
    scheduler = AgentScheduler(session_id="unreadable-parent", streaming=False)
    service = AgentService.for_test(tmp_path)
    scheduler._agent_service = service
    monkeypatch.setattr(scheduler.session, "get_items", failed_read)
    try:
        async with _GoalTurnLifecycle(scheduler):
            result = json.loads(
                await _agent_tool_impl(
                    description="Unreadable parent", prompt="Do work", context="fork"
                )
            )
        assert result["status"] == "error"
        assert seen == []
    finally:
        await scheduler.cleanup()
        await service.aclose()


@pytest.mark.asyncio
async def test_fork_does_not_outlive_the_parent_turn_binding(tmp_path, monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    seen = []

    async def execute(**kwargs):
        seen.append(kwargs)
        return "must not execute"

    async def delayed_read(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return [{"role": "user", "content": "retired parent"}]

    monkeypatch.setattr("koder_agent.harness.agents.service._execute_agent_run", execute)
    scheduler = AgentScheduler(session_id="retired-parent", streaming=False)
    service = AgentService.for_test(tmp_path)
    scheduler._agent_service = service
    monkeypatch.setattr(scheduler.session, "get_items", delayed_read)
    task = None
    try:
        async with _GoalTurnLifecycle(scheduler):
            task = asyncio.create_task(
                _agent_tool_impl(description="Late fork", prompt="Do work", context="fork")
            )
            await asyncio.wait_for(entered.wait(), 5)
        release.set()
        result = json.loads(await task)
        assert result["status"] == "error"
        assert seen == []
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await scheduler.cleanup()
        await service.aclose()


@pytest.mark.asyncio
async def test_fork_of_valid_empty_parent_is_not_a_missing_parent(tmp_path, monkeypatch):
    seen = []

    async def execute(**kwargs):
        seen.append(kwargs["seed_items"])
        return "empty parent accepted"

    monkeypatch.setattr("koder_agent.harness.agents.service._execute_agent_run", execute)
    scheduler = AgentScheduler(session_id="empty-parent", streaming=False)
    service = AgentService.for_test(tmp_path)
    scheduler._agent_service = service
    try:
        async with _GoalTurnLifecycle(scheduler):
            result = json.loads(
                await _agent_tool_impl(description="Empty fork", prompt="Do work", context="fork")
            )
        assert result["status"] == "completed"
        assert seen == [[]]
    finally:
        await scheduler.cleanup()
        await service.aclose()
