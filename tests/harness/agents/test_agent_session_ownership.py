"""A subagent owns its SDK session and must close it on every outcome."""

import asyncio
from types import SimpleNamespace

import pytest

from koder_agent.harness.agents import service
from koder_agent.harness.agents.definitions import AgentDefinition
from koder_agent.harness.agents.runtime_context import get_runtime_agent_session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["success", "error", "cancel", "cleanup-error", "error-and-cleanup-error"]
)
async def test_subagent_closes_its_actual_session(tmp_path, monkeypatch, outcome):
    monkeypatch.setenv("HOME", str(tmp_path))
    captured = []

    async def create_agent(*_args, **_kwargs):
        return SimpleNamespace()

    async def cleanup(*_args, **_kwargs):
        if outcome in {"cleanup-error", "error-and-cleanup-error"}:
            raise RuntimeError("synthetic MCP cleanup failure")
        return None

    async def run(_agent, _prompt, *, session, **_kwargs):
        captured.append(session)
        assert get_runtime_agent_session() is session
        await session.add_items([{"role": "user", "content": "owned child history"}])
        if outcome in {"error", "error-and-cleanup-error"}:
            raise RuntimeError("synthetic child failure")
        if outcome == "cancel":
            raise asyncio.CancelledError
        return SimpleNamespace(final_output="child result")

    monkeypatch.setattr(service, "create_dev_agent", create_agent)
    monkeypatch.setattr(service, "_cleanup_agent_mcp_servers", cleanup)
    monkeypatch.setattr(service, "get_all_tools", lambda: [])
    monkeypatch.setattr(service, "get_display_hooks", lambda: SimpleNamespace())
    monkeypatch.setattr(service.Runner, "run", run)
    definition = AgentDefinition(
        agent_type="owned-child",
        when_to_use="test",
        system_prompt="Synthetic",
        source="built-in",
    )
    try:
        invocation = service._execute_agent_run(
            agent_definition=definition,
            prompt="test",
            session_id="owned-child",
            seed_items=None,
            cwd=str(tmp_path),
        )
        if outcome == "success":
            assert await invocation == "child result"
        elif outcome in {"error", "error-and-cleanup-error"}:
            with pytest.raises(RuntimeError, match="synthetic child failure"):
                await invocation
        elif outcome == "cleanup-error":
            with pytest.raises(RuntimeError, match="synthetic MCP cleanup failure"):
                await invocation
        else:
            with pytest.raises(asyncio.CancelledError):
                await invocation
        assert len(captured) == 1
        assert captured[0]._closed, "subagent returned without closing its SQLite session"
        assert get_runtime_agent_session() is None
    finally:
        for session in captured:
            session.close()
