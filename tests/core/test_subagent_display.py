from types import SimpleNamespace

import pytest
from rich.console import Console

from koder_agent.agentic.hooks import get_subagent_display_hooks
from koder_agent.core.display_context import (
    SubagentDisplayEvent,
    SubagentDisplayIdentity,
    subagent_display_scope,
)
from koder_agent.core.streaming_display import OutputType, StreamingDisplayManager
from koder_agent.harness.agents.definitions import AgentDefinition


def _render(manager: StreamingDisplayManager) -> str:
    return manager.get_final_display()


def _display_manager(*, width: int = 100, height: int | None = None) -> StreamingDisplayManager:
    return StreamingDisplayManager(
        Console(force_terminal=False, width=width, height=height),
    )


def _ordered_identity(
    agent_id: str,
    label: str,
    order: int | None,
    *,
    group_id: str = "batch-ordered",
    parent_call_id: str | None = None,
) -> SubagentDisplayIdentity:
    return SubagentDisplayIdentity(
        group_id=group_id,
        agent_id=agent_id,
        label=label,
        parent_call_id=parent_call_id,
        order=order,
    )


def _subagent_tool_call(call_id: str, tool_name: str = "task_delegate") -> SimpleNamespace:
    return SimpleNamespace(
        raw_item=SimpleNamespace(
            name=tool_name,
            arguments='{"tasks": []}',
            call_id=call_id,
        )
    )


@pytest.mark.asyncio
async def test_subagent_hooks_emit_structured_progress_without_stdout(capsys):
    events = []
    hooks = get_subagent_display_hooks(
        group_id="batch-1",
        agent_id="agent-1",
        label="Inspect renderer",
        parent_call_id="call-parent",
    )
    agent = SimpleNamespace(name="explorer")
    tool = SimpleNamespace(name="grep_search")

    with subagent_display_scope(events.append):
        await hooks.on_agent_start(None, agent)
        await hooks.on_tool_start(None, agent, tool)
        await hooks.on_tool_end(None, agent, tool, "matches")
        hooks.finish("completed")

    assert capsys.readouterr().out == ""
    assert [event.kind for event in events] == [
        "started",
        "tool_started",
        "tool_finished",
        "completed",
    ]
    assert all(event.identity.parent_call_id == "call-parent" for event in events)


@pytest.mark.asyncio
async def test_subagent_hooks_ignore_late_activity_after_terminal_event():
    events = []
    hooks = get_subagent_display_hooks(
        group_id="batch-late",
        agent_id="agent-late",
        label="Late callbacks",
    )
    agent = SimpleNamespace(name="explorer")
    tool = SimpleNamespace(name="grep_search")

    with subagent_display_scope(events.append):
        hooks.finish("completed")
        await hooks.on_tool_start(None, agent, tool)
        await hooks.on_tool_end(None, agent, tool, "matches")
        hooks.finish("failed", "too late")

    assert [event.kind for event in events] == ["started", "completed"]


def test_streaming_display_groups_subagents_under_parent_tool_call():
    manager = _display_manager()
    first = SubagentDisplayIdentity(
        group_id="batch-1",
        agent_id="agent-1",
        label="Inspect Codex renderer",
        parent_call_id="call-parent",
    )
    second = SubagentDisplayIdentity(
        group_id="batch-1",
        agent_id="agent-2",
        label="Inspect Claude renderer",
        parent_call_id="call-parent",
    )
    # Child hooks can run in another task immediately after the SDK starts the
    # parent tool. The reducer must repair ordering even if that event is
    # observed before the parent's tool_called stream item.
    manager.handle_subagent_event(SubagentDisplayEvent(first, "started"))
    manager.handle_tool_called(_subagent_tool_call("call-parent"))

    manager.handle_subagent_event(
        SubagentDisplayEvent(first, "tool_started", tool_name="grep_search")
    )
    manager.handle_subagent_event(SubagentDisplayEvent(second, "started"))

    running = _render(manager)
    assert "Delegated agents · 2 running · 0 tool uses" in running
    assert "Inspect Codex renderer · grep_search" in running
    assert "Inspect Claude renderer · working" in running

    manager.handle_subagent_event(
        SubagentDisplayEvent(first, "tool_finished", tool_name="grep_search")
    )
    manager.handle_subagent_event(SubagentDisplayEvent(first, "completed"))
    manager.handle_subagent_event(SubagentDisplayEvent(second, "completed"))
    manager.handle_tool_output(
        SimpleNamespace(output="delegated results", tool_call_id="call-parent")
    )

    assert [section.type for section in manager.sections] == [
        OutputType.TOOL_CALL,
        OutputType.SUBAGENT_GROUP,
        OutputType.TOOL_OUTPUT,
    ]
    completed = _render(manager)
    assert "Delegated agents · Succeeded (2 agents, 1 tool use)" in completed
    assert "Inspect Codex renderer · succeeded (1 tool use)" in completed
    assert "Inspect Claude renderer · succeeded" in completed
    assert "Delegation complete" not in completed


def test_subagent_rows_use_declared_order_with_stable_fallback():
    manager = _display_manager(height=25)
    identities = [
        _ordered_identity("agent-third", "Third declared", 2),
        _ordered_identity("agent-zulu", "Zulu fallback", None),
        _ordered_identity("agent-first", "First declared", 0),
        _ordered_identity("agent-alpha", "Alpha fallback", None),
    ]

    for identity in identities:
        manager.handle_subagent_event(SubagentDisplayEvent(identity, "started"))

    rendered = _render(manager)
    assert rendered.index("First declared") < rendered.index("Third declared")
    assert rendered.index("Third declared") < rendered.index("Alpha fallback")
    assert rendered.index("Alpha fallback") < rendered.index("Zulu fallback")


@pytest.mark.parametrize(
    ("statuses", "expected_summary", "expected_color"),
    [
        (["completed"], "Succeeded (1 agent, 1 tool use)", "green"),
        (["failed"], "Failed (1 agent, 0 tool uses)", "red"),
        (["cancelled"], "Cancelled (1 agent, 0 tool uses)", "yellow"),
        (
            ["completed", "failed", "cancelled"],
            "Mixed results (3 agents, 0 tool uses) · 1 succeeded, 1 failed, 1 cancelled",
            "magenta",
        ),
    ],
)
def test_terminal_subagent_summaries_and_colors(statuses, expected_summary, expected_color):
    manager = _display_manager(height=25)

    for order, status in enumerate(statuses):
        identity = _ordered_identity(
            f"agent-{order}",
            f"Agent {order}",
            order,
            group_id="batch-terminal",
        )
        manager.handle_subagent_event(SubagentDisplayEvent(identity, "started"))
        if status == "completed" and len(statuses) == 1:
            manager.handle_subagent_event(
                SubagentDisplayEvent(identity, "tool_finished", tool_name="read_file")
            )
        manager.handle_subagent_event(SubagentDisplayEvent(identity, status))

    group = manager.subagent_groups["batch-terminal"]
    header = manager._format_subagent_group(group)[0]
    assert expected_summary in header.plain
    status_offset = header.plain.index(expected_summary)
    status_style = header.get_style_at_offset(manager.console, status_offset)
    assert expected_color in str(status_style)


def test_failed_subagent_row_shows_bounded_failure_detail():
    manager = _display_manager(width=120, height=25)
    identity = _ordered_identity("agent-failed", "Broken child", 0)
    manager.handle_subagent_event(SubagentDisplayEvent(identity, "started"))
    manager.handle_subagent_event(
        SubagentDisplayEvent(identity, "failed", detail="network\nfailed " + "x" * 100)
    )

    rendered = _render(manager)
    assert "Broken child · failed · network failed" in rendered
    assert "x" * 81 not in rendered


def test_small_viewport_uses_one_line_subagent_summary():
    compact = _display_manager(height=15)
    detailed = _display_manager(height=16)

    for manager in (compact, detailed):
        for order in range(3):
            identity = _ordered_identity(
                f"agent-{order}",
                f"Viewport agent {order}",
                order,
                group_id="batch-viewport",
            )
            manager.handle_subagent_event(SubagentDisplayEvent(identity, "started"))

    compact_rendered = _render(compact)
    assert compact_rendered.splitlines() == ["◉ Delegated agents · 3 running · 0 tool uses"]
    assert "Viewport agent" not in compact_rendered

    detailed_rendered = _render(detailed)
    assert "Viewport agent 0 · working" in detailed_rendered
    assert "Viewport agent 2 · working" in detailed_rendered


def test_default_viewport_bounds_detailed_subagent_rows():
    manager = _display_manager(height=25)

    for order in range(5):
        identity = _ordered_identity(
            f"agent-{order}",
            f"Bounded agent {order}",
            order,
            group_id="batch-bounded",
        )
        manager.handle_subagent_event(SubagentDisplayEvent(identity, "started"))

    rendered = _render(manager)
    for order in range(4):
        assert f"Bounded agent {order} · working" in rendered
    assert "Bounded agent 4" not in rendered
    assert "+1 more agent" in rendered
    assert "+1 more agents" not in rendered


def test_task_delegate_footer_is_suppressed_only_for_correlated_group():
    correlated = _display_manager(height=25)
    correlated.handle_tool_called(_subagent_tool_call("call-correlated"))
    identity = _ordered_identity(
        "agent-1",
        "Correlated agent",
        0,
        group_id="batch-correlated",
        parent_call_id="call-correlated",
    )
    correlated.handle_subagent_event(SubagentDisplayEvent(identity, "started"))
    correlated.handle_subagent_event(SubagentDisplayEvent(identity, "completed"))
    correlated.handle_tool_output(
        SimpleNamespace(output="delegated results", tool_call_id="call-correlated")
    )

    correlated_rendered = _render(correlated)
    assert "Correlated agent · succeeded" in correlated_rendered
    assert "Delegation complete" not in correlated_rendered

    uncorrelated = _display_manager(height=25)
    uncorrelated.handle_tool_called(_subagent_tool_call("call-uncorrelated"))
    uncorrelated.handle_tool_output(
        SimpleNamespace(output="delegated results", tool_call_id="call-uncorrelated")
    )
    assert "Delegation complete" in _render(uncorrelated)


def test_sync_agent_tool_footer_is_suppressed_for_correlated_group():
    manager = _display_manager(height=25)
    manager.handle_tool_called(_subagent_tool_call("call-agent", "agent_tool"))
    identity = _ordered_identity(
        "agent-1",
        "Explore · Inspect auth",
        0,
        group_id="agent-tool-group",
        parent_call_id="call-agent",
    )
    manager.handle_subagent_event(SubagentDisplayEvent(identity, "started"))
    manager.handle_subagent_event(SubagentDisplayEvent(identity, "completed"))
    manager.handle_tool_output(
        SimpleNamespace(
            output='{"status":"completed","result":"large child result"}',
            tool_call_id="call-agent",
        )
    )

    rendered = _render(manager)
    assert "Explore · Inspect auth · succeeded" in rendered
    assert "large child result" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("direct_display", [False, True])
async def test_unmanaged_agent_run_selects_silent_or_direct_hooks(
    monkeypatch, capsys, direct_display
):
    from koder_agent.harness.agents import service as service_module

    async def fake_create_dev_agent(*args, **kwargs):
        return SimpleNamespace(name="worker", _koder_mcp_servers=[])

    class FakeSession:
        def __init__(self, session_id):
            self.session_id = session_id

        def close(self):
            pass

        async def get_items(self):
            return []

        async def add_items(self, items):
            return None

    async def fake_run(agent, prompt, **kwargs):
        hooks = kwargs["hooks"]
        tool = SimpleNamespace(name="grep_search")
        await hooks.on_agent_start(None, agent)
        await hooks.on_tool_start(None, agent, tool)
        await hooks.on_tool_end(None, agent, tool, "matches")
        await hooks.on_agent_end(None, agent, "done")
        return SimpleNamespace(final_output="done")

    monkeypatch.setattr(service_module, "create_dev_agent", fake_create_dev_agent)
    monkeypatch.setattr(service_module, "EnhancedSQLiteSession", FakeSession)
    monkeypatch.setattr(service_module, "get_all_tools", lambda: [])
    monkeypatch.setattr(service_module.Runner, "run", fake_run)

    result = await service_module._execute_agent_run(
        agent_definition=AgentDefinition(
            agent_type="worker",
            when_to_use="test",
            system_prompt="test",
            source="built-in",
        ),
        prompt="inspect",
        session_id="background-worker",
        seed_items=None,
        cwd=None,
        direct_display=direct_display,
    )

    assert result == "done"
    output = capsys.readouterr().out
    if direct_display:
        assert "Agent: worker" in output
        assert "grep_search" in output
    else:
        assert output == ""
