import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from koder_agent.tools.todo import TodoRuntimeIdentity, TodoStore  # noqa: E402


def test_send_message_to_agent_by_name(tmp_path):
    from koder_agent.harness.agents.service import AgentService
    from koder_agent.tools.send_message import _send_message_impl

    service = AgentService.for_test(tmp_path)
    agent_id = service.spawn("worker")
    service.register_name("researcher", agent_id)

    result = asyncio.run(
        _send_message_impl(
            to="researcher",
            message="Please investigate the auth module",
            summary="Auth investigation",
            _agent_service=service,
        )
    )
    parsed = json.loads(result)
    assert parsed["status"] == "sent"
    messages = service.read_mailbox(agent_id)
    assert len(messages) == 1
    assert "auth module" in messages[0].content


def test_send_message_to_agent_by_id(tmp_path):
    from koder_agent.harness.agents.service import AgentService
    from koder_agent.tools.send_message import _send_message_impl

    service = AgentService.for_test(tmp_path)
    agent_id = service.spawn("worker")

    result = asyncio.run(
        _send_message_impl(
            to=agent_id,
            message="direct message",
            summary="Direct msg",
            _agent_service=service,
        )
    )
    parsed = json.loads(result)
    assert parsed["status"] == "sent"


def test_send_message_to_unknown_agent(tmp_path):
    from koder_agent.harness.agents.service import AgentService
    from koder_agent.tools.send_message import _send_message_impl

    service = AgentService.for_test(tmp_path)

    result = asyncio.run(
        _send_message_impl(
            to="nonexistent",
            message="hello",
            summary="test",
            _agent_service=service,
        )
    )
    parsed = json.loads(result)
    assert parsed["status"] == "error"


def test_send_message_to_team_mailbox(tmp_path):
    from koder_agent.harness.agents.teams.service import TeamService
    from koder_agent.tools.send_message import _send_message_impl

    team_svc = TeamService.for_test(root=tmp_path)
    team_id = team_svc.create_team("test-team")
    team_svc.add_member(team_id, "worker-1", name="worker")

    result = asyncio.run(
        _send_message_impl(
            to="worker",
            message="Team task assigned",
            summary="Task assignment",
            _team_service=team_svc,
            _team_name=team_id,
        )
    )
    parsed = json.loads(result)
    assert parsed["status"] == "sent"
    mailbox = team_svc.read_mailbox(team_id, recipient="worker")
    assert len(mailbox) == 1


def test_send_structured_shutdown_request(tmp_path):
    from koder_agent.harness.agents.teams.service import TeamService
    from koder_agent.tools.send_message import _send_message_impl

    team_svc = TeamService.for_test(root=tmp_path)
    team_id = team_svc.create_team("test-team")
    team_svc.add_member(team_id, "worker-1", name="worker")

    structured = json.dumps({"type": "shutdown_request", "reason": "work complete"})
    result = asyncio.run(
        _send_message_impl(
            to="worker",
            message=structured,
            _team_service=team_svc,
            _team_name=team_id,
        )
    )
    parsed = json.loads(result)
    assert parsed["status"] == "sent"


def test_send_message_broadcast_to_all_team_members(tmp_path):
    from koder_agent.harness.agents.teams.service import TeamService
    from koder_agent.tools.send_message import _send_message_impl

    team_svc = TeamService.for_test(root=tmp_path)
    team_id = team_svc.create_team("test-team")
    team_svc.add_member(team_id, "worker-1", name="alice")
    team_svc.add_member(team_id, "worker-2", name="bob")

    result = asyncio.run(
        _send_message_impl(
            to="*",
            message="All hands meeting",
            _team_service=team_svc,
            _team_name=team_id,
        )
    )
    parsed = json.loads(result)
    assert parsed["status"] == "sent"
    assert parsed["routing"] == "team_mailbox"
    assert parsed["broadcast"] is True
    assert set(parsed["recipients"]) == {"alice", "bob"}
    # Verify both mailboxes received the message
    assert len(team_svc.read_mailbox(team_id, recipient="alice")) == 1
    assert len(team_svc.read_mailbox(team_id, recipient="bob")) == 1


def test_send_message_teammate_to_teammate(tmp_path):
    """Teammates can message each other through the team mailbox."""
    from koder_agent.harness.agents.teams.service import TeamService
    from koder_agent.tools.send_message import _send_message_impl

    team_svc = TeamService.for_test(root=tmp_path)
    team_id = team_svc.create_team("collab-team")
    team_svc.add_member(team_id, "agent-1", name="frontend")
    team_svc.add_member(team_id, "agent-2", name="backend")

    # Frontend sends message to Backend (teammate→teammate, not lead→teammate)
    result = asyncio.run(
        _send_message_impl(
            to="backend",
            message="I need the API endpoint for user profile",
            summary="API request",
            _team_service=team_svc,
            _team_name=team_id,
        )
    )
    parsed = json.loads(result)
    assert parsed["status"] == "sent"
    assert parsed["routing"] == "team_mailbox"
    assert parsed["recipient"] == "backend"

    # Verify message arrived in backend's mailbox
    backend_mail = team_svc.read_mailbox(team_id, recipient="backend")
    assert len(backend_mail) == 1
    assert "API endpoint" in backend_mail[0].content

    # Backend responds to Frontend
    result2 = asyncio.run(
        _send_message_impl(
            to="frontend",
            message="The endpoint is /api/v1/users/:id/profile",
            summary="API response",
            _team_service=team_svc,
            _team_name=team_id,
        )
    )
    parsed2 = json.loads(result2)
    assert parsed2["status"] == "sent"

    # Verify response arrived in frontend's mailbox
    frontend_mail = team_svc.read_mailbox(team_id, recipient="frontend")
    assert len(frontend_mail) == 1
    assert "/api/v1/users" in frontend_mail[0].content


def test_send_message_uses_team_context_by_teammate_name(tmp_path):
    """Real tool calls inside teammates should route by team member name."""
    from koder_agent.harness.agents.teams.context import TeamToolContext, team_tool_context
    from koder_agent.harness.agents.teams.service import TeamService
    from koder_agent.tools.send_message import _send_message_impl

    team_svc = TeamService.for_test(root=tmp_path)
    team_id = team_svc.create_team("context-team")
    team_svc.add_member(team_id, "agent-integrator", name="integrator")
    team_svc.add_member(team_id, "agent-critic", name="critic")

    context = TeamToolContext(
        team_id=team_id,
        sender_name="integrator",
        sender_agent_id="agent-integrator",
        team_service=team_svc,
    )

    async def run_case():
        with team_tool_context(context):
            return await _send_message_impl(
                to="critic",
                message="DIRECT_PING_FROM_INTEGRATOR",
                summary="Direct ping",
            )

    result = asyncio.run(run_case())
    parsed = json.loads(result)
    assert parsed["status"] == "sent"
    assert parsed["routing"] == "team_mailbox"
    assert parsed["sender"] == "integrator"
    assert parsed["recipient"] == "critic"

    mailbox = team_svc.mailbox_entries(team_id, recipient="critic")
    assert len(mailbox) == 1
    assert mailbox[0].sender == "integrator"
    assert mailbox[0].content == "DIRECT_PING_FROM_INTEGRATOR"


def test_send_message_uses_team_context_by_agent_id(tmp_path):
    """Team context should keep agent-id addressing on the team mailbox path."""
    from koder_agent.harness.agents.teams.context import TeamToolContext, team_tool_context
    from koder_agent.harness.agents.teams.service import TeamService
    from koder_agent.tools.send_message import _send_message_impl

    team_svc = TeamService.for_test(root=tmp_path)
    team_id = team_svc.create_team("context-team")
    team_svc.add_member(team_id, "agent-integrator", name="integrator")
    team_svc.add_member(team_id, "agent-critic", name="critic")

    context = TeamToolContext(
        team_id=team_id,
        sender_name="integrator",
        sender_agent_id="agent-integrator",
        team_service=team_svc,
    )

    async def run_case():
        with team_tool_context(context):
            return await _send_message_impl(
                to="agent-critic",
                message="DIRECT_PING_BY_ID",
                summary="Direct ping",
            )

    result = asyncio.run(run_case())
    parsed = json.loads(result)
    assert parsed["status"] == "sent"
    assert parsed["routing"] == "team_mailbox"
    assert parsed["recipient"] == "agent-critic"

    mailbox = team_svc.mailbox_entries(team_id, recipient="agent-critic")
    assert len(mailbox) == 1
    assert mailbox[0].sender == "integrator"
    assert mailbox[0].content == "DIRECT_PING_BY_ID"


def test_send_message_team_context_broadcast_uses_actual_sender(tmp_path):
    """Team-context broadcasts should use the teammate sender and skip itself."""
    from koder_agent.harness.agents.teams.context import TeamToolContext, team_tool_context
    from koder_agent.harness.agents.teams.service import TeamService
    from koder_agent.tools.send_message import _send_message_impl

    team_svc = TeamService.for_test(root=tmp_path)
    team_id = team_svc.create_team("broadcast-team")
    team_svc.add_member(team_id, "agent-integrator", name="integrator")
    team_svc.add_member(team_id, "agent-critic", name="critic")
    team_svc.add_member(team_id, "agent-archived", name="archived", is_active=False)

    context = TeamToolContext(
        team_id=team_id,
        sender_name="integrator",
        sender_agent_id="agent-integrator",
        team_service=team_svc,
    )

    async def run_case():
        with team_tool_context(context):
            return await _send_message_impl(
                to="*",
                message="BROADCAST_FROM_INTEGRATOR",
                summary="Broadcast",
            )

    result = asyncio.run(run_case())
    parsed = json.loads(result)
    assert parsed["status"] == "sent"
    assert parsed["routing"] == "team_mailbox"
    assert parsed["sender"] == "integrator"
    assert parsed["recipients"] == ["critic"]

    critic_mail = team_svc.mailbox_entries(team_id, recipient="critic")
    assert len(critic_mail) == 1
    assert critic_mail[0].sender == "integrator"
    assert team_svc.mailbox_entries(team_id, recipient="integrator") == []
    assert team_svc.mailbox_entries(team_id, recipient="archived") == []


def test_send_message_team_context_unknown_recipient_does_not_create_orphan(tmp_path):
    """Unknown team recipients should fail before writing orphan inbox files."""
    from koder_agent.harness.agents.teams.context import TeamToolContext, team_tool_context
    from koder_agent.harness.agents.teams.service import TeamService
    from koder_agent.tools.send_message import _send_message_impl

    team_svc = TeamService.for_test(root=tmp_path)
    team_id = team_svc.create_team("strict-team")
    team_svc.add_member(team_id, "agent-integrator", name="integrator")

    context = TeamToolContext(
        team_id=team_id,
        sender_name="integrator",
        sender_agent_id="agent-integrator",
        team_service=team_svc,
    )

    async def run_case():
        with team_tool_context(context):
            return await _send_message_impl(
                to="ghost",
                message="you should not exist",
                summary="No orphan",
            )

    result = asyncio.run(run_case())
    parsed = json.loads(result)
    assert parsed["status"] == "error"
    assert parsed["routing"] == "team_mailbox"
    assert "Unknown team recipient" in parsed["error"]
    assert team_svc.mailbox_entries(team_id, recipient="ghost") == []
    assert not (tmp_path / "teams" / team_id / "inboxes" / "ghost.json").exists()


def test_execute_agent_run_exposes_team_context_to_send_message(tmp_path, monkeypatch):
    """The actual AgentService runner wrapper should set team context for tools."""
    from koder_agent.harness.agents.definitions import AgentDefinition
    from koder_agent.harness.agents.service import _execute_agent_run
    from koder_agent.harness.agents.teams.context import TeamToolContext
    from koder_agent.harness.agents.teams.service import TeamService
    from koder_agent.tools.send_message import _send_message_impl

    class FakeSession:
        def __init__(self, session_id):
            self.session_id = session_id
            self.items = []

        def close(self):
            pass

        async def get_items(self):
            return self.items

        async def add_items(self, items):
            self.items.extend(items)

    async def fake_create_dev_agent(*args, **kwargs):
        return object()

    async def fake_runner_run(*args, **kwargs):
        result = await _send_message_impl(
            to="critic",
            message="RUNNER_CONTEXT_PING",
            summary="Runner context",
        )
        return types.SimpleNamespace(final_output=result)

    monkeypatch.setattr(
        "koder_agent.harness.agents.service.EnhancedSQLiteSession",
        FakeSession,
    )
    monkeypatch.setattr(
        "koder_agent.harness.agents.service.create_dev_agent", fake_create_dev_agent
    )
    monkeypatch.setattr("koder_agent.harness.agents.service.Runner.run", fake_runner_run)

    team_svc = TeamService.for_test(root=tmp_path)
    team_id = team_svc.create_team("runner-context-team")
    team_svc.add_member(team_id, "agent-integrator", name="integrator")
    team_svc.add_member(team_id, "agent-critic", name="critic")
    context = TeamToolContext(
        team_id=team_id,
        sender_name="integrator",
        sender_agent_id="agent-integrator",
        team_service=team_svc,
    )
    definition = AgentDefinition(
        agent_type="general-purpose",
        when_to_use="General",
        system_prompt="Agent.",
        source="built-in",
    )

    result = asyncio.run(
        _execute_agent_run(
            agent_definition=definition,
            prompt="trigger tool",
            session_id="test-session",
            seed_items=None,
            cwd=str(tmp_path),
            team_context=context,
            todo_store=TodoStore(
                TodoRuntimeIdentity("test-session", "agent-integrator", "test-run")
            ),
        )
    )

    parsed = json.loads(result)
    assert parsed["status"] == "sent"
    assert parsed["routing"] == "team_mailbox"
    mailbox = team_svc.mailbox_entries(team_id, recipient="critic")
    assert len(mailbox) == 1
    assert mailbox[0].sender == "integrator"
    assert mailbox[0].content == "RUNNER_CONTEXT_PING"


def test_execute_agent_run_cleans_up_agent_mcp_servers(tmp_path, monkeypatch):
    """Subagent-owned MCP servers should be cleaned up in the runner task."""
    from koder_agent.harness.agents.definitions import AgentDefinition
    from koder_agent.harness.agents.service import _execute_agent_run

    class FakeSession:
        def __init__(self, session_id):
            self.session_id = session_id

        def close(self):
            pass

        async def get_items(self):
            return []

        async def add_items(self, items):
            raise AssertionError("seed items were not expected")

    class FakeMCPServer:
        name = "fake-mcp"

        def __init__(self):
            self.cleaned = False

        async def cleanup(self):
            self.cleaned = True

    fake_server = FakeMCPServer()

    async def fake_create_dev_agent(*args, **kwargs):
        return types.SimpleNamespace(mcp_servers=[fake_server])

    async def fake_runner_run(*args, **kwargs):
        return types.SimpleNamespace(final_output="done")

    monkeypatch.setattr(
        "koder_agent.harness.agents.service.EnhancedSQLiteSession",
        FakeSession,
    )
    monkeypatch.setattr(
        "koder_agent.harness.agents.service.create_dev_agent", fake_create_dev_agent
    )
    monkeypatch.setattr("koder_agent.harness.agents.service.Runner.run", fake_runner_run)
    definition = AgentDefinition(
        agent_type="general-purpose",
        when_to_use="General",
        system_prompt="Agent.",
        source="built-in",
    )

    result = asyncio.run(
        _execute_agent_run(
            agent_definition=definition,
            prompt="run",
            session_id="cleanup-session",
            seed_items=None,
            cwd=str(tmp_path),
            todo_store=TodoStore(
                TodoRuntimeIdentity("cleanup-session", "general-purpose", "test-run")
            ),
        )
    )

    assert result == "done"
    assert fake_server.cleaned is True


@pytest.mark.parametrize("failure_boundary", ["session", "get_items", "add_items"])
def test_execute_agent_run_partial_initialization_closes_owner_once(
    tmp_path, monkeypatch, failure_boundary
):
    from koder_agent.harness.agents.definitions import AgentDefinition
    from koder_agent.harness.agents.service import _execute_agent_run
    from koder_agent.mcp import MCPServerSet

    class FakeServer:
        name = "owned"

        def __init__(self):
            self.cleanup_count = 0

        async def cleanup(self):
            self.cleanup_count += 1

    server = FakeServer()
    owner = MCPServerSet([server])

    async def fake_create_dev_agent(*args, **kwargs):
        return types.SimpleNamespace(mcp_servers=[], _koder_mcp_servers=owner)

    class FakeSession:
        def __init__(self, session_id):
            if failure_boundary == "session":
                raise RuntimeError("session failed")

        def close(self):
            pass

        async def get_items(self):
            if failure_boundary == "get_items":
                raise RuntimeError("get failed")
            return []

        async def add_items(self, items):
            if failure_boundary == "add_items":
                raise RuntimeError("add failed")

    monkeypatch.setattr(
        "koder_agent.harness.agents.service.create_dev_agent", fake_create_dev_agent
    )
    monkeypatch.setattr("koder_agent.harness.agents.service.EnhancedSQLiteSession", FakeSession)
    definition = AgentDefinition(
        agent_type="general-purpose",
        when_to_use="General",
        system_prompt="Agent.",
        source="built-in",
    )

    with pytest.raises(RuntimeError):
        asyncio.run(
            _execute_agent_run(
                agent_definition=definition,
                prompt="run",
                session_id="partial-session",
                seed_items=[{"role": "user", "content": "seed"}],
                cwd=str(tmp_path),
            )
        )

    assert server.cleanup_count == 1


def test_agent_cleanup_helper_detaches_owner_before_await(monkeypatch):
    from koder_agent.harness.agents.service import _cleanup_agent_mcp_servers
    from koder_agent.mcp import MCPServerSet

    class FakeServer:
        name = "owned"

        def __init__(self):
            self.cleanup_count = 0

        async def cleanup(self):
            self.cleanup_count += 1

    server = FakeServer()
    owner = MCPServerSet([server])
    agent = types.SimpleNamespace(mcp_servers=[], _koder_mcp_servers=owner)

    async def scenario():
        await _cleanup_agent_mcp_servers(agent)
        await _cleanup_agent_mcp_servers(agent)

    asyncio.run(scenario())

    assert agent._koder_mcp_servers is None
    assert server.cleanup_count == 1


def test_send_message_to_stopped_agent_indicates_stopped_state(tmp_path, monkeypatch):
    """SendMessage to a stopped agent should indicate the agent is stopped."""

    async def fake_execute(*, agent_definition, prompt, session_id, seed_items, cwd, **_kwargs):
        return "done"

    monkeypatch.setattr("koder_agent.harness.agents.service._execute_agent_run", fake_execute)

    from koder_agent.harness.agents.definitions import AgentDefinition
    from koder_agent.harness.agents.service import AgentService
    from koder_agent.tools.send_message import _send_message_impl

    service = AgentService.for_test(tmp_path)
    definition = AgentDefinition(
        agent_type="general-purpose",
        when_to_use="General",
        system_prompt="Agent.",
        source="built-in",
    )

    async def run():
        # Launch and complete an agent
        record = await service.launch_background(
            agent_definition=definition,
            prompt="initial work",
            description="Test agent",
        )
        await service.wait(record.id)
        assert service.get(record.id).state == "completed"

        # Send message to the stopped agent
        result = await _send_message_impl(
            to=record.id,
            message="continue work",
            summary="Continue",
            _agent_service=service,
        )
        parsed = json.loads(result)
        assert parsed["status"] == "sent"
        assert parsed.get("agent_stopped") is True
        assert "agent_tool(resume=agent_id)" in parsed.get("note", "")
        assert parsed["delivery"] == "queued"

    asyncio.run(run())


def test_send_message_works_without_team_context(tmp_path):
    """SendMessage routes through agent service even without team context."""
    from koder_agent.harness.agents.service import AgentService
    from koder_agent.tools.send_message import _send_message_impl

    service = AgentService.for_test(tmp_path)
    agent_id = service.spawn("worker")
    service.register_name("helper", agent_id)

    result = asyncio.run(
        _send_message_impl(
            to="helper",
            message="work without teams",
            summary="No teams",
            _agent_service=service,
            # No _team_service or _team_name
        )
    )
    parsed = json.loads(result)
    assert parsed["status"] == "sent"
    assert parsed["routing"] == "agent_mailbox"
