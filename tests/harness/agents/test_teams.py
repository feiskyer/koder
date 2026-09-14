import asyncio
import sys
import types
from pathlib import Path

# Stub litellm before importing koder_agent to avoid optional dependency issues
if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from koder_agent.harness.agents.teams import TeamService


def test_team_service_can_create_delete_and_route_messages(tmp_path):
    service = TeamService.for_test(root=tmp_path / ".koder")
    team_id = service.create_team("reviewers")
    service.route(team_id, "sync")
    assert service.read_mailbox(team_id)[0].content == "sync"
    service.delete_team(team_id)


def test_team_membership_changes_are_preserved(tmp_path):
    service = TeamService.for_test(root=tmp_path / ".koder")
    team_id = service.create_team("reviewers")
    service.add_member(team_id, "agent-1")
    assert "agent-1" in service.members(team_id)


def test_team_service_persists_config_and_task_roots_under_koder(tmp_path):
    service = TeamService.for_test(root=tmp_path / ".koder", cwd=tmp_path)
    team_id = service.create_team("Reviewers")
    service.add_member(
        team_id,
        "agent-1",
        name="reviewer-1",
        agent_type="security-reviewer",
        model="sonnet",
        prompt="Review auth changes",
        plan_mode_required=True,
        cwd=tmp_path,
    )

    record = service.get(team_id)
    members = service.member_records(team_id)
    assert record.name == "Reviewers"
    assert record.config_path.endswith("config.json")
    assert str(service.teams_root).endswith("teams")
    assert str(service.tasks_root).endswith("tasks")
    assert (service.teams_root / team_id / "config.json").exists()
    assert (service.tasks_root / team_id).exists()
    assert members[0].agent_type == "security-reviewer"
    assert members[0].model == "sonnet"
    assert members[0].prompt == "Review auth changes"
    assert members[0].plan_mode_required is True


def test_team_service_mailboxes_are_recipient_scoped(tmp_path):
    service = TeamService.for_test(root=tmp_path / ".koder", cwd=tmp_path)
    team_id = service.create_team("reviewers")
    service.route(team_id, "sync with lead", recipient="team-lead", sender="agent-1")
    service.route(team_id, "sync with worker", recipient="agent-2", sender="team-lead")

    lead_mail = service.read_mailbox(team_id, recipient="team-lead")
    worker_mail = service.read_mailbox(team_id, recipient="agent-2")

    assert [message.content for message in lead_mail] == ["sync with lead"]
    assert [message.content for message in worker_mail] == ["sync with worker"]


def test_team_service_records_message_and_run_history(tmp_path):
    service = TeamService.for_test(root=tmp_path / ".koder", cwd=tmp_path)
    team_id = service.create_team("history-team")
    service.add_member(team_id, "agent-critic", name="critic", cwd=tmp_path)

    service.route(team_id, "review this", recipient="critic", sender="integrator")
    service.consume_next_mailbox_entry(team_id, recipient="critic")
    service.record_run(
        team_id,
        agent_id="agent-critic",
        member_name="critic",
        prompt="review this",
        output="looks good",
        state="completed",
        source="mailbox",
    )

    history = service.history_entries(team_id)
    assert [entry.event for entry in history] == [
        "message_sent",
        "message_read",
        "run_completed",
    ]
    assert history[0].sender == "integrator"
    assert history[0].recipient == "critic"
    assert history[1].read is True
    assert history[2].member_name == "critic"
    assert history[2].content == "looks good"


def test_team_service_tracks_plan_approval_requests_and_responses(tmp_path):
    service = TeamService.for_test(root=tmp_path / ".koder", cwd=tmp_path)
    team_id = service.create_team("reviewers")
    service.add_member(team_id, "agent-1", name="reviewer-1", cwd=tmp_path, mode="plan")

    service.request_plan_approval(
        team_id,
        agent_id="agent-1",
        plan="Refactor auth module",
        requested_permission_mode="default",
    )
    pending = service.list_plan_approvals(team_id)
    service.respond_plan_approval(
        team_id,
        agent_id="agent-1",
        approved=True,
        permission_mode="default",
    )

    assert "agent-1" in pending
    assert service.list_plan_approvals(team_id) == {}
    assert service.member_records(team_id)[0].mode == "default"
    inbox = service.read_mailbox(team_id, recipient="agent-1")
    assert any("plan approved" in message.content for message in inbox)


def test_team_service_tracks_shutdown_requests_and_rejections(tmp_path):
    service = TeamService.for_test(root=tmp_path / ".koder", cwd=tmp_path)
    team_id = service.create_team("reviewers")
    service.add_member(team_id, "agent-1", name="reviewer-1", cwd=tmp_path)

    service.request_shutdown(team_id, agent_id="agent-1", reason="done")
    pending = service.list_shutdown_requests(team_id)
    asyncio.run(
        service.respond_shutdown(
            team_id,
            agent_id="agent-1",
            approved=False,
            feedback="keep working",
        )
    )

    assert "agent-1" in pending
    assert service.list_shutdown_requests(team_id) == {}
    inbox = service.read_mailbox(team_id, recipient="agent-1")
    assert any("shutdown rejected" in message.content for message in inbox)
