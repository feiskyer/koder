"""Team identities must not reset existing work or outlive their membership."""

import asyncio

import pytest

from koder_agent.harness.agents.definitions import AgentDefinition
from koder_agent.harness.agents.service import AgentService
from koder_agent.harness.agents.teams.in_process import InProcessTeammateRunner
from koder_agent.harness.agents.teams.service import TeamService


def test_recreating_team_preserves_members_tasks_and_mailboxes(tmp_path):
    service = TeamService.for_test(root=tmp_path)
    team_id = service.create_team("reviewers")
    member = service.add_member(team_id, "agent-1", name="reviewer")
    task = service.task_service(team_id).create_task("Review")
    service.route(team_id, "continue", recipient=member.name)
    original = service.get(team_id)
    reloaded = TeamService.for_test(root=tmp_path)

    assert reloaded.create_team("reviewers") == team_id

    assert reloaded.get(team_id) == original
    assert reloaded.member_records(team_id) == [member]
    assert reloaded.task_service(team_id).get_task(task.id) == task
    assert reloaded.read_mailbox(team_id, recipient=member.name)[0].content == "continue"
    with pytest.raises(RuntimeError, match="active team"):
        reloaded.delete_team(team_id)


@pytest.mark.parametrize("name", ["reviewer", "Reviewer", "agent-1", "team-lead"])
def test_member_names_cannot_alias_another_recipient(tmp_path, name):
    service = TeamService.for_test(root=tmp_path)
    team_id = service.create_team("reviewers")
    original = service.add_member(team_id, "agent-1", name="reviewer")

    with pytest.raises(ValueError, match="recipient"):
        service.add_member(team_id, "agent-2", name=name)

    assert service.member_records(team_id) == [original]


def test_empty_team_name_does_not_create_config_at_teams_root(tmp_path):
    service = TeamService.for_test(root=tmp_path)

    with pytest.raises(ValueError, match="identifier"):
        service.create_team("")

    assert not (service.teams_root / "config.json").exists()


def test_normalized_team_name_collision_preserves_original(tmp_path):
    service = TeamService.for_test(root=tmp_path)
    team_id = service.create_team("Reviewers")
    original = service.add_member(team_id, "agent-1", name="reviewer")

    with pytest.raises(ValueError, match="aliases"):
        service.create_team("reviewers")

    assert service.member_records(team_id) == [original]


def _definition():
    return AgentDefinition(
        agent_type="worker",
        when_to_use="Tests",
        system_prompt="Synthetic worker",
        source="built-in",
    )


@pytest.mark.asyncio
async def test_spawning_into_missing_team_does_not_launch_an_agent(tmp_path, monkeypatch):
    executed = []

    async def execute(**kwargs):
        executed.append(kwargs["prompt"])
        return "done"

    monkeypatch.setattr("koder_agent.harness.agents.service._execute_agent_run", execute)
    monkeypatch.setattr(
        "koder_agent.harness.agents.service._redacted_model_config_snapshot", lambda _: {}
    )
    agents = AgentService.for_test(tmp_path)
    teams = TeamService.for_test(root=tmp_path)
    runner = InProcessTeammateRunner(agent_service=agents, team_service=teams)

    try:
        with pytest.raises(KeyError):
            await runner.spawn_teammate(
                team_id="missing",
                name="worker",
                agent_definition=_definition(),
                prompt="must not run",
                cwd=tmp_path,
            )
        await asyncio.sleep(0)
        assert executed == []
        assert agents.list_records() == []
        assert not runner._tasks
    finally:
        for record in agents.list_records():
            await agents.cancel_background(record.id)


@pytest.mark.asyncio
async def test_deactivated_idle_teammate_stops_before_consuming_more_work(tmp_path, monkeypatch):
    executed = []

    async def execute(**kwargs):
        executed.append(kwargs["prompt"])
        return "done"

    monkeypatch.setattr("koder_agent.harness.agents.service._execute_agent_run", execute)
    monkeypatch.setattr(
        "koder_agent.harness.agents.service._redacted_model_config_snapshot", lambda _: {}
    )
    agents = AgentService.for_test(tmp_path)
    teams = TeamService.for_test(root=tmp_path)
    team_id = teams.create_team("stopping")
    runner = InProcessTeammateRunner(agent_service=agents, team_service=teams)
    spawned = await runner.spawn_teammate(
        team_id=team_id,
        name="worker",
        agent_definition=_definition(),
        prompt="initial",
        cwd=tmp_path,
    )
    loop = runner._tasks[spawned.agent_id]
    try:
        await asyncio.wait_for(runner.wait(spawned.agent_id), timeout=2)
        teams.set_member_active(team_id, spawned.agent_id, False)
        teams.route(team_id, "must remain unread", recipient="worker")

        await asyncio.wait_for(asyncio.shield(loop), timeout=1)

        assert executed == ["initial"]
        assert not runner.manages(spawned.agent_id)
        assert not teams.mailbox_entries(team_id, recipient="worker")[0].read
    finally:
        await runner.terminate(spawned.agent_id)
        await asyncio.gather(loop, return_exceptions=True)
