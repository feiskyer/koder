"""Agent lifecycle regressions with synthetic execution and real temporary Git."""

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from koder_agent.harness.agents.definitions import AgentDefinition
from koder_agent.harness.agents.service import AgentService
from koder_agent.harness.agents.teams.in_process import InProcessTeammateRunner
from koder_agent.harness.agents.teams.service import TeamService


@pytest.fixture
def isolated_repo(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    repo = tmp_path / "repo"
    repo.mkdir()
    for arguments in (
        ["init", "-b", "main"],
        ["config", "user.name", "Test User"],
        ["config", "user.email", "test@example.com"],
    ):
        subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True)
    monkeypatch.setattr(
        "koder_agent.harness.agents.service._redacted_model_config_snapshot", lambda _: {}
    )
    return repo


def _definition():
    return AgentDefinition(
        agent_type="worker",
        when_to_use="Test isolated work",
        system_prompt="Synthetic test worker",
        source="built-in",
        isolation="worktree",
    )


@pytest.mark.asyncio
async def test_background_resume_recreates_cleaned_isolation(isolated_repo, tmp_path, monkeypatch):
    seen_cwds = []

    async def execute(*, cwd, **_kwargs):
        seen_cwds.append(Path(cwd))
        return "done"

    monkeypatch.setattr("koder_agent.harness.agents.service._execute_agent_run", execute)
    service = AgentService.for_test(tmp_path)
    record = await service.launch_background(
        agent_definition=_definition(),
        prompt="first",
        description="isolated",
        cwd=isolated_repo,
    )
    await service.wait(record.id)
    assert service.get(record.id).worktree_path is None

    await service.resume_background(
        agent_id=record.id,
        agent_definition=_definition(),
        prompt="follow-up",
        cwd=isolated_repo,
    )
    await service.wait(record.id)

    assert len(seen_cwds) == 2
    assert all(cwd != isolated_repo for cwd in seen_cwds)
    assert all(cwd.parent == isolated_repo / ".koder" / "worktrees" for cwd in seen_cwds)
    assert (isolated_repo / "README.md").read_text(encoding="utf-8") == "original\n"


@pytest.mark.asyncio
async def test_teammate_resumes_in_retained_dirty_worktree(isolated_repo, tmp_path, monkeypatch):
    seen_cwds = []
    follow_up = asyncio.Event()

    async def execute(*, prompt, cwd, **_kwargs):
        workspace = Path(cwd)
        seen_cwds.append(workspace)
        if prompt == "initial":
            (workspace / "result.txt").write_text("keep agent work\n", encoding="utf-8")
        else:
            assert (workspace / "result.txt").read_text(encoding="utf-8") == "keep agent work\n"
            follow_up.set()
        return "done"

    monkeypatch.setattr("koder_agent.harness.agents.service._execute_agent_run", execute)
    agent_service = AgentService.for_test(tmp_path)
    team_service = TeamService.for_test(root=tmp_path, cwd=isolated_repo)
    team_id = team_service.create_team("isolated-team")
    runner = InProcessTeammateRunner(agent_service=agent_service, team_service=team_service)
    spawned = await runner.spawn_teammate(
        team_id=team_id,
        name="worker",
        agent_definition=_definition(),
        prompt="initial",
        cwd=isolated_repo,
    )
    loop_task = runner._tasks[spawned.agent_id]
    try:
        await asyncio.wait_for(runner.wait(spawned.agent_id), timeout=2)
        team_service.route(team_id, "follow-up", recipient="worker")
        await asyncio.wait_for(follow_up.wait(), timeout=2)
        await asyncio.wait_for(runner.wait(spawned.agent_id), timeout=2)

        assert seen_cwds[0] == seen_cwds[1] != isolated_repo
        assert not (isolated_repo / "result.txt").exists()
        assert team_service.member_records(team_id)[0].is_active
    finally:
        await runner.terminate(spawned.agent_id)
        # Retrieve a failed loop as well, so the regression never leaks a task.
        await asyncio.gather(loop_task, return_exceptions=True)
