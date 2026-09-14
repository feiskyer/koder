"""Controlled team task outcomes; no providers, user profiles or tmux."""

import asyncio
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from koder_agent.harness.agents.definitions import AgentDefinition
from koder_agent.harness.agents.service import AgentService
from koder_agent.harness.agents.teams.in_process import InProcessTeammateRunner
from koder_agent.harness.agents.teams.service import TeamService
from koder_agent.harness.agents.teams.task_service import TeamTaskService


@pytest.fixture
def teams(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "koder_agent.harness.agents.teams.service.dispatch_project_hook_event",
        lambda **_: SimpleNamespace(blocked=False),
    )
    monkeypatch.setattr(
        "koder_agent.harness.agents.teams.task_service.dispatch_project_hook_event",
        lambda **_: SimpleNamespace(blocked=False),
    )
    monkeypatch.setattr(
        "koder_agent.harness.agents.service._redacted_model_config_snapshot", lambda _: {}
    )
    service = TeamService.for_test(root=tmp_path)
    service.create_team("outcomes")
    return service


async def _spawn(tmp_path, teams, monkeypatch, execute):
    monkeypatch.setattr("koder_agent.harness.agents.service._execute_agent_run", execute)
    agents = AgentService.for_test(tmp_path)
    runner = InProcessTeammateRunner(agent_service=agents, team_service=teams)
    spawned = await runner.spawn_teammate(
        team_id="outcomes",
        name="worker",
        agent_definition=AgentDefinition(
            agent_type="worker",
            when_to_use="test",
            system_prompt="synthetic",
            source="built-in",
        ),
        prompt="boot",
        cwd=tmp_path,
    )
    await asyncio.wait_for(runner.wait(spawned.agent_id), 2)
    return agents, runner, spawned


async def _eventually(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), 2)


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_terminal_tasks_require_explicit_pending_retry(tmp_path, status):
    tasks = TeamTaskService.for_test("outcomes", root=tmp_path)
    task = tasks.create_task("work")
    tasks.claim_task(task.id, "worker")
    tasks.update_status(task.id, status)

    assert not tasks.claim_task(task.id, "worker").success
    retry = tasks.update_status(task.id, "pending")
    assert retry.owner is None
    assert tasks.claim_task(task.id, "replacement").success


@pytest.mark.asyncio
async def test_failed_task_is_not_reexecuted_and_other_work_keeps_moving(
    tmp_path, teams, monkeypatch
):
    attempts = []
    stop_duplicate = asyncio.Event()

    async def execute(**kwargs):
        prompt = kwargs["prompt"]
        attempts.append(prompt)
        if prompt == "bad":
            if attempts.count("bad") > 1:
                await stop_duplicate.wait()
            raise RuntimeError("synthetic failure")
        return "done"

    _, runner, spawned = await _spawn(tmp_path, teams, monkeypatch, execute)
    tasks = teams.task_service("outcomes")
    bad = tasks.create_task("bad")
    good = tasks.create_task("good")
    try:
        await _eventually(
            lambda: tasks.get_task(good.id).status == "completed" or attempts.count("bad") > 1
        )
        assert tasks.get_task(bad.id).status == "failed"
        assert tasks.get_task(good.id).status == "completed"
        assert attempts.count("bad") == 1
    finally:
        await runner.terminate(spawned.agent_id)


@pytest.mark.asyncio
async def test_termination_persists_cancelled_claim(tmp_path, teams, monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute(**kwargs):
        if kwargs["prompt"] == "work":
            started.set()
            await release.wait()
        return "done"

    _, runner, spawned = await _spawn(tmp_path, teams, monkeypatch, execute)
    tasks = teams.task_service("outcomes")
    task = tasks.create_task("work")
    await asyncio.wait_for(started.wait(), 2)
    await runner.terminate(spawned.agent_id)
    assert tasks.get_task(task.id).status == "cancelled"
    assert not tasks.claim_task(task.id, spawned.agent_id).success


@pytest.mark.parametrize("same_owner", [False, True])
@pytest.mark.asyncio
async def test_late_result_cannot_complete_reassigned_or_retried_claim(
    tmp_path, teams, monkeypatch, same_owner
):
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute(**kwargs):
        if kwargs["prompt"] == "work":
            started.set()
            await release.wait()
        elif kwargs["prompt"] == "replacement work":
            await asyncio.Event().wait()
        return "done"

    _, runner, spawned = await _spawn(tmp_path, teams, monkeypatch, execute)
    tasks = teams.task_service("outcomes")
    task = tasks.create_task("work")
    try:
        await asyncio.wait_for(started.wait(), 2)
        # The old execution must not consume any replacement work on its next poll.
        teams.add_member("outcomes", "replacement")
        owner = spawned.agent_id if same_owner else "replacement"
        tasks.update_task(task.id, status="pending", owner=owner, active_form="replacement work")
        new_claim = tasks.claim_task(task.id, owner).task
        release.set()
        await asyncio.wait_for(runner.wait(spawned.agent_id), 2)
        assert tasks.get_task(task.id).status == "in_progress"
        assert tasks.get_task(task.id).owner == owner
        assert tasks.get_task(task.id).created_at == new_claim.created_at
        assert tasks.get_task(task.id).claim_id == new_claim.claim_id
    finally:
        await runner.terminate(spawned.agent_id)


@pytest.mark.parametrize("delete", [False, True])
@pytest.mark.asyncio
async def test_deactivating_busy_member_revokes_work_before_more_side_effects(
    tmp_path, teams, monkeypatch, delete
):
    started = asyncio.Event()
    release = asyncio.Event()
    side_effects = []

    async def execute(**kwargs):
        if kwargs["prompt"] == "work":
            started.set()
            await release.wait()
            side_effects.append("executed after revocation")
        return "done"

    _, runner, spawned = await _spawn(tmp_path, teams, monkeypatch, execute)
    tasks = teams.task_service("outcomes")
    task = tasks.create_task("work")
    loop = runner._tasks[spawned.agent_id]
    try:
        await asyncio.wait_for(started.wait(), 2)
        teams.set_member_active("outcomes", spawned.agent_id, False)
        if delete:
            teams.delete_team("outcomes")
        await _eventually(lambda: loop.done())
        release.set()
        await asyncio.gather(loop, return_exceptions=True)
        assert side_effects == []
        if delete:
            assert not (teams.tasks_root / "outcomes").exists()
            assert not (teams.teams_root / "outcomes").exists()
        else:
            assert tasks.get_task(task.id).status == "cancelled"
    finally:
        await runner.terminate(spawned.agent_id)
        await asyncio.gather(loop, return_exceptions=True)


def test_deleted_team_rejects_fresh_and_retained_task_handles(teams):
    tasks = teams.task_service("outcomes")
    tasks.create_task("work")
    teams.delete_team("outcomes")
    with pytest.raises(KeyError):
        teams.task_service("outcomes")
    with pytest.raises(KeyError):
        tasks.create_task("late work")
    assert not (teams.tasks_root / "outcomes").exists()


def test_recreated_team_does_not_accept_old_task_handle(teams):
    tasks = teams.task_service("outcomes")
    tasks.create_task("old")
    teams.delete_team("outcomes")
    teams.create_team("outcomes")
    current = teams.task_service("outcomes")
    replacement = current.create_task("new")
    with pytest.raises(KeyError):
        tasks.update_status(replacement.id, "completed")
    assert current.get_task(replacement.id).status == "pending"


def test_completion_hook_cannot_roll_back_concurrent_reassignment(teams, monkeypatch):
    tasks = teams.task_service("outcomes")
    task = tasks.create_task("work")
    tasks.update_task(task.id, owner="first", status="in_progress")

    def hook(**kwargs):
        tasks.update_task(task.id, owner="replacement", status="pending")
        return SimpleNamespace(blocked=True, block_reason="not accepted")

    monkeypatch.setattr(
        "koder_agent.harness.agents.teams.task_service.dispatch_project_hook_event", hook
    )
    with pytest.raises(RuntimeError):
        tasks.update_status(task.id, "completed")
    persisted = tasks.get_task(task.id)
    assert persisted.owner == "replacement"
    assert persisted.status == "pending"


def test_creation_hook_cannot_expose_unaccepted_task(teams, monkeypatch):
    tasks = teams.task_service("outcomes")
    visible = []

    def hook(**kwargs):
        visible.extend(tasks.list_tasks())
        return SimpleNamespace(blocked=True, block_reason="not accepted")

    monkeypatch.setattr(
        "koder_agent.harness.agents.teams.task_service.dispatch_project_hook_event", hook
    )
    with pytest.raises(RuntimeError):
        tasks.create_task("work")
    assert visible == []
    assert tasks.list_tasks() == []


def test_failed_dependency_stays_blocked_but_does_not_keep_owner_busy(tmp_path):
    tasks = TeamTaskService.for_test("outcomes", root=tmp_path)
    first = tasks.create_task("first")
    dependent = tasks.create_task("dependent", blocked_by=[first.id])
    independent = tasks.create_task("independent")
    tasks.claim_task(first.id, "worker")
    tasks.update_status(first.id, "failed")
    assert tasks.claim_task(dependent.id, "worker").reason == "blocked"
    assert tasks.claim_task(independent.id, "worker", check_agent_busy=True).success


def test_claim_is_idempotent_and_retry_rotates_identity(tmp_path):
    tasks = TeamTaskService.for_test("outcomes", root=tmp_path)
    task = tasks.create_task("work")
    first = tasks.claim_task(task.id, "worker").task
    assert tasks.claim_task(task.id, "worker").task == first
    tasks.update_status(task.id, "failed")
    with pytest.raises(ValueError, match="pending"):
        tasks.update_status(task.id, "in_progress")
    tasks.update_task(task.id, status="pending", owner="worker")
    second = tasks.claim_task(task.id, "worker").task
    assert second.claim_id != first.claim_id
    with pytest.raises(RuntimeError, match="claim"):
        tasks.finish_claim(task.id, "worker", first.claim_id, "completed")
    assert tasks.finish_claim(task.id, "worker", second.claim_id, "completed").status == "completed"


def test_member_bound_callbacks_do_not_survive_reactivation_or_team_recreation(teams):
    teams.add_member("outcomes", "worker")
    old = teams.bind_member("outcomes", "worker")
    teams.set_member_active("outcomes", "worker", False)
    teams.set_member_active("outcomes", "worker", True)
    with pytest.raises(KeyError):
        old.route("outcomes", "late result", sender="worker")
    current = teams.bind_member("outcomes", "worker")
    teams.set_member_active("outcomes", "worker", False)
    teams.delete_team("outcomes")
    teams.create_team("outcomes")
    teams.add_member("outcomes", "worker")
    with pytest.raises(KeyError):
        current.record_run(
            "outcomes",
            agent_id="worker",
            member_name="worker",
            prompt="old work",
            output="late",
            state="completed",
        )
    assert teams.history_entries("outcomes") == []


@pytest.mark.parametrize("operation", ["create", "complete"])
def test_delayed_hooks_cannot_publish_into_a_recreated_team(teams, monkeypatch, operation):
    tasks = teams.task_service("outcomes")
    task = tasks.create_task("old work")
    entered, release = Event(), Event()

    def hook(**kwargs):
        entered.set()
        assert release.wait(3)
        return SimpleNamespace(blocked=False)

    monkeypatch.setattr(
        "koder_agent.harness.agents.teams.task_service.dispatch_project_hook_event", hook
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            tasks.create_task if operation == "create" else tasks.update_status,
            *("late work",) if operation == "create" else (task.id, "completed"),
        )
        try:
            assert entered.wait(2)
            teams.delete_team("outcomes")
            teams.create_team("outcomes")
        finally:
            release.set()
        with pytest.raises(KeyError):
            future.result(timeout=2)
    assert teams.task_service("outcomes").list_tasks() == []


def test_idle_hook_does_not_overwrite_new_member_settings(teams, monkeypatch):
    teams.add_member("outcomes", "worker", mode="default")
    entered, release = Event(), Event()

    def hook(**kwargs):
        entered.set()
        assert release.wait(3)
        return SimpleNamespace(blocked=False)

    monkeypatch.setattr(
        "koder_agent.harness.agents.teams.service.dispatch_project_hook_event", hook
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(teams.set_member_active, "outcomes", "worker", False)
        try:
            assert entered.wait(2)
            teams.set_member_mode("outcomes", "worker", "plan")
        finally:
            release.set()
        future.result(timeout=2)
    member = teams.member_records("outcomes")[0]
    assert member.mode == "plan"
    assert not member.is_active


def _write_mode_in_process(root, ready, begin, entered, release):
    teams = TeamService.for_test(root=root)
    write = teams._write_config

    def paused(team_id, payload):
        entered.set()
        if not release.wait(3):
            raise TimeoutError("controlled writer not released")
        write(team_id, payload)

    teams._write_config = paused
    ready.set()
    if not begin.wait(30):
        raise TimeoutError("writer was not started")
    teams.set_member_mode("outcomes", "worker", "plan")


def _delete_in_process(root, ready, begin, entered, done, startup_delay):
    teams = TeamService.for_test(root=root)
    time.sleep(startup_delay)
    ready.set()
    if not begin.wait(30):
        raise TimeoutError("deleter was not started")
    entered.set()
    teams.delete_team("outcomes")
    done.set()


@pytest.mark.parametrize("startup_delay", [0.0, 3.5], ids=["ready", "slow-startup"])
def test_delete_serializes_with_inflight_config_write_across_processes(
    tmp_path, teams, monkeypatch, startup_delay
):
    teams.add_member("outcomes", "worker", is_active=False)
    # Full pytest runs can have live SQLite/worker threads. Fresh interpreters
    # avoid inheriting their locks and get an explicit synthetic profile.
    child_home = tmp_path / "child-home"
    child_home.mkdir()
    monkeypatch.setenv("HOME", str(child_home))
    ctx = multiprocessing.get_context("spawn")
    writing, release, deleting, deleted = (ctx.Event() for _ in range(4))
    writer_ready, deleter_ready, begin_write, begin_delete = (ctx.Event() for _ in range(4))
    writer = ctx.Process(
        target=_write_mode_in_process,
        args=(tmp_path, writer_ready, begin_write, writing, release),
    )
    deleter = ctx.Process(
        target=_delete_in_process,
        args=(tmp_path, deleter_ready, begin_delete, deleting, deleted, startup_delay),
    )
    try:
        writer.start()
        deleter.start()
        # Interpreter imports and deliberate slow startup must finish before
        # the writer's three-second critical-section watchdog starts.
        assert writer_ready.wait(10)
        assert deleter_ready.wait(10)
        begin_write.set()
        assert writing.wait(10)
        begin_delete.set()
        assert deleting.wait(10)
        # Before the fix deletion finished while the paused writer was still
        # able to recreate config.json. Now deletion must wait for the writer.
        assert not deleted.wait(0.2)
    finally:
        begin_write.set()
        begin_delete.set()
        release.set()
        for process in (writer, deleter):
            if process.pid is not None:
                process.join(10)
                if process.is_alive():
                    process.terminate()
                    process.join(2)
        exitcodes = [process.exitcode for process in (writer, deleter)]
        for process in (writer, deleter):
            process.close()
    assert exitcodes == [0, 0]
    assert deleted.is_set()
    assert not (teams.teams_root / "outcomes").exists()
    assert not (teams.tasks_root / "outcomes").exists()


@pytest.mark.asyncio
async def test_aclose_joins_own_busy_and_idle_members_but_not_foreign_agent(
    tmp_path, teams, monkeypatch
):
    started = asyncio.Event()

    async def execute(**kwargs):
        if kwargs["prompt"] in {"busy", "foreign"}:
            started.set()
            await asyncio.Event().wait()
        return "done"

    agents, runner, first = await _spawn(tmp_path, teams, monkeypatch, execute)
    definition = runner._runtimes[first.agent_id].agent_definition
    busy = await runner.spawn_teammate(
        team_id="outcomes",
        name="busy-worker",
        agent_definition=definition,
        prompt="busy",
        cwd=tmp_path,
    )
    await asyncio.wait_for(started.wait(), 2)
    foreign = await agents.launch_background(
        agent_definition=definition, prompt="foreign", description="not owned", cwd=tmp_path
    )
    loops = tuple(runner._tasks.values())
    try:
        await asyncio.wait_for(runner.aclose(), 2)
        await runner.aclose()
        assert all(loop.done() for loop in loops)
        assert not runner._tasks
        assert not runner._runtimes
        assert not any(m.is_active for m in teams.member_records("outcomes"))
        assert agents.get(busy.agent_id).state == "cancelled"
        assert not agents._tasks[foreign.id].done()
        with pytest.raises(RuntimeError, match="closed"):
            await runner.spawn_teammate(
                team_id="outcomes",
                name="new",
                agent_definition=definition,
                prompt="no work",
                cwd=tmp_path,
            )
    finally:
        await agents.cancel_background(foreign.id)
        await runner.aclose()


@pytest.mark.asyncio
async def test_resume_failure_marks_task_failed_then_picks_up_next(tmp_path, teams, monkeypatch):
    async def execute(**kwargs):
        return "done"

    agents, runner, spawned = await _spawn(tmp_path, teams, monkeypatch, execute)
    resume = agents.resume_background

    async def reject_bad(**kwargs):
        if kwargs["prompt"] == "bad":
            raise RuntimeError("synthetic launch failure")
        return await resume(**kwargs)

    monkeypatch.setattr(agents, "resume_background", reject_bad)
    tasks = teams.task_service("outcomes")
    bad = tasks.create_task("bad")
    good = tasks.create_task("good")
    try:
        await _eventually(lambda: tasks.get_task(good.id).status == "completed")
        assert tasks.get_task(bad.id).status == "failed"
        assert runner.manages(spawned.agent_id)
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_cancelled_work_can_be_explicitly_retried_without_restarting_member(
    tmp_path, teams, monkeypatch
):
    attempts = 0

    async def execute(**kwargs):
        nonlocal attempts
        if kwargs["prompt"] == "work":
            attempts += 1
            if attempts == 1:
                raise asyncio.CancelledError
        return "done"

    _, runner, spawned = await _spawn(tmp_path, teams, monkeypatch, execute)
    tasks = teams.task_service("outcomes")
    task = tasks.create_task("work")
    next_task = tasks.create_task("next")
    try:
        await _eventually(lambda: tasks.get_task(next_task.id).status == "completed")
        assert tasks.get_task(task.id).status == "cancelled"
        assert attempts == 1
        tasks.update_task(task.id, status="pending", owner=spawned.agent_id)
        await _eventually(lambda: tasks.get_task(task.id).status == "completed")
        assert attempts == 2
        assert runner.manages(spawned.agent_id)
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_rejected_completion_is_failed_not_reexecuted(tmp_path, teams, monkeypatch):
    calls = []

    async def execute(**kwargs):
        calls.append(kwargs["prompt"])
        return "done"

    _, runner, _ = await _spawn(tmp_path, teams, monkeypatch, execute)
    tasks = teams.task_service("outcomes")
    rejected = tasks.create_task("rejected")
    good = tasks.create_task("good")

    def hook(**kwargs):
        blocked = kwargs["event_name"] == "TaskCompleted" and kwargs["match_value"] == rejected.id
        return SimpleNamespace(blocked=blocked, block_reason="synthetic rejection")

    monkeypatch.setattr(
        "koder_agent.harness.agents.teams.task_service.dispatch_project_hook_event", hook
    )
    try:
        await _eventually(lambda: tasks.get_task(good.id).status == "completed")
        assert tasks.get_task(rejected.id).status == "failed"
        assert calls.count("rejected") == 1
    finally:
        await runner.aclose()


def test_lifetimes_are_unique_even_when_clock_does_not_advance(teams, monkeypatch):
    monkeypatch.setattr("koder_agent.harness.agents.teams.models._utc_now_iso", lambda: "frozen")
    monkeypatch.setattr("koder_agent.harness.agents.teams.service._utc_now_iso", lambda: "frozen")
    teams.delete_team("outcomes")
    teams.create_team("outcomes")
    first_team = teams.get("outcomes")
    first_member = teams.add_member("outcomes", "worker")
    bound = teams.bind_member("outcomes", "worker")
    teams.set_member_active("outcomes", "worker", False)
    teams.set_member_active("outcomes", "worker", True)
    second_member = teams.member_records("outcomes")[0]
    assert first_member.joined_at == second_member.joined_at
    assert first_member.generation != second_member.generation
    with pytest.raises(KeyError):
        bound.get("outcomes")
    teams.set_member_active("outcomes", "worker", False)
    teams.delete_team("outcomes")
    teams.create_team("outcomes")
    second_team = teams.get("outcomes")
    assert first_team.created_at == second_team.created_at
    assert first_team.generation != second_team.generation


@pytest.mark.asyncio
async def test_late_shutdown_ack_does_not_target_reactivated_membership(teams):
    teams.add_member("outcomes", "worker")
    teams.request_shutdown("outcomes", agent_id="worker")

    class Canceller:
        async def cancel_background(self, agent_id):
            teams.set_member_active("outcomes", agent_id, True)

    with pytest.raises(KeyError):
        await teams.respond_shutdown(
            "outcomes", agent_id="worker", approved=True, agent_service=Canceller()
        )
    assert teams.member_records("outcomes")[0].is_active
    assert not any(
        "shutdown approved" in m.content
        for m in teams.mailbox_entries("outcomes", recipient="worker")
    )
