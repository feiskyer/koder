"""Actual agent tools must share live state and deliver queued follow-up work."""

import asyncio
import json
from pathlib import Path

import pytest

from koder_agent.harness.agents import service as service_module
from koder_agent.harness.agents.definitions import AgentDefinition
from koder_agent.tools.agent import agent_tool
from koder_agent.tools.send_message import _send_message_impl, send_message

DEFINITION = AgentDefinition(
    agent_type="general-purpose",
    when_to_use="Synthetic tests",
    system_prompt="Synthetic worker",
    source="built-in",
)


@pytest.fixture
def runtime_profile(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(service_module, "_redacted_model_config_snapshot", lambda _definition: {})
    return tmp_path


async def invoke(tool, **arguments):
    return json.loads(await tool.on_invoke_tool(None, json.dumps(arguments)))


@pytest.mark.asyncio
async def test_separate_tool_calls_share_name_and_consume_followup(runtime_profile, monkeypatch):
    created = []
    original_service = service_module.AgentService
    started, release = asyncio.Event(), asyncio.Event()
    prompts = []

    class TrackedService(original_service):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            created.append(self)

    async def execute(**kwargs):
        prompts.append(kwargs["prompt"])
        if len(prompts) == 1:
            started.set()
            await release.wait()
        return "synthetic result"

    monkeypatch.setattr(service_module, "AgentService", TrackedService)
    monkeypatch.setattr(service_module, "_execute_agent_run", execute)
    try:
        launched = await invoke(
            agent_tool,
            description="Test worker",
            prompt="initial",
            name="researcher",
            run_in_background=True,
        )
        await asyncio.wait_for(started.wait(), timeout=3)
        sent = await invoke(send_message, to="researcher", message="follow up marker")
        assert sent["status"] == "sent", sent
        assert sent["delivery"] == "queued"
        release.set()
        await asyncio.wait_for(created[0].wait(launched["agent_id"]), timeout=3)
        assert len(created) == 1
        assert len(prompts) == 2
        assert "follow up marker" in prompts[1]
    finally:
        release.set()
        tasks = [task for service in created for task in service._tasks.values()]
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_historical_copy_cannot_report_message_delivery(runtime_profile):
    first = service_module.AgentService.for_test(runtime_profile)
    agent_id = first.spawn("worker")
    other = service_module.AgentService(output_root=first.output_root)
    result = json.loads(
        await _send_message_impl(
            to=agent_id,
            message="must not be lost in a disconnected mailbox",
            _agent_service=other,
        )
    )
    assert result["status"] == "error"
    assert first.read_mailbox(agent_id) == []


@pytest.mark.asyncio
async def test_cancel_waiter_does_not_cancel_observed_agent(runtime_profile):
    service = service_module.AgentService.for_test(runtime_profile)
    started, release = asyncio.Event(), asyncio.Event()

    async def execute(**_kwargs):
        started.set()
        await release.wait()
        return "finished"

    record = await service.launch_background(
        agent_definition=DEFINITION,
        prompt="initial",
        description="observe",
        cwd=runtime_profile,
        executor=execute,
    )
    await asyncio.wait_for(started.wait(), timeout=3)
    waiter = asyncio.create_task(service.wait(record.id))
    await asyncio.sleep(0)
    try:
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert service.get(record.id).state == "in_progress"
        release.set()
        assert (await service.wait(record.id)).state == "completed"
    finally:
        release.set()
        await asyncio.gather(*service._tasks.values(), return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelling_completed_agent_preserves_result(runtime_profile):
    service = service_module.AgentService.for_test(runtime_profile)

    async def execute(**_kwargs):
        return "completed evidence"

    record = await service.launch_background(
        agent_definition=DEFINITION,
        prompt="initial",
        description="completed",
        cwd=runtime_profile,
        executor=execute,
    )
    await service.wait(record.id)
    result = await service.cancel_background(record.id)
    assert result.state == "completed"
    assert Path(record.output_path).read_text() == "completed evidence"


@pytest.mark.asyncio
async def test_queued_messages_survive_cancellation_before_delivery(runtime_profile):
    service = service_module.AgentService.for_test(runtime_profile)
    started = asyncio.Event()

    async def execute(**_kwargs):
        started.set()
        await asyncio.Event().wait()

    record = await service.launch_background(
        agent_definition=DEFINITION,
        prompt="initial",
        description="queued",
        cwd=runtime_profile,
        executor=execute,
    )
    await asyncio.wait_for(started.wait(), timeout=3)
    service.send(record.id, "queued before cancellation")
    await service.cancel_background(record.id)
    observed = []

    async def resume(**kwargs):
        observed.append(kwargs["prompt"])
        return "resumed"

    await service.resume_background(
        agent_id=record.id,
        agent_definition=DEFINITION,
        prompt="resume explicitly",
        cwd=runtime_profile,
        executor=resume,
    )
    await service.wait(record.id)
    assert any("queued before cancellation" in prompt for prompt in observed)


@pytest.mark.asyncio
async def test_tool_explicit_resume_preserves_agent_and_pending_message(
    runtime_profile, monkeypatch
):
    from koder_agent.harness.agents.runtime_context import get_runtime_agent_service

    prompts = []

    async def execute(**kwargs):
        prompts.append(kwargs["prompt"])
        return "result"

    monkeypatch.setattr(service_module, "_execute_agent_run", execute)
    launched = await invoke(
        agent_tool,
        description="Resume test",
        prompt="initial",
        name="resumable",
        run_in_background=True,
    )
    service = get_runtime_agent_service()
    await service.wait(launched["agent_id"])
    before = service.get(launched["agent_id"])
    sent = await invoke(send_message, to="resumable", message="queued while stopped")
    assert sent["agent_stopped"]
    resumed = await invoke(
        agent_tool,
        description="Continue test",
        prompt="continue",
        resume="resumable",
    )
    assert resumed["status"] == "async_resumed"
    assert resumed["agent_id"] == before.id
    after = await service.wait(before.id)
    assert after.session_id == before.session_id
    assert len(prompts) == 2
    assert "queued while stopped" in prompts[-1]


@pytest.mark.asyncio
async def test_pending_message_persists_for_explicit_service_recovery(runtime_profile):
    service = service_module.AgentService.for_test(runtime_profile)

    async def execute(**_kwargs):
        return "first result"

    record = await service.launch_background(
        agent_definition=DEFINITION,
        prompt="first",
        description="persistent queue",
        cwd=runtime_profile,
        executor=execute,
    )
    await service.wait(record.id)
    service.send(record.id, "saved pending message")
    recovered = service_module.AgentService(output_root=service.output_root)
    observed = []

    async def resumed(**kwargs):
        observed.append(kwargs["prompt"])
        return "recovered"

    await recovered.resume_background(
        agent_id=record.id,
        agent_definition=DEFINITION,
        prompt="explicit resume",
        cwd=runtime_profile,
        executor=resumed,
    )
    await recovered.wait(record.id)
    assert len(observed) == 1 and "saved pending message" in observed[0]
    assert recovered.get(record.id).pending_messages == ()


@pytest.mark.asyncio
async def test_service_close_joins_only_owned_runs_and_blocks_admission(runtime_profile):
    first = service_module.AgentService.for_test(runtime_profile / "first")
    second = service_module.AgentService.for_test(runtime_profile / "second")
    first_started, second_started = asyncio.Event(), asyncio.Event()
    first_closed, second_closed = asyncio.Event(), asyncio.Event()

    async def execute(started, closed, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    a = await first.launch_background(
        agent_definition=DEFINITION,
        prompt="first",
        description="close",
        cwd=runtime_profile,
        executor=lambda **kw: execute(first_started, first_closed, **kw),
    )
    b = await second.launch_background(
        agent_definition=DEFINITION,
        prompt="second",
        description="keep",
        cwd=runtime_profile,
        executor=lambda **kw: execute(second_started, second_closed, **kw),
    )
    await asyncio.wait_for(asyncio.gather(first_started.wait(), second_started.wait()), timeout=3)
    try:
        await first.aclose()
        await first.aclose()
        assert first_closed.is_set() and not second_closed.is_set()
        assert first.get(a.id).state == "cancelled"
        assert second.get(b.id).state == "in_progress"
        with pytest.raises(RuntimeError, match="closed"):
            await first.launch_background(
                agent_definition=DEFINITION,
                prompt="rejected",
                description="closed",
                cwd=runtime_profile,
            )
        with pytest.raises(RuntimeError, match="closed"):
            first.send(a.id, "must not be accepted")
    finally:
        await second.aclose()


@pytest.mark.asyncio
async def test_foreign_runtime_cannot_resume_an_active_execution(runtime_profile):
    first = service_module.AgentService.for_test(runtime_profile)
    started, release = asyncio.Event(), asyncio.Event()

    async def execute(**_kwargs):
        started.set()
        await release.wait()
        return "first"

    record = await first.launch_background(
        agent_definition=DEFINITION,
        prompt="initial",
        description="lease",
        cwd=runtime_profile,
        executor=execute,
    )
    await asyncio.wait_for(started.wait(), timeout=3)
    second = service_module.AgentService(output_root=first.output_root)
    before = first._record_path(record.id).read_bytes()
    try:
        with pytest.raises(RuntimeError, match="another runtime"):
            await second.resume_background(
                agent_id=record.id,
                agent_definition=DEFINITION,
                prompt="must not run",
                cwd=runtime_profile,
                executor=execute,
            )
        assert first._record_path(record.id).read_bytes() == before
        release.set()
        await first.wait(record.id)

        async def resumed(**_kwargs):
            return "second"

        await second.resume_background(
            agent_id=record.id,
            agent_definition=DEFINITION,
            prompt="explicit takeover",
            cwd=runtime_profile,
            executor=resumed,
        )
        await second.wait(record.id)
        with pytest.raises(ValueError, match="another runtime"):
            first.send(record.id, "stale sender must not publish over new owner")
        assert second.get(record.id).pending_messages == ()
    finally:
        release.set()
        await first.aclose()
        await second.aclose()


@pytest.mark.asyncio
async def test_close_before_executor_starts_records_cancellation(runtime_profile):
    service = service_module.AgentService.for_test(runtime_profile)
    executed = False

    async def execute(**_kwargs):
        nonlocal executed
        executed = True
        await asyncio.Event().wait()

    record = await service.launch_background(
        agent_definition=DEFINITION,
        prompt="initial",
        description="pre-start close",
        cwd=runtime_profile,
        executor=execute,
    )
    await service.aclose()
    assert service.get(record.id).state == "cancelled"
    assert not executed
    assert not service._tasks


@pytest.mark.asyncio
async def test_failed_launch_binding_does_not_leave_a_running_record(runtime_profile):
    service = service_module.AgentService.for_test(runtime_profile)

    def rejected(_record):
        raise RuntimeError("synthetic owner disappeared")

    with pytest.raises(RuntimeError, match="synthetic owner"):
        await service.launch_background(
            agent_definition=DEFINITION,
            prompt="initial",
            description="rejected",
            cwd=runtime_profile,
            team_context_builder=rejected,
        )
    assert not service._tasks
    assert not service._run_claims
    records = service.list_records()
    assert len(records) == 1 and records[0].state == "failed"
    restored = service_module.AgentService(output_root=service.output_root)
    assert restored.get(records[0].id).state == "failed"


def test_invalid_record_identity_and_output_path_are_not_loaded(runtime_profile):
    import dataclasses

    service = service_module.AgentService.for_test(runtime_profile)
    valid_id = service.spawn("worker")
    data = dataclasses.asdict(service.get(valid_id))
    external = runtime_profile / "external.txt"
    external.write_text("not an agent artifact")
    data["output_path"] = str(external)
    service._record_path(valid_id).write_text(json.dumps(data))
    invalid = dict(data, id="../outside")
    (service.output_root / "agent-invalid.json").write_text(json.dumps(invalid))
    loaded = service_module.AgentService(output_root=service.output_root)
    assert loaded.list_records() == []
    assert external.read_text() == "not an agent artifact"
