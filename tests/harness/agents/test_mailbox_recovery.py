"""Explicit recovery retains a mailbox message whose assigned run failed."""

import asyncio
from pathlib import Path

import pytest

from koder_agent.harness.agents.definitions import AgentDefinition
from koder_agent.harness.agents.service import AgentService


@pytest.mark.asyncio
async def test_claimed_message_survives_failed_attempt_and_service_recreation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    monkeypatch.setattr(
        "koder_agent.harness.agents.service._redacted_model_config_snapshot", lambda _: {}
    )
    service = AgentService.for_test(tmp_path)
    definition = AgentDefinition("worker", "synthetic", "synthetic", "built-in")
    started, release = asyncio.Event(), asyncio.Event()

    async def execute(**kwargs):
        if kwargs["prompt"] == "initial":
            started.set()
            await release.wait()
            return "initial result"
        raise RuntimeError("synthetic failure before model accepted the message")

    record = await service.launch_background(
        agent_definition=definition,
        prompt="initial",
        description="recovery",
        cwd=tmp_path,
        executor=execute,
    )
    await asyncio.wait_for(started.wait(), timeout=3)
    service.send(record.id, "accepted message must survive")
    release.set()
    assert (await service.wait(record.id)).state == "failed"

    recovered = AgentService(output_root=service.output_root)
    prompts = []

    async def resume(**kwargs):
        prompts.append(kwargs["prompt"])
        return "recovered result"

    await recovered.resume_background(
        agent_id=record.id,
        agent_definition=definition,
        prompt="recover explicitly",
        cwd=tmp_path,
        executor=resume,
    )
    assert (await recovered.wait(record.id)).state == "completed"
    assert any("accepted message must survive" in prompt for prompt in prompts)
