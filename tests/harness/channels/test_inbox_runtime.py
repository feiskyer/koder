"""Channel backlog admission must not wait for the model or hide incomplete work."""

import asyncio
import json

import anyio
import pytest
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCNotification, JSONRPCResponse
from pydantic import TypeAdapter

from koder_agent.harness.channels.interceptor import ChannelInterceptingStream
from koder_agent.harness.channels.notification import CHANNEL_NOTIFICATION_METHOD

from .test_consumer_liveness import _send
from .test_consumer_liveness import channel_runtime as channel_runtime


async def eventually(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_burst_is_staged_without_retaining_payloads_in_the_runtime_queue(channel_runtime):
    runtime = channel_runtime
    runtime.first_outcome = "blocked"
    await _send(runtime, "hold the consumer")
    await asyncio.wait_for(runtime.first_started.wait(), 2)
    inbox = runtime.owner.channel_inbox
    for number in range(128):
        await _send(runtime, f"{number}:" + "x" * 4096)
    snapshot = inbox.snapshot()
    assert snapshot["running"] == 1
    assert snapshot["pending"] == 128
    assert len(runtime.messages) == 1
    assert all(not hasattr(entry, "content") for entry in inbox._entries.values())
    assert len(list(inbox.directory.glob("*.pending.json"))) == 128


@pytest.mark.parametrize(
    "channel_runtime",
    [
        {
            "KODER_CHANNEL_MAX_PENDING_MESSAGES": 2,
            "KODER_CHANNEL_MAX_PENDING_BYTES": 8192,
            "KODER_CHANNEL_MAX_MESSAGE_BYTES": 4096,
        }
    ],
    indirect=True,
)
@pytest.mark.asyncio
async def test_full_inbox_does_not_block_the_following_mcp_response(channel_runtime, caplog):
    runtime = channel_runtime
    runtime.first_outcome = "blocked"
    await _send(runtime, "active model turn")
    await asyncio.wait_for(runtime.first_started.wait(), 2)
    await _send(runtime, "pending turn")
    inbox = runtime.owner.channel_inbox
    send, receive = anyio.create_memory_object_stream(2)
    rejected = SessionMessage(
        TypeAdapter(JSONRPCMessage).validate_python(
            JSONRPCNotification(
                jsonrpc="2.0", method=CHANNEL_NOTIFICATION_METHOD, params={"content": "overflow"}
            ).model_dump(by_alias=True)
        )
    )
    response = SessionMessage(
        TypeAdapter(JSONRPCMessage).validate_python(
            JSONRPCResponse(jsonrpc="2.0", id=7, result={}).model_dump(by_alias=True)
        )
    )
    send.send_nowait(rejected)
    send.send_nowait(response)
    async with (
        send,
        ChannelInterceptingStream(
            receive, runtime.router.dispatch_raw_notification, "synthetic"
        ) as stream,
    ):
        assert await asyncio.wait_for(stream.receive(), 2) is response
    assert inbox.snapshot()["rejected"] == 1
    assert inbox.snapshot()["accepted"] == 2
    assert "message_capacity" in caplog.text
    assert len(runtime.messages) == 1


@pytest.mark.parametrize(
    "outcome", ["failed", "cancelled", "reported_failed", "reported_cancelled"]
)
@pytest.mark.asyncio
async def test_non_success_is_retained_while_later_messages_still_run(channel_runtime, outcome):
    runtime = channel_runtime
    runtime.first_outcome = outcome
    inbox = runtime.owner.channel_inbox
    await _send(runtime, "unsuccessful first")
    await asyncio.wait_for(runtime.first_started.wait(), 2)
    await _send(runtime, "successful second")
    await eventually(lambda: inbox.snapshot()["completed"] == 1)
    snapshot = inbox.snapshot()
    expected = "cancelled" if outcome in {"cancelled", "reported_cancelled"} else "failed"
    assert snapshot[expected] == 1
    assert snapshot["retained_messages"] == 1
    assert not runtime.owner.channel_task.done()
    retained = next(
        path
        for path in inbox.directory.glob("*.json")
        if ".failed." in path.name or ".cancelled." in path.name
    )
    assert "unsuccessful first" in json.loads(retained.read_text())["content"]


@pytest.mark.asyncio
async def test_runtime_shutdown_preserves_backlog_and_revokes_callback(channel_runtime):
    runtime = channel_runtime
    runtime.first_outcome = "blocked"
    await _send(runtime, "interrupted active")
    await asyncio.wait_for(runtime.first_started.wait(), 2)
    await _send(runtime, "pending at exit")
    inbox = runtime.owner.channel_inbox
    directory = inbox.directory
    runtime.release_startup.set()
    assert await asyncio.wait_for(asyncio.shield(runtime.application), 5) == 0
    assert runtime.owner.channel_inbox is None
    assert inbox.snapshot()["status"] == "closed"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert [entry["state"] for entry in manifest["entries"]] == ["interrupted", "pending"]
    before = inbox.snapshot()
    await _send(runtime, "after retirement")
    assert inbox.snapshot() == before


@pytest.mark.asyncio
async def test_storage_fault_does_not_accept_more_work_or_spin_consumer(
    channel_runtime, monkeypatch, caplog
):
    runtime = channel_runtime
    inbox = runtime.owner.channel_inbox
    original = inbox._publish

    def fail_pending(path, data):
        if path.name.endswith(".pending.json"):
            raise OSError("synthetic storage failure")
        original(path, data)

    monkeypatch.setattr(inbox, "_publish", fail_pending)
    await _send(runtime, "cannot persist")
    await eventually(lambda: runtime.owner.channel_task.done())
    await _send(runtime, "must not be acknowledged")
    snapshot = inbox.snapshot()
    assert snapshot["status"] == "faulted"
    assert snapshot["accepted"] == 0
    assert snapshot["rejected"] == 2
    assert runtime.messages == []
    assert "not accepted" in caplog.text
    assert "consumer stopped" in caplog.text
