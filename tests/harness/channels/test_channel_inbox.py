"""Owned channel staging, quotas, cancellation and retained outcomes."""

import asyncio
import json
import os
import stat
import threading

import pytest
import pytest_asyncio

from koder_agent.harness.channels.inbox import (
    ChannelInbox,
    ChannelInboxLimits,
    ChannelInboxRejectedError,
    ChannelInboxStorageError,
)


@pytest_asyncio.fixture
async def inbox(tmp_path):
    instance = ChannelInbox(root=tmp_path / "inboxes")
    try:
        await instance.open()
        yield instance
    finally:
        if instance.snapshot()["status"] != "closed":
            await instance.aclose()


async def finish_all(inbox, count):
    for _ in range(count):
        message = await inbox.get()
        await inbox.finish(message.id, "completed")


@pytest.mark.asyncio
async def test_burst_keeps_only_bounded_descriptors_in_memory_and_preserves_fifo(inbox):
    payload = "x" * 4096
    identifiers = [await inbox.put("synthetic", f"{number}:{payload}") for number in range(128)]
    assert inbox.snapshot()["pending"] == 128
    assert inbox.snapshot()["retained_bytes"] > 128 * len(payload)
    assert all(
        set(vars(entry)) == {"id", "size", "digest", "state", "reason"}
        for entry in inbox._entries.values()
    )
    assert len(list(inbox.directory.glob("*.pending.json"))) == 128
    for number, identifier in enumerate(identifiers):
        message = await inbox.get()
        assert message.id == identifier
        assert message.source == "synthetic"
        assert message.content == f"{number}:{payload}"
        await inbox.finish(identifier, "completed")
    assert inbox.snapshot()["completed"] == 128
    assert inbox.snapshot()["retained_bytes"] == 0
    directory = inbox.directory
    await inbox.aclose()
    assert not directory.exists()


@pytest.mark.asyncio
async def test_capacity_rejects_immediately_without_waiting_for_a_running_consumer(tmp_path):
    inbox = ChannelInbox(
        root=tmp_path,
        limits=ChannelInboxLimits(max_messages=1, max_bytes=8192, max_message_bytes=4096),
    )
    await inbox.open()
    try:
        await inbox.put("source", "first")
        first = await inbox.get()
        with pytest.raises(ChannelInboxRejectedError, match="message_capacity"):
            await asyncio.wait_for(inbox.put("source", "second"), 1)
        assert inbox.snapshot()["running"] == 1
        assert inbox.snapshot()["accepted"] == 1
        await inbox.finish(first.id, "completed")
        await inbox.put("source", "third")
        third = await inbox.get()
        assert third.content == "third"
        await inbox.finish(third.id, "completed")
    finally:
        await inbox.aclose()
    manifest = json.loads((inbox.directory / "manifest.json").read_text())
    assert manifest["rejections"] == {"message_capacity": 1}
    assert manifest["entries"] == []


@pytest.mark.asyncio
async def test_byte_quota_is_released_only_after_successful_delivery(tmp_path):
    inbox = ChannelInbox(
        root=tmp_path,
        limits=ChannelInboxLimits(max_messages=10, max_bytes=512, max_message_bytes=512),
    )
    await inbox.open()
    try:
        await inbox.put("source", "x" * 200)
        with pytest.raises(ChannelInboxRejectedError, match="byte_capacity"):
            await inbox.put("source", "y" * 200)
        message = await inbox.get()
        await inbox.finish(message.id, "failed", reason="SyntheticError")
        with pytest.raises(ChannelInboxRejectedError, match="byte_capacity"):
            await inbox.put("source", "y" * 200)
        assert inbox.snapshot()["failed"] == 1
    finally:
        await inbox.aclose()


@pytest.mark.asyncio
async def test_oversized_and_invalid_text_is_not_stored(tmp_path):
    inbox = ChannelInbox(
        root=tmp_path,
        limits=ChannelInboxLimits(max_messages=4, max_bytes=4096, max_message_bytes=256),
    )
    await inbox.open()
    try:
        with pytest.raises(ChannelInboxRejectedError, match="message_too_large"):
            await inbox.put("source", "x" * 257)
        with pytest.raises(ChannelInboxRejectedError, match="invalid_encoding"):
            await inbox.put("source", "\ud800")
        assert inbox.snapshot()["retained_messages"] == 0
        assert list(inbox.directory.iterdir()) == []
    finally:
        await inbox.aclose()
    assert json.loads((inbox.directory / "manifest.json").read_text())["rejected"] == 2


@pytest.mark.parametrize("outcome", ["failed", "cancelled", "interrupted"])
@pytest.mark.asyncio
async def test_non_success_retains_original_content_and_explicit_state(inbox, outcome):
    await inbox.put("source", "original message")
    message = await inbox.get()
    await inbox.finish(message.id, outcome, reason="SyntheticError")
    directory = inbox.directory
    await inbox.aclose()
    manifest = json.loads((directory / "manifest.json").read_text())
    entry = manifest["entries"][0]
    assert entry["state"] == outcome
    assert entry["reason"] == "SyntheticError"
    assert not manifest["automatic_replay"]
    payload = json.loads((directory / entry["file"]).read_text())
    assert payload["content"] == "original message"


@pytest.mark.asyncio
async def test_shutdown_preserves_pending_and_marks_inflight_uncertain(inbox):
    await inbox.put("source", "possibly started")
    await inbox.put("source", "not started")
    message = await inbox.get()
    await inbox.aclose()
    assert inbox.snapshot()["interrupted"] == 1
    assert inbox.snapshot()["pending"] == 1
    assert await inbox.get() is None
    with pytest.raises(ChannelInboxRejectedError, match="closed"):
        await inbox.put("source", "too late")
    manifest = json.loads((inbox.directory / "manifest.json").read_text())
    assert [(item["id"], item["state"]) for item in manifest["entries"]] == [
        (message.id, "interrupted"),
        (message.id + 1, "pending"),
    ]


@pytest.mark.asyncio
async def test_cancelled_put_joins_write_and_adopts_committed_message(inbox, monkeypatch):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = inbox._publish

    def delayed(path, data):
        started.set()
        assert release.wait(3)
        try:
            original(path, data)
        finally:
            finished.set()

    monkeypatch.setattr(inbox, "_publish", delayed)
    task = asyncio.create_task(inbox.put("source", "committed despite waiter cancellation"))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert inbox.snapshot()["accepted"] == inbox.snapshot()["pending"] == 1
        message = await inbox.get()
        assert message.content == "committed despite waiter cancellation"
        await inbox.finish(message.id, "completed")
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_get_preserves_its_claim_for_shutdown_review(inbox, monkeypatch):
    await inbox.put("source", "retained")
    started, release = threading.Event(), threading.Event()
    original = inbox._read_and_claim

    def delayed(entry):
        started.set()
        assert release.wait(3)
        return original(entry)

    monkeypatch.setattr(inbox, "_read_and_claim", delayed)
    task = asyncio.create_task(inbox.get())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert inbox.snapshot()["running"] == 1
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    await inbox.aclose()
    assert inbox.snapshot()["interrupted"] == 1
    assert len(list(inbox.directory.glob("*.interrupted.json"))) == 1


@pytest.mark.asyncio
async def test_shutdown_joins_admission_and_rejects_later_work(inbox, monkeypatch):
    started, release = threading.Event(), threading.Event()
    original = inbox._publish

    def delayed(path, data):
        if path.name.endswith(".pending.json"):
            started.set()
            assert release.wait(3)
        original(path, data)

    monkeypatch.setattr(inbox, "_publish", delayed)
    producer = asyncio.create_task(inbox.put("source", "started before close"))
    close = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        close = asyncio.create_task(inbox.aclose())
        await asyncio.sleep(0)
        with pytest.raises(ChannelInboxRejectedError, match="closed"):
            await inbox.put("source", "late")
        assert not close.done()
        release.set()
        await producer
        await close
        assert inbox.snapshot()["pending"] == 1
    finally:
        release.set()
        await asyncio.gather(producer, *([close] if close else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_storage_failure_stops_admission_and_keeps_previously_accepted_data(
    inbox, monkeypatch
):
    await inbox.put("source", "keep this")
    original = inbox._publish

    def fail_payload(path, data):
        if path.name.endswith(".pending.json"):
            raise OSError("synthetic full disk")
        original(path, data)

    monkeypatch.setattr(inbox, "_publish", fail_payload)
    with pytest.raises(ChannelInboxStorageError):
        await inbox.put("source", "cannot store")
    with pytest.raises(ChannelInboxRejectedError, match="storage_unavailable"):
        await inbox.put("source", "also cannot store")
    with pytest.raises(ChannelInboxStorageError):
        await inbox.get()
    await inbox.aclose()
    files = list(inbox.directory.glob("*.pending.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["content"] == "keep this"
    assert inbox.snapshot()["rejected"] == 2


@pytest.mark.parametrize("kind", ["content", "symlink"])
@pytest.mark.asyncio
async def test_altered_payload_is_not_delivered(inbox, tmp_path, kind):
    await inbox.put("source", "original")
    path = next(inbox.directory.glob("*.pending.json"))
    if kind == "content":
        path.write_text(path.read_text().replace("original", "modified"))
    else:
        other = tmp_path / "other.txt"
        other.write_text("must not be read as a channel")
        path.unlink()
        path.symlink_to(other)
    with pytest.raises(ChannelInboxStorageError):
        await inbox.get()
    assert inbox.snapshot()["completed"] == 0
    assert inbox.snapshot()["pending"] == 1


@pytest.mark.asyncio
async def test_disk_work_does_not_block_event_loop(inbox, monkeypatch, event_loop_progress_probe):
    wrap, observed = event_loop_progress_probe
    monkeypatch.setattr(inbox, "_publish", wrap(inbox._publish))
    await inbox.put("source", "heartbeat")
    assert observed == [True]
    await finish_all(inbox, 1)


@pytest.mark.asyncio
async def test_private_modes_and_unknown_files_are_preserved(inbox):
    await inbox.put("source", "private")
    if os.name == "posix":
        assert stat.S_IMODE(inbox.directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(next(inbox.directory.glob("*.json")).stat().st_mode) == 0o600
    await finish_all(inbox, 1)
    unknown = inbox.directory / "unrelated.txt"
    unknown.write_text("do not remove")
    with pytest.raises(OSError):
        await inbox.aclose()
    assert unknown.read_text() == "do not remove"
    # The failed close is a terminal, repeatable result, not another cleanup attempt.
    with pytest.raises(OSError):
        await inbox.aclose()


@pytest.mark.parametrize(
    "values",
    [
        {"max_messages": 0},
        {"max_messages": -1},
        {"max_bytes": 0},
        {"max_message_bytes": 0},
        {"max_message_bytes": 100, "max_bytes": 99},
    ],
)
def test_limits_cannot_be_disabled_or_inverted(values):
    with pytest.raises(ValueError):
        ChannelInboxLimits(**values)


def test_limits_read_only_documented_environment_keys(monkeypatch):
    monkeypatch.setenv("KODER_CHANNEL_MAX_PENDING_MESSAGES", "8")
    monkeypatch.setenv("KODER_CHANNEL_MAX_PENDING_BYTES", "8192")
    monkeypatch.setenv("KODER_CHANNEL_MAX_MESSAGE_BYTES", "4096")
    assert ChannelInboxLimits.from_env() == ChannelInboxLimits(8, 8192, 4096)
    monkeypatch.setenv("KODER_CHANNEL_MAX_PENDING_MESSAGES", "unbounded")
    with pytest.raises(ValueError):
        ChannelInboxLimits.from_env()


@pytest.mark.asyncio
async def test_cancelled_open_is_joined_before_its_directory_is_closed(tmp_path, monkeypatch):
    inbox = ChannelInbox(root=tmp_path)
    started, release = threading.Event(), threading.Event()
    original = inbox._allocate_directory
    allocated = []

    def delayed():
        directory = original()
        allocated.append(directory)
        started.set()
        assert release.wait(3)
        return directory

    monkeypatch.setattr(inbox, "_allocate_directory", delayed)
    opener = asyncio.create_task(inbox.open())
    closer = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        opener.cancel()
        closer = asyncio.create_task(inbox.aclose())
        await asyncio.sleep(0)
        assert not opener.done() and not closer.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await opener
        await closer
        assert inbox.snapshot()["status"] == "closed"
        assert not allocated[0].exists()
    finally:
        release.set()
        await asyncio.gather(opener, *([closer] if closer else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_concurrent_admission_keeps_first_writer_order(inbox, monkeypatch):
    started, release = threading.Event(), threading.Event()
    original = inbox._publish

    def delayed(path, data):
        if path.name.startswith("0000000000000001."):
            started.set()
            assert release.wait(3)
        original(path, data)

    monkeypatch.setattr(inbox, "_publish", delayed)
    first = asyncio.create_task(inbox.put("source", "first"))
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        second = asyncio.create_task(inbox.put("source", "second"))
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        await asyncio.gather(first, second)
        one, two = await inbox.get(), await inbox.get()
        assert (one.content, two.content) == ("first", "second")
        await inbox.finish(one.id, "completed")
        await inbox.finish(two.id, "completed")
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_same_root_runtimes_do_not_read_or_close_each_others_backlog(tmp_path):
    first, second = ChannelInbox(root=tmp_path), ChannelInbox(root=tmp_path)
    await first.open()
    await second.open()
    try:
        assert first.directory != second.directory
        await first.put("first", "retained in first")
        await second.put("second", "delivered only in second")
        await first.aclose()
        message = await second.get()
        assert message.source == "second"
        assert message.content == "delivered only in second"
        await second.finish(message.id, "completed")
        assert first.snapshot()["pending"] == 1
        assert second.snapshot()["completed"] == 1
    finally:
        await first.aclose()
        await second.aclose()


@pytest.mark.parametrize(
    "reason,limits",
    [
        ("message_capacity", ChannelInboxLimits(1, 8192, 4096)),
        ("byte_capacity", ChannelInboxLimits(4, 512, 512)),
    ],
)
@pytest.mark.asyncio
async def test_inflight_admission_reserves_capacity_before_waiting_for_disk(
    tmp_path, monkeypatch, reason, limits
):
    inbox = ChannelInbox(root=tmp_path, limits=limits)
    await inbox.open()
    started, release = threading.Event(), threading.Event()
    original = inbox._publish

    def delayed(path, data):
        if path.name.endswith(".pending.json"):
            started.set()
            assert release.wait(3)
        original(path, data)

    monkeypatch.setattr(inbox, "_publish", delayed)
    first = asyncio.create_task(inbox.put("first", "x" * 200))
    second = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        second = asyncio.create_task(inbox.put("second", "y" * 200))
        with pytest.raises(ChannelInboxRejectedError, match=reason):
            await asyncio.wait_for(asyncio.shield(second), 0.5)
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
        await inbox.aclose()


@pytest.mark.asyncio
async def test_cancelled_reserved_waiter_is_owned_through_shutdown(inbox, monkeypatch):
    started, release = threading.Event(), threading.Event()
    original = inbox._publish

    def delayed(path, data):
        if path.name.startswith("0000000000000001."):
            started.set()
            assert release.wait(3)
        original(path, data)

    monkeypatch.setattr(inbox, "_publish", delayed)
    first = asyncio.create_task(inbox.put("source", "first"))
    second = close = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        second = asyncio.create_task(inbox.put("source", "reserved before cancellation"))
        await asyncio.sleep(0)
        assert inbox.snapshot()["admitting"] == 2
        second.cancel()
        close = asyncio.create_task(inbox.aclose())
        await asyncio.sleep(0)
        assert not second.done() and not close.done()
        release.set()
        await first
        with pytest.raises(asyncio.CancelledError):
            await second
        await close
        assert inbox.snapshot()["admitting"] == 0
        assert inbox.snapshot()["admitting_bytes"] == 0
        assert inbox.snapshot()["pending"] == 2
        assert len(list(inbox.directory.glob("*.pending.json"))) == 2
    finally:
        release.set()
        await asyncio.gather(
            first,
            *([second] if second else []),
            *([close] if close else []),
            return_exceptions=True,
        )
