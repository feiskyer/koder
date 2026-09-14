"""The session's channel consumer survives one failed or cancelled message."""

import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

from koder_agent.harness import session_flow
from koder_agent.harness.channels.notification import (
    CHANNEL_NOTIFICATION_METHOD,
    wrap_channel_message,
)
from koder_agent.harness.cli.entrypoint import build_runtime_request, run_harness_runtime
from koder_agent.mcp.notifications import get_notification_handler


@pytest_asyncio.fixture
async def channel_runtime(tmp_path, monkeypatch, request):
    monkeypatch.chdir(tmp_path)
    for name, value in getattr(request, "param", {}).items():
        monkeypatch.setenv(name, str(value))
    observed = SimpleNamespace(
        ready=asyncio.Event(),
        release_startup=asyncio.Event(),
        first_started=asyncio.Event(),
        second_delivered=asyncio.Event(),
        blocked_cancelled=asyncio.Event(),
        first_outcome="success",
        messages=[],
        cleanups=0,
    )

    class MetadataSession:
        def __init__(self, session_id):
            self.session_id = session_id

        async def get_cwd(self):
            return None

        async def get_agent(self):
            return None

        @classmethod
        async def record_session_cwd(cls, session_id, cwd):
            pass

        def close(self):
            pass

    class ActiveSession(MetadataSession):
        async def get_items(self):
            # The production flow has registered and started its consumer here.
            # Hold startup without holding the scheduler lifecycle lock.
            observed.ready.set()
            await observed.release_startup.wait()
            return []

    class Scheduler:
        def __init__(self, session_id, **kwargs):
            self.session = ActiveSession(session_id)
            self.usage_tracker = SimpleNamespace()
            self.agent_definition = kwargs.get("agent_definition")

        async def _ensure_agent_initialized(self):
            observed.router = get_notification_handler().channel_router

        async def handle(self, prompt, **kwargs):
            self._last_turn_errored = False
            self._last_turn_cancelled = False
            observed.messages.append(prompt)
            if len(observed.messages) == 1:
                observed.first_started.set()
                if observed.first_outcome == "cancelled":
                    raise asyncio.CancelledError("synthetic message cancellation")
                if observed.first_outcome == "failed":
                    raise RuntimeError("synthetic message failure")
                if observed.first_outcome == "reported_failed":
                    self._last_turn_errored = True
                    return "synthetic failure rendered as text"
                if observed.first_outcome == "reported_cancelled":
                    self._last_turn_cancelled = True
                    return "synthetic cancellation rendered as text"
                if observed.first_outcome == "blocked":
                    try:
                        await asyncio.Event().wait()
                    finally:
                        observed.blocked_cancelled.set()
            else:
                observed.second_delivered.set()
            return "delivered"

        async def cleanup(self):
            observed.cleanups += 1

    original_owner = session_flow._SessionCleanupOwner

    class ObservedOwner(original_owner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observed.owner = self

        async def _run_auto_dream(self):
            pass

    monkeypatch.setattr("koder_agent.utils.setup_openai_client", lambda: None)
    monkeypatch.setattr("koder_agent.core.session.EnhancedSQLiteSession", MetadataSession)
    monkeypatch.setattr("koder_agent.core.scheduler.AgentScheduler", Scheduler)
    monkeypatch.setattr(
        "koder_agent.core.interactive.InteractivePrompt", lambda *args, **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(session_flow, "_SessionCleanupOwner", ObservedOwner)
    monkeypatch.setattr(session_flow, "_read_piped_stdin", lambda: None)
    request = build_runtime_request(
        ["--bare", "--session", "channel-owner", "--channels", "server:synthetic", "-p", "/exit"]
    )
    observed.application = asyncio.create_task(run_harness_runtime(request))
    try:
        await asyncio.wait_for(observed.ready.wait(), timeout=5)
        yield observed
    finally:
        observed.release_startup.set()
        await asyncio.wait_for(observed.application, timeout=5)


async def _send(runtime, content):
    await runtime.router.dispatch_raw_notification(
        "synthetic", CHANNEL_NOTIFICATION_METHOD, {"content": content}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("first_outcome", ["success", "failed", "cancelled"])
async def test_runtime_delivers_later_messages_after_one_turn_ends(channel_runtime, first_outcome):
    runtime = channel_runtime
    runtime.first_outcome = first_outcome
    await _send(runtime, "first")
    await asyncio.wait_for(runtime.first_started.wait(), timeout=2)
    assert not runtime.owner.channel_task.done(), "one message stopped the channel consumer"

    await _send(runtime, "second")
    await asyncio.wait_for(runtime.second_delivered.wait(), timeout=2)
    assert runtime.messages == [
        wrap_channel_message("synthetic", "first"),
        wrap_channel_message("synthetic", "second"),
    ]


@pytest.mark.asyncio
async def test_runtime_shutdown_cancels_and_joins_an_active_channel_turn(channel_runtime):
    runtime = channel_runtime
    runtime.first_outcome = "blocked"
    await _send(runtime, "active")
    await asyncio.wait_for(runtime.first_started.wait(), timeout=2)
    consumer = runtime.owner.channel_task
    runtime.release_startup.set()
    assert await asyncio.wait_for(asyncio.shield(runtime.application), timeout=5) == 0

    assert consumer.done()
    assert runtime.owner.channel_task is None
    assert runtime.blocked_cancelled.is_set()
    assert runtime.cleanups == 1
    await _send(runtime, "after shutdown")
    assert runtime.messages == [wrap_channel_message("synthetic", "active")]
