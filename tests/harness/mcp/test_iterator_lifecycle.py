"""Authorization and resource ownership at public MCP iterator boundaries."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from anyio import fail_after

from koder_agent.mcp.project_approvals import reset_project_choices, set_project_approval
from koder_agent.mcp.reconnection import LiveMCPServer
from koder_agent.mcp.runtime_authorization import (
    MCPAuthorizationError,
    attach_project_authorization_validator,
)
from koder_agent.mcp.server_manager import MCPServerManager


class _Transport:
    name = "iterator-test"

    def __init__(self):
        self.closed_streams = 0
        self.tool_calls = 0
        self.cleanups = 0
        self.session = SimpleNamespace(stream_events=self.stream_events)
        self.experimental = SimpleNamespace(stream_events=self.stream_events)

    async def stream_events(self):
        try:
            yield "first"
            yield "last"
        finally:
            self.closed_streams += 1

    async def call_tool(self, name, arguments=None):
        self.tool_calls += 1
        return "contacted"

    async def cleanup(self):
        self.cleanups += 1


def _stream(handle, path):
    for part in path:
        handle = getattr(handle, part)
    return handle().__aiter__()


async def _approved_handle(monkeypatch, tmp_path, *, live=True):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    project.mkdir()
    (project / ".mcp.json").write_text(
        json.dumps(
            {"mcpServers": {"iterator-test": {"type": "http", "url": "https://example.test/mcp"}}}
        ),
        encoding="utf-8",
    )
    config = await MCPServerManager().get_server("iterator-test", cwd=project, scope="project")
    assert config is not None
    set_project_approval(
        project_root=config.project_root,
        source_path=config.source_path,
        source_digest=config.source_digest,
        approved=True,
    )
    transport = _Transport()
    handle = LiveMCPServer("iterator-test", transport) if live else transport
    validator = attach_project_authorization_validator(handle, config)
    assert validator is not None
    return project, transport, handle, validator


_STREAM_PATHS = [
    ("session", "stream_events"),
    ("stream_events",),
    ("experimental", "stream_events"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", _STREAM_PATHS)
@pytest.mark.parametrize("drift", ["reset", "source"])
async def test_paused_stream_does_not_authorize_new_call_after_reset(
    monkeypatch, tmp_path, path, drift
):
    project, transport, handle, validator = await _approved_handle(monkeypatch, tmp_path)
    iterator = _stream(handle, path)
    try:
        assert await anext(iterator) == "first"
        assert validator.in_flight == 1
        if drift == "reset":
            assert reset_project_choices(project) == 1
        else:
            source = project / ".mcp.json"
            source.write_text("{}\n", encoding="utf-8")

        # Stay in the consumer task: create_task/wait_for would hide context
        # leaked across yield by changing the task identity checked by admission.
        with pytest.raises(MCPAuthorizationError):
            with fail_after(1):
                await handle.call_tool("new-request", {})
        assert transport.tool_calls == 0
        assert transport.cleanups == 0
    finally:
        # Already-admitted work may finish after reset. Exhaustion also keeps
        # this regression's baseline cleanup independent of early-close bugs.
        async for _ in iterator:
            pass
        await handle.cleanup()
    assert validator.in_flight == 0
    assert transport.cleanups == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("other_task", [False, True])
async def test_authorized_stream_close_drains_admission_even_from_another_task(
    monkeypatch, tmp_path, live, other_task
):
    _project, transport, handle, validator = await _approved_handle(
        monkeypatch, tmp_path, live=live
    )
    iterator = _stream(handle, ("session", "stream_events"))
    try:
        assert await anext(iterator) == "first"
        if other_task:
            await asyncio.create_task(iterator.aclose())
        else:
            await iterator.aclose()
        assert validator.in_flight == 0
        assert transport.closed_streams == 1
    finally:
        await iterator.aclose()
        await asyncio.wait_for(handle.cleanup(), timeout=1)
    assert transport.cleanups == 1


@pytest.mark.asyncio
async def test_admitted_iterator_keeps_internal_calls_authorized_after_reset(monkeypatch, tmp_path):
    project, transport, handle, validator = await _approved_handle(monkeypatch, tmp_path)

    async def events():
        yield "started"
        yield await handle.call_tool("part-of-admitted-operation", {})

    transport.session.stream_events = events
    iterator = _stream(handle, ("session", "stream_events"))
    try:
        assert await anext(iterator) == "started"
        assert reset_project_choices(project) == 1
        assert await anext(iterator) == "contacted"
    finally:
        await iterator.aclose()
        await handle.cleanup()
    assert transport.tool_calls == 1
    assert validator.in_flight == 0


@pytest.mark.asyncio
async def test_cancelled_iterator_step_closes_generator_and_releases_admission(
    monkeypatch, tmp_path, cancellation_observer
):
    observe, cancellations = cancellation_observer
    _project, transport, handle, validator = await _approved_handle(monkeypatch, tmp_path)
    entered = asyncio.Event()
    closed = asyncio.Event()

    async def events():
        try:
            yield "started"
            entered.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    transport.session.stream_events = events
    iterator = _stream(handle, ("session", "stream_events"))
    try:
        assert await anext(iterator) == "started"
        step = asyncio.create_task(observe(anext(iterator)))
        await entered.wait()
        step.cancel("stop-stream")
        with pytest.raises(asyncio.CancelledError):
            await step
        assert [error.args for error in cancellations] == [("stop-stream",)]
        assert closed.is_set()
        assert validator.in_flight == 0
        assert handle._active_calls == {}
    finally:
        await iterator.aclose()
        await handle.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", _STREAM_PATHS)
async def test_public_stream_close_releases_transport_and_inner_generator_immediately(path):
    transport = _Transport()
    handle = LiveMCPServer("iterator-test", transport)
    iterator = _stream(handle, path)
    try:
        assert await anext(iterator) == "first"
        await iterator.aclose()

        # Cleanup must be complete when aclose returns, not left to event-loop
        # async-generator finalization, which can run in a different context.
        assert transport.closed_streams == 1
        assert handle._active_calls == {}
    finally:
        await iterator.aclose()
        await asyncio.wait_for(handle.cleanup(), timeout=1)
    assert transport.cleanups == 1
