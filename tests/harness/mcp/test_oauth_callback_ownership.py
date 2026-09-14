"""MCP OAuth owns real loopback listeners, but never calls a real provider."""

import asyncio
import socket
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from koder_agent.mcp import oauth


@pytest_asyncio.fixture
async def callback_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    state = SimpleNamespace(
        outcome="success",
        shutdown_error=False,
        wrap_shutdown=None,
        authorizing=asyncio.Event(),
        servers=[],
    )
    flow = oauth.MCPOAuthFlow(
        "owned-callback",
        "https://resource.example/mcp",
        oauth.MCPOAuthConfig(client_id="synthetic-client"),
    )
    metadata = {
        "issuer": "https://auth.example/",
        "authorization_endpoint": "https://auth.example/authorize",
        "token_endpoint": "https://auth.example/token",
    }
    monkeypatch.setattr(flow, "_discover_fresh_metadata", AsyncMock(return_value=metadata))
    monkeypatch.setattr(flow, "_ensure_client", AsyncMock(return_value=("synthetic-client", None)))
    monkeypatch.setattr(oauth, "_load_bound_tokens", lambda *_args: None)
    monkeypatch.setattr(oauth, "_merge_bound_tokens", lambda *_args: None)

    async def authorize(*_args, **_kwargs):
        state.authorizing.set()
        if state.outcome == "error":
            raise RuntimeError("synthetic authorization failure")
        if state.outcome == "cancel":
            await asyncio.Event().wait()
        return {"access_token": "synthetic-token"}

    monkeypatch.setattr(flow, "_authorization_code_flow", authorize)
    real_start = oauth._start_callback_server

    def start(port):
        server, actual_port = real_start(port)
        original_shutdown = server.shutdown
        state.servers.append((server, original_shutdown))

        def shutdown():
            original_shutdown()
            if state.shutdown_error:
                raise RuntimeError("synthetic shutdown failure")

        server.shutdown = state.wrap_shutdown(shutdown) if state.wrap_shutdown else shutdown
        return server, actual_port

    monkeypatch.setattr(oauth, "_start_callback_server", start)
    try:
        yield flow, state
    finally:
        for server, shutdown in state.servers:
            try:
                await asyncio.to_thread(shutdown)
            finally:
                server.server_close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
@pytest.mark.parametrize("shutdown_error", [False, True])
async def test_authentication_closes_listener_and_preserves_primary_outcome(
    callback_flow, outcome, shutdown_error
):
    flow, state = callback_flow
    state.outcome = outcome
    state.shutdown_error = shutdown_error
    task = asyncio.create_task(flow.authenticate())
    if outcome == "cancel":
        await asyncio.wait_for(state.authorizing.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif outcome == "error":
        with pytest.raises(RuntimeError, match="synthetic authorization failure"):
            await task
    elif shutdown_error:
        with pytest.raises(RuntimeError, match="synthetic shutdown failure"):
            await task
    else:
        assert await task == {"Authorization": "Bearer synthetic-token"}

    assert len(state.servers) == 1
    server, _shutdown = state.servers[0]
    assert server.socket.fileno() == -1, "OAuth completed without releasing its listener"


@pytest.mark.asyncio
async def test_callback_shutdown_keeps_the_event_loop_responsive(
    callback_flow, event_loop_progress_probe
):
    flow, state = callback_flow
    wrap, progress = event_loop_progress_probe
    state.wrap_shutdown = wrap
    assert await flow.authenticate() == {"Authorization": "Bearer synthetic-token"}
    assert progress == [True], "callback shutdown blocked the authentication event loop"


def test_callback_thread_start_failure_closes_the_allocated_listener(monkeypatch):
    created = []
    factory_name = (
        "_OAuthCallbackServer" if hasattr(oauth, "_OAuthCallbackServer") else "HTTPServer"
    )
    real_server = getattr(oauth, factory_name)

    def server(*args, **kwargs):
        instance = real_server(*args, **kwargs)
        created.append(instance)
        return instance

    def thread(*_args, **_kwargs):
        raise RuntimeError("synthetic thread startup failure")

    monkeypatch.setattr(oauth, factory_name, server)
    monkeypatch.setattr(oauth, "Thread", thread)
    try:
        with pytest.raises(RuntimeError, match="thread startup failure"):
            oauth._start_callback_server(None)
        assert len(created) == 1
        assert created[0].socket.fileno() == -1
    finally:
        for instance in created:
            instance.server_close()


@pytest.mark.asyncio
async def test_idle_callback_request_has_a_finite_read_timeout(monkeypatch):
    timeout = oauth._OAuthCallbackHandler.timeout
    assert isinstance(timeout, (float, int)) and 0 < timeout <= 10
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    original_handler = oauth._OAuthCallbackHandler

    class Handler(original_handler):
        timeout = 0.05

        def handle(self):
            loop.call_soon_threadsafe(entered.set)
            super().handle()

    monkeypatch.setattr(oauth, "_OAuthCallbackHandler", Handler)
    server, port = oauth._start_callback_server(None)
    client = socket.create_connection(("127.0.0.1", port), timeout=2)
    try:
        # Connect without sending an HTTP request; the read timeout must release
        # the single server thread so shutdown can join it.
        await asyncio.wait_for(entered.wait(), timeout=2)
        await asyncio.wait_for(asyncio.to_thread(server.shutdown), timeout=2)
    finally:
        client.close()
        await asyncio.to_thread(server.shutdown)
        server.server_close()


@pytest.mark.asyncio
async def test_shutdown_interrupts_an_active_read_without_waiting_for_its_timeout(monkeypatch):
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    original_handler = oauth._OAuthCallbackHandler

    class Handler(original_handler):
        timeout = 60

        def handle(self):
            loop.call_soon_threadsafe(entered.set)
            super().handle()

    monkeypatch.setattr(oauth, "_OAuthCallbackHandler", Handler)
    server, port = oauth._start_callback_server(None)
    client = socket.create_connection(("127.0.0.1", port), timeout=2)
    stop = getattr(oauth, "_stop_callback_server", lambda current: current.shutdown())
    stop_task = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        stop_task = asyncio.create_task(asyncio.to_thread(stop, server))
        await asyncio.wait_for(asyncio.shield(stop_task), timeout=2)
        assert server.socket.fileno() == -1
    finally:
        # Also release the deliberate read stall on a failing pre-fix run.
        client.close()
        if stop_task is not None:
            await asyncio.wait_for(asyncio.shield(stop_task), timeout=2)
        else:
            await asyncio.to_thread(server.shutdown)
        server.server_close()


@pytest.mark.asyncio
async def test_authentication_joins_the_callback_thread_before_returning(
    callback_flow, monkeypatch
):
    flow, _state = callback_flow
    release = threading.Event()
    join_requested = asyncio.Event()
    loop = asyncio.get_running_loop()
    created = []
    original_thread = oauth.Thread

    class CallbackThread(original_thread):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

        def run(self):
            try:
                super().run()
            finally:
                release.wait(timeout=5)

        def join(self, timeout=None):
            loop.call_soon_threadsafe(join_requested.set)
            return super().join(timeout)

    monkeypatch.setattr(oauth, "Thread", CallbackThread)
    request = asyncio.create_task(flow.authenticate())
    observed_join = asyncio.create_task(join_requested.wait())
    try:
        done, _pending = await asyncio.wait(
            {request, observed_join}, timeout=2, return_when=asyncio.FIRST_COMPLETED
        )
        assert observed_join in done, "authentication returned without joining its callback thread"
        assert not request.done()
        release.set()
        assert await request == {"Authorization": "Bearer synthetic-token"}
        assert len(created) == 1 and not created[0].is_alive()
    finally:
        release.set()
        observed_join.cancel()
        await asyncio.gather(observed_join, return_exceptions=True)
        await asyncio.gather(request, return_exceptions=True)
