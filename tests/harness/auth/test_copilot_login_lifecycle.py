"""Copilot login owns its async HTTP work and uses only synthetic caches."""

import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

from koder_agent.harness.auth import commands


@pytest_asyncio.fixture
async def copilot_io(tmp_path, monkeypatch):
    from litellm.llms.github_copilot import authenticator as sdk

    calls = []
    clients = []
    pending = asyncio.Event()
    release = asyncio.Event()
    state = {"mode": "success", "polls": 0}
    cleanup_releases = []
    tasks = []
    access_file = tmp_path / "access-token"
    api_file = tmp_path / "api-key.json"

    class FakeAuthenticator:
        def __init__(self):
            calls.append("init")
            self.token_dir = str(tmp_path)
            self.access_token_file = str(access_file)
            self.api_key_file = str(api_file)

        def _get_github_headers(self, access_token=None):
            return {"authorization": f"token {access_token}"} if access_token else {}

        def _login(self):
            raise AssertionError("Login must not start the legacy synchronous poller")

        def _refresh_api_key(self):
            raise AssertionError("Refresh must not start the legacy synchronous client")

    async def request(request):
        calls.append(request.url.path)
        if request.url.path.endswith("/device/code"):
            data = {
                "device_code": "synthetic-device",
                "user_code": "DISPLAY-CODE",
                "verification_uri": "https://example.invalid/activate",
                "interval": 0.01,
            }
            if state["mode"] == "expired":
                data.update(expires_in=0.02, interval=0.1)
            if state["mode"] == "bad-interval":
                data["interval"] = 0
            return httpx.Response(200, json=data)
        if request.url.path.endswith("/oauth/access_token"):
            state["polls"] += 1
            if state["mode"] == "timeout":
                await asyncio.sleep(0.1)
            if state["mode"] == "cancel":
                pending.set()
                await release.wait()
            if state["mode"] == "denied":
                return httpx.Response(200, json={"error": "access_denied"})
            if state["mode"] == "slow-down":
                return httpx.Response(200, json={"error": "slow_down"})
            if state["mode"] == "expired" or (state["mode"] == "pending" and state["polls"] == 1):
                return httpx.Response(200, json={"error": "authorization_pending"})
            return httpx.Response(200, json={"access_token": "synthetic-access"})
        if state["mode"] == "service-error":
            return httpx.Response(401, text="synthetic-secret-response")
        data = {
            "token": "synthetic-api",
            "expires_at": 2000000000,
            "endpoints": {"api": "https://example.invalid/api"},
        }
        if state["mode"] == "bad-key":
            data.pop("token")
        if state["mode"] == "bad-expiry":
            data["expires_at"] = "invalid"
        if state["mode"] == "bad-endpoint":
            data["endpoints"]["api"] = 0
        return httpx.Response(200, json=data)

    client_type = httpx.AsyncClient

    def client(**kwargs):
        result = client_type(transport=httpx.MockTransport(request), trust_env=False, **kwargs)
        clients.append(result)
        return result

    monkeypatch.setattr(sdk, "Authenticator", FakeAuthenticator)
    monkeypatch.setattr(httpx, "AsyncClient", client)

    def start(timeout=5):
        task = asyncio.create_task(commands.handle_github_copilot_login(timeout=timeout))
        tasks.append(task)
        return task

    try:
        yield SimpleNamespace(
            state=state,
            calls=calls,
            clients=clients,
            pending=pending,
            release=release,
            access=access_file,
            api=api_file,
            start=start,
            cleanup_releases=cleanup_releases,
        )
    finally:
        release.set()
        for cleanup_release in cleanup_releases:
            cleanup_release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for client in clients:
            if not client.is_closed:
                await client.aclose()


@pytest.mark.asyncio
async def test_copilot_timeout_returns_failure_without_publishing(copilot_io):
    copilot_io.state["mode"] = "timeout"
    assert await commands.handle_github_copilot_login(timeout=0.02) is False
    assert not copilot_io.access.exists() and not copilot_io.api.exists()
    assert all(client.is_closed for client in copilot_io.clients)


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
@pytest.mark.asyncio
async def test_invalid_copilot_deadline_never_initializes_storage(copilot_io, timeout):
    assert await commands.handle_github_copilot_login(timeout=timeout) is False
    assert copilot_io.calls == []
    assert not copilot_io.access.exists() and not copilot_io.api.exists()


@pytest.mark.asyncio
async def test_copilot_cancellation_does_not_leave_a_later_cache_writer(copilot_io):
    copilot_io.state["mode"] = "cancel"
    task = copilot_io.start()
    await asyncio.wait_for(copilot_io.pending.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    copilot_io.release.set()
    assert not copilot_io.access.exists() and not copilot_io.api.exists()
    assert all(client.is_closed for client in copilot_io.clients)


@pytest.mark.asyncio
async def test_successful_copilot_login_preserves_sdk_cache_shape(copilot_io, capsys):
    assert await commands.handle_github_copilot_login(timeout=5) is True
    assert copilot_io.access.read_text() == "synthetic-access"
    assert json.loads(copilot_io.api.read_text())["token"] == "synthetic-api"
    assert all(client.is_closed for client in copilot_io.clients)
    output = capsys.readouterr().out
    assert "synthetic-access" not in output
    assert "synthetic-api" not in output
    assert copilot_io.access.stat().st_mode & 0o777 == 0o600
    assert copilot_io.api.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "mode", ["denied", "expired", "bad-interval", "bad-key", "bad-expiry", "bad-endpoint"]
)
@pytest.mark.asyncio
async def test_rejected_or_malformed_responses_do_not_publish(copilot_io, mode):
    copilot_io.state["mode"] = mode
    assert await commands.handle_github_copilot_login(timeout=5) is False
    assert not copilot_io.access.exists() and not copilot_io.api.exists()
    assert all(client.is_closed for client in copilot_io.clients)


@pytest.mark.asyncio
async def test_pending_authorization_can_complete_within_the_deadline(copilot_io):
    copilot_io.state["mode"] = "pending"
    assert await commands.handle_github_copilot_login(timeout=5) is True
    assert copilot_io.state["polls"] == 2


@pytest.mark.asyncio
async def test_slow_down_waits_until_the_owner_deadline(copilot_io, capsys):
    copilot_io.state["mode"] = "slow-down"
    assert await commands.handle_github_copilot_login(timeout=0.2) is False
    assert copilot_io.state["polls"] == 1
    assert "timed out" in capsys.readouterr().out
    assert not copilot_io.access.exists() and not copilot_io.api.exists()


@pytest.mark.asyncio
async def test_http_errors_do_not_print_response_credentials(copilot_io, capsys):
    copilot_io.state["mode"] = "service-error"
    assert await commands.handle_github_copilot_login(timeout=5) is False
    output = capsys.readouterr().out
    assert "HTTP 401" in output
    assert "synthetic-secret-response" not in output
    assert not copilot_io.access.exists() and not copilot_io.api.exists()


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_http_cleanup(copilot_io, monkeypatch):
    copilot_io.state["mode"] = "cancel"
    task = copilot_io.start()
    await asyncio.wait_for(copilot_io.pending.wait(), timeout=5)
    client = copilot_io.clients[-1]
    original = client.aclose
    closing = asyncio.Event()
    release = asyncio.Event()
    copilot_io.cleanup_releases.append(release)

    async def close():
        closing.set()
        await release.wait()
        await original()

    monkeypatch.setattr(client, "aclose", close)
    task.cancel()
    await asyncio.wait_for(closing.wait(), timeout=5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.is_closed
    assert not copilot_io.access.exists() and not copilot_io.api.exists()


@pytest.mark.asyncio
async def test_cancellation_joins_an_already_started_cache_publication(copilot_io, monkeypatch):
    from koder_agent.auth import github_copilot

    original = github_copilot._publish_cache
    publishing = asyncio.Event()
    release = threading.Event()
    copilot_io.cleanup_releases.append(release)
    loop = asyncio.get_running_loop()
    finished = []

    def publish(*args):
        loop.call_soon_threadsafe(publishing.set)
        assert release.wait(timeout=5)
        original(*args)
        finished.append(True)

    monkeypatch.setattr(github_copilot, "_publish_cache", publish)
    task = copilot_io.start()
    await asyncio.wait_for(publishing.wait(), timeout=5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished == [True]
    assert copilot_io.access.read_text() == "synthetic-access"
    assert json.loads(copilot_io.api.read_text())["token"] == "synthetic-api"
