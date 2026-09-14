"""Callback boundaries, exercised without a browser or listening socket."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import make_mocked_request

from koder_agent.auth import callback_server
from koder_agent.auth.callback_server import OAuthCallbackServer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    ["error=denied&error_description=%3Cscript%3Ecanary%3C/script%3E", ""],
)
async def test_callback_error_page_renders_and_escapes(query):
    server = OAuthCallbackServer(0)
    response = await server._handle_callback(make_mocked_request("GET", f"/auth/callback?{query}"))

    assert response.status == 400
    assert "<script>" not in response.text
    if query:
        assert "&lt;script&gt;canary&lt;/script&gt;" in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["code=canary", "code=canary&state=wrong", "error=denied"])
async def test_unrelated_callback_does_not_consume_pending_login(query):
    server = OAuthCallbackServer(0, expected_state="expected")
    response = await server._handle_callback(make_mocked_request("GET", f"/auth/callback?{query}"))

    assert response.status == 400
    assert not server._callback_received.is_set()
    assert server._result is None

    await server._handle_callback(
        make_mocked_request("GET", "/auth/callback?code=legitimate&state=expected")
    )
    result = await server.wait_for_callback(timeout=0.1)
    assert result.code == "legitimate"


@pytest.mark.asyncio
async def test_first_valid_callback_cannot_be_overwritten():
    server = OAuthCallbackServer(0)
    await server._handle_callback(make_mocked_request("GET", "/auth/callback?code=first"))
    response = await server._handle_callback(
        make_mocked_request("GET", "/auth/callback?code=second")
    )
    assert response.status == 409
    assert (await server.wait_for_callback(timeout=0.1)).code == "first"


@pytest.mark.asyncio
async def test_flow_derives_expected_state_from_authorization_url(monkeypatch):
    observed = {}

    class Server:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def wait_for_callback(self, **_kwargs):
            return callback_server.CallbackResult(success=True, code="synthetic")

    monkeypatch.setattr(callback_server, "OAuthCallbackServer", Server)
    await callback_server.run_oauth_flow(
        "https://synthetic.invalid/authorize?state=expected", 0, open_browser=False
    )
    assert observed["expected_state"] == "expected"


@pytest.mark.asyncio
async def test_failed_bind_cleans_up_runner(monkeypatch):
    runner = SimpleNamespace(setup=AsyncMock(), cleanup=AsyncMock())
    site = SimpleNamespace(start=AsyncMock(side_effect=OSError("synthetic bind failure")))
    monkeypatch.setattr(callback_server.web, "AppRunner", lambda _app: runner)
    monkeypatch.setattr(callback_server.web, "TCPSite", lambda *_args: site)

    with pytest.raises(OSError, match="synthetic bind failure"):
        await OAuthCallbackServer(0).start()

    runner.cleanup.assert_awaited_once()
