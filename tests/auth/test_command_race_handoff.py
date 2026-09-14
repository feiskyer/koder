"""CLI remote results must not resurrect locally revoked credentials."""

import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest

from koder_agent.auth import providers, token_storage
from koder_agent.auth.base import OAuthResult, OAuthTokens
from koder_agent.harness.auth import commands


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["list", "status"])
@pytest.mark.parametrize("pause_at", ["refresh", "models"])
@pytest.mark.parametrize("delete_during_request", [True, False])
async def test_command_conditionally_publishes_remote_results(
    tmp_path, monkeypatch, command, pause_at, delete_during_request
):
    keychain = Mock()
    keychain.is_available.return_value = False
    monkeypatch.setattr(token_storage, "SecureStorage", lambda: keychain)
    storage = token_storage.TokenStorage(tmp_path / "tokens")
    original = OAuthTokens(
        "google",
        "synthetic-old",
        "synthetic-refresh",
        1 if pause_at == "refresh" else 4000000000000,
        models=["synthetic-model"] if pause_at == "refresh" else [],
        models_fetched_at=int(time.time() * 1000) if pause_at == "refresh" else None,
    )
    storage.save(original)
    monkeypatch.setattr(commands, "get_token_storage", lambda: storage)
    monkeypatch.setattr(commands, "console", Mock())
    started, release = asyncio.Event(), asyncio.Event()

    async def pause():
        started.set()
        await asyncio.wait_for(release.wait(), timeout=3)

    async def refresh(_refresh):
        if pause_at == "refresh":
            await pause()
        return OAuthResult(
            True,
            OAuthTokens(
                "google",
                "synthetic-late",
                "synthetic-rotated",
                4000000000000,
                models=["synthetic-model"],
                models_fetched_at=int(time.time() * 1000),
            ),
        )

    async def models(_access):
        if pause_at == "models":
            await pause()
        return ["synthetic-model"], {"source": "api"}

    provider = Mock(refresh_tokens=refresh, list_models=models)
    monkeypatch.setattr(commands, "get_provider", lambda _provider: provider)
    monkeypatch.setattr(providers, "get_provider", lambda _provider: provider)
    pending = asyncio.create_task(
        commands.handle_list() if command == "list" else commands.handle_status("google")
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=3)
        if delete_during_request:
            assert storage.delete("google")
    finally:
        release.set()
        await asyncio.wait_for(pending, timeout=3)
    saved = storage.load("google")
    if delete_during_request:
        assert saved is None
    else:
        assert saved is not None
        assert saved.models == ["synthetic-model"]
        assert saved.access_token == (
            "synthetic-late" if pause_at == "refresh" else "synthetic-old"
        )


@pytest.mark.asyncio
async def test_failed_refresh_does_not_query_models_with_expired_credentials(tmp_path, monkeypatch):
    keychain = Mock()
    keychain.is_available.return_value = False
    monkeypatch.setattr(token_storage, "SecureStorage", lambda: keychain)
    storage = token_storage.TokenStorage(tmp_path / "tokens")
    tokens = OAuthTokens("google", "synthetic-expired", "synthetic-refresh", 1)
    storage.save(tokens)
    provider = Mock(
        refresh_tokens=AsyncMock(return_value=OAuthResult(False)),
        list_models=AsyncMock(side_effect=AssertionError("expired credentials used")),
    )
    monkeypatch.setattr(commands, "get_provider", lambda _: provider)
    monkeypatch.setattr(providers, "get_provider", lambda _: provider)
    monkeypatch.setattr(commands, "get_token_storage", lambda: storage)
    monkeypatch.setattr(commands, "console", Mock())
    await commands.handle_status("google")
    provider.list_models.assert_not_awaited()
    assert storage.load("google") == tokens
