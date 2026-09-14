"""Auth command tests use in-memory tokens and provider doubles only."""

import argparse
import time
from unittest.mock import AsyncMock, Mock

import pytest

from koder_agent.auth.base import OAuthTokens
from koder_agent.harness.auth import commands


@pytest.mark.asyncio
async def test_human_status_does_not_print_even_short_tokens(monkeypatch, capsys):
    tokens = OAuthTokens(
        "google",
        "access-canary",
        "refresh-canary",
        4000000000000,
        models=["synthetic-model"],
        models_fetched_at=int(time.time() * 1000),
    )
    storage = Mock(load=Mock(return_value=tokens))
    monkeypatch.setattr(commands, "get_token_storage", lambda: storage)
    monkeypatch.setattr(commands, "get_provider", Mock(side_effect=AssertionError("no network")))

    await commands.handle_status("google")

    output = capsys.readouterr().out
    assert "access-canary" not in output
    assert "refresh-canary" not in output
    assert "Access token: present" in output


@pytest.mark.asyncio
@pytest.mark.parametrize("delete_failure", [False, OSError("synthetic-secret-canary")])
async def test_revoke_does_not_claim_success_when_local_delete_fails(
    monkeypatch, capsys, delete_failure
):
    tokens = OAuthTokens("google", "canary", "refresh-canary", 1)
    storage = Mock(load=Mock(return_value=tokens))
    if isinstance(delete_failure, Exception):
        storage.delete.side_effect = delete_failure
    else:
        storage.delete.return_value = delete_failure
    provider = Mock(revoke_token=AsyncMock(return_value=True))
    monkeypatch.setattr(commands, "get_token_storage", lambda: storage)
    monkeypatch.setattr(commands, "get_provider", lambda _provider: provider)

    assert not await commands.handle_revoke("google")
    output = capsys.readouterr().out
    assert "Tokens revoked" not in output
    assert "synthetic-secret-canary" not in output


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["list", "status", "status_json", "revoke", "token_login"])
async def test_auth_command_storage_keeps_event_loop_responsive(
    monkeypatch, event_loop_progress_probe, command
):
    tokens = OAuthTokens(
        "google",
        "synthetic-access",
        "synthetic-refresh",
        4000000000000,
        models=["synthetic-model"],
        models_fetched_at=int(time.time() * 1000),
    )
    wrap, observed = event_loop_progress_probe
    storage = Mock(
        load=wrap(lambda _provider: tokens),
        get_all_tokens=wrap(lambda: {"google": tokens}),
        save=wrap(lambda _tokens: None),
        delete=wrap(lambda _provider: True),
    )
    monkeypatch.setattr(commands, "get_token_storage", wrap(lambda: storage))
    provider = Mock(revoke_token=AsyncMock(return_value=True))
    monkeypatch.setattr(commands, "get_provider", lambda _provider: provider)

    if command == "list":
        await commands.handle_list()
    elif command == "status":
        await commands.handle_status("google")
    elif command == "status_json":
        await commands.handle_status_json("google")
    elif command == "revoke":
        assert await commands.handle_revoke("google")
    else:
        args = argparse.Namespace(
            auth_command="login", provider="google", timeout=300, token="synthetic-token"
        )
        assert await commands.handle_auth_subcommand(args) == 0
    assert len(observed) >= 2
    assert all(observed), "an auth command performed credential I/O on the event loop"
