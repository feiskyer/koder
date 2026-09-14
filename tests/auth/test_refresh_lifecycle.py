"""Synchronous refresh must work in threads without an ambient event loop."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from koder_agent.auth import client_integration, providers
from koder_agent.auth.base import OAuthResult, OAuthTokens


@pytest.mark.asyncio
async def test_sync_refresh_in_worker_without_current_event_loop(monkeypatch):
    old = OAuthTokens("google", "old-canary", "refresh-canary", 1)
    fresh = OAuthTokens("google", "new-canary", "rotated-canary", 4000000000000)
    provider = Mock(refresh_tokens=AsyncMock(return_value=OAuthResult(True, fresh)))
    storage = Mock()
    storage.load.return_value = old
    monkeypatch.setattr(providers, "get_provider", lambda _provider: provider)
    monkeypatch.setattr(client_integration, "get_token_storage", lambda: storage)

    result = await asyncio.to_thread(client_integration._sync_refresh_token, "google", old)

    assert result is fresh
    provider.refresh_tokens.assert_awaited_once_with("refresh-canary")
    storage.save_if_current.assert_called_once_with(old, fresh)
    storage.save.assert_not_called()
