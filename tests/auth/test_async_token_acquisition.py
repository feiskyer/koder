"""OAuth maintenance must be asynchronous, serialized and conditionally published."""

from __future__ import annotations

import asyncio
import base64
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from filelock import FileLock

from koder_agent.auth import client_integration, providers
from koder_agent.auth.base import OAuthResult, OAuthTokens
from koder_agent.auth.providers.antigravity import AntigravityOAuthLLM
from koder_agent.auth.providers.chatgpt import ChatGPTOAuthLLM
from koder_agent.auth.providers.claude import ClaudeOAuthLLM
from koder_agent.auth.providers.google import GoogleOAuthLLM


def _tokens(provider="google", *, fresh=False):
    payload = (
        base64.urlsafe_b64encode(
            json.dumps(
                {"https://api.openai.com/auth": {"chatgpt_account_id": "fixture-account"}}
            ).encode()
        )
        .decode()
        .rstrip("=")
    )
    return OAuthTokens(
        provider=provider,
        access_token=f"fixture.{payload}.{'fresh' if fresh else 'old'}",
        refresh_token="fixture-new-refresh" if fresh else "fixture-old-refresh",
        expires_at=4000000000000 if fresh else 1,
        extra={"project_id": "fixture-project"},
    )


class _Store:
    """Memory-backed credentials with a real, private refresh lease."""

    def __init__(self, directory: Path, tokens: OAuthTokens):
        self.base_dir = directory
        directory.mkdir(exist_ok=True)
        self.current = tokens
        self.saved = []
        self.on_load = None
        self.on_save = None
        self.mutex = threading.Lock()

    def refresh_lock(self, provider):
        return FileLock(
            str(self.base_dir / f".{provider}.refresh.lock"),
            timeout=0,
            thread_local=False,
            mode=0o600,
        )

    def load(self, _provider):
        if self.on_load:
            self.on_load()
        with self.mutex:
            return self.current

    def save_if_current(self, previous, fresh):
        if self.on_save:
            self.on_save()
        with self.mutex:
            if self.current != previous:
                return False
            self.current = fresh
            self.saved.append(fresh)
            return True


@pytest.fixture
def store(monkeypatch, tmp_path):
    storage = _Store(tmp_path / "tokens", _tokens())
    monkeypatch.setattr(client_integration, "get_token_storage", lambda: storage)
    return storage


async def _drain(baseline, *owned):
    tasks = (asyncio.all_tasks() - baseline) | set(owned)
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=3)


class _TransportReached(BaseException):
    """Stop before transport creation, including providers' fallback handlers."""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "llm_class", [GoogleOAuthLLM, ClaudeOAuthLLM, ChatGPTOAuthLLM, AntigravityOAuthLLM]
)
@pytest.mark.parametrize("stream", [False, True])
async def test_async_provider_keeps_loop_responsive_during_token_read(
    monkeypatch, tmp_path, llm_class, stream
):
    loop = asyncio.get_running_loop()
    llm = llm_class()
    storage = _Store(tmp_path / "provider-tokens", _tokens(llm.provider_id, fresh=True))
    observed_progress = []

    def blocking_read():
        heartbeat = threading.Event()
        loop.call_soon_threadsafe(heartbeat.set)
        # Bounds the unfixed case rather than hanging the event loop forever.
        observed_progress.append(heartbeat.wait(timeout=1))

    def stop_transport(*_args, **_kwargs):
        raise _TransportReached

    storage.on_load = blocking_read
    monkeypatch.setattr(client_integration, "get_token_storage", lambda: storage)
    monkeypatch.setattr("aiohttp.ClientSession", stop_transport)
    messages = [{"role": "user", "content": "fixture"}]

    with pytest.raises(_TransportReached):
        if stream:
            iterator = llm.astreaming("fixture-model", messages)
            try:
                await anext(iterator)
            finally:
                await iterator.aclose()
        else:
            await llm.acompletion("fixture-model", messages)

    assert observed_progress
    assert all(observed_progress), "OAuth storage blocked the provider's event loop"


@pytest.mark.asyncio
async def test_concurrent_refreshes_reuse_one_rotated_result(store, monkeypatch):
    baseline = asyncio.all_tasks()
    previous = store.current
    fresh = _tokens(fresh=True)
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def refresh(refresh_token):
        calls.append(refresh_token)
        started.set()
        await release.wait()
        return OAuthResult(True, fresh)

    monkeypatch.setattr(
        providers, "get_provider", lambda _name: SimpleNamespace(refresh_tokens=refresh)
    )
    first = asyncio.create_task(client_integration.async_refresh_token("google", previous))
    second = None
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        second = asyncio.create_task(client_integration.async_refresh_token("google", previous))
        await asyncio.sleep(0.05)
        assert calls == [previous.refresh_token], "the same refresh token was sent concurrently"
        release.set()
        assert await first == await second == fresh
        assert store.saved == [fresh]
    finally:
        release.set()
        await _drain(baseline, first, *((second,) if second is not None else ()))


@pytest.mark.asyncio
async def test_successful_refresh_still_uses_conditional_publication(store, monkeypatch):
    previous, fresh = store.current, _tokens(fresh=True)

    async def refresh(_refresh_token):
        return OAuthResult(True, fresh)

    monkeypatch.setattr(
        providers, "get_provider", lambda _name: SimpleNamespace(refresh_tokens=refresh)
    )
    assert await client_integration.async_refresh_token("google", previous) == fresh
    assert store.saved == [fresh]


@pytest.mark.asyncio
async def test_refresh_keeps_its_original_store_across_awaits(store, monkeypatch, tmp_path):
    baseline = asyncio.all_tasks()
    other = _Store(tmp_path / "other-profile", store.current)
    active = [store]
    started, release = asyncio.Event(), asyncio.Event()
    fresh = _tokens(fresh=True)
    monkeypatch.setattr(client_integration, "get_token_storage", lambda: active[0])

    async def refresh(_refresh_token):
        started.set()
        await release.wait()
        return OAuthResult(True, fresh)

    monkeypatch.setattr(
        providers, "get_provider", lambda _name: SimpleNamespace(refresh_tokens=refresh)
    )
    owner = asyncio.create_task(client_integration.async_refresh_token("google", store.current))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        active[0] = other
        release.set()
        assert await owner == fresh
        assert store.saved == [fresh]
        assert other.saved == [], "an unrelated profile received another request's credentials"
    finally:
        release.set()
        await _drain(baseline, owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("logout", [False, True])
async def test_cancelled_active_refresh_settles_rotation_without_resurrecting_logout(
    store, monkeypatch, cancellation_observer, logout
):
    observe, cancellations = cancellation_observer
    baseline = asyncio.all_tasks()
    previous, fresh = store.current, _tokens(fresh=True)
    started, release = asyncio.Event(), asyncio.Event()

    async def refresh(_refresh_token):
        started.set()
        await release.wait()
        return OAuthResult(True, fresh)

    monkeypatch.setattr(
        providers, "get_provider", lambda _name: SimpleNamespace(refresh_tokens=refresh)
    )
    owner = asyncio.create_task(observe(client_integration.async_refresh_token("google", previous)))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        owner.cancel("cancel model request")
        await asyncio.sleep(0)
        owner.cancel("cancel again")
        await asyncio.sleep(0.02)
        assert not owner.done(), "a started token rotation was abandoned"
        if logout:
            with store.mutex:
                store.current = None
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert [error.args for error in cancellations] == [("cancel model request",)]
        assert store.current == (None if logout else fresh)
        assert store.saved == ([] if logout else [fresh])
    finally:
        release.set()
        await _drain(baseline, owner)


@pytest.mark.asyncio
async def test_cancelled_lease_waiter_never_starts_another_refresh(store, monkeypatch):
    baseline = asyncio.all_tasks()
    previous, fresh = store.current, _tokens(fresh=True)
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def refresh(refresh_token):
        calls.append(refresh_token)
        started.set()
        await release.wait()
        return OAuthResult(True, fresh)

    monkeypatch.setattr(
        providers, "get_provider", lambda _name: SimpleNamespace(refresh_tokens=refresh)
    )
    first = asyncio.create_task(client_integration.async_refresh_token("google", previous))
    second = None
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        second = asyncio.create_task(client_integration.async_refresh_token("google", previous))
        await asyncio.sleep(0.05)
        second.cancel()
        await asyncio.wait({second}, timeout=0.3)
        assert second.done(), "cancelled lock acquisition waited for somebody else's network call"
        with pytest.raises(asyncio.CancelledError):
            await second
        assert calls == [previous.refresh_token]
        release.set()
        assert await first == fresh
    finally:
        release.set()
        await _drain(baseline, first, *((second,) if second is not None else ()))


@pytest.mark.asyncio
async def test_async_refresh_has_a_deadline_and_joins_provider_cleanup(store, monkeypatch):
    baseline = asyncio.all_tasks()
    previous = store.current
    started, cleanup_started, cleanup_release, settled = (asyncio.Event() for _ in range(4))

    async def refresh(_refresh_token):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await cleanup_release.wait()
            settled.set()

    monkeypatch.setattr(
        providers, "get_provider", lambda _name: SimpleNamespace(refresh_tokens=refresh)
    )
    # Leave enough admission time for real executor/lock scheduling. The
    # started barrier distinguishes a provider timeout from an admission timeout.
    monkeypatch.setattr(client_integration, "OAUTH_REFRESH_TIMEOUT_SECONDS", 0.5)
    owner = asyncio.create_task(client_integration.async_refresh_token("google", previous))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        await asyncio.wait_for(cleanup_started.wait(), timeout=2)
        assert not owner.done(), "deadline detached provider cleanup"
        cleanup_release.set()
        await asyncio.wait({owner}, timeout=2)
        assert owner.done(), "refresh did not finish after cleanup"
        assert await owner is None
        assert settled.is_set()
        assert store.current is previous
        assert not store.saved
    finally:
        cleanup_release.set()
        await _drain(baseline, owner)


@pytest.mark.asyncio
async def test_async_refresh_does_not_block_loop_during_conditional_publication(store, monkeypatch):
    loop = asyncio.get_running_loop()
    previous, fresh = store.current, _tokens(fresh=True)
    observed_progress = []

    def blocking_publish():
        heartbeat = threading.Event()
        loop.call_soon_threadsafe(heartbeat.set)
        observed_progress.append(heartbeat.wait(timeout=1))

    async def refresh(_refresh_token):
        return OAuthResult(True, fresh)

    store.on_save = blocking_publish
    monkeypatch.setattr(
        providers, "get_provider", lambda _name: SimpleNamespace(refresh_tokens=refresh)
    )
    result = await client_integration.async_refresh_token("google", previous)
    assert result == fresh
    assert observed_progress == [True], "conditional token publication blocked the event loop"


@pytest.mark.asyncio
async def test_refresh_wait_deadline_does_not_send_request_or_detach_acquisition(
    store, monkeypatch
):
    calls = []
    fresh = _tokens(fresh=True)

    async def refresh(value):
        calls.append(value)
        return OAuthResult(True, fresh)

    monkeypatch.setattr(
        providers, "get_provider", lambda _name: SimpleNamespace(refresh_tokens=refresh)
    )
    monkeypatch.setattr(client_integration, "OAUTH_REFRESH_TIMEOUT_SECONDS", 0.1)
    lease = store.refresh_lock("google")
    lease.acquire(timeout=0)
    try:
        result = await asyncio.wait_for(
            client_integration.async_refresh_token("google", store.current), timeout=2
        )
        assert result is None
        assert not calls
        assert not store.saved
        assert lease.is_locked
    finally:
        lease.release()
    # A timed-out waiter must not acquire the lease later and strand the retry.
    monkeypatch.setattr(client_integration, "OAUTH_REFRESH_TIMEOUT_SECONDS", 2)
    assert await client_integration.async_refresh_token("google", store.current) == fresh
    assert calls == ["fixture-old-refresh"]


@pytest.mark.asyncio
async def test_mismatched_input_provider_is_rejected_before_storage_or_network(store, monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("a mismatched provider reached storage or the provider factory")

    monkeypatch.setattr(client_integration, "get_token_storage", unexpected)
    monkeypatch.setattr(providers, "get_provider", unexpected)
    assert await client_integration.async_refresh_token("claude", store.current) is None
    assert not store.saved


@pytest.mark.asyncio
@pytest.mark.parametrize("separation", ["provider", "profile"])
async def test_independent_refresh_leases_do_not_serialize_unrelated_credentials(
    store, monkeypatch, tmp_path, separation
):
    baseline = asyncio.all_tasks()
    second_provider = "claude" if separation == "provider" else "google"
    second_dir = store.base_dir if separation == "provider" else tmp_path / "second-profile"
    second_store = _Store(second_dir, _tokens(second_provider))
    entered = 0
    both_entered, release = asyncio.Event(), asyncio.Event()

    def provider(name):
        async def refresh(_refresh):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await release.wait()
            return OAuthResult(True, _tokens(name, fresh=True))

        return SimpleNamespace(refresh_tokens=refresh)

    monkeypatch.setattr(providers, "get_provider", provider)
    owners = [
        asyncio.create_task(
            client_integration.async_refresh_token(
                storage.current.provider, storage.current, storage=storage
            )
        )
        for storage in (store, second_store)
    ]
    try:
        await asyncio.wait_for(both_entered.wait(), timeout=2)
        release.set()
        assert await asyncio.gather(*owners) == [
            _tokens(fresh=True),
            _tokens(second_provider, fresh=True),
        ]
    finally:
        release.set()
        await _drain(baseline, *owners)


@pytest.mark.asyncio
@pytest.mark.parametrize("logout", [False, True])
async def test_cancellation_during_publication_keeps_cas_owned(
    store, monkeypatch, cancellation_observer, logout
):
    baseline = asyncio.all_tasks()
    loop = asyncio.get_running_loop()
    publishing, release = asyncio.Event(), threading.Event()
    fresh = _tokens(fresh=True)
    observe, cancellations = cancellation_observer

    def blocked_publication():
        loop.call_soon_threadsafe(publishing.set)
        assert release.wait(timeout=5), "test did not release conditional publication"

    async def refresh(_refresh):
        return OAuthResult(True, fresh)

    store.on_save = blocked_publication
    monkeypatch.setattr(
        providers, "get_provider", lambda _name: SimpleNamespace(refresh_tokens=refresh)
    )
    owner = asyncio.create_task(
        observe(client_integration.async_refresh_token("google", store.current))
    )
    try:
        await asyncio.wait_for(publishing.wait(), timeout=2)
        owner.cancel("cancel publication")
        await asyncio.sleep(0)
        owner.cancel("cancel publication again")
        await asyncio.sleep(0)
        assert not owner.done()
        if logout:
            with store.mutex:
                store.current = None
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert [error.args for error in cancellations] == [("cancel publication",)]
        assert store.current == (None if logout else fresh)
        assert store.saved == ([] if logout else [fresh])
        # Completion includes releasing the exact lease, even after cancellation.
        with store.refresh_lock("google"):
            pass
    finally:
        release.set()
        await _drain(baseline, owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["acquire", "read"])
async def test_cancellation_during_lease_io_never_leaves_a_lock_or_starts_network(
    store, monkeypatch, phase
):
    baseline = asyncio.all_tasks()
    loop = asyncio.get_running_loop()
    started, release = asyncio.Event(), threading.Event()
    lease = store.refresh_lock("google")
    acquire = lease.acquire

    def blocked():
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5), "test did not release synthetic lease I/O"

    def blocked_acquire(*args, **kwargs):
        result = acquire(*args, **kwargs)
        blocked()
        return result

    def unexpected_provider(_name):
        pytest.fail("a cancelled lease waiter started network work")

    monkeypatch.setattr(store, "refresh_lock", lambda _name: lease)
    monkeypatch.setattr(providers, "get_provider", unexpected_provider)
    if phase == "acquire":
        monkeypatch.setattr(lease, "acquire", blocked_acquire)
    else:
        store.on_load = blocked
    owner = asyncio.create_task(client_integration.async_refresh_token("google", store.current))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        owner.cancel()
        await asyncio.sleep(0)
        owner.cancel()
        await asyncio.sleep(0)
        assert not owner.done()
        assert lease.is_locked
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert not lease.is_locked
        assert not store.saved
    finally:
        release.set()
        await _drain(baseline, owner)
