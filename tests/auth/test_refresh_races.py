"""Public refresh paths with synthetic stores/providers and controlled barriers."""

import asyncio
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty
from threading import Event, Lock, Thread
from unittest.mock import Mock

import pytest
from filelock import FileLock, Timeout

from koder_agent.auth import client_integration, providers, token_storage
from koder_agent.auth.base import OAuthResult, OAuthTokens


def token(access="old", refresh="old-refresh", expires=1):
    return OAuthTokens("google", access, refresh, expires)


@pytest.fixture
def threaded_parent(recwarn):
    """Keep the cross-process tests safe even inside a threaded test harness."""
    started, stop = Event(), Event()

    def background():
        started.set()
        stop.wait()

    thread = Thread(target=background, name="auth-process-test-parent")
    thread.start()
    try:
        assert started.wait(5)
        yield
    finally:
        stop.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        # On the pinned runtime, -W error alone does not fail os.fork().
        assert not recwarn.list, [str(warning.message) for warning in recwarn]


@pytest.fixture
def process_profile(tmp_path, monkeypatch):
    """Spawn inherits synthetic paths before it imports this test module."""
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(project)
    return home, project


def _refresh_process(base_dir, home, project, started, release, output):
    """Create process-local mocks; only the token files and barriers are shared."""
    assert multiprocessing.get_start_method() == "spawn"
    assert Path.home() == home
    assert Path.cwd() == project
    keychain = Mock()
    keychain.is_available.return_value = False

    async def refresh(_refresh):
        started.set()
        assert release.wait(10), "parent did not release synthetic refresh"
        return OAuthResult(True, token("late", "late-refresh", 4000000000000))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(token_storage, "SecureStorage", lambda: keychain)
        storage = token_storage.TokenStorage(base_dir)
        patch.setattr(client_integration, "get_token_storage", lambda: storage)
        patch.setattr(providers, "get_provider", lambda _name: Mock(refresh_tokens=refresh))
        result = client_integration.get_oauth_token("google")
        output.put((os.getpid(), None if result is None else result.access_token))


def _cas_process(base_dir, home, project, publishing, release, output):
    assert multiprocessing.get_start_method() == "spawn"
    assert Path.home() == home
    assert Path.cwd() == project
    keychain = Mock()
    keychain.is_available.return_value = False
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(token_storage, "SecureStorage", lambda: keychain)
        storage = token_storage.TokenStorage(base_dir)
        save = storage._save

        def paused_save(tokens):
            publishing.set()
            assert release.wait(10), "parent did not release CAS publication"
            save(tokens)

        patch.setattr(storage, "_save", paused_save)
        committed = storage.save_if_current(token(), token("winner", "rotated", 4000000000000))
        output.put((os.getpid(), committed))


def _competing_refresh_process(
    base_dir, home, project, mode, started, contended, release, calls, output
):
    """Exercise independent event loops and real locks without native credentials."""
    assert multiprocessing.get_start_method() == "spawn"
    assert Path.home() == home
    assert Path.cwd() == project
    keychain = Mock()
    keychain.is_available.return_value = False

    async def refresh(_refresh):
        calls.put(os.getpid())
        started.set()
        assert await asyncio.to_thread(release.wait, 10), "parent did not release refresh"
        return OAuthResult(True, token("winner", "rotated", 4000000000000))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(token_storage, "SecureStorage", lambda: keychain)
        storage = token_storage.TokenStorage(base_dir)
        make_lease = storage.refresh_lock

        def observed_lease(provider):
            lease = make_lease(provider)
            acquire = lease.acquire

            def observed_acquire(*args, **kwargs):
                try:
                    return acquire(*args, **kwargs)
                except Timeout:
                    contended.set()
                    raise

            patch.setattr(lease, "acquire", observed_acquire)
            return lease

        patch.setattr(storage, "refresh_lock", observed_lease)
        patch.setattr(client_integration, "get_token_storage", lambda: storage)
        patch.setattr(providers, "get_provider", lambda _name: Mock(refresh_tokens=refresh))
        result = invoke_refresh(mode, token())
        output.put((os.getpid(), result.access_token, result.refresh_token))


@pytest.fixture(params=["file", "keychain", "fallback"])
def stores(tmp_path, monkeypatch, request):
    entries = {}
    keychain = Mock()
    keychain.is_available.return_value = request.param != "file"

    def store(_service, account, value):
        if request.param == "fallback" and account in entries:
            return False
        entries[account] = value
        return True

    def delete(_service, account):
        return entries.pop(account, None) is not None

    keychain.store_checked.side_effect = store
    keychain.retrieve.side_effect = lambda _service, account: entries.get(account)
    keychain.retrieve_checked.side_effect = keychain.retrieve.side_effect
    keychain.delete.side_effect = delete
    keychain.delete_checked.side_effect = delete
    monkeypatch.setattr(token_storage, "SecureStorage", lambda: keychain)
    first = token_storage.TokenStorage(tmp_path / "tokens")
    second = token_storage.TokenStorage(tmp_path / "tokens")
    first.save(token())
    if request.param == "fallback":
        first.save(token())
    monkeypatch.setattr(client_integration, "get_token_storage", lambda: first)
    return first, second


def invoke_refresh(mode, original):
    if mode == "sync":
        return client_integration.get_oauth_token("google")
    return asyncio.run(client_integration.async_refresh_token("google", original))


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_late_refresh_cannot_resurrect_local_logout(stores, monkeypatch, mode):
    first, second = stores
    started, release = Event(), Event()
    original = first.load("google")

    async def refresh(_refresh):
        started.set()
        assert release.wait(3), "test did not release synthetic refresh"
        return OAuthResult(True, token("late", "late-refresh", 4000000000000))

    monkeypatch.setattr(providers, "get_provider", lambda _name: Mock(refresh_tokens=refresh))
    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(invoke_refresh, mode, original)
        try:
            assert started.wait(3)
            assert second.delete("google")
        finally:
            release.set()
        result = pending.result(timeout=3)

    assert second.load("google") is None
    assert result is None


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_older_refresh_cannot_overwrite_successful_rotation(stores, monkeypatch, mode):
    first, second = stores
    started, release = Event(), Event()
    original = first.load("google")
    winner = token("winner", "winner-rotated", 4000000000000)
    count = 0
    call_lock = Lock()

    async def refresh(_refresh):
        nonlocal count
        with call_lock:
            count += 1
            call = count
        if call == 1:
            started.set()
            assert release.wait(3)
            return OAuthResult(True, token("stale", "stale-rotated", 4000000000000))
        return OAuthResult(True, winner)

    monkeypatch.setattr(providers, "get_provider", lambda _name: Mock(refresh_tokens=refresh))
    with ThreadPoolExecutor(max_workers=2) as executor:
        older = executor.submit(invoke_refresh, mode, original)
        try:
            assert started.wait(3)
            # An external login/legacy writer does not take the refresh lease.
            # The older response must still lose its conditional publication.
            second.save(winner)
            newer = executor.submit(invoke_refresh, mode, original)
        finally:
            release.set()
        assert newer.result(timeout=3) == winner
        late_result = older.result(timeout=3)

    assert second.load("google") == winner
    assert late_result == winner
    assert count == 1, "a queued caller refreshed credentials that already had a valid winner"


def test_refresh_rejects_mismatched_result_provider(stores, monkeypatch):
    first, second = stores
    original = first.load("google")
    wrong = OAuthTokens("claude", "wrong", "wrong-refresh", 4000000000000)

    async def refresh(_refresh):
        return OAuthResult(True, wrong)

    monkeypatch.setattr(providers, "get_provider", lambda _name: Mock(refresh_tokens=refresh))
    result = asyncio.run(client_integration.async_refresh_token("google", original))
    assert result is None
    assert second.load("google") == original
    assert second.load("claude") is None


@pytest.mark.parametrize("operation", ["load", "save", "delete", "save_if_current", "update"])
def test_storage_lock_wait_is_bounded(stores, monkeypatch, operation):
    first, second = stores
    original = first.load("google")
    monkeypatch.setattr(token_storage, "TOKEN_LOCK_TIMEOUT_SECONDS", 0.02)
    actions = {
        "load": lambda: second.load("google"),
        "save": lambda: second.save(token("new")),
        "delete": lambda: second.delete("google"),
        "save_if_current": lambda: second.save_if_current(original, token("new")),
        "update": lambda: second.update_access_token("google", "new", 4000000000000),
    }
    with FileLock(str(first.base_dir / ".google.lock"), timeout=0):
        started = time.monotonic()
        with pytest.raises(Timeout):
            actions[operation]()
        assert time.monotonic() - started < 1
    assert second.load("google") == original


def test_failed_cas_does_not_touch_winner(stores):
    first, second = stores
    original = first.load("google")
    winner = token("winner", "rotated", 4000000000000)
    second.save(winner)
    assert not first.save_if_current(original, token("loser"))
    assert second.load("google") == winner


@pytest.mark.parametrize("intervention", ["logout", "rotation"])
def test_refresh_coordinates_with_another_process(
    tmp_path, monkeypatch, intervention, threaded_parent, process_profile
):
    """A late refresh cannot undo another process's logout or rotation."""
    keychain = Mock()
    keychain.is_available.return_value = False
    monkeypatch.setattr(token_storage, "SecureStorage", lambda: keychain)
    storage = token_storage.TokenStorage(tmp_path / "tokens")
    storage.save(token())
    context = multiprocessing.get_context("spawn")
    started, release = context.Event(), context.Event()
    output = context.Queue()

    child = context.Process(
        target=_refresh_process,
        args=(storage.base_dir, *process_profile, started, release, output),
    )
    try:
        child.start()
        assert started.wait(10)
        other = token_storage.TokenStorage(tmp_path / "tokens")
        if intervention == "logout":
            assert other.delete("google")
            expected = None
        else:
            other.save(token("winner", "rotated", 4000000000000))
            expected = "winner"
        release.set()
        worker_pid, result = output.get(timeout=10)
        assert worker_pid == child.pid
        assert worker_pid != os.getpid()
        assert result == expected
        child.join(timeout=5)
        assert child.exitcode == 0
        current = storage.load("google")
        assert current == (
            None if intervention == "logout" else token("winner", "rotated", 4000000000000)
        )
    finally:
        release.set()
        if child.is_alive():
            child.terminate()
            child.join(timeout=3)
        assert not child.is_alive()
        child.close()
        output.close()
        output.join_thread()


def test_cas_excludes_cross_process_delete_until_publish(
    tmp_path, monkeypatch, threaded_parent, process_profile
):
    keychain = Mock()
    keychain.is_available.return_value = False
    monkeypatch.setattr(token_storage, "SecureStorage", lambda: keychain)
    storage = token_storage.TokenStorage(tmp_path / "tokens")
    storage.save(token())
    context = multiprocessing.get_context("spawn")
    publishing, release = context.Event(), context.Event()
    output = context.Queue()
    child = context.Process(
        target=_cas_process,
        args=(storage.base_dir, *process_profile, publishing, release, output),
    )
    try:
        child.start()
        assert publishing.wait(10)
        other = token_storage.TokenStorage(tmp_path / "tokens")
        monkeypatch.setattr(token_storage, "TOKEN_LOCK_TIMEOUT_SECONDS", 0.02)
        with pytest.raises(Timeout):
            other.delete("google")
        release.set()
        worker_pid, committed = output.get(timeout=10)
        assert worker_pid == child.pid
        assert worker_pid != os.getpid()
        assert committed is True
        child.join(timeout=5)
        assert child.exitcode == 0
        assert other.load("google") == token("winner", "rotated", 4000000000000)
        assert other.delete("google")
        assert other.load("google") is None
    finally:
        release.set()
        if child.is_alive():
            child.terminate()
            child.join(timeout=3)
        assert not child.is_alive()
        child.close()
        output.close()
        output.join_thread()


@pytest.mark.parametrize("modes", [("async", "async"), ("sync", "async"), ("sync", "sync")])
def test_two_processes_share_one_refresh_and_reuse_the_persisted_winner(
    tmp_path, monkeypatch, threaded_parent, process_profile, modes
):
    keychain = Mock()
    keychain.is_available.return_value = False
    monkeypatch.setattr(token_storage, "SecureStorage", lambda: keychain)
    storage = token_storage.TokenStorage(tmp_path / "tokens")
    storage.save(token())
    context = multiprocessing.get_context("spawn")
    started, contended, release = context.Event(), context.Event(), context.Event()
    calls = context.Queue()
    output = context.Queue()
    children = [
        context.Process(
            target=_competing_refresh_process,
            args=(
                storage.base_dir,
                *process_profile,
                mode,
                started,
                contended,
                release,
                calls,
                output,
            ),
        )
        for mode in modes
    ]
    launched = []
    try:
        children[0].start()
        launched.append(children[0])
        assert started.wait(10), "first process never entered refresh"
        children[1].start()
        launched.append(children[1])
        assert contended.wait(10), "second process never attempted the held lease"
        release.set()
        results = [output.get(timeout=10) for _ in children]
        assert {result[0] for result in results} == {child.pid for child in children}
        assert os.getpid() not in {result[0] for result in results}
        assert all(result[1:] == ("winner", "rotated") for result in results)
        for child in children:
            child.join(timeout=5)
            assert child.exitcode == 0
        assert calls.get(timeout=2) == children[0].pid
        with pytest.raises(Empty):
            calls.get_nowait()
        assert storage.load("google") == token("winner", "rotated", 4000000000000)
    finally:
        release.set()
        for child in launched:
            if child.is_alive():
                child.terminate()
                child.join(timeout=3)
            assert not child.is_alive()
        for child in children:
            child.close()
        output.close()
        output.join_thread()
        calls.close()
        calls.join_thread()
