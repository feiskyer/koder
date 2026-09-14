"""Manual OAuth input must release its reader without touching real terminals."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import prompt_toolkit
import pytest
import pytest_asyncio
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from koder_agent.harness.auth import commands
from koder_agent.harness.auth.commands import _handle_manual_code_flow

AUTH_URL = "https://auth.example.invalid/synthetic"


@pytest_asyncio.fixture
async def prompt_io(monkeypatch):
    started = asyncio.Queue()
    prompts = []
    tasks = []
    cleanup_releases = []
    monkeypatch.setattr("webbrowser.open", lambda _url: False)

    def no_blocking_input(*_args, **_kwargs):
        raise AssertionError("Manual auth must not start a blocking input worker")

    monkeypatch.setattr("builtins.input", no_blocking_input)

    class ObservedPrompt(PromptSession):
        async def prompt_async(self, *args, **kwargs):
            kwargs["pre_run"] = lambda: started.put_nowait(self)
            return await super().prompt_async(*args, **kwargs)

    with create_pipe_input() as pipe:

        def create_prompt(**kwargs):
            prompt = ObservedPrompt(input=pipe, output=DummyOutput(), **kwargs)
            prompts.append(prompt)
            return prompt

        monkeypatch.setattr(prompt_toolkit, "PromptSession", create_prompt)

        def track(coroutine):
            task = asyncio.create_task(coroutine)
            tasks.append(task)
            return task

        def start(timeout=10):
            return track(_handle_manual_code_flow(AUTH_URL, timeout))

        async def wait_started():
            return await asyncio.wait_for(started.get(), timeout=5)

        try:
            yield SimpleNamespace(
                pipe=pipe,
                start=start,
                track=track,
                wait_started=wait_started,
                releases=cleanup_releases,
            )
        finally:
            for release in cleanup_releases:
                release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            assert all(not prompt.app.is_running for prompt in prompts)


@pytest.mark.parametrize("code", ["synthetic-code", "  synthetic-code#state  "])
@pytest.mark.asyncio
async def test_manual_code_is_returned_with_its_opaque_state_suffix(prompt_io, code):
    task = prompt_io.start()
    prompt = await prompt_io.wait_started()
    prompt_io.pipe.send_text(code + "\n")
    result = await asyncio.wait_for(task, timeout=5)
    assert result.success
    assert result.code == code.strip()
    assert not prompt.app.is_running


@pytest.mark.asyncio
async def test_empty_manual_code_reports_failure(prompt_io):
    task = prompt_io.start()
    prompt = await prompt_io.wait_started()
    prompt_io.pipe.send_text("\n")
    result = await asyncio.wait_for(task, timeout=5)
    assert not result.success and result.error == "empty_code"
    assert not prompt.app.is_running


@pytest.mark.asyncio
async def test_closed_input_reports_failure(prompt_io):
    task = prompt_io.start()
    prompt = await prompt_io.wait_started()
    prompt_io.pipe.close()
    result = await asyncio.wait_for(task, timeout=5)
    assert not result.success and result.error == "input_closed"
    assert not prompt.app.is_running


@pytest.mark.asyncio
async def test_timeout_finishes_prompt_cleanup_before_returning(prompt_io):
    task = prompt_io.start(timeout=0.1)
    prompt = await prompt_io.wait_started()
    result = await asyncio.wait_for(task, timeout=5)
    assert not result.success and result.error == "timeout"
    assert not prompt.app.is_running


@pytest.mark.asyncio
async def test_cancelled_prompt_does_not_consume_the_next_prompts_input(prompt_io):
    task = prompt_io.start()
    first = await prompt_io.wait_started()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not first.app.is_running

    following = prompt_io.start()
    second = await prompt_io.wait_started()
    prompt_io.pipe.send_text("fresh-code\n")
    result = await asyncio.wait_for(following, timeout=5)
    assert result.success and result.code == "fresh-code"
    assert not second.app.is_running


@pytest.mark.parametrize("initial_stop", ["cancel", "timeout"])
@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_prompt_cleanup(prompt_io, monkeypatch, initial_stop):
    task = prompt_io.start(timeout=0.1 if initial_stop == "timeout" else 10)
    prompt = await prompt_io.wait_started()
    cleanup_started = asyncio.Event()
    release = asyncio.Event()
    prompt_io.releases.append(release)
    original_cleanup = prompt.app.cancel_and_wait_for_background_tasks

    async def delayed_cleanup():
        cleanup_started.set()
        await release.wait()
        await original_cleanup()

    monkeypatch.setattr(prompt.app, "cancel_and_wait_for_background_tasks", delayed_cleanup)
    if initial_stop == "cancel":
        task.cancel()
    await asyncio.wait_for(cleanup_started.wait(), timeout=5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    # An additional user cancellation during timeout cleanup is not downgraded
    # to a returned "timeout" error.
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not prompt.app.is_running


@pytest.mark.parametrize("outcome", ["empty", "eof", "timeout", "cancel"])
@pytest.mark.asyncio
async def test_failed_manual_input_never_starts_token_exchange(prompt_io, monkeypatch, outcome):
    exchange = AsyncMock(side_effect=AssertionError("Token exchange must not run"))
    provider = SimpleNamespace(
        get_authorization_url=lambda: (AUTH_URL, "synthetic-verifier"),
        exchange_code=exchange,
    )
    monkeypatch.setattr(commands, "get_provider", lambda _provider: provider)

    storage = Mock(side_effect=AssertionError("Token storage must not be accessed"))
    monkeypatch.setattr(commands, "get_token_storage", storage)
    task = prompt_io.track(
        commands.handle_login("claude", timeout=0.1 if outcome == "timeout" else 10)
    )
    prompt = await prompt_io.wait_started()
    if outcome == "empty":
        prompt_io.pipe.send_text("\n")
    elif outcome == "eof":
        prompt_io.pipe.close()
    elif outcome == "cancel":
        task.cancel()

    if outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert await asyncio.wait_for(task, timeout=5) is False
    exchange.assert_not_awaited()
    storage.assert_not_called()
    assert not prompt.app.is_running
