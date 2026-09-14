"""Use actual prompt-toolkit pipe readers, never the developer's terminal."""

import asyncio
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import prompt_toolkit
import pytest
import pytest_asyncio
from mcp.types import ElicitRequestFormParams
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from koder_agent.mcp.elicitation import ElicitationHandler


@pytest_asyncio.fixture
async def input_flow(monkeypatch):
    started = asyncio.Queue()
    prompts = []
    tasks = []
    releases = []
    handler = ElicitationHandler(console=Console(file=StringIO(), force_terminal=True))
    handler._try_hook_auto_response = AsyncMock(return_value=None)
    handler._dispatch_result_hook = AsyncMock()

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

        def start(schema=None, message="Synthetic form"):
            task = asyncio.create_task(
                handler(
                    None, ElicitRequestFormParams(message=message, requestedSchema=schema or {})
                )
            )
            tasks.append(task)
            return task

        async def next_prompt():
            return await asyncio.wait_for(started.get(), timeout=5)

        try:
            yield SimpleNamespace(
                handler=handler,
                pipe=pipe,
                start=start,
                next_prompt=next_prompt,
                started=started,
                releases=releases,
            )
        finally:
            for release in releases:
                release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            assert all(not prompt.app.is_running for prompt in prompts)


@pytest.mark.asyncio
async def test_actual_async_form_retains_array_values(input_flow):
    task = input_flow.start(
        {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
    )
    await input_flow.next_prompt()
    input_flow.pipe.send_text("one, two\n")
    await input_flow.next_prompt()
    input_flow.pipe.send_text("\n")  # Accept the displayed confirmation default.
    result = await asyncio.wait_for(task, timeout=5)
    assert result.action == "accept" and result.content == {"tags": ["one", "two"]}


@pytest.mark.parametrize("stop", ["eof", "ctrl-c"])
@pytest.mark.asyncio
async def test_input_dismissal_returns_cancel_without_stopping_the_loop(input_flow, stop):
    task = input_flow.start()
    prompt = await input_flow.next_prompt()
    if stop == "eof":
        input_flow.pipe.close()
    else:
        input_flow.pipe.send_text("\x03")
    result = await asyncio.wait_for(task, timeout=5)
    assert result.action == "cancel" and result.content is None
    assert not prompt.app.is_running
    alive = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(alive.set_result, True)
    assert await alive


@pytest.mark.asyncio
async def test_simultaneous_requests_serialize_their_terminal_reader(input_flow):
    first = input_flow.start(message="first")
    await input_flow.next_prompt()
    second_entered = asyncio.Event()

    async def no_hook(params):
        if params.message == "second":
            second_entered.set()
        return None

    input_flow.handler._try_hook_auto_response = no_hook
    second = input_flow.start(message="second")
    await asyncio.wait_for(second_entered.wait(), timeout=5)
    await asyncio.sleep(0)
    assert input_flow.started.empty()
    input_flow.pipe.send_text("\n")
    assert (await first).action == "accept"
    await input_flow.next_prompt()
    input_flow.pipe.send_text("\n")
    assert (await second).action == "accept"


@pytest.mark.asyncio
async def test_repeated_task_cancellation_joins_reader_cleanup(input_flow, monkeypatch):
    task = input_flow.start()
    prompt = await input_flow.next_prompt()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    input_flow.releases.append(release)
    original = prompt.app.cancel_and_wait_for_background_tasks

    async def cleanup():
        cleaning.set()
        await release.wait()
        await original()

    monkeypatch.setattr(prompt.app, "cancel_and_wait_for_background_tasks", cleanup)
    task.cancel()
    await asyncio.wait_for(cleaning.wait(), timeout=5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not prompt.app.is_running
    input_flow.handler._dispatch_result_hook.assert_not_awaited()
    following = input_flow.start()
    await input_flow.next_prompt()
    input_flow.pipe.send_text("\n")
    assert (await following).action == "accept"
