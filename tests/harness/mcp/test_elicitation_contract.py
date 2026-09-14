"""Elicitation must preserve schema values and release the event loop."""

import asyncio
from io import StringIO
from unittest.mock import AsyncMock

import pytest
from mcp.types import ElicitRequestFormParams, ElicitRequestURLParams, ElicitResult
from rich.console import Console

from koder_agent.mcp import elicitation as module


def make_handler(monkeypatch, *, terminal=True):
    handler = module.ElicitationHandler(console=Console(file=StringIO(), force_terminal=terminal))
    for method, value in (("_try_hook_auto_response", None), ("_dispatch_result_hook", None)):
        monkeypatch.setattr(handler, method, AsyncMock(return_value=value))
    return handler


@pytest.mark.asyncio
async def test_array_answers_retain_the_requested_array_type(monkeypatch):
    handler = make_handler(monkeypatch)
    params = ElicitRequestFormParams(
        message="Synthetic tags",
        requestedSchema={
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["tags"],
        },
    )
    monkeypatch.setattr(handler, "_read_input", AsyncMock(side_effect=["one, two", "yes"]))
    result = await handler(None, params)
    assert result.action == "accept"
    assert result.content == {"tags": ["one", "two"]}


@pytest.mark.asyncio
async def test_prompt_wait_does_not_freeze_its_event_loop(monkeypatch):
    handler = make_handler(monkeypatch)
    loop = asyncio.get_running_loop()
    progress = []

    async def asynchronous_input(*_args, **_kwargs):
        advanced = loop.create_future()
        loop.call_soon(advanced.set_result, "yes")
        answer = await advanced
        progress.append(True)
        return answer

    monkeypatch.setattr(handler, "_read_input", asynchronous_input)
    result = await handler(None, ElicitRequestFormParams(message="Synthetic", requestedSchema={}))
    assert result.action == "accept"
    assert progress and all(progress)


@pytest.mark.asyncio
async def test_closed_input_cancels_instead_of_escaping_or_submitting_defaults(monkeypatch):
    handler = make_handler(monkeypatch)
    params = ElicitRequestFormParams(
        message="Synthetic", requestedSchema={"properties": {"name": {"type": "string"}}}
    )
    monkeypatch.setattr(handler, "_read_input", AsyncMock(side_effect=EOFError))
    result = await handler(None, params)
    assert result is not None and result.action == "cancel" and result.content is None


@pytest.mark.parametrize("content", [{}, {"count": "not-an-integer"}, {"count": 100}])
@pytest.mark.asyncio
async def test_hook_acceptance_cannot_publish_schema_invalid_content(monkeypatch, content):
    handler = make_handler(monkeypatch)
    params = ElicitRequestFormParams(
        message="Synthetic",
        requestedSchema={
            "type": "object",
            "properties": {"count": {"type": "integer", "maximum": 5}},
            "required": ["count"],
        },
    )
    response = ElicitResult(action="accept", content=content)
    monkeypatch.setattr(handler, "_try_hook_auto_response", AsyncMock(return_value=response))
    result = await handler(None, params)
    assert result.action == "cancel" and result.content is None


@pytest.mark.asyncio
async def test_noninteractive_requests_do_not_consume_stdin(monkeypatch):
    handler = make_handler(monkeypatch, terminal=False)
    read = AsyncMock(side_effect=AssertionError("stdin"))
    monkeypatch.setattr(handler, "_read_input", read)
    result = await handler(None, ElicitRequestFormParams(message="Synthetic", requestedSchema={}))
    assert result.action in {"cancel", "decline"} and result.content is None
    read.assert_not_awaited()


@pytest.mark.asyncio
async def test_unsupported_url_mode_is_declined_without_opening_a_browser(monkeypatch):
    handler = make_handler(monkeypatch)
    monkeypatch.setattr("webbrowser.open", lambda *_a: pytest.fail("Must not open a browser"))
    params = ElicitRequestURLParams(
        message="Synthetic", url="https://example.invalid/", elicitationId="synthetic-id"
    )
    result = await handler(None, params)
    assert result is not None and result.action == "decline" and result.content is None


@pytest.mark.asyncio
async def test_schema_references_cannot_fetch_remote_documents(monkeypatch):
    handler = make_handler(monkeypatch, terminal=False)
    fetched = []
    monkeypatch.setattr("urllib.request.urlopen", lambda *_a, **_k: fetched.append(True))
    monkeypatch.setattr(
        handler,
        "_try_hook_auto_response",
        AsyncMock(return_value=ElicitResult(action="accept", content={"count": 1})),
    )
    params = ElicitRequestFormParams(
        message="Synthetic", requestedSchema={"$ref": "https://example.invalid/schema.json"}
    )
    result = await handler(None, params)
    assert result.action == "cancel" and result.content is None
    assert not fetched


@pytest.mark.asyncio
async def test_invalid_enum_does_not_silently_choose_the_first_option(monkeypatch):
    handler = make_handler(monkeypatch)
    read = AsyncMock(side_effect=["invalid", "blue", "yes"])
    monkeypatch.setattr(handler, "_read_input", read)
    params = ElicitRequestFormParams(
        message="Synthetic",
        requestedSchema={
            "type": "object",
            "properties": {"color": {"type": "string", "enum": ["red", "blue"]}},
            "required": ["color"],
        },
    )
    result = await handler(None, params)
    assert result.action == "accept" and result.content == {"color": "blue"}
    assert read.await_count == 3


@pytest.mark.asyncio
async def test_declined_hook_does_not_send_content(monkeypatch):
    handler = make_handler(monkeypatch)
    monkeypatch.setattr(
        handler,
        "_try_hook_auto_response",
        AsyncMock(return_value=ElicitResult(action="decline", content={"unused": "not sent"})),
    )
    result = await handler(None, ElicitRequestFormParams(message="Synthetic", requestedSchema={}))
    assert result.action == "decline" and result.content is None
