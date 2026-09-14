"""MCP notification delivery over an in-memory AnyIO stream."""

import asyncio

import anyio
import pytest
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCNotification
from pydantic import TypeAdapter

from koder_agent.harness.channels.interceptor import ChannelInterceptingStream
from koder_agent.harness.channels.notification import CHANNEL_NOTIFICATION_METHOD


def message(method, params):
    return SessionMessage(
        TypeAdapter(JSONRPCMessage).validate_python(
            JSONRPCNotification(jsonrpc="2.0", method=method, params=params).model_dump(
                by_alias=True
            )
        )
    )


@pytest.mark.asyncio
async def test_interceptor_consumes_only_channel_messages_and_closes_inner_stream():
    send, receive = anyio.create_memory_object_stream(3)
    channel = message(CHANNEL_NOTIFICATION_METHOD, {"content": "synthetic"})
    ordinary = message("notifications/tools/list_changed", {})
    transport_error = ValueError("synthetic transport failure")
    send.send_nowait(channel)
    send.send_nowait(ordinary)
    send.send_nowait(transport_error)
    await send.aclose()
    delivered = []

    async def callback(*args):
        delivered.append(args)

    interceptor = ChannelInterceptingStream(receive, callback, "synthetic-server")
    async with interceptor:
        assert await interceptor.receive() is ordinary
        assert await interceptor.receive() is transport_error
        with pytest.raises(StopAsyncIteration):
            await interceptor.__anext__()
    assert delivered == [
        ("synthetic-server", CHANNEL_NOTIFICATION_METHOD, {"content": "synthetic"})
    ]
    with pytest.raises(anyio.ClosedResourceError):
        await receive.receive()


@pytest.mark.asyncio
async def test_callback_failure_does_not_break_nonchannel_delivery():
    send, receive = anyio.create_memory_object_stream(2)
    send.send_nowait(message(CHANNEL_NOTIFICATION_METHOD, {"content": "synthetic"}))
    ordinary = message("notifications/tools/list_changed", {})
    send.send_nowait(ordinary)

    async def callback(*_args):
        raise ValueError("synthetic handler failure")

    async with send, ChannelInterceptingStream(receive, callback) as interceptor:
        assert await interceptor.receive() is ordinary


@pytest.mark.asyncio
async def test_cancellation_propagates_through_channel_callback():
    send, receive = anyio.create_memory_object_stream(1)
    send.send_nowait(message(CHANNEL_NOTIFICATION_METHOD, {"content": "synthetic"}))
    entered = asyncio.Event()

    async def callback(*_args):
        entered.set()
        await asyncio.Event().wait()

    async with send, ChannelInterceptingStream(receive, callback) as interceptor:
        task = asyncio.create_task(interceptor.receive())
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
