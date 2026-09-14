"""The terminal fixture must honor the real OpenAI client's wire protocol."""

import threading

import pytest
from openai import AsyncOpenAI

from scripts.fake_openai_chat_server import _Handler, _ReusableHTTPServer


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_single_fixture_serves_real_client_stream_and_completion(stream, api):
    class Handler(_Handler):
        response_text = "fixture protocol verified"
        log_file = None
        scenario = "single"

    server = _ReusableHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        async with AsyncOpenAI(
            api_key="synthetic-fixture",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            max_retries=0,
            timeout=5,
        ) as client:
            if api == "chat":
                result = await client.chat.completions.create(
                    model="fixture",
                    messages=[{"role": "user", "content": "hello"}],
                    stream=stream,
                )
            else:
                result = await client.responses.create(
                    model="fixture", input="hello", stream=stream
                )
            if stream:
                async with result:
                    chunks = [chunk async for chunk in result]
                if api == "chat":
                    assert (
                        "".join(
                            choice.delta.content or ""
                            for chunk in chunks
                            for choice in chunk.choices
                        )
                        == Handler.response_text
                    )
                    assert chunks[-1].choices[0].finish_reason == "stop"
                else:
                    assert chunks[-1].type == "response.completed"
                    assert chunks[-1].response.output_text == Handler.response_text
                    assert any(chunk.type == "response.output_text.delta" for chunk in chunks)
            else:
                text = result.choices[0].message.content if api == "chat" else result.output_text
                assert text == Handler.response_text
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    assert not worker.is_alive()
