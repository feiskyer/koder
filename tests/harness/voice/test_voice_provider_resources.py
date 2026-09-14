"""Exercise SDK/httpx resource ownership with in-memory HTTP only."""

import asyncio
import time
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI

from koder_agent.harness.voice import service


def config_for(*, provider="openai", base_url=None):
    return SimpleNamespace(
        voice=SimpleNamespace(model="synthetic-model", base_url=None, api_version=None),
        model=SimpleNamespace(
            provider=provider, base_url=base_url, azure_api_version="model-api-version"
        ),
    )


def test_azure_voice_does_not_inherit_another_providers_endpoint(monkeypatch):
    config = config_for(provider="openai", base_url="https://chat.example.invalid/v1")
    monkeypatch.setenv("AZURE_API_BASE", "https://voice.example.invalid")
    monkeypatch.setenv("AZURE_API_VERSION", "voice-api-version")
    endpoint, version = service._resolve_azure_endpoint_and_api_version(config)
    assert endpoint == "https://voice.example.invalid"
    assert version == "voice-api-version"


def test_azure_voice_still_inherits_azure_chat_endpoint(monkeypatch):
    config = config_for(provider="azure", base_url="https://azure.example.invalid/openai/v1")
    monkeypatch.setenv("AZURE_API_BASE", "https://fallback.example.invalid")
    assert service._resolve_azure_endpoint_and_api_version(config) == (
        "https://azure.example.invalid",
        "model-api-version",
    )


@pytest.mark.parametrize(
    "base_url, endpoint",
    [
        (
            "https://openai-prod.example.invalid/openai/deployments/transcribe",
            "https://openai-prod.example.invalid",
        ),
        ("https://openai-prod.example.invalid", "https://openai-prod.example.invalid"),
        (
            "https://proxy.example.invalid/openai-proxy/openai/deployments/transcribe",
            "https://proxy.example.invalid/openai-proxy",
        ),
    ],
)
def test_azure_endpoint_only_removes_the_openai_path_segment(base_url, endpoint):
    config = config_for(provider="azure", base_url=base_url)
    assert service._resolve_azure_endpoint_and_api_version(config)[0] == endpoint


async def _assert_openai_client_resource_lifecycle(monkeypatch, cancel):
    entered = asyncio.Event()
    requests = []

    async def respond(request):
        requests.append(request)
        entered.set()
        if cancel:
            await asyncio.Event().wait()
        return httpx.Response(200, text="synthetic transcript")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    client = AsyncOpenAI(
        api_key="synthetic-test-key",
        base_url="https://voice.example.invalid/v1",
        http_client=http_client,
        max_retries=0,
    )
    monkeypatch.setattr(service, "AsyncOpenAI", lambda **_kwargs: client)
    monkeypatch.setattr(service, "get_config", config_for)
    monkeypatch.setattr(
        service, "resolve_voice_credentials", lambda _provider: ("synthetic-test-key", {}, None)
    )
    monkeypatch.setattr(service, "get_all_keyterms", lambda _cwd: ["Koder"])
    task = asyncio.create_task(
        service.ProviderVoiceTranscriber().transcribe(audio_bytes=b"synthetic", provider="openai")
    )
    readiness = asyncio.create_task(entered.wait())
    try:
        # SDK metadata discovery precedes the transport and is not the close
        # contract under test. Observe early task failure as well as readiness,
        # rather than masking the real exception with a two-second event wait.
        done, _ = await asyncio.wait(
            {task, readiness}, return_when=asyncio.FIRST_COMPLETED, timeout=10
        )
        if task in done and not entered.is_set():
            assert http_client.is_closed
            await task  # Propagate the original pre-request error.
        assert entered.is_set(), "Transcription did not reach the mock transport"
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        else:
            assert await asyncio.wait_for(task, timeout=2) == "synthetic transcript"
        # Assert product-owned closure before the fixture's safety cleanup.
        assert http_client.is_closed
        assert requests[0].url.path == "/v1/audio/transcriptions"
    finally:
        for owned in (task, readiness):
            if not owned.done():
                owned.cancel()
        await asyncio.gather(task, readiness, return_exceptions=True)
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_openai_client_is_closed_on_success_or_cancellation(monkeypatch, cancel):
    await _assert_openai_client_resource_lifecycle(monkeypatch, cancel)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_openai_client_closes_after_slow_sdk_metadata(monkeypatch, cancel):
    from openai import _base_client

    def slow_platform_probe():
        # The pinned SDK runs this before sending to the in-memory transport.
        # Resource ownership is not a two-second platform-startup benchmark.
        time.sleep(2.1)
        return "MacOS"

    monkeypatch.setattr(_base_client, "get_platform", slow_platform_probe)
    await _assert_openai_client_resource_lifecycle(monkeypatch, cancel)


@pytest.mark.asyncio
async def test_openai_request_failure_is_not_masked_as_readiness_timeout(monkeypatch):
    from openai import _base_client

    def failed_platform_probe():
        raise RuntimeError("synthetic SDK metadata failure")

    monkeypatch.setattr(_base_client, "get_platform", failed_platform_probe)
    with pytest.raises(RuntimeError, match="synthetic SDK metadata failure"):
        await _assert_openai_client_resource_lifecycle(monkeypatch, False)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "cancel", "malformed"])
async def test_gemini_stream_closes_on_every_exit(monkeypatch, outcome):
    entered = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b'data: {"candidates":[{"content":{"parts":[{"text":"hello"}]}}]}\n\n'
            entered.set()
            if outcome == "cancel":
                await asyncio.Event().wait()
            if outcome == "malformed":
                yield b"data: invalid-json\n\n"
            else:
                yield b'data: {"candidates":[{"content":{"parts":[{"text":" world"}]}}]}\n\n'

        async def aclose(self):
            self.closed = True

    stream = Stream()
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))
    )
    monkeypatch.setattr(service.httpx, "AsyncClient", lambda **_kwargs: client)
    partials = []
    task = asyncio.create_task(
        service.ProviderVoiceTranscriber()._stream_gemini_transcript(
            url="https://voice.example.invalid/stream",
            payload={},
            headers={},
            params={},
            on_partial=partials.append,
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=2)
    if outcome == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif outcome == "malformed":
        with pytest.raises(ValueError):
            await task
    else:
        assert await task == "hello world"
        assert partials == ["hello", "hello world"]
    assert stream.closed
    assert client.is_closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_gemini_honors_explicit_voice_base_url(monkeypatch, streaming):
    requests = []
    clients = []
    real_async_client = httpx.AsyncClient

    def respond(request):
        requests.append(request)
        data = {"candidates": [{"content": {"parts": [{"text": "synthetic transcript"}]}}]}
        if streaming:
            return httpx.Response(
                200,
                text='data: {"candidates":[{"content":{"parts":[{"text":"synthetic transcript"}]}}]}\n\n',
            )
        return httpx.Response(200, json=data)

    def create_client(**_kwargs):
        client = real_async_client(transport=httpx.MockTransport(respond))
        clients.append(client)
        return client

    monkeypatch.setattr(service.httpx, "AsyncClient", create_client)
    monkeypatch.setattr(service, "get_config", config_for)
    monkeypatch.setattr(
        service,
        "resolve_voice_credentials",
        lambda _provider: ("synthetic-test-key", {}, "https://voice.example.invalid/custom-api/"),
    )
    partials = []
    result = await service.ProviderVoiceTranscriber().transcribe(
        audio_bytes=b"synthetic",
        provider="google",
        on_partial=partials.append if streaming else None,
    )
    assert result == "synthetic transcript"
    assert len(requests) == 1
    assert requests[0].url.host == "voice.example.invalid"
    method = "streamGenerateContent" if streaming else "generateContent"
    assert requests[0].url.path == f"/custom-api/models/synthetic-model:{method}"
    assert all(client.is_closed for client in clients)
