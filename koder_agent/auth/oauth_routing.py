"""Request-local OAuth routing that does not repurpose SDK-native provider names.

Public model identities stay unchanged. Only the SDK wire name is private, and
the proxy restores the raw model before invoking a Koder-owned OAuth handler.
These names prevent dispatcher collisions; they are not an authorization boundary
against other Python code in the same process.
"""

from __future__ import annotations

import asyncio
import inspect
from copy import copy
from typing import Any

from litellm.llms.custom_llm import CustomLLM

from ..litellm_cost_map import get_litellm
from ..utils.async_tasks import await_owned_task
from .constants import SUPPORTED_PROVIDERS

_WIRE_PREFIX = "koder_oauth_"
_MODEL_PREFIX = "model-"
_WIRE_OWNERSHIP_ATTRIBUTE = "_koder_private_oauth_handlers"
_MAX_MODEL_BYTES = 4096


def _public_parts(model: str) -> tuple[str, str] | None:
    if not isinstance(model, str):
        return None
    unwrapped = model.removeprefix("litellm/")
    provider, separator, raw_model = unwrapped.partition("/")
    if provider.lower() not in SUPPORTED_PROVIDERS:
        return None
    if not separator or not raw_model:
        raise ValueError("OAuth model must include its provider and model name")
    if len(raw_model.encode("utf-8")) > _MAX_MODEL_BYTES or "\x00" in raw_model:
        raise ValueError("Invalid OAuth model identifier")
    return provider.lower(), raw_model


def is_oauth_model(model: str) -> bool:
    return _public_parts(model) is not None


def _decode_wire_model(model: str) -> str:
    if not model.startswith(_MODEL_PREFIX):
        raise ValueError("Invalid private OAuth routing identifier")
    try:
        raw = bytes.fromhex(model[len(_MODEL_PREFIX) :])
        decoded = raw.decode("utf-8")
    except (ValueError, UnicodeError) as error:
        raise ValueError("Invalid private OAuth routing identifier") from error
    if not decoded or len(raw) > _MAX_MODEL_BYTES or "\x00" in decoded:
        raise ValueError("Invalid private OAuth routing identifier")
    return decoded


async def _close_stream(stream: Any) -> None:
    close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
    if close is not None:
        result = close()
        if inspect.isawaitable(result):
            await await_owned_task(asyncio.ensure_future(result))


def _public_model_result(result, public_model: str):
    if not hasattr(result, "model"):
        return result
    # SDK logging and partial-usage assembly retain the original response/chunks.
    # Mutating them can make those later paths resolve the public OAuth prefix
    # through a native provider even though the request used a private route.
    public = copy(result)
    public.model = public_model
    return public


class _WireOAuthHandler(CustomLLM):
    def __init__(self, provider: str) -> None:
        super().__init__()
        self.provider = provider

    def _handler(self):
        from .providers import _oauth_handlers

        return _oauth_handlers()[self.provider]

    async def acompletion(self, model, messages, **kwargs):
        return await self._handler().acompletion(
            model=_decode_wire_model(model), messages=messages, **kwargs
        )

    async def astreaming(self, model, messages, **kwargs):
        stream = self._handler().astreaming(
            model=_decode_wire_model(model), messages=messages, **kwargs
        )
        if inspect.isawaitable(stream):
            stream = await stream
        try:
            async for chunk in stream:
                yield chunk
        finally:
            await _close_stream(stream)


_WIRE_HANDLERS = {
    _WIRE_PREFIX + provider: _WireOAuthHandler(provider) for provider in SUPPORTED_PROVIDERS
}


class _PublicModelStream:
    def __init__(self, stream, public_model: str):
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._public_model = public_model
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        try:
            chunk = await self._iterator.__anext__()
        except BaseException:
            await self.aclose()
            raise
        return _public_model_result(chunk, self._public_model)

    async def aclose(self):
        if self._closed:
            return
        self._closed = True
        await _close_stream(self._stream)


async def acompletion(*, model: str, messages: list, **kwargs):
    """Run a normal SDK completion, isolating only Koder OAuth routing names."""
    sdk = get_litellm()
    parts = _public_parts(model)
    if parts is None:
        return await sdk.acompletion(model=model, messages=messages, **kwargs)

    provider, raw_model = parts
    private_provider = _WIRE_PREFIX + provider
    encoded_model = _MODEL_PREFIX + raw_model.encode("utf-8").hex()
    wire_model = f"{private_provider}/{encoded_model}"
    if (
        private_provider in sdk.openai_compatible_providers
        or encoded_model in sdk.open_ai_chat_completion_models
        or encoded_model in sdk.model_list_set
        or wire_model in sdk.model_list_set
    ):
        raise ValueError("Private OAuth routing collision with an SDK-native route")
    request = dict(kwargs)
    if request.pop("custom_llm_provider", None) is not None:
        raise ValueError("OAuth routing cannot be overridden by request options")
    metadata = request.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("OAuth request metadata must be a mapping")
    request["metadata"] = {**(metadata or {}), "koder_public_model": model}
    # OAuth endpoints/credentials belong to the selected Koder handler, not an
    # unrelated API client or the SDK's native/default credential discovery.
    for key in ("api_key", "api_base", "base_url", "client"):
        request.pop(key, None)
    request["api_key"] = "koder-oauth-managed"
    request["custom_llm_provider"] = private_provider

    from .providers import _register_handler_mapping

    _register_handler_mapping(_WIRE_HANDLERS, ownership_attribute=_WIRE_OWNERSHIP_ATTRIBUTE)
    result = await sdk.acompletion(model=wire_model, messages=messages, **request)
    if request.get("stream"):
        return _PublicModelStream(result, model)
    return _public_model_result(result, model)
