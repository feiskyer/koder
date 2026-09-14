"""Agents SDK chat conversion with a Koder-owned OAuth transport boundary.

The request contract follows the installed LitellmModel._fetch_response.
Run the parity tests against the actual application runtime on upgrades. OAuth dispatch and
stream ownership are replaced; SDK response/event conversion and retry advice
stay inherited.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from copy import copy
from dataclasses import dataclass
from typing import Any, Protocol, cast

from agents import _debug
from agents.agent_output import AgentOutputSchemaBase
from agents.extensions.models.litellm_model import LitellmModel, litellm
from agents.handoffs import Handoff
from agents.items import TResponseInputItem, TResponseStreamEvent
from agents.logger import logger
from agents.model_settings import ModelSettings
from agents.models._retry_runtime import should_disable_provider_managed_retries
from agents.models.chatcmpl_converter import Converter as ChatCompletionsConverter
from agents.models.fake_id import FAKE_RESPONSES_ID
from agents.models.interface import ModelTracing
from agents.models.openai_responses import Converter as ResponsesConverter
from agents.tool import Tool
from agents.tracing.span_data import GenerationSpanData
from agents.tracing.spans import Span
from agents.util._json import _to_dump_compatible
from openai import AsyncStream, omit
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import Response

from koder_agent.utils.async_tasks import await_owned_task


class _AsyncCloseable(Protocol):
    async def aclose(self) -> None: ...


@dataclass
class _OwnedOAuthStream:
    """One request's iterator and shared close, used by both Koder and the SDK."""

    transport: _AsyncCloseable | None = None
    _iterator: AsyncIterator | None = None
    _close_task: asyncio.Task | None = None

    def attach(self, transport: Any) -> None:
        self.transport = transport
        self._iterator = transport.__aiter__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        assert self._iterator is not None
        return await anext(self._iterator)

    async def aclose(self) -> None:
        # New SDK releases also close their input stream, sometimes in a
        # background task. Both owners must join the same physical close.
        if self._close_task is None:
            transport = self.transport
            if transport is None:
                return
            self._close_task = asyncio.create_task(transport.aclose())
        await await_owned_task(self._close_task)

    async def close_transport(self) -> None:
        await self.aclose()


async def _owned_events(
    events: AsyncIterator[TResponseStreamEvent], owner: _OwnedOAuthStream
) -> AsyncIterator[TResponseStreamEvent]:
    try:
        async for event in events:
            yield event
    finally:
        try:
            # SDK generators own tracing ContextVar tokens. Finalize that scope
            # in the consuming context, not a newly created cleanup task.
            await cast(_AsyncCloseable, events).aclose()
        finally:
            # Transport shutdown can wait through repeated cancellation without
            # moving the SDK generator's context-bound scope into another task.
            await await_owned_task(asyncio.create_task(owner.close_transport()))


class OAuthAwareLitellmModel(LitellmModel):
    """Keep public model identity while routing Koder OAuth requests explicitly."""

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any | None = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        from koder_agent.auth.oauth_routing import is_oauth_model

        kwargs = dict(
            system_instructions=system_instructions,
            input=input,
            model_settings=model_settings,
            tools=tools,
            output_schema=output_schema,
            handoffs=handoffs,
            tracing=tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )
        if not is_oauth_model(self.model):
            return super().stream_response(**kwargs)

        # Concurrent streams share configuration, not lifecycle state. Do not use
        # a ContextVar token across yields: aclose may run in a different task.
        request_model = copy(self)
        owner = _OwnedOAuthStream()
        request_model._oauth_stream_owner = owner
        events = LitellmModel.stream_response(request_model, **kwargs)
        return _owned_events(events, owner)

    def _converted_chat_request(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: list,
        handoffs: list,
    ) -> tuple[list, list]:
        """Shared by actual OAuth requests and the retrying model's preflight."""
        preserve_thinking_blocks = bool(
            getattr(model_settings, "reasoning", None) is not None
            and getattr(getattr(model_settings, "reasoning", None), "effort", None) is not None
        )
        converted_messages = ChatCompletionsConverter.items_to_messages(
            input,
            base_url=getattr(self, "base_url", None),
            preserve_thinking_blocks=preserve_thinking_blocks,
            preserve_tool_output_all_content=True,
            model=self.model,
            should_replay_reasoning_content=getattr(
                self,
                "should_replay_reasoning_content",
                None,
            ),
        )
        if any(name in str(self.model).lower() for name in ["anthropic", "claude", "gemini"]):
            converted_messages = self._fix_tool_message_ordering(converted_messages)
        if "gemini" in str(self.model).lower():
            converted_messages = self._convert_gemini_extra_content_to_provider_specific_fields(
                converted_messages
            )
        if system_instructions:
            converted_messages.insert(0, {"content": system_instructions, "role": "system"})
        converted_tools = (
            [ChatCompletionsConverter.tool_to_openai(tool) for tool in tools] if tools else []
        )
        converted_tools.extend(
            ChatCompletionsConverter.convert_handoff_tool(handoff) for handoff in handoffs
        )
        return _to_dump_compatible(converted_messages), _to_dump_compatible(converted_tools)

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: bool = False,
        prompt: Any | None = None,
    ) -> litellm.types.utils.ModelResponse | tuple[Response, AsyncStream[ChatCompletionChunk]]:
        # Import at the request boundary, not during package/model discovery.
        from koder_agent.auth.oauth_routing import acompletion, is_oauth_model

        if not is_oauth_model(self.model):
            return await super()._fetch_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                span,
                tracing,
                stream=stream,
                prompt=prompt,
            )

        converted_messages, converted_tools = self._converted_chat_request(
            system_instructions, input, model_settings, tools, handoffs
        )
        if tracing.include_data():
            span.span_data.input = converted_messages

        parallel_tool_calls = model_settings.parallel_tool_calls if converted_tools else None
        tool_choice = ChatCompletionsConverter.convert_tool_choice(model_settings.tool_choice)
        response_format = ChatCompletionsConverter.convert_response_format(output_schema)
        if _debug.DONT_LOG_MODEL_DATA:
            logger.debug("Calling LLM")
        else:
            logger.debug(
                "Calling Litellm model: %s\n%s\nTools:\n%s\nStream: %s\n"
                "Tool choice: %s\nResponse format: %s\n",
                self.model,
                json.dumps(converted_messages, indent=2, ensure_ascii=False),
                json.dumps(converted_tools, indent=2, ensure_ascii=False),
                stream,
                tool_choice,
                response_format,
            )

        reasoning_effort = self._get_reasoning_effort(model_settings)
        stream_options = None
        if stream and model_settings.include_usage is not None:
            stream_options = {"include_usage": model_settings.include_usage}

        extra_kwargs: dict[str, Any] = {}
        if model_settings.extra_query:
            extra_kwargs["extra_query"] = copy(model_settings.extra_query)
        if model_settings.metadata:
            extra_kwargs["metadata"] = copy(model_settings.metadata)
        if model_settings.extra_body is not None:
            extra_body = copy(model_settings.extra_body)
            if isinstance(extra_body, dict) and reasoning_effort is not None:
                extra_body.pop("reasoning_effort", None)
                if not extra_body:
                    extra_body = None
            if extra_body is not None:
                extra_kwargs["extra_body"] = extra_body
        if model_settings.extra_args:
            extra_kwargs.update(model_settings.extra_args)
        if converted_tools:
            # These tools were already converted by the Agents SDK; LiteLLM
            # must not run its separate proxy-only MCP discovery on them.
            extra_kwargs.setdefault("_skip_mcp_handler", True)
        if should_disable_provider_managed_retries():
            extra_kwargs["num_retries"] = 0
            extra_kwargs["max_retries"] = 0
        extra_kwargs.pop("reasoning_effort", None)
        if model_settings.top_logprobs is not None:
            extra_kwargs.setdefault("logprobs", True)

        result = await acompletion(
            model=self.model,
            messages=converted_messages,
            tools=converted_tools or None,
            temperature=model_settings.temperature,
            top_p=model_settings.top_p,
            frequency_penalty=model_settings.frequency_penalty,
            presence_penalty=model_settings.presence_penalty,
            max_tokens=model_settings.max_tokens,
            tool_choice=self._remove_not_given(tool_choice),
            response_format=self._remove_not_given(response_format),
            parallel_tool_calls=parallel_tool_calls,
            stream=stream,
            stream_options=stream_options,
            reasoning_effort=reasoning_effort,
            top_logprobs=model_settings.top_logprobs,
            extra_headers=self._merge_headers(model_settings),
            api_key=self.api_key,
            base_url=self.base_url,
            **extra_kwargs,
        )
        if isinstance(result, litellm.types.utils.ModelResponse):
            return result

        # Direct _fetch_response callers retain ownership. Only the model clone
        # created by stream_response captures a transport for automatic cleanup.
        owner = getattr(self, "_oauth_stream_owner", None)
        if owner is not None:
            owner.attach(result)
            result = owner

        responses_tool_choice = ResponsesConverter.convert_tool_choice(model_settings.tool_choice)
        if responses_tool_choice is None or responses_tool_choice is omit:
            responses_tool_choice = "auto"
        response = Response(
            id=FAKE_RESPONSES_ID,
            created_at=time.time(),
            model=self.model,
            object="response",
            output=[],
            tool_choice=responses_tool_choice,
            top_p=model_settings.top_p,
            temperature=model_settings.temperature,
            tools=[],
            parallel_tool_calls=parallel_tool_calls or False,
            reasoning=model_settings.reasoning,
        )
        return response, result
