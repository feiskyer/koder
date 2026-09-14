"""Agent definitions and hooks for Koder."""

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import uuid
from collections import Counter, defaultdict
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

from agents import Agent, ModelBehaviorError, ModelSettings
from agents.items import ItemHelpers, ModelResponse, TResponseStreamEvent
from agents.models._openai_shared import get_default_openai_client
from agents.models.openai_chatcompletions import Converter as ChatCompletionsConverter
from agents.models.openai_responses import (
    Converter as ResponsesConverter,
)
from agents.models.openai_responses import (
    OpenAIResponsesModel,
)
from agents.tracing import generation_span
from agents.usage import model_usage_to_span_usage
from agents.util._json import _to_dump_compatible
from openai import AsyncOpenAI, omit
from openai._models import construct_type
from openai.types.shared import Reasoning
from rich.console import Console
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ..auth.tool_utils import clean_json_schema
from ..config import get_config
from ..harness.agents.definitions import get_agent_definitions
from ..harness.execution_context import get_execution_cwd
from ..harness.memory.budget import ContextPreflightError, estimate_model_request_preflight
from ..harness.output_styles import load_active_output_style_body
from ..harness.reasoning_display import normalize_reasoning_display_mode
from ..litellm_cost_map import get_litellm
from ..mcp import MCPServerSet, close_mcp_servers, load_mcp_servers
from ..tools.skill import build_skills_metadata_prompt, discover_merged_skills
from ..utils.async_tasks import run_sync_owned
from ..utils.client import (
    GITHUB_COPILOT_HEADERS,
    LITELLM_RETRYABLE_ERRORS,
    get_configured_context_window,
    get_model_client_snapshot,
)
from ..utils.model_info import get_maximum_output_tokens, should_use_reasoning_param
from ..utils.prompts import KODER_SYSTEM_PROMPT
from .oauth_model import OAuthAwareLitellmModel
from .responses_compat import (
    check_responses_request_budget,
    close_provider_stream,
    present_request_value,
    responses_usage,
    terminal_response_error,
)

litellm = get_litellm()
console = Console()
logger = logging.getLogger(__name__)

_MCP_PUBLIC_TOOL_NAME_MAX_LENGTH = 64
_MCP_PUBLIC_TOOL_HASH_LENGTH = 8
_MCP_PUBLIC_TOOL_DISAMBIGUATION_HASH_LENGTH = 40


@dataclass(frozen=True)
class _MCPToolNameRecord:
    key: tuple[int, int]
    identity: tuple[str, str, str]
    base_name: str
    seed: str
    initial_name: str


def _present_request_value(value: Any) -> Any:
    """Normalize SDK omission sentinels before request-budget estimation."""
    return present_request_value(value)


class PreflightOpenAIResponsesModel(OpenAIResponsesModel):
    """Native Responses model with an exact, per-provider-call budget gate.

    Hooking the SDK request builder keeps native auth, tracing, streaming,
    request IDs, and response parsing intact. The returned kwargs are the same
    object the SDK passes to ``client.responses.create`` immediately afterward.
    """

    def __init__(self, *args, context_window: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.context_window = max(1, int(context_window))

    def _build_response_create_kwargs(self, *args, **kwargs) -> dict[str, Any]:
        create_kwargs = super()._build_response_create_kwargs(*args, **kwargs)
        check_responses_request_budget(
            create_kwargs,
            context_window=self.context_window,
            model=str(self.model),
        )
        return create_kwargs


def _safe_mcp_name_part(value: str) -> str:
    """Collapse non-identifier chars to ``_`` for use in ``mcp__server__tool`` ids."""
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", value or "").strip("_")
    return cleaned or "unknown"


def prefixed_mcp_tool_name(server_name: str, tool_name: str) -> str:
    """Return the ``mcp__<server>__<tool>`` public name for an MCP tool."""
    base_name = f"mcp__{_safe_mcp_name_part(server_name)}__{_safe_mcp_name_part(tool_name)}"
    return _shorten_mcp_tool_name(base_name, f"{server_name}\0{tool_name}")


def _shorten_mcp_tool_name(
    base_name: str,
    seed: str,
    *,
    force_hash: bool = False,
    hash_length: int = _MCP_PUBLIC_TOOL_HASH_LENGTH,
) -> str:
    if not force_hash and len(base_name) <= _MCP_PUBLIC_TOOL_NAME_MAX_LENGTH:
        return base_name

    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:hash_length]
    return _mcp_tool_name_with_suffix(base_name, digest)


def _mcp_tool_name_with_suffix(base_name: str, suffix_value: str) -> str:
    suffix = f"_{suffix_value}"
    stem_length = _MCP_PUBLIC_TOOL_NAME_MAX_LENGTH - len(suffix)
    if stem_length < 1:
        raise RuntimeError("MCP tool-name limit cannot represent a unique hashed name")
    stem = base_name[:stem_length].rstrip("_") or "mcp"
    return f"{stem}{suffix}"


def _canonical_mcp_tool_metadata(tool: Any) -> str:
    """Serialize stable MCP metadata/schema used to distinguish duplicate names."""
    model_dump = getattr(tool, "model_dump", None)
    if callable(model_dump):
        try:
            payload = model_dump(mode="json", by_alias=True, exclude_none=False)
        except TypeError:
            payload = model_dump(by_alias=True, exclude_none=False)
    else:
        fields = (
            "title",
            "description",
            "inputSchema",
            "outputSchema",
            "icons",
            "annotations",
            "meta",
            "execution",
        )
        payload = {field: getattr(tool, field) for field in fields if hasattr(tool, field)}

    if isinstance(payload, dict):
        payload = dict(payload)
        payload.pop("name", None)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _mcp_tool_identity(server_name: str, tool: Any) -> tuple[str, str, str]:
    return (server_name, tool.name, _canonical_mcp_tool_metadata(tool))


def _deduplicate_mcp_tool_entries(entries):
    """Deterministically keep one copy of truly identical public tool identities."""
    unique_entries = []
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        identity = _mcp_tool_identity(entry[2], entry[3])
        if identity in seen:
            logger.warning(
                "Duplicate identical MCP tool skipped: server=%s tool=%s",
                entry[2],
                entry[3].name,
            )
            continue
        seen.add(identity)
        unique_entries.append(entry)
    return unique_entries


def _allocate_mcp_tool_names(entries, reserved_names: set[str]) -> dict[tuple[int, int], str]:
    """Allocate deterministic, provider-safe names for a complete MCP tool batch.

    Allocation is intentionally set-based: every identity is known before any
    public name is claimed. If multiple identities have the same preferred name
    (including a collision in the short truncation hash), every member receives
    a full identity digest. No encounter-order winner keeps the preferred name.
    """
    name_counts = Counter((server_name, tool.name) for _, _, server_name, tool in entries)
    records: list[_MCPToolNameRecord] = []
    seen_identities: set[tuple[str, str, str]] = set()

    for server_index, tool_index, server_name, tool in sorted(
        entries,
        key=lambda entry: (_mcp_tool_identity(entry[2], entry[3]), entry[0], entry[1]),
    ):
        identity = _mcp_tool_identity(server_name, tool)
        if identity in seen_identities:
            raise ValueError(
                f"Duplicate identical MCP tool identity: server={server_name} tool={tool.name}"
            )
        seen_identities.add(identity)
        identity_seed = f"{server_name}\0{tool.name}"
        seed = identity_seed
        if name_counts[(server_name, tool.name)] > 1:
            seed = f"{seed}\0metadata:{identity[2]}"
        base_name = f"mcp__{_safe_mcp_name_part(server_name)}__{_safe_mcp_name_part(tool.name)}"
        records.append(
            _MCPToolNameRecord(
                key=(server_index, tool_index),
                identity=identity,
                base_name=base_name,
                seed=seed,
                initial_name=_shorten_mcp_tool_name(base_name, identity_seed),
            )
        )

    allocated: dict[tuple[int, int], str] = {}
    used = set(reserved_names)
    initial_groups: dict[str, list[_MCPToolNameRecord]] = defaultdict(list)
    for record in records:
        initial_groups[record.initial_name].append(record)

    pending: list[_MCPToolNameRecord] = []
    for initial_name in sorted(initial_groups):
        group = initial_groups[initial_name]
        if len(group) == 1 and initial_name not in reserved_names:
            allocated[group[0].key] = initial_name
            used.add(initial_name)
        else:
            # Promote the whole collision group. Choosing one preferred-name
            # winner here would make allocation depend on list-tools order.
            pending.extend(group)

    primary_groups: dict[str, list[_MCPToolNameRecord]] = defaultdict(list)
    for record in pending:
        candidate = _shorten_mcp_tool_name(
            record.base_name,
            record.seed,
            force_hash=True,
            hash_length=_MCP_PUBLIC_TOOL_DISAMBIGUATION_HASH_LENGTH,
        )
        primary_groups[candidate].append(record)

    unresolved: list[_MCPToolNameRecord] = []
    for candidate in sorted(primary_groups):
        group = primary_groups[candidate]
        if len(group) == 1 and candidate not in used:
            allocated[group[0].key] = candidate
            used.add(candidate)
        else:
            unresolved.extend(group)

    # A monkeypatched/broken SHA-1 implementation can make even full digests
    # identical. Resolve that finite set with independent SHA-256 material plus
    # a stable identity rank. The bounded probe only skips pre-reserved names;
    # it cannot become an unbounded salted retry loop.
    unresolved.sort(key=lambda record: record.identity)
    population = max(1, len(unresolved))
    for rank, record in enumerate(unresolved):
        secondary = hashlib.sha256(record.seed.encode("utf-8")).hexdigest()[:12]
        for probe in range(len(used) + 1):
            discriminator = rank + (probe * population)
            candidate = _mcp_tool_name_with_suffix(
                record.base_name,
                f"{secondary}_{discriminator:08x}",
            )
            if candidate not in used:
                allocated[record.key] = candidate
                used.add(candidate)
                break
        else:  # pragma: no cover - pigeonhole bound makes this unreachable
            raise RuntimeError("Unable to allocate a unique MCP tool name")

    return allocated


async def _build_prefixed_mcp_tools(mcp_servers, guardrails, reserved_names=None):
    """Wrap MCP tools as Koder-named FunctionTools with guardrails applied."""
    from agents import FunctionTool
    from agents.mcp.util import MCPUtil

    if not mcp_servers:
        return []

    results = await asyncio.gather(
        *[server.list_tools() for server in mcp_servers], return_exceptions=True
    )
    entries = []
    for server_index, (server, result) in enumerate(zip(mcp_servers, results)):
        server_name = getattr(server, "name", "unknown")
        if isinstance(result, BaseException):
            logger.warning("MCP server '%s' failed to list tools: %s", server_name, result)
            continue
        for tool_index, mcp_tool in enumerate(result):
            entries.append((server_index, tool_index, server_name, mcp_tool))

    entries = _deduplicate_mcp_tool_entries(entries)
    public_names = _allocate_mcp_tool_names(entries, set(reserved_names or set()))
    built: list[Any] = []
    for server_index, tool_index, server_name, mcp_tool in entries:
        server = mcp_servers[server_index]
        public_name = public_names[(server_index, tool_index)]
        try:
            fn_tool = MCPUtil.to_function_tool(
                mcp_tool,
                server,
                convert_schemas_to_strict=False,
                tool_name_override=public_name,
            )
        except Exception as exc:
            logger.warning(
                "Failed to build MCP tool %s from server %s: %s",
                mcp_tool.name,
                server_name,
                exc,
            )
            continue
        if isinstance(fn_tool, FunctionTool):
            fn_tool.tool_input_guardrails = list(guardrails)
        built.append(fn_tool)
    return built


# GITHUB_COPILOT_HEADERS is defined in ..utils.client (single source of truth)
# and re-imported above; the is_copilot branch below spreads it and appends a
# per-request x-request-id.

_BRIEF_MODE_INSTRUCTION = (
    "# Brief Mode\n"
    "You are in brief mode. Be extremely concise:\n"
    "- Give the shortest possible answer that is still correct and complete\n"
    "- Skip preambles, summaries, and sign-offs\n"
    "- No bullet lists when a single sentence suffices\n"
    "- For code changes, just make the change with minimal explanation\n"
    "- Only elaborate when the user explicitly asks for detail"
)


def _log_api_error_on_retry(details):
    """Log user-friendly error messages before retry attempts.

    Called before a scheduled retry.
    """
    from ..agentic.api_errors import classify_api_error

    exc_info = details.get("exception")
    if exc_info is None:
        return

    # Extract status code if available
    status_code = None
    if hasattr(exc_info, "status_code"):
        status_code = exc_info.status_code
    elif hasattr(exc_info, "response") and hasattr(exc_info.response, "status_code"):
        status_code = exc_info.response.status_code

    classified = classify_api_error(exc_info, status_code=status_code)

    # Log user-friendly message
    if classified.should_retry:
        logger.info(
            "API error (will retry): %s [attempt %d/%d]",
            classified.user_message,
            details.get("tries", 0),
            details.get("max_tries", 0),
        )
    else:
        logger.error("API error (not retryable): %s", classified.user_message)


def _log_nonstream_api_retry(state: RetryCallState) -> None:
    _log_api_error_on_retry(
        {
            "exception": state.outcome.exception() if state.outcome is not None else None,
            "tries": state.attempt_number,
            "max_tries": 3,
        }
    )


class RetryingLitellmModel(OAuthAwareLitellmModel):
    """LitellmModel with backoff retry logic."""

    def __init__(self, *args, context_window: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.context_window = (
            get_configured_context_window(str(self.model), use_main_override=False)
            if context_window is None
            else context_window
        )

    def _effective_context_window(self) -> int:
        configured = getattr(self, "context_window", None)
        if configured is not None:
            return int(configured)
        return get_configured_context_window(str(self.model), use_main_override=False)

    def _effective_output_cap(self, model_settings: ModelSettings) -> int:
        context_window = self._effective_context_window()
        configured = getattr(model_settings, "max_tokens", None)
        if configured is not None:
            return max(0, int(configured))
        return get_maximum_output_tokens(
            str(self.model),
            max_context_size=context_window,
        )

    @staticmethod
    def _converted_responses_request(
        input: str | list, tools: list, handoffs: list
    ) -> tuple[list, list]:
        list_input = _to_dump_compatible(ItemHelpers.input_to_new_input_list(input))
        converted_tools = ResponsesConverter.convert_tools(tools, handoffs)
        return list_input, _to_dump_compatible(converted_tools.tools)

    def _preflight_request(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: list,
        output_schema,
        handoffs: list,
    ) -> None:
        """Reject each actual provider request before any provider I/O."""
        if self._should_use_responses_api():
            input_items, converted_tools = self._converted_responses_request(input, tools, handoffs)
            instructions = system_instructions
            response_format = ResponsesConverter.get_response_format(output_schema)
        else:
            input_items, converted_tools = self._converted_chat_request(
                system_instructions,
                input,
                model_settings,
                tools,
                handoffs,
            )
            instructions = None  # Included in converted chat messages above.
            response_format = ChatCompletionsConverter.convert_response_format(output_schema)
        estimate = estimate_model_request_preflight(
            context_window=self._effective_context_window(),
            response_reserve=self._effective_output_cap(model_settings),
            instructions=instructions,
            input_items=input_items,
            tools=converted_tools,
            response_format=_present_request_value(response_format),
            model=str(self.model),
        )
        if not estimate.fits:
            raise ContextPreflightError(estimate, subject="Provider request")

    def _is_github_copilot(self) -> bool:
        """Check if the current model is using GitHub Copilot."""
        return str(self.model).lower().removeprefix("litellm/").startswith("github_copilot/")

    def _clean_tools_for_github_copilot(self, tools: list) -> list:
        """Clean tool schemas for GitHub Copilot compatibility.

        GitHub Copilot doesn't support $ref/$defs in JSON schemas.
        """
        if not tools or not self._is_github_copilot():
            return tools

        for tool in tools:
            if not hasattr(tool, "params_json_schema"):
                continue

            try:
                tool.params_json_schema = clean_json_schema(tool.params_json_schema)
                if hasattr(tool, "strict_json_schema"):
                    tool.strict_json_schema = False

            except Exception as exc:
                logger.debug(
                    "Failed to clean tool schema for %s: %s",
                    getattr(tool, "name", "unknown"),
                    exc,
                )

        return tools

    def _should_use_responses_api(self) -> bool:
        """
        GitHub Copilot Codex models are not accessible via /chat/completions.
        Route them through LiteLLM's Responses API instead.
        """
        model_lower = str(self.model).lower()
        return self._is_github_copilot() and "codex" in model_lower

    async def _fetch_responses_api(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: list,
        output_schema,
        handoffs: list,
        *,
        previous_response_id: str | None,
        stream: bool,
        prompt: Any | None,
    ):
        if not hasattr(litellm, "aresponses"):
            raise RuntimeError(
                "GitHub Copilot Codex models require LiteLLM Responses API support. "
                "Please upgrade litellm to a version that provides `aresponses`."
            )
        list_input, converted_tools_payload = self._converted_responses_request(
            input,
            tools,
            handoffs,
        )

        parallel_tool_calls = (
            model_settings.parallel_tool_calls
            if converted_tools_payload or prompt is not None
            else None
        )
        tool_choice = ResponsesConverter.convert_tool_choice(
            model_settings.tool_choice, tools=tools, handoffs=handoffs
        )
        if tool_choice is omit:
            tool_choice = None

        converted_tools = ResponsesConverter.convert_tools(tools, handoffs)
        tools_payload = converted_tools_payload
        if not tools_payload:
            tools_payload = None

        text_param = ResponsesConverter.get_response_format(output_schema)
        if text_param is omit:
            text_param = None

        include_set = set(converted_tools.includes)
        response_include = getattr(model_settings, "response_include", None)
        if response_include is not None:
            include_set.update(response_include)
        top_logprobs = getattr(model_settings, "top_logprobs", None)
        if top_logprobs is not None:
            include_set.add("message.output_text.logprobs")
        include = list(include_set) if include_set else None

        extra_args: dict[str, Any] = dict(model_settings.extra_args or {})
        if top_logprobs is not None:
            extra_args["top_logprobs"] = top_logprobs
        verbosity = getattr(model_settings, "verbosity", None)
        if verbosity is not None:
            if text_param is not None and isinstance(text_param, dict):
                text_param["verbosity"] = verbosity
            else:
                text_param = {"verbosity": verbosity}

        aresponses_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": list_input,
            "include": include,
            "instructions": system_instructions,
            "tools": tools_payload,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
            "temperature": model_settings.temperature,
            "top_p": model_settings.top_p,
            "truncation": getattr(model_settings, "truncation", None),
            "max_output_tokens": model_settings.max_tokens,
            "reasoning": getattr(model_settings, "reasoning", None),
            "metadata": model_settings.metadata,
            "previous_response_id": previous_response_id,
            "prompt": prompt,
            "text": text_param,
            "stream": stream,
            "extra_headers": self._merge_headers(model_settings),
            "extra_query": model_settings.extra_query,
            "extra_body": model_settings.extra_body,
        }
        for key in (
            "store",
            "prompt_cache_retention",
            "prompt_cache_options",
            "context_management",
        ):
            value = getattr(model_settings, key, None)
            if value is not None:
                aresponses_kwargs[key] = value
        duplicate_keys = sorted(
            key
            for key in extra_args
            if key in aresponses_kwargs
            and present_request_value(aresponses_kwargs[key]) is not None
        )
        if duplicate_keys:
            raise TypeError(
                "Responses request received multiple values for: " + ", ".join(duplicate_keys)
            )
        aresponses_kwargs.update(extra_args)
        if self.api_key:
            aresponses_kwargs["api_key"] = self.api_key
        if self.base_url:
            aresponses_kwargs["base_url"] = self.base_url

        check_responses_request_budget(
            aresponses_kwargs,
            context_window=self._effective_context_window(),
            model=str(self.model),
        )
        return await litellm.aresponses(**aresponses_kwargs)

    @retry(
        retry=retry_if_exception_type(LITELLM_RETRYABLE_ERRORS),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1),
        before_sleep=_log_nonstream_api_retry,
        reraise=True,
    )
    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: list,
        output_schema,
        handoffs: list,
        tracing,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,  # unused for LiteLLM responses
        prompt: Any | None = None,
    ) -> ModelResponse:
        # Clean tools for GitHub Copilot compatibility
        cleaned_tools = self._clean_tools_for_github_copilot(tools)

        if not self._should_use_responses_api():
            self._preflight_request(
                system_instructions,
                input,
                model_settings,
                cleaned_tools,
                output_schema,
                handoffs,
            )
            return await super().get_response(
                system_instructions,
                input,
                model_settings,
                cleaned_tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
            )

        with generation_span(
            model=str(self.model),
            model_config=model_settings.to_json_dict()
            | {"base_url": str(self.base_url or ""), "model_impl": "litellm-responses"},
            disabled=tracing.is_disabled(),
        ) as span_generation:
            response = await self._fetch_responses_api(
                system_instructions,
                input,
                model_settings,
                cleaned_tools,
                output_schema,
                handoffs,
                previous_response_id=previous_response_id,
                stream=False,
                prompt=prompt,
            )

            failure = terminal_response_error(response)
            if failure is not None:
                span_generation.set_error({"message": str(failure), "data": {}})
                raise failure
            usage = responses_usage(getattr(response, "usage", None))
            span_generation.span_data.usage = model_usage_to_span_usage(usage)

            if tracing.include_data():
                try:
                    span_generation.span_data.output = (
                        [response.model_dump()] if hasattr(response, "model_dump") else [response]
                    )
                except Exception:
                    pass

        return ModelResponse(
            output=getattr(response, "output", []) or [],
            usage=usage,
            response_id=getattr(response, "id", None),
            request_id=getattr(response, "_request_id", None),
        )

    async def _stream_via_responses_api(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        cleaned_tools: list,
        output_schema,
        handoffs: list,
        tracing,
        *,
        previous_response_id: str | None,
        prompt: Any | None,
    ):
        """Stream via the custom LiteLLM Responses API path.

        Yields ``TResponseStreamEvent`` objects identical to the original
        implementation. The caller is responsible for retry handling.
        """
        with generation_span(
            model=str(self.model),
            model_config=model_settings.to_json_dict()
            | {"base_url": str(self.base_url or ""), "model_impl": "litellm-responses"},
            disabled=tracing.is_disabled(),
        ) as span_generation:
            stream = await self._fetch_responses_api(
                system_instructions,
                input,
                model_settings,
                cleaned_tools,
                output_schema,
                handoffs,
                previous_response_id=previous_response_id,
                stream=True,
                prompt=prompt,
            )

            yielded_terminal = False
            failed = False
            try:
                async for chunk in stream:
                    if hasattr(chunk, "model_dump"):
                        try:
                            data = chunk.model_dump()
                        except Exception:
                            data = chunk
                    else:
                        data = chunk

                    if isinstance(data, dict):
                        data = dict(data)
                        event_type = data.get("type")
                        if hasattr(event_type, "value"):
                            data["type"] = event_type.value
                        elif not isinstance(event_type, str):
                            data["type"] = str(event_type)
                        event = construct_type(value=data, type_=TResponseStreamEvent)
                    else:
                        event = chunk

                    event_type = getattr(event, "type", None)
                    final_response = getattr(event, "response", None)
                    failure = terminal_response_error(final_response, event_type=event_type)
                    yielded_terminal = failure is not None or event_type == "response.completed"
                    if failure is not None:
                        span_generation.set_error({"message": str(failure), "data": {}})
                    elif event_type == "response.completed":
                        usage = responses_usage(getattr(final_response, "usage", None))
                        # Record before yielding: callers can close at the terminal event.
                        span_generation.span_data.usage = model_usage_to_span_usage(usage)
                        if tracing.include_data() and final_response is not None:
                            span_generation.span_data.output = (
                                [final_response.model_dump()]
                                if hasattr(final_response, "model_dump")
                                else [final_response]
                            )
                    yield event
                    if failure is not None:
                        raise failure
                    if yielded_terminal:
                        break
                if not yielded_terminal:
                    raise ModelBehaviorError(
                        "Responses stream ended before a terminal response event."
                    )
            except BaseException:
                failed = True
                raise
            finally:
                await close_provider_stream(stream, suppress_errors=failed or yielded_terminal)

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list,
        model_settings: ModelSettings,
        tools: list,
        output_schema,
        handoffs: list,
        tracing,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,  # unused for LiteLLM responses
        prompt: Any | None = None,
    ):
        """Stream model output with retry-before-first-chunk semantics.

        Creating an async generator does not run its body. Keep retry ownership
        around iteration itself, rather than decorating generator creation.

        A retryable error is only retried while NO chunk has been yielded
        downstream yet. Once any chunk has reached the consumer, retrying would
        replay partial output (duplicate tokens), so the error is re-raised
        instead.
        """
        # Clean tools for GitHub Copilot compatibility
        cleaned_tools = self._clean_tools_for_github_copilot(tools)
        if not self._should_use_responses_api():
            self._preflight_request(
                system_instructions,
                input,
                model_settings,
                cleaned_tools,
                output_schema,
                handoffs,
            )

        max_tries = 5
        attempt = 0

        while True:
            attempt += 1
            yielded_any = False
            try:
                if self._should_use_responses_api():
                    source = self._stream_via_responses_api(
                        system_instructions,
                        input,
                        model_settings,
                        cleaned_tools,
                        output_schema,
                        handoffs,
                        tracing,
                        previous_response_id=previous_response_id,
                        prompt=prompt,
                    )
                else:
                    source = super().stream_response(
                        system_instructions,
                        input,
                        model_settings,
                        cleaned_tools,
                        output_schema,
                        handoffs,
                        tracing,
                        previous_response_id=previous_response_id,
                        conversation_id=conversation_id,
                        prompt=prompt,
                    )

                # SDK generators own task-local tracing scopes. Closing them in
                # the consuming task avoids delayed GC cleanup in a new context.
                async with aclosing(source):
                    async for chunk in source:
                        yielded_any = True
                        yield chunk
                return
            except LITELLM_RETRYABLE_ERRORS as exc:
                # Once any chunk has been emitted downstream, a retry would
                # replay partial output. Propagate instead.
                if yielded_any or attempt >= max_tries:
                    raise
                _log_api_error_on_retry(
                    {
                        "exception": exc,
                        "tries": attempt,
                        "max_tries": max_tries,
                    }
                )
                # Same full-jitter ceilings as non-streaming retries: 1, 2, 4, 8.
                wait = random.uniform(0, 2 ** (attempt - 1))
                await asyncio.sleep(wait)
                continue


def _get_skills_metadata(config) -> str:
    """Load and return skills metadata from configured directories.

    Priority: project skills directory > user skills directory.
    Skills with the same name in project dir override user dir.
    """
    if os.environ.get("KODER_SIMPLE") == "1":
        return "Skills are disabled in bare mode."
    if not config.skills.enabled:
        return "Skills are disabled."

    all_skills = discover_merged_skills(
        cwd=get_execution_cwd(),
        user_dir=config.skills.user_skills_dir,
        project_dir=config.skills.project_skills_dir,
    )

    if not all_skills:
        return "No skills are currently available."

    return build_skills_metadata_prompt(all_skills)


def _get_environment_info(model_name: str) -> str:
    """Render environment facts for the {ENVIRONMENT_INFO} prompt placeholder."""
    import platform
    from datetime import date

    cwd = get_execution_cwd()
    lines = [
        f"Working directory: {cwd}",
        f"Is a git repository: {'true' if (cwd / '.git').exists() else 'false'}",
        f"Platform: {platform.system().lower()} ({platform.release()})",
        f"Today's date: {date.today().isoformat()}",
        f"Model: {model_name}",
        "When you need other model names or IDs (e.g. when building AI applications), "
        "verify them against documentation instead of guessing.",
    ]
    return "\n".join(lines)


def _get_agents_metadata() -> str:
    if os.environ.get("KODER_SIMPLE") == "1":
        return "Agents metadata is disabled in bare mode."
    try:
        definitions = get_agent_definitions(cwd=get_execution_cwd())
    except Exception:
        return "No agents are currently available."

    if not definitions.active_agents:
        return "No agents are currently available."

    lines = ["Available agents:", ""]
    for agent in sorted(definitions.active_agents, key=lambda item: item.agent_type.lower()):
        lines.append(f"- {agent.agent_type}: {agent.when_to_use}")
    return "\n".join(lines)


async def create_dev_agent(
    tools,
    *,
    name: str = "Koder",
    instructions_override: str | None = None,
    instructions_append: str | None = None,
    model_override: str | None = None,
    extra_mcp_server_configs=None,
    _mcp_servers: MCPServerSet | None = None,
) -> Agent:
    """Create the main development agent or an overridden subagent with MCP servers."""
    if _mcp_servers is None:
        if os.environ.get("KODER_SIMPLE") == "1":
            owner = MCPServerSet()
        else:
            owner = (
                await load_mcp_servers(extra_configs=extra_mcp_server_configs)
                if extra_mcp_server_configs
                else await load_mcp_servers()
            )
        try:
            return await create_dev_agent(
                tools,
                name=name,
                instructions_override=instructions_override,
                instructions_append=instructions_append,
                model_override=model_override,
                _mcp_servers=owner,
            )
        except BaseException:
            try:
                await close_mcp_servers(owner, propagate_cancellation=False)
            except BaseException:
                logger.debug(
                    "Failed to close MCP owner after agent construction failed", exc_info=True
                )
            raise

    config = get_config()
    mcp_servers = _mcp_servers

    # Populate ToolSearch deferred tools registry with all available tools.
    # KODER_ENABLE_TOOL_SEARCH controls deferred loading behaviour:
    #   unset/true  — MCP tools deferred (names only), discovered via ToolSearch
    #   false       — all MCP tools loaded upfront, no deferral
    #   auto        — threshold mode: upfront if ≤10% of context, deferred otherwise
    #   auto:N      — threshold mode with custom % (e.g. auto:5)
    from koder_agent.tools.tool_search import _set_deferred_tools

    tool_search_mode = (
        (
            os.environ.get("KODER_ENABLE_TOOL_SEARCH")
            or os.environ.get("ENABLE_TOOL_SEARCH")
            or "true"
        )
        .strip()
        .lower()
    )

    from koder_agent.agentic.hook_guardrail import hook_pretool_input_guardrail
    from koder_agent.agentic.plan_guardrail import plan_mode_restriction_guardrail
    from koder_agent.agentic.skill_guardrail import skill_restriction_guardrail

    mcp_guardrails = [
        plan_mode_restriction_guardrail,
        skill_restriction_guardrail,
        hook_pretool_input_guardrail,
    ]
    mcp_tools = await _build_prefixed_mcp_tools(
        mcp_servers,
        mcp_guardrails,
        reserved_names={getattr(tool, "name", "") for tool in tools},
    )

    tools = [*tools, *mcp_tools]

    all_deferred = list(tools)  # Regular tools + prefixed MCP tools
    _set_deferred_tools(all_deferred if tool_search_mode != "false" else None)

    model_override_value = None if model_override in (None, "", "inherit") else str(model_override)
    model_client = await run_sync_owned(get_model_client_snapshot, model_override_value)
    effective_model_name = model_client["model_name"]
    context_window = model_client.get("context_window") or get_configured_context_window(
        effective_model_name
    )
    max_output_tokens = model_client.get("max_output_tokens") or get_maximum_output_tokens(
        effective_model_name,
        max_context_size=context_window,
    )

    # Every tool-capable path receives an explicit model object so no Runner
    # request can bypass the per-provider-call preflight boundary.
    resolved_extra_headers = None
    if model_client["native_openai"]:
        # Reuse the configured client only for the inherited main model. An
        # explicit override can resolve different credentials or an endpoint,
        # which must not be replaced by the main model's shared client.
        native_client = get_default_openai_client() if model_override_value is None else None
        if native_client is None:
            native_client = AsyncOpenAI(
                api_key=model_client.get("api_key") or "sk-koder-unconfigured",
                base_url=model_client.get("base_url"),
                max_retries=3,
            )
        model = PreflightOpenAIResponsesModel(
            model=effective_model_name,
            openai_client=native_client,
            context_window=context_window,
        )
    else:
        # Use LitellmModel with explicit base_url and api_key
        litellm_kwargs = model_client["litellm_kwargs"]
        resolved_extra_headers = litellm_kwargs.get("extra_headers")
        model = RetryingLitellmModel(
            model=litellm_kwargs["model"],
            base_url=litellm_kwargs["base_url"],
            api_key=litellm_kwargs["api_key"],
            context_window=context_window,
        )

    # Build model_settings with reasoning if configured
    model_name_str = effective_model_name
    model_settings = ModelSettings(
        metadata={"source": "koder"},
        max_tokens=max_output_tokens,
        include_usage=True,
    )
    reasoning_display = normalize_reasoning_display_mode(
        os.environ.get("KODER_REASONING_DISPLAY")
        or getattr(getattr(config, "harness", None), "reasoning_display", "off")
    )
    # Only set reasoning parameters for native OpenAI providers. LiteLLM-based providers
    # can expose reasoning through their own deltas, but the OpenAI Reasoning object can
    # cause schema validation errors when routed through LiteLLM.
    if (
        config.model.reasoning_effort is not None or reasoning_display != "off"
    ) and should_use_reasoning_param():
        reasoning_kwargs: dict[str, Any] = {"summary": "detailed"}
        if config.model.reasoning_effort is not None:
            reasoning_kwargs["effort"] = (
                None if config.model.reasoning_effort == "none" else config.model.reasoning_effort
            )
        model_settings.reasoning = Reasoning(**reasoning_kwargs)
    if resolved_extra_headers:
        model_settings.extra_headers = dict(resolved_extra_headers)

    # Build system prompt with skills metadata (Progressive Disclosure Level 1)
    skills_metadata = _get_skills_metadata(config)
    agents_metadata = _get_agents_metadata()
    environment_info = _get_environment_info(model_name_str)
    system_prompt = (
        instructions_override
        if instructions_override is not None
        else KODER_SYSTEM_PROMPT.replace("{SKILLS_METADATA}", skills_metadata)
        .replace("{AGENTS_METADATA}", agents_metadata)
        .replace("{ENVIRONMENT_INFO}", environment_info)
    )
    if instructions_append:
        system_prompt = f"{system_prompt.rstrip()}\n\n{instructions_append.strip()}"

    # Inject brief mode instruction when enabled
    from koder_agent.harness.config.service import RuntimeConfigService

    _harness_config = RuntimeConfigService().load().harness
    brief_enabled = (
        os.environ.get("KODER_BRIEF", "").lower() in ("1", "true")
        or _harness_config.brief_mode_enabled
    )
    if brief_enabled:
        system_prompt = f"{system_prompt.rstrip()}\n\n{_BRIEF_MODE_INSTRUCTION}"

    # Inject the active output-style persona body into the system prompt.
    # Only for the main agent — subagents that pass a full instructions_override
    # already carry their own persona/instructions.
    if instructions_override is None:
        try:
            persona_body = load_active_output_style_body(get_execution_cwd())
        except Exception:
            persona_body = None
        if persona_body:
            system_prompt = f"{system_prompt.rstrip()}\n\n{persona_body.strip()}"

    dev_agent = Agent(
        name=name,
        model=model,
        instructions=system_prompt,
        tools=tools,
        mcp_servers=[],
        model_settings=model_settings,
    )
    # Retained for lifecycle cleanup (scheduler/subagent teardown).
    dev_agent._koder_mcp_servers = mcp_servers

    if "github_copilot" in model_name_str:
        # NOTE: x-request-id is generated once here and reused for all requests
        # in this agent's lifetime. The openai-agents SDK applies extra_headers
        # statically — it does not support per-request dynamic header generation.
        # A truly rotating request id would require SDK-level support (e.g. a
        # callable header factory). This is a known limitation.
        dev_agent.model_settings.extra_headers = {
            **GITHUB_COPILOT_HEADERS,
            "x-request-id": str(uuid.uuid4()),
        }

    # planner.handoffs.append(dev_agent)
    return dev_agent
