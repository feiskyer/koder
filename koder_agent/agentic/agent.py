"""Agent definitions and hooks for Koder."""

import logging
import uuid
from pathlib import Path
from typing import Any

import backoff
import litellm
from agents import Agent, ModelSettings
from agents.extensions.models.litellm_model import LitellmModel
from agents.items import ItemHelpers, ModelResponse, TResponseStreamEvent
from agents.models.openai_responses import Converter as ResponsesConverter
from agents.tracing import generation_span
from agents.usage import Usage
from agents.util._json import _to_dump_compatible
from openai import omit
from openai._models import construct_type
from openai.types.shared import Reasoning
from rich.console import Console

from ..config import get_config
from ..mcp import load_mcp_servers
from ..tools.skill import SkillLoader
from ..utils.client import get_litellm_model_kwargs, get_model_name, is_native_openai_provider
from ..utils.model_info import get_maximum_output_tokens, should_use_reasoning_param
from ..utils.prompts import KODER_SYSTEM_PROMPT

console = Console()
logger = logging.getLogger(__name__)


class RetryingLitellmModel(LitellmModel):
    """LitellmModel with backoff retry logic."""

    _EXC = getattr(litellm, "exceptions", litellm)
    _EXC_TUPLE = (
        getattr(_EXC, "ServiceUnavailableError", Exception),
        getattr(_EXC, "RateLimitError", Exception),
        getattr(_EXC, "APIConnectionError", Exception),
        getattr(_EXC, "Timeout", Exception),
        getattr(_EXC, "InternalServerError", Exception),
    )

    def _should_use_responses_api(self) -> bool:
        """
        GitHub Copilot Codex models are not accessible via /chat/completions.
        Route them through LiteLLM's Responses API instead.
        """
        model_lower = str(self.model).lower()
        return "github_copilot/" in model_lower and "codex" in model_lower

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
        list_input = ItemHelpers.input_to_new_input_list(input)
        list_input = _to_dump_compatible(list_input)

        if model_settings.parallel_tool_calls and tools:
            parallel_tool_calls: bool | None = True
        elif model_settings.parallel_tool_calls is False:
            parallel_tool_calls = False
        else:
            parallel_tool_calls = None

        tool_choice = ResponsesConverter.convert_tool_choice(model_settings.tool_choice)
        if tool_choice is omit:
            tool_choice = None

        converted_tools = ResponsesConverter.convert_tools(tools, handoffs)
        tools_payload = _to_dump_compatible(converted_tools.tools)
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
            "stream": stream,
            "extra_headers": self._merge_headers(model_settings),
            "extra_query": model_settings.extra_query,
            "extra_body": model_settings.extra_body,
            **extra_args,
        }
        if self.api_key:
            aresponses_kwargs["api_key"] = self.api_key
        if self.base_url:
            aresponses_kwargs["base_url"] = self.base_url

        return await litellm.aresponses(**aresponses_kwargs)

    @backoff.on_exception(
        backoff.expo,
        _EXC_TUPLE,
        max_tries=3,
        jitter=backoff.full_jitter,
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
        if not self._should_use_responses_api():
            return await super().get_response(
                system_instructions,
                input,
                model_settings,
                tools,
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
                tools,
                output_schema,
                handoffs,
                previous_response_id=previous_response_id,
                stream=False,
                prompt=prompt,
            )

            response_usage = getattr(response, "usage", None)
            if response_usage:
                usage_kwargs: dict[str, Any] = {
                    "requests": 1,
                    "input_tokens": getattr(response_usage, "input_tokens", 0) or 0,
                    "output_tokens": getattr(response_usage, "output_tokens", 0) or 0,
                    "total_tokens": getattr(response_usage, "total_tokens", 0) or 0,
                }
                usage = Usage(**usage_kwargs)
                span_generation.span_data.usage = {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                }
            else:
                usage = Usage()

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
        )

    @backoff.on_exception(
        backoff.expo,
        _EXC_TUPLE,
        max_tries=5,
        jitter=backoff.full_jitter,
    )
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
        if not self._should_use_responses_api():
            async for chunk in super().stream_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
            ):
                yield chunk
            return

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
                tools,
                output_schema,
                handoffs,
                previous_response_id=previous_response_id,
                stream=True,
                prompt=prompt,
            )

            final_response = None
            async for chunk in stream:
                if hasattr(chunk, "model_dump"):
                    try:
                        data = chunk.model_dump()
                    except Exception:
                        data = chunk
                else:
                    data = chunk

                if isinstance(data, dict):
                    event_type = data.get("type")
                    if hasattr(event_type, "value"):
                        data["type"] = event_type.value
                    elif not isinstance(event_type, str):
                        data["type"] = str(event_type)
                    event = construct_type(value=data, type_=TResponseStreamEvent)
                else:
                    event = chunk

                if getattr(event, "type", None) == "response.completed":
                    final_response = getattr(event, "response", None)
                yield event

            if final_response is not None and getattr(final_response, "usage", None):
                usage_obj = final_response.usage
                span_generation.span_data.usage = {
                    "input_tokens": getattr(usage_obj, "input_tokens", 0) or 0,
                    "output_tokens": getattr(usage_obj, "output_tokens", 0) or 0,
                }
            if tracing.include_data() and final_response is not None:
                try:
                    span_generation.span_data.output = (
                        [final_response.model_dump()]
                        if hasattr(final_response, "model_dump")
                        else [final_response]
                    )
                except Exception:
                    pass


def _get_skills_metadata(config) -> str:
    """Load and return skills metadata from configured directories.

    Priority: project skills directory > user skills directory.
    Skills with the same name in project dir override user dir.
    """
    if not config.skills.enabled:
        return "Skills are disabled."

    all_skills = {}

    # Load user skills first (lower priority)
    user_dir = Path(config.skills.user_skills_dir).expanduser()
    if user_dir.exists():
        user_loader = SkillLoader(user_dir)
        for skill in user_loader.discover_skills():
            all_skills[skill.name] = skill

    # Load project skills (higher priority - overrides user skills)
    project_dir = Path(config.skills.project_skills_dir)
    if project_dir.exists():
        project_loader = SkillLoader(project_dir)
        for skill in project_loader.discover_skills():
            all_skills[skill.name] = skill

    if not all_skills:
        return "No skills are currently available."

    lines = ["Available skills:", ""]
    for skill in sorted(all_skills.values(), key=lambda s: s.name.lower()):
        description = skill.description.strip()
        lines.append(f"- {skill.name}: {description}")

    return "\n".join(lines)


async def create_dev_agent(tools) -> Agent:
    """Create the main development agent with MCP servers."""
    config = get_config()
    mcp_servers = await load_mcp_servers()

    # Determine the model to use: native OpenAI string or LitellmModel instance
    if is_native_openai_provider():
        # Use string model name for native OpenAI providers (handled by default client)
        model = get_model_name()
    else:
        # Use LitellmModel with explicit base_url and api_key
        litellm_kwargs = get_litellm_model_kwargs()
        model = RetryingLitellmModel(
            model=litellm_kwargs["model"],
            base_url=litellm_kwargs["base_url"],
            api_key=litellm_kwargs["api_key"],
        )

    # Build model_settings with reasoning if configured
    model_name_str = get_model_name()  # Always get string name for max_tokens lookup
    model_settings = ModelSettings(
        metadata={"source": "koder"},
        max_tokens=get_maximum_output_tokens(model_name_str),
    )
    # Only set reasoning parameter for native OpenAI providers
    # LiteLLM-based providers (GitHub Copilot, Anthropic, etc.) don't support the Reasoning object
    if config.model.reasoning_effort is not None and should_use_reasoning_param():
        effort = None if config.model.reasoning_effort == "none" else config.model.reasoning_effort
        model_settings.reasoning = Reasoning(effort=effort, summary="detailed")

    # Build system prompt with skills metadata (Progressive Disclosure Level 1)
    skills_metadata = _get_skills_metadata(config)
    system_prompt = KODER_SYSTEM_PROMPT.replace("{SKILLS_METADATA}", skills_metadata)

    dev_agent = Agent(
        name="Koder",
        model=model,
        instructions=system_prompt,
        tools=tools,
        mcp_servers=mcp_servers,
        model_settings=model_settings,
    )

    if "github_copilot" in model_name_str:
        dev_agent.model_settings.extra_headers = {
            "copilot-integration-id": "vscode-chat",
            "editor-version": "vscode/1.98.1",
            "editor-plugin-version": "copilot-chat/0.26.7",
            "user-agent": "GitHubCopilotChat/0.26.7",
            "openai-intent": "conversation-panel",
            "x-github-api-version": "2025-04-01",
            "x-request-id": str(uuid.uuid4()),
            "x-vscode-user-agent-library-version": "electron-fetch",
        }

    # planner.handoffs.append(dev_agent)
    return dev_agent
