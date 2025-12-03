"""Agent definitions and hooks for Koder."""

import logging
import uuid
from pathlib import Path

import backoff
import litellm
from agents import Agent, ModelSettings
from agents.extensions.models.litellm_model import LitellmModel
from openai.types.shared import Reasoning
from rich.console import Console

from ..config import get_config
from ..mcp import load_mcp_servers
from ..tools.skill import SkillLoader
from ..utils.client import get_litellm_model_kwargs, get_model_name, is_native_openai_provider
from ..utils.model_info import get_maximum_output_tokens
from ..utils.prompts import KODER_SYSTEM_PROMPT
from .skill_guardrail import skill_restriction_guardrail

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

    @backoff.on_exception(
        backoff.expo,
        _EXC_TUPLE,
        max_tries=3,
        jitter=backoff.full_jitter,
    )
    async def get_response(self, *args, **kwargs):
        return await super().get_response(*args, **kwargs)

    @backoff.on_exception(
        backoff.expo,
        _EXC_TUPLE,
        max_tries=5,
        jitter=backoff.full_jitter,
    )
    async def stream_response(self, *args, **kwargs):
        async for chunk in super().stream_response(*args, **kwargs):
            yield chunk


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
    if config.model.reasoning_effort is not None:
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
        # Add skill-based tool restriction enforcement
        input_guardrails=[skill_restriction_guardrail],
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
