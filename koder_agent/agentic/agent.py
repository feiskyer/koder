"""Agent definitions and hooks for Koder."""

import logging
import uuid

from agents import Agent, ModelSettings
from agents.extensions.models.litellm_model import LitellmModel
from openai.types.shared import Reasoning
from rich.console import Console

from ..config import get_config
from ..mcp import load_mcp_servers
from ..utils.client import get_litellm_model_kwargs, get_model_name, is_native_openai_provider
from ..utils.model_info import get_maximum_output_tokens
from ..utils.prompts import KODER_SYSTEM_PROMPT

console = Console()
logger = logging.getLogger(__name__)


async def create_dev_agent(tools) -> Agent:
    """Create the main development agent with MCP servers."""
    config = get_config()
    mcp_servers = await load_mcp_servers()

    # Determine the model to use: native OpenAI string or LitellmModel instance
    if is_native_openai_provider():
        # Use string model name for native OpenAI providers (handled by default client)
        model = get_model_name()
    else:
        # Use LitellmModel with explicit base_url and api_key for all other providers
        litellm_kwargs = get_litellm_model_kwargs()
        model = LitellmModel(
            model=litellm_kwargs["model"],
            base_url=litellm_kwargs["base_url"],
            api_key=litellm_kwargs["api_key"],
        )

    # Build model_settings with reasoning if configured
    model_name_str = get_model_name()  # Always get string name for max_tokens lookup
    model_settings = ModelSettings(max_tokens=get_maximum_output_tokens(model_name_str))
    if config.model.reasoning_effort is not None:
        model_settings.reasoning = Reasoning(effort=config.model.reasoning_effort)

    dev_agent = Agent(
        name="Koder",
        model=model,
        instructions=KODER_SYSTEM_PROMPT,
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
