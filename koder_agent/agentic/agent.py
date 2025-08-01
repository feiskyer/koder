"""Agent definitions and hooks for Koder."""

import logging
import uuid

from agents import Agent
from rich.console import Console

from ..mcp import load_mcp_servers
from ..utils.client import get_model_name
from ..utils.prompts import KODER_SYSTEM_PROMPT

console = Console()
logger = logging.getLogger(__name__)


async def create_dev_agent(tools) -> Agent:
    """Create the main development agent with MCP servers."""
    # Get the appropriate model based on environment
    model = get_model_name()
    mcp_servers = await load_mcp_servers()

    dev_agent = Agent(
        name="Koder",
        model=model,
        instructions=KODER_SYSTEM_PROMPT,
        tools=tools,
        mcp_servers=mcp_servers,
    )

    if "github_copilot" in model:
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
