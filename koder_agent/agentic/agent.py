"""Agent definitions and hooks for Koder."""

import logging

from agents import Agent
from rich.console import Console

from ..mcp import load_mcp_servers
from ..utils.client import get_model_name
from ..utils.prompts import KODER_SYSTEM_PROMPT

console = Console()
logger = logging.getLogger(__name__)


# def create_planner_agent(model, tools, mcp_servers) -> Agent:
#     """Create the planner agent."""
#     return Agent(
#         name="Planner",
#         instructions=f"{KODER_SYSTEM_PROMPT}\nReturn numbered list or 'no planning needed'.",
#         tools=tools,
#         mcp_servers=mcp_servers,
#         model=model,
#         handoffs=[],
#     )


async def create_dev_agent(tools) -> Agent:
    """Create the main development agent with MCP servers."""
    # Get the appropriate model based on environment
    model = get_model_name()
    mcp_servers = await load_mcp_servers()
    # planner = create_planner_agent(model, tools, mcp_servers)

    dev_agent = Agent(
        name="Koder",
        model=model,
        instructions=KODER_SYSTEM_PROMPT,
        tools=tools,
        # handoffs=[planner],
        mcp_servers=mcp_servers,
    )
    # planner.handoffs.append(dev_agent)
    return dev_agent
