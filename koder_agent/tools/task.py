"""Task delegation operations."""

import asyncio
import uuid
from typing import List, Union

from agents import Agent, ModelSettings, RunConfig, Runner, function_tool
from openai.types.shared import Reasoning
from pydantic import BaseModel

from ..config import get_config
from ..mcp import load_mcp_servers
from ..utils import get_model_name


class TaskModel(BaseModel):
    description: str
    prompt: str


class TaskDelegateModel(BaseModel):
    tasks: Union[List[TaskModel], TaskModel]


@function_tool
async def task_delegate(tasks: Union[List[TaskModel], TaskModel]) -> str:
    """
    Delegate one or more tasks to specialized agents and return the aggregated results.
    If multiple tasks are provided, they will be run in parallel.
    """
    try:
        # Normalize input to always be a list
        if isinstance(tasks, TaskModel):
            task_list = [tasks]
        else:
            task_list = tasks

        config = get_config()
        mcp_servers = await load_mcp_servers()

        # Import tools dynamically to avoid circular imports
        from ..agentic import get_display_hooks
        from . import get_all_tools

        async def run_single_task(task: TaskModel) -> tuple[str, str]:
            """Run a single task and return (description, result)."""
            try:
                tools = get_all_tools()
                # remove task_delegate tool
                tools = [tool for tool in tools if tool.name != "task_delegate"]

                # Build model_settings with reasoning if configured
                # Only set reasoning parameter for native OpenAI providers
                # LiteLLM-based providers (GitHub Copilot, Anthropic, etc.) don't support it
                from ..utils.model_info import should_use_reasoning_param

                model_settings = ModelSettings()
                if config.model.reasoning_effort is not None and should_use_reasoning_param():
                    model_settings.reasoning = Reasoning(effort=config.model.reasoning_effort)

                # Create agent for the delegated task
                delegated_agent = Agent(
                    name=f"Delegated Agent - {task.description[:30]}...",
                    model=get_model_name(),
                    instructions=f"""You are a task agent handling this task: {task.description}

You have access to tools to help complete this task effectively.
Be concise and focused on the specific task at hand.
Return your findings or results directly without unnecessary explanation.""",
                    tools=tools,
                    mcp_servers=mcp_servers,
                    model_settings=model_settings,
                )
                if "github_copilot" in get_model_name():
                    delegated_agent.model_settings.extra_headers = {
                        "copilot-integration-id": "vscode-chat",
                        "editor-version": "vscode/1.98.1",
                        "editor-plugin-version": "copilot-chat/0.26.7",
                        "user-agent": "GitHubCopilotChat/0.26.7",
                        "openai-intent": "conversation-panel",
                        "x-github-api-version": "2025-04-01",
                        "x-request-id": str(uuid.uuid4()),
                        "x-vscode-user-agent-library-version": "electron-fetch",
                    }

                # Run the delegated agent
                result = await Runner.run(
                    delegated_agent,
                    task.prompt,
                    max_turns=50,
                    run_config=RunConfig(),
                    hooks=get_display_hooks(),
                )

                return task.description, result.final_output

            except Exception as e:
                return task.description, f"Error: {e}"

        # Run all tasks in parallel
        results = await asyncio.gather(*[run_single_task(task) for task in task_list])

        # Aggregate results
        if len(results) == 1:
            description, result = results[0]
            return f"Delegated task '{description}' completed successfully:\n\n{result}"
        else:
            # Multiple tasks - format as aggregated report
            aggregated_result = "# Delegated Tasks Results\n\n"
            for i, (description, result) in enumerate(results, 1):
                aggregated_result += f"## Task {i}: {description}\n\n{result}\n\n"

            return aggregated_result

    except Exception as e:
        error_msg = f"Error delegating tasks: {e}"
        return error_msg
