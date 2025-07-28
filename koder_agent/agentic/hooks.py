"""Agent definitions and hooks for Koder."""

from agents import (
    Agent,
    RunContextWrapper,
    RunHooks,
    Tool,
)
from rich.console import Console
from rich.panel import Panel

console = Console()


class ToolDisplayHooks(RunHooks):
    """RunHooks implementation to display tool input/output with rich formatting."""

    def __init__(self, streaming_mode: bool = False):
        self.streaming_mode = streaming_mode

    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        """Called before the agent is invoked. Called each time the current agent changes."""
        if self.streaming_mode:
            # In streaming mode, don't print to avoid Panel conflicts
            return

        console.print(
            Panel(
                f"🚀 [dark_cyan]AGENT {agent.name} START[/dark_cyan]",
                title="🚀 Agent START",
                border_style="dark_cyan",
            )
        )

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        """Display tool execution start."""
        if self.streaming_mode:
            # In streaming mode, don't print to avoid Panel conflicts
            return

        console.print(
            Panel(
                f"\nTOOL START: {agent.name} calling '{tool.name}'\n",
                title=f"🔧{agent.name}: Tool {tool.name} START",
                border_style="cyan",
            )
        )

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        """Display tool execution result."""
        if self.streaming_mode:
            # In streaming mode, don't print to avoid Panel conflicts
            return

        # Show tool output
        display_result = str(result)
        if len(display_result) > 200:
            display_result = display_result[:200] + "..."

        console.print(
            Panel(
                f"TOOL END: {agent.name} finished '{tool.name}' → {display_result}",
                title=f"🔧{agent.name}: Tool {tool.name} END",
                border_style="cyan",
            )
        )


def get_display_hooks(streaming_mode: bool = False) -> RunHooks:
    """Get the display hooks instance."""
    return ToolDisplayHooks(streaming_mode=streaming_mode)
