"""Agent definitions and hooks for Koder."""

from agents import (
    Agent,
    RunContextWrapper,
    RunHooks,
    Tool,
)
from rich.console import Console
from rich.text import Text

console = Console()


def _truncate_agent_name(name: str, max_len: int = 40) -> str:
    """Truncate agent name if it's too long."""
    if len(name) <= max_len:
        return name
    return name[: max_len - 3] + "..."


class ToolDisplayHooks(RunHooks):
    """RunHooks implementation to display tool input/output with rich formatting."""

    def __init__(self, streaming_mode: bool = False):
        self.streaming_mode = streaming_mode

    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        """Called before the agent is invoked. Called each time the current agent changes."""
        if self.streaming_mode:
            # In streaming mode, don't print to avoid conflicts
            return

        # Use simple bullet-point style consistent with streaming_display.py
        agent_text = Text()
        agent_text.append("● ", style="green")
        agent_text.append("Agent: ", style="bold cyan")
        agent_text.append(_truncate_agent_name(agent.name), style="cyan")
        console.print(agent_text)

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        """Display tool execution start."""
        # Don't print on tool start - wait for tool end to print both name and result together
        # This ensures tool name and result are always paired even with parallel execution
        pass

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        """Display tool execution result."""
        if self.streaming_mode:
            # In streaming mode, don't print to avoid conflicts
            return

        # Print tool name and result together to ensure they're always paired
        tool_text = Text()
        tool_text.append("● ", style="green")
        tool_text.append(tool.name, style="bold cyan")
        console.print(tool_text)

        # Show tool output with arrow style consistent with streaming_display.py
        display_result = str(result).strip()
        if len(display_result) > 200:
            display_result = display_result[:200] + "..."

        # Determine if this is an error output
        is_error = _is_error_output(display_result)
        style = "red" if is_error else "dim green"
        arrow_style = "red" if is_error else "dim green"

        output_text = Text()
        output_text.append("  ╰─ ", style=arrow_style)
        output_text.append(display_result, style=style)
        console.print(output_text)


def _is_error_output(output: str) -> bool:
    """Check if the output indicates an error."""
    if not output:
        return False

    error_indicators = [
        "error:",
        "Error:",
        "ERROR:",
        "failed:",
        "Failed:",
        "FAILED:",
        "exception:",
        "Exception:",
        "traceback",
        "Traceback",
        "not found",
        "Not found",
        "permission denied",
        "Permission denied",
        "No such file",
        "fatal:",
    ]

    return any(indicator in output for indicator in error_indicators)


def get_display_hooks(streaming_mode: bool = False) -> RunHooks:
    """Get the display hooks instance."""
    return ToolDisplayHooks(streaming_mode=streaming_mode)
