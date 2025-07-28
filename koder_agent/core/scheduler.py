"""Agent scheduler for managing agent execution."""

import asyncio

from agents import (
    AgentUpdatedStreamEvent,
    RawResponsesStreamEvent,
    RunConfig,
    RunItemStreamEvent,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
)
from openai._utils import is_dict
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from ..agentic import create_dev_agent, get_display_hooks
from ..core.context import ContextManager
from ..tools import get_all_tools

console = Console()


class AgentScheduler:
    """Scheduler for managing agent execution with context and security."""

    def __init__(self, session_id: str = "default", streaming: bool = False):
        self.semaphore = asyncio.Semaphore(10)
        self.context_manager = ContextManager(session_id)
        self.tools = get_all_tools()
        self.dev_agent = None  # Will be initialized in async method
        self.streaming = streaming
        self.hooks = get_display_hooks(streaming_mode=streaming)
        self._agent_initialized = False
        self._mcp_servers = []  # Track MCP servers for cleanup

    async def _ensure_agent_initialized(self):
        """Ensure the dev agent is initialized."""
        if not self._agent_initialized:
            self.dev_agent = await create_dev_agent(self.tools)
            # Track MCP servers for cleanup
            if hasattr(self.dev_agent, "mcp_servers") and self.dev_agent.mcp_servers:
                self._mcp_servers = self.dev_agent.mcp_servers
            self._agent_initialized = True

    async def handle(self, user_input: str) -> str:
        """Handle user input and execute agent."""
        # Ensure agent is initialized with MCP servers
        await self._ensure_agent_initialized()

        if self.dev_agent is None:
            console.print("[dim red]Agent not initialized[/dim red]")
            return "Agent not initialized"

        # Note: Input panel is now displayed in InteractivePrompt, so we skip showing it here

        # Load conversation history
        history = await self.context_manager.load()

        console.print("[yellow]🤖 Agent is thinking...[/yellow]")

        # Build context from history
        context_str = ""
        if history:
            context_str = "Previous conversation:\n"
            for msg in history:
                role = msg["role"].capitalize()
                content = msg["content"]
                context_str += f"{role}: {content}\n"
            context_str += "\nCurrent request:\n"

        # Combine context with current user input
        full_input = context_str + user_input if context_str else user_input

        # Run the agent
        async with self.semaphore:
            if self.streaming:
                response = await self._handle_streaming(full_input)
            else:
                result = await Runner.run(
                    self.dev_agent,
                    full_input,
                    run_config=RunConfig(),
                    hooks=self.hooks,
                )
                # Filter output for security
                response = self._filter_output(result.final_output)

                console.print(
                    Panel(
                        f"[dark_cyan]{response}[/dark_cyan]",
                        title="🤖 Agent Response",
                        border_style="dark_cyan",
                    )
                )

        # Save conversation to context
        await self.context_manager.save(
            history
            + [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response},
            ]
        )

        return response

    async def _handle_streaming(self, full_input: str) -> str:
        """Handle streaming execution with real-time output."""
        # TODO: upstream issue https://github.com/openai/openai-agents-python/issues/1016
        # TODO: upstream issue https://github.com/openai/openai-agents-python/issues/824
        # Create initial display content
        streaming_text = Text()
        current_response = ""
        # Track text content by output_index to avoid duplication
        output_texts = {}
        # Track non-text indicators (tool calls, status messages, etc.)
        indicators = []

        def rebuild_display():
            """Rebuild the complete display with indicators and text content."""
            display_text = Text()

            # Add all indicators first
            for indicator in indicators:
                display_text.append(indicator)

            # Add text content from all outputs
            for idx in sorted(output_texts.keys()):
                filtered_text = self._filter_output(output_texts[idx])
                display_text.append(filtered_text)

            return display_text

        # Use Rich Live to update the display in real-time
        with Live(
            Panel(
                streaming_text,
                title="🤖 Agent Response",
                border_style="dark_cyan",
                style="dark_cyan",
            ),
            refresh_per_second=10,
            console=console,
        ) as live:
            # Run the agent in streaming mode
            if self.dev_agent is None:
                console.print("[dim red]Agent not initialized[/dim red]")
                return "Agent not initialized"

            result = Runner.run_streamed(
                self.dev_agent,
                full_input,
                run_config=RunConfig(),
                hooks=self.hooks,
            )

            try:
                # Process streaming events
                async for event in result.stream_events():
                    try:
                        # Handle raw response events (token-by-token streaming)
                        if isinstance(event, RawResponsesStreamEvent):
                            if isinstance(event.data, ResponseTextDeltaEvent):
                                delta_text = event.data.delta
                                output_index = event.data.output_index

                                if delta_text:
                                    # Initialize output text for this index if not exists
                                    if output_index not in output_texts:
                                        output_texts[output_index] = ""

                                    # Append delta to the specific output stream
                                    output_texts[output_index] += delta_text

                                    # Update current_response for final result
                                    current_response = "".join(output_texts.values())

                                    # Rebuild and update display
                                    live.update(
                                        Panel(
                                            rebuild_display(),
                                            title="🤖 Agent Response",
                                            border_style="dark_cyan",
                                        )
                                    )

                        # Handle run item events (tool calls, outputs, etc.)
                        elif isinstance(event, RunItemStreamEvent):
                            if event.name == "tool_called":
                                if (
                                    hasattr(event, "item")
                                    and isinstance(event.item, ToolCallItem)
                                    and isinstance(event.item.raw_item, ResponseFunctionToolCall)
                                ):
                                    tool_name = event.item.raw_item.name
                                    tool_id = event.item.raw_item.call_id
                                    tool_args = event.item.raw_item.arguments
                                    tool_indicator = Text(
                                        f"\n[🔧 Tool {tool_name} called (ID: {tool_id}): {tool_args}]\n",
                                        style="dim yellow",
                                    )
                                    indicators.append(tool_indicator)
                                    live.update(
                                        Panel(
                                            rebuild_display(),
                                            title="🤖 Agent Response",
                                            border_style="dark_cyan",
                                        )
                                    )
                            elif event.name == "tool_output":
                                if hasattr(event, "item") and isinstance(
                                    event.item, ToolCallOutputItem
                                ):
                                    if (
                                        is_dict(event.item.raw_item)
                                        and "call_id" in event.item.raw_item
                                        and "output" in event.item.raw_item
                                    ):
                                        tool_id = event.item.raw_item.get("call_id", "unknown")
                                        tool_result = event.item.raw_item.get("output", "unknown")
                                        tool_result = str(tool_result)[:150]
                                        tool_indicator = Text(
                                            f"\n[📤 Tool {tool_id} output:\n{tool_result}]\n",
                                            style="dim green",
                                        )
                                        indicators.append(tool_indicator)
                                        live.update(
                                            Panel(
                                                rebuild_display(),
                                                title="🤖 Agent Response",
                                                border_style="dark_cyan",
                                            )
                                        )
                            elif event.name == "message_output_created":
                                pass
                            elif event.name == "handoff_requested":
                                handoff_indicator = Text(
                                    "\n[🤝 Handoff requested]", style="dim magenta"
                                )
                                indicators.append(handoff_indicator)
                                live.update(
                                    Panel(
                                        rebuild_display(),
                                        title="🤖 Agent Response",
                                        border_style="dark_cyan",
                                    )
                                )
                            elif event.name == "handoff_occured":
                                handoff_indicator = Text(
                                    "\n[✋ Handoff occurred]", style="dim magenta"
                                )
                                indicators.append(handoff_indicator)
                                live.update(
                                    Panel(
                                        rebuild_display(),
                                        title="🤖 Agent Response",
                                        border_style="dark_cyan",
                                    )
                                )
                            elif event.name == "reasoning_item_created":
                                reasoning_indicator = Text(
                                    "\n[🧠 Reasoning step]", style="dim purple"
                                )
                                indicators.append(reasoning_indicator)
                                live.update(
                                    Panel(
                                        rebuild_display(),
                                        title="🤖 Agent Response",
                                        border_style="dark_cyan",
                                    )
                                )

                        # Handle agent updates (handoffs, etc.)
                        elif isinstance(event, AgentUpdatedStreamEvent):
                            agent_indicator = Text(
                                f"\n[🤖 Entering Agent {event.new_agent.name}]\n", style="dim cyan"
                            )
                            indicators.append(agent_indicator)
                            live.update(
                                Panel(
                                    rebuild_display(),
                                    title="🤖 Agent Response",
                                    border_style="dark_cyan",
                                )
                            )

                    except Exception as e:
                        # Log event processing errors but continue streaming
                        console.print(f"[dim red]Event processing error: {e}[/dim red]")
                        continue
            except Exception as e:
                # Handle streaming errors
                console.print(f"[dim red]Streaming error: {e}[/dim red]")

        # Get final result and filter it
        if current_response:
            final_response = self._filter_output(current_response)
        else:
            final_response = self._filter_output(result.final_output)
        return final_response

    def _get_display_input(self, user_input: str) -> str:
        """Get a filtered version of user input for display purposes."""
        # Check if input contains KODER.md content
        if "KODER.md content:" in user_input:
            lines = user_input.split("\n")
            filtered_lines = []
            skip_koder_content = False

            for line in lines:
                if "KODER.md content:" in line:
                    skip_koder_content = True
                    continue
                elif skip_koder_content and line.startswith("User request:"):
                    skip_koder_content = False
                    filtered_lines.append(line)
                elif not skip_koder_content:
                    filtered_lines.append(line)

            return "\n".join(filtered_lines)

        return user_input

    def _filter_output(self, text: str) -> str:
        """Filter sensitive information from output."""
        import re

        # Filter API keys and tokens
        text = re.sub(r"sk-\w{10,}", "[TOKEN]", text)
        text = re.sub(
            r"(api[_-]?key|token|secret)[\s:=]+[\w-]{10,}", "[REDACTED]", text, flags=re.IGNORECASE
        )
        return text

    async def cleanup(self):
        """Clean up resources, including MCP servers."""
        if self._mcp_servers:
            for server in self._mcp_servers:
                try:
                    await server.cleanup()
                except Exception as e:
                    console.print(f"[dim red]Error disconnecting MCP server: {e}[/dim red]")
            self._mcp_servers.clear()
