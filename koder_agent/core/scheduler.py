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
from openai.types.responses import ResponseFunctionToolCall
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from ..agentic import ApprovalHooks, ToolApprovalError, create_dev_agent, get_display_hooks
from ..core.context import ContextManager
from ..core.permissions import PermissionManager
from ..core.streaming_display import StreamingDisplayManager
from ..tools import get_all_tools

console = Console()


class AgentScheduler:
    """Scheduler for managing agent execution with context and security."""

    def __init__(self, session_id: str = "default", streaming: bool = False):
        self.semaphore = asyncio.Semaphore(10)
        self.context_manager = ContextManager(session_id)
        self.permission_manager = PermissionManager()
        self.tools = get_all_tools()
        self.dev_agent = None  # Will be initialized in async method
        self.streaming = streaming
        # Create approval hooks that wrap display hooks
        display_hooks = get_display_hooks(streaming_mode=streaming)
        self.hooks = ApprovalHooks(self.permission_manager, display_hooks)
        self._agent_initialized = False
        self._mcp_servers = []  # Track MCP servers for cleanup

    def _has_content(self, content) -> bool:
        """Check if Rich or string content has any content."""
        if isinstance(content, str):
            return bool(content.strip())
        elif isinstance(content, Text):
            return bool(str(content).strip())
        elif isinstance(content, Group):
            return bool(content.renderables)
        else:
            return content is not None

    def _get_line_count(self, content) -> int:
        """Get line count for Rich or string content."""
        if isinstance(content, str):
            return content.count("\n") + 1
        elif isinstance(content, Text):
            return str(content).count("\n") + 1
        elif isinstance(content, Group):
            return len(content.renderables) * 2
        else:
            return 50  # Conservative estimate

    async def _ensure_agent_initialized(self):
        """Ensure the dev agent is initialized."""
        if not self._agent_initialized:
            self.dev_agent = await create_dev_agent(self.tools)
            # Track MCP servers for cleanup
            if hasattr(self.dev_agent, "mcp_servers") and self.dev_agent.mcp_servers:
                self._mcp_servers = list(self.dev_agent.mcp_servers)  # Create a copy
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

        console.print("[dim]thinking...[/dim]")

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
            try:
                if self.streaming:
                    response = await self._handle_streaming(full_input)
                else:
                    result = await Runner.run(
                        self.dev_agent,
                        full_input,
                        run_config=RunConfig(),
                        hooks=self.hooks,
                        max_turns=50,
                    )
                    # Filter output for security
                    response = self._filter_output(result.final_output)

                    # Clean response output without heavy panels
                    print()  # Add space before response
                    console.print(response)
                    print()  # Add space after response
            except ToolApprovalError as e:
                # Handle tool denial gracefully
                response = (
                    f"[red]Execution stopped: {str(e)}[/red]\n\nPlease provide new instructions."
                )
                console.print(response)
                return response

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
        """Handle streaming execution with Rich Live and smart cleanup."""
        import os
        import sys

        # Create the streaming display manager
        display_manager = StreamingDisplayManager(console)

        # Detect terminal capabilities
        terminal_type = os.environ.get("TERM_PROGRAM", "unknown")
        supports_advanced_clearing = terminal_type in ["iTerm.app", "Apple_Terminal", "vscode"]

        # Add space before streaming starts
        print()

        # Capture cursor position before Rich Live starts (if supported)
        if supports_advanced_clearing:
            sys.stdout.write("\033[6n")  # Request cursor position
            sys.stdout.flush()

        # Run the agent in streaming mode
        if self.dev_agent is None:
            console.print("[dim red]Agent not initialized[/dim red]")
            return "Agent not initialized"

        result = Runner.run_streamed(
            self.dev_agent,
            full_input,
            run_config=RunConfig(),
            hooks=self.hooks,
            max_turns=50,
        )

        # Track if we displayed content during streaming
        content_displayed = False

        # Use Rich Live for proper formatting during streaming
        with Live(
            "",
            console=console,
            refresh_per_second=8,  # Reasonable refresh rate
            transient=False,  # Keep content visible during streaming
            vertical_overflow="visible",  # Show all content even if it exceeds screen height
        ) as live:
            try:
                # Process streaming events
                async for event in result.stream_events():
                    try:
                        should_update = False

                        # Handle raw response events (token-by-token streaming)
                        if isinstance(event, RawResponsesStreamEvent):
                            if isinstance(event.data, ResponseTextDeltaEvent):
                                delta_text = event.data.delta
                                output_index = event.data.output_index

                                if delta_text:
                                    should_update = display_manager.handle_text_delta(
                                        output_index, delta_text
                                    )

                        # Handle run item events (tool calls, outputs, etc.)
                        elif isinstance(event, RunItemStreamEvent):
                            if event.name == "tool_called":
                                if (
                                    hasattr(event, "item")
                                    and isinstance(event.item, ToolCallItem)
                                    and isinstance(event.item.raw_item, ResponseFunctionToolCall)
                                ):
                                    should_update = display_manager.handle_tool_called(event.item)

                            elif event.name == "tool_output":
                                if hasattr(event, "item") and isinstance(
                                    event.item, ToolCallOutputItem
                                ):
                                    should_update = display_manager.handle_tool_output(event.item)

                            elif event.name == "message_output_created":
                                pass
                            elif event.name == "handoff_requested":
                                # Handle agent handoff as a special tool call
                                should_update = display_manager.handle_tool_called(
                                    type(
                                        "HandoffItem",
                                        (),
                                        {
                                            "raw_item": type(
                                                "RawItem",
                                                (),
                                                {
                                                    "name": "agent_handoff",
                                                    "arguments": "{}",
                                                    "id": "handoff",
                                                },
                                            )()
                                        },
                                    )()
                                )
                            elif event.name == "handoff_occured":
                                # Handle as tool output
                                should_update = display_manager.handle_tool_output(
                                    type(
                                        "HandoffOutput",
                                        (),
                                        {"output": "Agent switched", "tool_call_id": "handoff"},
                                    )()
                                )
                            elif event.name == "reasoning_item_created":
                                # Don't show reasoning steps in display
                                pass

                        # Handle agent updates (handoffs, etc.)
                        elif isinstance(event, AgentUpdatedStreamEvent):
                            # This is handled by handoff events above
                            pass

                        # Update Rich Live display
                        if should_update:
                            current_content = display_manager.get_display_content()
                            # Handle both string and renderable objects
                            if isinstance(current_content, str):
                                if current_content.strip():
                                    live.update(current_content)
                                    content_displayed = True
                            elif current_content:  # For renderables like Group
                                live.update(current_content)
                                content_displayed = True

                    except Exception as e:
                        # Log event processing errors but continue streaming
                        console.print(f"[dim red]Event processing error: {e}[/dim red]")
                        continue

            except ToolApprovalError as e:
                # Handle tool denial in streaming mode
                error_msg = f"Execution stopped: {str(e)}"
                console.print(f"[red]{error_msg}[/red]")
                # Clear the display manager and return early
                display_manager.finalize_text_sections()
                return f"{error_msg}\n\nPlease provide new instructions."
            except Exception as e:
                # Handle streaming errors
                console.print(f"[dim red]Streaming error: {e}[/dim red]")
                # Continue to try to get some output

        # After Rich Live context ends, perform intelligent cleanup
        display_manager.finalize_text_sections()

        # Get final content for permanent display
        final_content = display_manager.get_display_content()

        # Clear the Rich Live region and show final content cleanly
        # Check if we have content to display
        has_content = self._has_content(final_content)

        if has_content:
            # Strategy 1: For advanced terminals, clear the scroll buffer region
            if supports_advanced_clearing:
                try:
                    # Clear recent lines from scrollback (terminal-specific)
                    if terminal_type == "iTerm.app":
                        # iTerm2 specific: Clear last N lines from scrollback
                        lines_count = self._get_line_count(final_content)
                        sys.stdout.write(f"\033]1337;ClearScrollback=lines:{lines_count * 3}\007")
                    elif terminal_type == "Apple_Terminal":
                        # Terminal.app: Use scrollback clearing if available
                        sys.stdout.write("\033[3J")  # Clear scrollback

                    sys.stdout.flush()
                except Exception:
                    pass  # Fallback to simple approach

            # Strategy 2: Print final clean content only if not already displayed
            # Skip if we already showed content during streaming
            if not content_displayed:
                # No content was displayed during streaming, show it now
                print()  # Add spacing
                console.print(final_content)
                print()  # Add spacing after

        # Get final text response for context saving
        final_response = display_manager.get_final_text()
        if not final_response:
            # Fallback to result.final_output if no text was captured
            final_response = self._filter_output(result.final_output)
        else:
            final_response = self._filter_output(final_response)

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

        # Handle None or non-string input
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        # Filter API keys and tokens
        text = re.sub(r"sk-\w{10,}", "[TOKEN]", text)
        text = re.sub(
            r"(api[_-]?key|token|secret)[\s:=]+[\w-]{10,}", "[REDACTED]", text, flags=re.IGNORECASE
        )
        return text

    async def cleanup(self):
        """Clean up resources, including MCP servers."""
        try:
            # Clean up MCP servers one by one to avoid task group issues
            if self._mcp_servers:
                for server in self._mcp_servers:
                    try:
                        if hasattr(server, "cleanup"):
                            # Try cleanup with a timeout to avoid hanging
                            try:
                                await asyncio.wait_for(server.cleanup(), timeout=3.0)
                            except asyncio.TimeoutError:
                                console.print(
                                    f"[dim red]MCP server {getattr(server, 'name', 'unknown')} cleanup timed out[/dim red]"
                                )
                            except Exception as cleanup_error:
                                console.print(
                                    f"[dim red]Error cleaning up MCP server {getattr(server, 'name', 'unknown')}: {cleanup_error}[/dim red]"
                                )
                    except Exception as e:
                        console.print(
                            f"[dim red]Error accessing MCP server for cleanup: {e}[/dim red]"
                        )

                self._mcp_servers.clear()

            # Reset agent state to force re-initialization
            if self.dev_agent:
                self.dev_agent = None
                self._agent_initialized = False

        except Exception as e:
            console.print(f"[dim red]Unexpected error during scheduler cleanup: {e}[/dim red]")
