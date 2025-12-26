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
from rich.console import Group
from rich.live import Live
from rich.text import Text

from ..agentic import ApprovalHooks, create_dev_agent, get_display_hooks
from ..core.keyboard_listener import CancellationToken, escape_listener, iter_with_cancellation
from ..core.session import EnhancedSQLiteSession, migrate_legacy_sessions
from ..core.streaming_display import StreamingDisplayManager
from ..core.usage_tracker import UsageTracker
from ..tools import BackgroundShellManager, get_all_tools
from ..utils.terminal_theme import get_adaptive_console

console = get_adaptive_console()


class AgentScheduler:
    """Scheduler for managing agent execution with context and security."""

    def __init__(self, session_id: str = "default", streaming: bool = False):
        self.semaphore = asyncio.Semaphore(10)
        self.session = EnhancedSQLiteSession(session_id=session_id)
        self.tools = get_all_tools()
        self.dev_agent = None  # Will be initialized in async method
        self.streaming = streaming
        # Create hooks that wrap display hooks (no permissions)
        display_hooks = get_display_hooks(streaming_mode=streaming)
        self.hooks = ApprovalHooks(display_hooks)
        self._agent_initialized = False
        self._mcp_servers = []  # Track MCP servers for cleanup
        self.usage_tracker = UsageTracker()  # Track token usage and cost
        self._title_generation_task: asyncio.Task | None = None  # Async title generation
        self._migration_done = False  # Track if migration has been performed

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
        """Ensure the dev agent is initialized and migration is complete."""
        # Run migration once per process
        if not self._migration_done:
            await migrate_legacy_sessions(self.session.db_path)
            self._migration_done = True

        if not self._agent_initialized:
            self.dev_agent = await create_dev_agent(self.tools)
            # Track MCP servers for cleanup
            if hasattr(self.dev_agent, "mcp_servers") and self.dev_agent.mcp_servers:
                self._mcp_servers = list(self.dev_agent.mcp_servers)  # Create a copy
            self._agent_initialized = True

    async def _generate_title_background(self, user_input: str) -> None:
        """Background task to generate and save session title."""
        try:
            title = await self.session.generate_title(user_input)
            if title:
                await self.session.set_title(title)
        except Exception:
            pass  # Silent failure - best effort

    async def handle(self, user_input: str) -> str:
        """Handle user input and execute agent."""
        # Ensure agent is initialized with MCP servers and migration complete
        await self._ensure_agent_initialized()

        if self.dev_agent is None:
            console.print("[dim red]Agent not initialized[/dim red]")
            return "Agent not initialized"

        # Note: Input panel is now displayed in InteractivePrompt, so we skip showing it here

        # Check if this is the first message for title generation
        history = await self.session.get_items()
        if not history and self._title_generation_task is None:
            # Extract actual user request (strip context prefix if present)
            actual_request = user_input
            if "User request:" in user_input:
                actual_request = user_input.split("User request:")[-1].strip()
            self._title_generation_task = asyncio.create_task(
                self._generate_title_background(actual_request)
            )

        console.print()
        console.print("[dim]thinking...[/dim]")

        # Run the agent with session - history is managed automatically
        async with self.semaphore:
            try:
                if self.streaming:
                    response = await self._handle_streaming(user_input)
                else:
                    result = await Runner.run(
                        self.dev_agent,
                        user_input,  # Just current input - session handles history
                        session=self.session,  # Automatic history management
                        run_config=RunConfig(),
                        hooks=self.hooks,
                        max_turns=50,
                    )
                    # Capture token usage from result
                    await self._capture_usage(result)

                    # Filter output for security
                    response = self._filter_output(result.final_output)

                    # Clean response output without heavy panels
                    print()  # Add space before response
                    console.print(response)
                    print()  # Add space after response
            except Exception as e:
                # Handle execution errors gracefully
                response = (
                    f"[red]Execution error: {str(e)}[/red]\n\nPlease provide new instructions."
                )
                console.print(response)
                return response

        # History is automatically saved by the session
        # No manual save needed!

        return response

    async def _handle_streaming(self, user_input: str) -> str:
        """Handle streaming execution with Rich Live, smart cleanup, and ESC cancellation."""
        import os
        import sys

        # Create the streaming display manager
        display_manager = StreamingDisplayManager(console)

        # Detect terminal capabilities
        terminal_type = os.environ.get("TERM_PROGRAM", "unknown")
        supports_advanced_clearing = terminal_type in ["iTerm.app", "Apple_Terminal", "vscode"]

        # Check if ESC listener should be enabled (Unix TTY only)
        esc_enabled = sys.platform != "win32" and sys.stdin.isatty()

        # Add space before streaming starts
        print()

        # Run the agent in streaming mode
        if self.dev_agent is None:
            console.print("[dim red]Agent not initialized[/dim red]")
            return "Agent not initialized"

        result = Runner.run_streamed(
            self.dev_agent,
            user_input,  # Just current input - session handles history
            session=self.session,  # Automatic history management
            run_config=RunConfig(),
            hooks=self.hooks,
            max_turns=50,
        )

        # Track if we displayed content during streaming
        content_displayed = False

        # Track cancellation state with token for immediate response
        cancel_token = CancellationToken()
        cancelled = False

        async def handle_escape():
            """Callback when ESC key is pressed."""
            nonlocal cancelled
            cancelled = True
            cancel_token.cancel()  # Signal to break out of iterator immediately
            result.cancel(mode="immediate")  # Also cancel the underlying stream

        # Show ESC hint if enabled (will be cleared after streaming)
        esc_hint_shown = False
        if esc_enabled:
            console.print("[dim]Press ESC to cancel[/dim]")
            esc_hint_shown = True

        def clear_esc_hint():
            """Clear the ESC hint line using ANSI escape sequences."""
            if esc_hint_shown:
                # Move cursor up one line and clear it
                sys.stdout.write("\033[A\033[2K")
                sys.stdout.flush()

        # Use Rich Live for proper formatting during streaming
        with Live(
            "",
            console=console,
            refresh_per_second=8,
            transient=True,
            vertical_overflow="crop",
        ) as live:
            try:
                # Wrap event loop with ESC listener
                async with escape_listener(on_escape=handle_escape, enabled=esc_enabled):
                    # Process streaming events with cancellation support
                    # iter_with_cancellation allows immediate break when ESC is pressed
                    stream_iter = result.stream_events()
                    async for event in iter_with_cancellation(stream_iter, cancel_token):
                        # Check if cancelled (redundant but explicit)
                        if cancelled:
                            break

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
                                        and isinstance(
                                            event.item.raw_item, ResponseFunctionToolCall
                                        )
                                    ):
                                        should_update = display_manager.handle_tool_called(
                                            event.item
                                        )

                                elif event.name == "tool_output":
                                    if hasattr(event, "item") and isinstance(
                                        event.item, ToolCallOutputItem
                                    ):
                                        should_update = display_manager.handle_tool_output(
                                            event.item
                                        )

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

            except Exception as e:
                # Handle execution errors in streaming mode
                error_msg = f"Execution error: {str(e)}"
                console.print(f"[red]{error_msg}[/red]")
                # Clear the display manager and return early
                display_manager.finalize_text_sections()
                clear_esc_hint()  # Clear ESC hint on error path too
                return f"{error_msg}\n\nPlease provide new instructions."

        # After Rich Live context ends, perform intelligent cleanup
        display_manager.finalize_text_sections()

        # Clear the ESC hint line
        clear_esc_hint()

        # Handle cancellation case
        if cancelled:
            # Rich Live with transient=True clears content on exit, so we need to re-print
            # Get partial content that was accumulated during streaming
            partial_display = display_manager.get_final_display()
            partial_text = display_manager.get_final_text()

            # Show the partial output (text + tool summaries)
            if partial_display and partial_display.strip():
                print()  # Add spacing
                console.print(partial_display)
            elif partial_text and partial_text.strip():
                print()  # Add spacing
                console.print(partial_text)

            # Show cancellation message
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
            console.print()

            # Capture any usage data we can
            await self._capture_usage(result)

            # Return partial text for session history
            return partial_text or "Operation cancelled. You can provide additional instructions."

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

            # Strategy 2: Always show final response with tools, but avoid duplicate tool output
            # Get the final display content including tools
            final_display_response = display_manager.get_final_display()
            final_text_response = display_manager.get_final_text()

            # Show the complete display including tools if available, otherwise fallback to text only
            if final_display_response and final_display_response.strip():
                # Show the complete response including tool summaries
                print()  # Add spacing
                console.print(final_display_response)
                print()  # Add spacing after
            elif final_text_response and final_text_response.strip():
                # Fallback to text-only response
                print()  # Add spacing
                console.print(final_text_response)
                print()  # Add spacing after
            elif not content_displayed:
                # Fallback: show full content if no separate text and nothing was displayed
                print()  # Add spacing
                console.print(final_content)
                print()  # Add spacing after

        # Capture token usage from streaming result
        await self._capture_usage(result)

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
        # Check if input contains AGENTS.md content
        if "AGENTS.md content:" in user_input:
            lines = user_input.split("\n")
            filtered_lines = []
            skip_koder_content = False

            for line in lines:
                if "AGENTS.md content:" in line:
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

    async def _capture_usage(self, result) -> None:
        """Capture token usage from a Runner result.

        If the API doesn't return usage data, falls back to tiktoken estimation
        using the session's existing _estimate_tokens method.
        """
        try:
            input_tokens = 0
            output_tokens = 0
            context_tokens = None

            # Try to get usage from the API response
            if hasattr(result, "context_wrapper") and hasattr(result.context_wrapper, "usage"):
                usage = result.context_wrapper.usage
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0

                if hasattr(usage, "request_usage_entries") and usage.request_usage_entries:
                    last_req = usage.request_usage_entries[-1]
                    context_tokens = last_req.total_tokens

            # Fallback: estimate tokens using session's tiktoken encoder
            if input_tokens <= 0 and output_tokens <= 0:
                # Estimate output tokens from final_output
                final_output = getattr(result, "final_output", None)
                if final_output and hasattr(self.session, "encoder"):
                    output_text = str(final_output)
                    output_tokens = len(self.session.encoder.encode(output_text))

                # Estimate input/context tokens from session history
                try:
                    session_items = await self.session.get_items()
                    if session_items:
                        input_tokens = self.session._estimate_tokens(session_items)
                        context_tokens = input_tokens + output_tokens
                except Exception:
                    pass

            # Record usage if we have any tokens
            if input_tokens > 0 or output_tokens > 0:
                self.usage_tracker.record_usage(
                    input_tokens, output_tokens, context_tokens=context_tokens
                )
        except Exception:
            # Silently ignore usage capture errors
            pass

    async def cleanup(self):
        """Clean up resources, including MCP servers."""
        try:
            # Cancel pending title generation task
            if self._title_generation_task and not self._title_generation_task.done():
                self._title_generation_task.cancel()
                self._title_generation_task = None

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

            # Clean up background shells
            for shell_id in list(BackgroundShellManager.get_available_ids()):
                try:
                    await BackgroundShellManager.terminate(shell_id)
                except Exception:
                    pass  # Best effort cleanup

            # Reset agent state to force re-initialization
            if self.dev_agent:
                self.dev_agent = None
                self._agent_initialized = False

        except Exception as e:
            console.print(f"[dim red]Unexpected error during scheduler cleanup: {e}[/dim red]")
