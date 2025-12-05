"""Command-line interface for Koder Agent."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from rich.panel import Panel

from .config import get_config, get_config_manager
from .core.commands import slash_handler
from .core.interactive import InteractivePrompt
from .core.scheduler import AgentScheduler
from .core.session import EnhancedSQLiteSession
from .utils import (
    default_session_local_ms,
    parse_session_dt,
    picker_arrows_with_titles,
    setup_openai_client,
)
from .utils.terminal_theme import get_adaptive_console

logging.basicConfig(level=logging.FATAL)
console = get_adaptive_console()


async def _prompt_select_session() -> Optional[str]:
    sessions_with_titles = await EnhancedSQLiteSession.list_sessions_with_titles()
    if not sessions_with_titles:
        console.print(Panel("No sessions found.", title="Sessions", border_style="yellow"))
        return None

    # Sort by datetime descending
    sessions_with_titles.sort(
        key=lambda x: (parse_session_dt(x[0])[0], parse_session_dt(x[0])[1] or None),
        reverse=True,
    )

    return picker_arrows_with_titles(sessions_with_titles)


async def load_context() -> str:
    """Load context information from the project directory.

    Returns:
        str: The loaded context information.
    """
    context_info = []
    current_dir = os.getcwd()
    context_info.append(f"Working directory: {current_dir}")
    koder_md_path = Path(current_dir) / "AGENTS.md"
    if koder_md_path.exists():
        try:
            koder_content = koder_md_path.read_text("utf-8", errors="ignore")
            context_info.append(f"AGENTS.md content:\n{koder_content}")
        except Exception as e:
            context_info.append(f"Error reading AGENTS.md: {e}")
    return "\n\n".join(context_info)


def create_mcp_subparsers(subparsers):
    """Create MCP subcommand parsers."""
    mcp_parser = subparsers.add_parser("mcp", help="Manage MCP servers")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_action", help="MCP actions")

    add_parser = mcp_subparsers.add_parser("add", help="Add an MCP server")
    add_parser.add_argument("name", help="Server name")
    add_parser.add_argument("command_or_url", help="Command for stdio or URL for SSE/HTTP")
    add_parser.add_argument("args", nargs="*", help="Arguments for stdio command")
    add_parser.add_argument(
        "--transport", choices=["stdio", "sse", "http"], default="stdio", help="Transport type"
    )
    add_parser.add_argument(
        "-e", "--env", action="append", help="Environment variables (KEY=VALUE)"
    )
    add_parser.add_argument("--header", action="append", help="HTTP headers (Key: Value)")
    add_parser.add_argument("--cache-tools", action="store_true", help="Cache tools list")
    add_parser.add_argument("--allow-tool", action="append", help="Allowed tools")
    add_parser.add_argument("--block-tool", action="append", help="Blocked tools")

    mcp_subparsers.add_parser("list", help="List all MCP servers")

    get_parser = mcp_subparsers.add_parser("get", help="Get details for a specific server")
    get_parser.add_argument("name", help="Server name")

    remove_parser = mcp_subparsers.add_parser("remove", help="Remove an MCP server")
    remove_parser.add_argument("name", help="Server name")


def create_config_subparsers(subparsers):
    """Create config subcommand parsers."""
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_subparsers = config_parser.add_subparsers(dest="config_action", help="Config actions")

    config_subparsers.add_parser("show", help="Show current configuration")
    config_subparsers.add_parser("path", help="Show config file path")
    config_subparsers.add_parser("edit", help="Open config file in default editor")
    config_subparsers.add_parser("init", help="Initialize config file with defaults")

    set_parser = config_subparsers.add_parser("set", help="Set a configuration value")
    set_parser.add_argument("key", help="Configuration key (e.g., model.name)")
    set_parser.add_argument("value", help="Value to set")


async def main():
    """Run the Koder CLI.

    Returns:
        int: The exit code.
    """
    first_arg = sys.argv[1] if len(sys.argv) > 1 else None
    is_config_command = first_arg == "config"

    if not is_config_command:
        try:
            setup_openai_client()
        except ValueError as e:
            console.print(Panel(f"[red]{e}[/red]", title="Error", border_style="red"))
            return 1

    config_manager = get_config_manager()
    config = get_config() if not is_config_command else None

    # Check if first argument is "mcp" or "config" to decide parser strategy
    if first_arg in ("mcp", "config"):
        # Use subcommand parser
        parser = argparse.ArgumentParser(description="Koder - AI Coding Assistant")
        parser.add_argument("--session", "-s", default=None, help="Session ID for context")
        parser.add_argument(
            "--resume", action="store_true", help="List and select a previous session to resume"
        )
        parser.add_argument("--no-stream", action="store_true", help="Disable streaming mode")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")

        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        create_mcp_subparsers(subparsers)
        create_config_subparsers(subparsers)
    else:
        # Use simple parser for prompt mode
        parser = argparse.ArgumentParser(description="Koder - AI Coding Assistant")
        parser.add_argument("--session", "-s", default=None, help="Session ID for context")
        parser.add_argument(
            "--resume", action="store_true", help="List and select a previous session to resume"
        )
        parser.add_argument("--no-stream", action="store_true", help="Disable streaming mode")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")

        parser.add_argument(
            "prompt", nargs="*", help="Prompt text (if not provided, starts interactive mode)"
        )

    args = parser.parse_args()

    # Configure logging level based on --debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled")

    # Set command to None for prompt mode
    if not hasattr(args, "command"):
        args.command = None

    # Handle config command before touching configuration-dependent defaults
    if args.command == "config":
        from .config.cli_handler import handle_config_command

        return await handle_config_command(args)

    if getattr(args, "resume", False):
        selected = await _prompt_select_session()
        if selected:
            args.session = selected
        else:
            if not getattr(args, "session", None):
                args.session = (
                    config_manager.get_effective_value(config.cli.session, None)
                    or default_session_local_ms()
                )

    if not getattr(args, "session", None):
        args.session = (
            config_manager.get_effective_value(config.cli.session, None)
            or default_session_local_ms()
        )

    # Handle MCP command
    if args.command == "mcp":
        from .mcp.cli_handler import handle_mcp_command

        return await handle_mcp_command(args)

    context = await load_context()

    # Determine streaming setting: --no-stream overrides config default
    streaming = config.cli.stream and not args.no_stream

    scheduler = AgentScheduler(session_id=args.session, streaming=streaming)

    try:
        command_list = slash_handler.get_command_list()
        commands_dict = {name: desc for name, desc in command_list}

        # Create interactive prompt with status line
        interactive_prompt = InteractivePrompt(
            commands_dict,
            usage_tracker=scheduler.usage_tracker,
            session_id=args.session,
        )

        prompt_text = getattr(args, "prompt", None)
        if prompt_text and len(prompt_text) > 0:
            prompt = " ".join(prompt_text)

            # Check if this is a slash command
            if slash_handler.is_slash_command(prompt):
                slash_response = await slash_handler.handle_slash_input(prompt, scheduler)
                if slash_response:
                    # Handle special session switch response
                    if slash_response.startswith("session_switch:"):
                        new_session_id = slash_response.split(":", 1)[1]
                        console.print(f"[dim]Switched to session: {new_session_id}[/dim]")
                    else:
                        console.print(
                            Panel(
                                f"[bold green]{slash_response}[/bold green]",
                                title="Command Response",
                                border_style="green",
                            )
                        )
            else:
                if context:
                    prompt = f"Context:\n{context}\n\nUser request: {prompt}"
                await scheduler.handle(prompt)
        else:
            while True:
                try:
                    user_input = await interactive_prompt.get_input()
                    if not user_input and not sys.stdin.isatty():
                        break
                except (EOFError, KeyboardInterrupt):
                    break

                if user_input.lower() in {"exit", "quit"}:
                    break

                # Check for completed title generation (might have finished while waiting for input)
                if scheduler._title_generation_task and scheduler._title_generation_task.done():
                    try:
                        display_name = await scheduler.session.get_display_name()
                        if interactive_prompt.status_line:
                            interactive_prompt.status_line.update_display_name(display_name)
                    except Exception:
                        pass
                    scheduler._title_generation_task = None

                if user_input:
                    if slash_handler.is_slash_command(user_input):
                        slash_response = await slash_handler.handle_slash_input(
                            user_input, scheduler
                        )
                        if slash_response:
                            # Handle special session switch response
                            if slash_response.startswith("session_switch:"):
                                new_session_id = slash_response.split(":", 1)[1]
                                # Clean up the old scheduler before creating a new one
                                await scheduler.cleanup()
                                scheduler = AgentScheduler(
                                    session_id=new_session_id, streaming=streaming
                                )
                                # Update interactive prompt with new session and usage tracker
                                interactive_prompt.update_session(new_session_id)
                                if interactive_prompt.status_line:
                                    interactive_prompt.status_line.usage_tracker = (
                                        scheduler.usage_tracker
                                    )
                                    # Load existing title for the session if available
                                    try:
                                        display_name = await scheduler.session.get_display_name()
                                        interactive_prompt.status_line.update_display_name(
                                            display_name
                                        )
                                    except Exception:
                                        pass
                                console.print(f"[dim]Switched to session: {new_session_id}[/dim]")
                            else:
                                console.print(
                                    Panel(
                                        f"[bold green]{slash_response}[/bold green]",
                                        title="Command Response",
                                        border_style="green",
                                    )
                                )
                    else:
                        await scheduler.handle(user_input)

                        # Refresh title display if generation completed
                        if (
                            scheduler._title_generation_task
                            and scheduler._title_generation_task.done()
                        ):
                            try:
                                display_name = await scheduler.session.get_display_name()
                                if interactive_prompt.status_line:
                                    interactive_prompt.status_line.update_display_name(display_name)
                            except Exception:
                                pass
                            scheduler._title_generation_task = None
    finally:
        try:
            await scheduler.cleanup()
        except (asyncio.CancelledError, KeyboardInterrupt, EOFError):
            # Silently ignore interruption during cleanup
            pass
        except Exception:
            # Ignore other cleanup errors on exit
            pass

    return 0


def run():
    """Run the Koder CLI."""
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except (KeyboardInterrupt, EOFError, asyncio.CancelledError):
        # Silently exit on Ctrl+C, Ctrl+D, or task cancellation
        exit(0)
    except SystemExit:
        # Re-raise SystemExit to allow normal exit
        raise
    except Exception as e:
        console.print(
            Panel(f"[red]Fatal error: {e}[/red]", title="Fatal Error", border_style="red")
        )
        exit(1)


if __name__ == "__main__":
    run()
