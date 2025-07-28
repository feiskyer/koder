"""Command-line interface for Koder Agent."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from .core.commands import slash_handler
from .core.interactive import InteractivePrompt
from .core.scheduler import AgentScheduler
from .utils import setup_openai_client

console = Console()


async def load_context() -> str:
    """Load context information including current directory and KODER.md."""
    context_info = []

    # Add current directory
    current_dir = os.getcwd()
    context_info.append(f"Working directory: {current_dir}")

    # Check for KODER.md
    koder_md_path = Path(current_dir) / "KODER.md"
    if koder_md_path.exists():
        try:
            koder_content = koder_md_path.read_text("utf-8", errors="ignore")
            context_info.append(f"KODER.md content:\n{koder_content}")
        except Exception as e:
            context_info.append(f"Error reading KODER.md: {e}")

    return "\n\n".join(context_info)


async def main():
    """Main entry point for the CLI."""
    # Set up OpenAI client
    try:
        setup_openai_client()
    except ValueError as e:
        console.print(Panel(f"[red]{e}[/red]", title="❌ Error", border_style="red"))
        return 1

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Koder - AI Coding Assistant")

    # Add root-level options (from old chat command)
    parser.add_argument("--session", "-s", default="default", help="Session ID for context")
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming mode for real-time output"
    )

    # Create subparsers for specific commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # MCP management commands
    mcp_parser = subparsers.add_parser("mcp", help="Manage MCP servers")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_action", help="MCP actions")

    # MCP add command
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

    # MCP list command
    mcp_subparsers.add_parser("list", help="List all MCP servers")

    # MCP get command
    get_parser = mcp_subparsers.add_parser("get", help="Get details for a specific server")
    get_parser.add_argument("name", help="Server name")

    # MCP remove command
    remove_parser = mcp_subparsers.add_parser("remove", help="Remove an MCP server")
    remove_parser.add_argument("name", help="Server name")

    # Try to parse arguments. If it fails due to unrecognized command,
    # treat everything as a prompt for backward compatibility
    try:
        args = parser.parse_args()
    except SystemExit:
        # Check if the first non-flag argument is a known command
        known_commands = {"mcp"}
        non_flag_args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

        # If it starts with a known command, re-raise the exception (it's a real parsing error)
        if non_flag_args and non_flag_args[0] in known_commands:
            raise

        # Otherwise, treat everything as a prompt for backward compatibility
        if non_flag_args:
            # Parse only the flags
            flag_args = [arg for arg in sys.argv[1:] if arg.startswith("-")]
            # Add flag values
            i = 0
            while i < len(sys.argv[1:]):
                arg = sys.argv[1:][i]
                if (
                    arg.startswith("-")
                    and i + 1 < len(sys.argv[1:])
                    and not sys.argv[1:][i + 1].startswith("-")
                ):
                    flag_args.append(sys.argv[1:][i + 1])
                    i += 1
                i += 1

            args = parser.parse_args(flag_args)
            args.command = None
            args.prompt = non_flag_args
        else:
            raise

    # Handle MCP commands
    if args.command == "mcp":
        from .mcp.cli_handler import handle_mcp_command

        return await handle_mcp_command(args)

    # Load context for chat mode
    context = await load_context()
    console.print(
        Panel(
            f"[bold cyan]🚀 Koder Started[/bold cyan]\n"
            f"[dim]Working in: {os.getcwd()}[/dim]\n"
            "[dim]Type 'exit' or 'quit' to stop, Ctrl+C to interrupt[/dim]",
            title="Koder",
            border_style="cyan",
        )
    )

    # Create scheduler
    scheduler = AgentScheduler(session_id=args.session, streaming=args.stream)

    try:
        # Create interactive prompt with slash commands
        command_list = slash_handler.get_command_list()
        commands_dict = {name: desc for name, desc in command_list}
        interactive_prompt = InteractivePrompt(commands_dict)

        # Handle initial prompt if provided
        prompt_text = getattr(args, "prompt", None)
        if prompt_text:
            prompt = " ".join(prompt_text)
            if context:
                prompt = f"Context:\n{context}\n\nUser request: {prompt}"
            await scheduler.handle(prompt)
        else:
            # Interactive mode
            while True:
                try:
                    user_input = await interactive_prompt.get_input()
                    # If we get empty input, treat it as EOF for piped input
                    if not user_input and not sys.stdin.isatty():
                        break
                except (EOFError, KeyboardInterrupt):
                    console.print(
                        Panel(
                            "[yellow]👋 Goodbye![/yellow]",
                            title="👋 Farewell",
                            border_style="yellow",
                        )
                    )
                    break

                if user_input.lower() in {"exit", "quit"}:
                    console.print(
                        Panel(
                            "[yellow]👋 Goodbye![/yellow]",
                            title="👋 Farewell",
                            border_style="yellow",
                        )
                    )
                    break

                if user_input:
                    # Check if it's a slash command
                    if slash_handler.is_slash_command(user_input):
                        slash_response = await slash_handler.handle_slash_input(
                            user_input, scheduler
                        )
                        if slash_response:
                            console.print(
                                Panel(
                                    f"[bold green]{slash_response}[/bold green]",
                                    title="⚡ Command Response",
                                    border_style="green",
                                )
                            )
                    else:
                        await scheduler.handle(user_input)
    finally:
        # Always cleanup MCP servers, regardless of how we exit
        await scheduler.cleanup()

    return 0


def run():
    """Run the CLI application."""
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        console.print(
            Panel("[yellow]👋 Interrupted![/yellow]", title="⚠️ Interruption", border_style="yellow")
        )
        exit(0)
    except Exception as e:
        console.print(
            Panel(f"[red]Fatal error: {e}[/red]", title="💥 Fatal Error", border_style="red")
        )
        exit(1)


if __name__ == "__main__":
    run()
