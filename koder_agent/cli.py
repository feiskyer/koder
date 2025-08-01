"""Command-line interface for Koder Agent."""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from .core.commands import slash_handler
from .core.context import ContextManager
from .core.interactive import InteractivePrompt
from .core.scheduler import AgentScheduler
from .utils import default_session_local_ms, picker_arrows, setup_openai_client, sort_sessions_desc

console = Console()


async def _prompt_select_session() -> Optional[str]:
    ctx = ContextManager()
    sessions = await ctx.list_sessions()
    if not sessions:
        console.print(Panel("No sessions found.", title="Sessions", border_style="yellow"))
        return None

    sessions = sort_sessions_desc(sessions)

    return picker_arrows(sessions)


async def load_context() -> str:
    context_info = []
    current_dir = os.getcwd()
    context_info.append(f"Working directory: {current_dir}")
    koder_md_path = Path(current_dir) / "KODER.md"
    if koder_md_path.exists():
        try:
            koder_content = koder_md_path.read_text("utf-8", errors="ignore")
            context_info.append(f"KODER.md content:\n{koder_content}")
        except Exception as e:
            context_info.append(f"Error reading KODER.md: {e}")
    return "\n\n".join(context_info)


async def main():
    try:
        setup_openai_client()
    except ValueError as e:
        console.print(Panel(f"[red]{e}[/red]", title="❌ Error", border_style="red"))
        return 1

    parser = argparse.ArgumentParser(description="Koder - AI Coding Assistant")
    parser.add_argument("--session", "-s", default=None, help="Session ID for context")
    parser.add_argument(
        "--resume", action="store_true", help="List and select a previous session to resume"
    )
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming mode")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

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

    try:
        args = parser.parse_args()
    except SystemExit:
        known_commands = {"mcp"}
        non_flag_args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
        if non_flag_args and non_flag_args[0] in known_commands:
            raise
        if non_flag_args:
            flag_args = [arg for arg in sys.argv[1:] if arg.startswith("-")]
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

    if getattr(args, "resume", False):
        selected = await _prompt_select_session()
        if selected:
            args.session = selected
        else:
            if not getattr(args, "session", None):
                args.session = default_session_local_ms()

    if not getattr(args, "session", None):
        args.session = default_session_local_ms()

    if args.command == "mcp":
        from .mcp.cli_handler import handle_mcp_command

        return await handle_mcp_command(args)

    context = await load_context()
    console.print(f"[dim]koder session: {args.session}[/dim]")
    console.print(f"[dim]working in: {os.getcwd()}[/dim]")
    print()

    scheduler = AgentScheduler(session_id=args.session, streaming=not args.no_stream)

    try:
        command_list = slash_handler.get_command_list()
        commands_dict = {name: desc for name, desc in command_list}
        interactive_prompt = InteractivePrompt(commands_dict)

        prompt_text = getattr(args, "prompt", None)
        if prompt_text:
            prompt = " ".join(prompt_text)
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
                                    session_id=new_session_id, streaming=not args.no_stream
                                )
                                console.print(f"[dim]Switched to session: {new_session_id}[/dim]")
                            else:
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
        await scheduler.cleanup()

    return 0


def run():
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
