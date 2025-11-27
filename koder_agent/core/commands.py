"""Slash command system for Koder CLI."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..tools import get_all_tools
from ..utils.client import get_model_name

console = Console()


class SlashCommand:
    """Base class for slash commands."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def execute(self, scheduler, *_args) -> str:
        """Execute the command."""
        raise NotImplementedError


class InitCommand(SlashCommand):
    """Initialize project by reading codebase and generating AGENTS.md."""

    def __init__(self):
        super().__init__("init", "Read through the project and generate AGENTS.md")

    async def execute(self, scheduler, *_args) -> str:
        """Execute the /init command."""
        koder_md_path = Path(os.getcwd()) / "AGENTS.md"
        if koder_md_path.exists():
            console.print("[yellow]⚠️ AGENTS.md already exists.[/yellow]")
            return "⚠️ AGENTS.md already exists."

        # Use the agent to analyze the project and generate AGENTS.md
        prompt = """Please analyze this codebase and create a AGENTS.md file containing:
1. Build/lint/test commands - especially for running a single test
2. Code style guidelines including imports, formatting, types, naming conventions, error handling, etc.

Usage notes:
- The file you create will be given to agentic coding agents (such as yourself) that operate in this repository. Make it about 20 lines long.\
- If there's already a AGENTS.md, improve it.\
- If there are Cursor rules (in .cursor/rules/ or .cursorrules) or Copilot rules (in .github/copilot-instructions.md), make sure to include them.\
- Be sure to prefix the file with the following text:

# AGENTS.md

This file provides guidance to Koder when working with code in this repository.
"""

        await scheduler.handle(prompt)
        return "✅ AGENTS.md generated."


class ClearCommand(SlashCommand):
    """Clear the current session context."""

    def __init__(self):
        super().__init__("clear", "Start a new session with clean context")

    async def execute(self, scheduler, *_args) -> str:
        """Execute the /clear command."""
        from ..utils.sessions import default_session_local_ms

        new_sid = default_session_local_ms()

        # Clear screen
        try:
            import sys as _sys

            _sys.stdout.write("\033[2J\033[H")
            _sys.stdout.flush()
        except Exception:
            pass

        return f"session_switch:{new_sid}"  # Special return value for CLI to handle


class StatusCommand(SlashCommand):
    """Show current model and configuration status."""

    def __init__(self):
        super().__init__("status", "Show model and configuration details")

    async def execute(self, scheduler, *_args) -> str:
        """Execute the /status command."""
        from ..utils.sessions import parse_session_dt

        # Wait for any pending title generation to complete
        if (
            hasattr(scheduler, "_title_generation_task")
            and scheduler._title_generation_task
            and not scheduler._title_generation_task.done()
        ):
            try:
                await scheduler._title_generation_task
            except Exception:
                pass
            scheduler._title_generation_task = None

        # Get current model
        model = get_model_name()

        # Get session info
        session_id = scheduler.session.session_id
        streaming = scheduler.streaming

        # Get session title for current session
        session_title = await scheduler.session.get_title()
        session_display = session_title or session_id

        # Get available sessions with titles
        from .session import EnhancedSQLiteSession

        sessions_with_titles = await EnhancedSQLiteSession.list_sessions_with_titles()

        # Get tools count
        tools = get_all_tools()
        tool_count = len(tools)

        # Create status table
        table = Table(title="Koder Status", show_header=True, header_style="bold cyan")
        table.add_column("Setting", style="cyan", width=20)
        table.add_column("Value", style="green")

        table.add_row("Model", model)
        table.add_row("Current Session", session_display)
        table.add_row("Streaming Mode", "✅ Enabled" if streaming else "❌ Disabled")
        table.add_row("Available Tools", str(tool_count))
        table.add_row("Total Sessions", str(len(sessions_with_titles)))
        table.add_row("Working Directory", os.getcwd())

        if sessions_with_titles:
            # Sort by datetime descending
            sessions_with_titles.sort(
                key=lambda x: (parse_session_dt(x[0])[0], parse_session_dt(x[0])[1] or None),
                reverse=True,
            )
            table.add_section()
            table.add_row("[bold]Available Sessions[/bold]", "[bold]Title / ID[/bold]")
            for sid, title in sessions_with_titles[:5]:  # Show max 5 sessions
                if title:
                    _, dt = parse_session_dt(sid)
                    if dt:
                        display = f"{title} - {dt.strftime('%Y-%m-%d %H:%M')}"
                    else:
                        display = title
                else:
                    display = sid
                marker = "-> " if sid == session_id else "   "
                table.add_row("", f"{marker}{display}")
            if len(sessions_with_titles) > 5:
                table.add_row("", f"... and {len(sessions_with_titles) - 5} more")

        console.print(table)
        return "Status information displayed above."


class McpCommand(SlashCommand):
    """Show MCP servers configuration."""

    def __init__(self):
        super().__init__("mcp", "Show configured MCP servers")

    async def execute(self, scheduler, *_args) -> str:
        """Execute the /mcp command."""
        from ..mcp.server_manager import MCPServerManager

        try:
            manager = MCPServerManager()
            servers = await manager.list_servers()

            if not servers:
                return "No MCP servers configured. Use 'koder mcp add' to add servers."

            # Create simple table for display
            table = Table(title="MCP Servers", show_header=True, header_style="bold cyan")
            table.add_column("Name", style="cyan", width=20)
            table.add_column("Type", style="green", width=10)
            table.add_column("Configuration", style="white")

            for server in servers:
                # Format configuration
                if server.transport_type.value == "stdio":
                    config_str = f"{server.command} {' '.join(server.args or [])}"
                else:
                    config_str = server.url or ""

                table.add_row(
                    server.name,
                    server.transport_type.value.upper(),
                    config_str[:40] + "..." if len(config_str) > 40 else config_str,
                )

            console.print(table)
            return f"Found {len(servers)} MCP server(s). Use 'koder mcp get <name>' for details."

        except Exception as e:
            return f"Error listing MCP servers: {e}"


class ConfigCommand(SlashCommand):
    """Show current configuration."""

    def __init__(self):
        super().__init__("config", "Show current configuration")

    async def execute(self, scheduler, *_args) -> str:
        """Execute the /config command."""
        from ..config.cli_handler import show_config_status

        try:
            return await show_config_status()
        except Exception as e:
            return f"Error showing config: {e}"


class HelpCommand(SlashCommand):
    """Show help and usage instructions."""

    def __init__(self):
        super().__init__("help", "Show help and usage instructions")

    async def execute(self, scheduler, *_args) -> str:
        """Execute the /help command."""
        # Show general usage instructions
        usage_text = """[bold cyan]Koder - AI Coding Assistant[/bold cyan]

[bold]How to use:[/bold]
• Type your coding questions or requests in natural language
• Use slash commands (/) for special functions
• Press Ctrl+C to exit interactive mode
• Type 'exit' or 'quit' to end the session

[bold]Examples:[/bold]
• "Help me implement a login feature"
• "Fix the bug in auth.py"
• "Explain this code"
• "/status" - show current configuration
• "/clear" - start a new session"""

        console.print(Panel(usage_text, title="📖 Help", border_style="cyan"))

        # Get the command handler to access all commands
        handler = scheduler.slash_handler if hasattr(scheduler, "slash_handler") else slash_handler
        commands = handler.get_command_list()

        if commands:
            # Show available slash commands in a table
            table = Table(show_header=True, header_style="bold cyan", box=None)
            table.add_column("Command", style="cyan", width=12)
            table.add_column("Description", style="white")

            for name, description in commands:
                table.add_row(f"/{name}", description)

            console.print()
            console.print(table)

        return ""


class SlashCommandHandler:
    """Handles slash command detection and execution."""

    def __init__(self):
        self.commands: Dict[str, SlashCommand] = {
            "help": HelpCommand(),
            "init": InitCommand(),
            "clear": ClearCommand(),
            "status": StatusCommand(),
            "config": ConfigCommand(),
            "mcp": McpCommand(),
            "session": SlashCommand("session", "Switch session via picker and recreate scheduler"),
        }

    def get_command_list(self) -> List[Tuple[str, str]]:
        """Get list of available commands with descriptions."""
        return [(name, cmd.description) for name, cmd in self.commands.items()]

    async def handle_slash_input(self, user_input: str, scheduler) -> Optional[str]:
        """Handle slash command input. Returns response if it's a slash command, None otherwise."""
        if not user_input.startswith("/"):
            return None

        # Special case: just "/" shows command selection
        if user_input.strip() == "/":
            return await self._show_command_selection(scheduler)

        # Parse command
        parts = user_input[1:].split()
        if not parts:
            return await self._show_command_selection(scheduler)

        command_name = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []

        if command_name not in self.commands:
            available = ", ".join(self.commands.keys())
            return f"❌ Unknown command '{command_name}'. Available commands: {available}"

        try:
            command = self.commands[command_name]
            if command_name == "session":
                from ..utils.sessions import parse_session_dt, picker_arrows_with_titles

                # Wait for any pending title generation to complete
                if (
                    hasattr(scheduler, "_title_generation_task")
                    and scheduler._title_generation_task
                    and not scheduler._title_generation_task.done()
                ):
                    try:
                        await scheduler._title_generation_task
                    except Exception:
                        pass
                    scheduler._title_generation_task = None

                # Get sessions with titles and prompt for selection
                from .session import EnhancedSQLiteSession

                sessions_with_titles = await EnhancedSQLiteSession.list_sessions_with_titles()
                if not sessions_with_titles:
                    return "No sessions found."

                # Sort by datetime descending
                sessions_with_titles.sort(
                    key=lambda x: (parse_session_dt(x[0])[0], parse_session_dt(x[0])[1] or None),
                    reverse=True,
                )
                selected = picker_arrows_with_titles(sessions_with_titles)

                if selected:
                    return f"session_switch:{selected}"  # Special return value for CLI to handle
                return "Session switch cancelled"
            return await command.execute(scheduler, *args)
        except Exception as e:
            return f"❌ Error executing command '{command_name}': {str(e)}"

    async def _show_command_selection(self, _scheduler) -> str:
        """Show interactive command selection menu."""
        commands = self.get_command_list()

        if not commands:
            return "No slash commands available."

        # Show available commands in a simple table
        table = Table(
            title="📋 Available Slash Commands", show_header=True, header_style="bold cyan"
        )
        table.add_column("Command", style="cyan", width=12)
        table.add_column("Description", style="white")

        for name, description in commands:
            table.add_row(f"/{name}", description)

        console.print(table)
        console.print(
            Panel(
                "[dim]Commands are auto-completed when you type / - just start typing![/dim]",
                border_style="dim",
            )
        )

        return "Commands available with auto-completion. Type / and see the popup!"

    def is_slash_command(self, user_input: str) -> bool:
        """Check if input is a slash command."""
        return user_input.strip().startswith("/")


# Global instance
slash_handler = SlashCommandHandler()
