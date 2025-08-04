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
    """Initialize project by reading codebase and generating KODER.md."""

    def __init__(self):
        super().__init__("init", "Read through the project and generate KODER.md")

    async def execute(self, scheduler, *_args) -> str:
        """Execute the /init command."""
        koder_md_path = Path(os.getcwd()) / "KODER.md"
        if koder_md_path.exists():
            console.print("[yellow]⚠️ KODER.md already exists.[/yellow]")
            return "⚠️ KODER.md already exists."

        # Use the agent to analyze the project and generate KODER.md
        prompt = """Please analyze this codebase and create a KODER.md file containing:
1. Build/lint/test commands - especially for running a single test
2. Code style guidelines including imports, formatting, types, naming conventions, error handling, etc.

Usage notes:
- The file you create will be given to agentic coding agents (such as yourself) that operate in this repository. Make it about 20 lines long.\
- If there's already a KODER.md, improve it.\
- If there are Cursor rules (in .cursor/rules/ or .cursorrules) or Copilot rules (in .github/copilot-instructions.md), make sure to include them.\
- Be sure to prefix the file with the following text:

# KODER.md

This file provides guidance to Koder when working with code in this repository.
"""

        await scheduler.handle(prompt)
        return "✅ KODER.md generated."


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
        # Get current model
        model = get_model_name()

        # Get session info
        session_id = scheduler.context_manager.session_id
        streaming = scheduler.streaming

        # Get available sessions
        sessions = await scheduler.context_manager.list_sessions()

        # Get tools count
        tools = get_all_tools()
        tool_count = len(tools)

        # Create status table
        table = Table(title="Koder Status", show_header=True, header_style="bold cyan")
        table.add_column("Setting", style="cyan", width=20)
        table.add_column("Value", style="green")

        table.add_row("Model", model)
        table.add_row("Current Session", session_id)
        table.add_row("Streaming Mode", "✅ Enabled" if streaming else "❌ Disabled")
        table.add_row("Available Tools", str(tool_count))
        table.add_row("Total Sessions", str(len(sessions)))
        table.add_row("Working Directory", os.getcwd())


        if sessions:
            table.add_section()
            table.add_row("[bold]Available Sessions[/bold]", "[bold]ID[/bold]")
            for session in sessions[:5]:  # Show max 5 sessions
                marker = "👉 " if session == session_id else "   "
                table.add_row("", f"{marker}{session}")
            if len(sessions) > 5:
                table.add_row("", f"... and {len(sessions) - 5} more")

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


class SlashCommandHandler:
    """Handles slash command detection and execution."""

    def __init__(self):
        self.commands: Dict[str, SlashCommand] = {
            "init": InitCommand(),
            "clear": ClearCommand(),
            "status": StatusCommand(),
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
                from ..utils.sessions import picker_arrows, sort_sessions_desc

                # Get sessions and prompt for selection
                sessions = await scheduler.context_manager.list_sessions()
                if not sessions:
                    return "No sessions found."

                sessions = sort_sessions_desc(sessions)
                selected = picker_arrows(sessions)

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
