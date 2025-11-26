"""CLI handler for config commands."""

import argparse
import os
import subprocess
import sys

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ..utils.client import get_provider_api_env_var
from .manager import get_config, get_config_manager
from .models import KoderConfig

console = Console()


async def handle_config_command(args: argparse.Namespace) -> int:
    """Handle config CLI commands.

    Args:
        args: Parsed command line arguments.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        if args.config_action == "show":
            return await handle_show_command()
        elif args.config_action == "path":
            return await handle_path_command()
        elif args.config_action == "edit":
            return await handle_edit_command()
        elif args.config_action == "init":
            return await handle_init_command()
        elif args.config_action == "set":
            return await handle_set_command(args)
        else:
            console.print(
                Panel(
                    "[yellow]Usage: koder config <show|path|edit|init|set>[/yellow]\n\n"
                    "Commands:\n"
                    "  show  - Show current configuration\n"
                    "  path  - Show config file path\n"
                    "  edit  - Open config file in editor\n"
                    "  init  - Initialize config file with defaults\n"
                    "  set   - Set a configuration value",
                    title="Config Help",
                    border_style="cyan",
                )
            )
            return 0
    except Exception as e:
        console.print(Panel(f"[red]Error: {e}[/red]", title="Error", border_style="red"))
        return 1


async def handle_show_command() -> int:
    """Show current configuration.

    Returns:
        Exit code.
    """
    config = get_config()
    config_manager = get_config_manager()

    # Convert to YAML for display
    data = config.model_dump(exclude_none=False)
    yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)

    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
    console.print(
        Panel(syntax, title=f"Configuration ({config_manager.config_path})", border_style="cyan")
    )
    return 0


async def handle_path_command() -> int:
    """Show config file path.

    Returns:
        Exit code.
    """
    config_manager = get_config_manager()
    console.print(str(config_manager.config_path))
    return 0


async def handle_edit_command() -> int:
    """Open config file in default editor.

    Returns:
        Exit code.
    """
    config_manager = get_config_manager()

    # Ensure config file exists
    if not config_manager.config_path.exists():
        config_manager.save()
        console.print(f"[dim]Created default config at {config_manager.config_path}[/dim]")

    # Determine editor
    editor = os.environ.get("EDITOR")
    if not editor:
        if sys.platform == "win32":
            editor = "notepad"
        elif sys.platform == "darwin":
            editor = "open -e"  # Opens in TextEdit on macOS
        else:
            editor = "nano"

    try:
        subprocess.run([editor, str(config_manager.config_path)], check=True)
    except FileNotFoundError:
        # Try with shell=True for compound commands like "open -e"
        subprocess.run(f"{editor} {config_manager.config_path}", shell=True, check=True)

    console.print(
        Panel(
            f"[green]Config file edited: {config_manager.config_path}[/green]",
            title="Success",
            border_style="green",
        )
    )
    return 0


async def handle_init_command() -> int:
    """Initialize config file with defaults.

    Returns:
        Exit code.
    """
    config_manager = get_config_manager()

    if config_manager.config_path.exists():
        console.print(
            Panel(
                f"[yellow]Config file already exists at {config_manager.config_path}[/yellow]\n\n"
                "Use 'koder config edit' to modify it, or delete it manually to reinitialize.",
                title="Warning",
                border_style="yellow",
            )
        )
        return 1

    config_manager.save()
    console.print(
        Panel(
            f"[green]Created config file at {config_manager.config_path}[/green]\n\n"
            "Use 'koder config show' to view or 'koder config edit' to modify.",
            title="Success",
            border_style="green",
        )
    )
    return 0


async def handle_set_command(args: argparse.Namespace) -> int:
    """Set a configuration value.

    Args:
        args: Parsed arguments with key and value.

    Returns:
        Exit code.
    """
    config_manager = get_config_manager()
    config = config_manager.load()

    # Parse key path (e.g., "model.name" -> ["model", "name"])
    keys = args.key.split(".")

    # Convert config to dict for manipulation
    data = config.model_dump()

    # Navigate to the parent and set the value
    current = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        if current[key] is None:
            current[key] = {}
        current = current[key]

    # Parse the value (convert types as needed)
    value = args.value
    if value.lower() == "true":
        value = True
    elif value.lower() == "false":
        value = False
    elif value.lower() == "null" or value.lower() == "none":
        value = None
    else:
        # Try to convert to int or float if possible
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string

    # Set the value
    final_key = keys[-1]
    current[final_key] = value

    # Rebuild config and save
    try:
        new_config = KoderConfig(**data)
        config_manager.save(new_config)

        console.print(
            Panel(
                f"[green]Set {args.key} = {args.value}[/green]",
                title="Success",
                border_style="green",
            )
        )
        return 0
    except Exception as e:
        console.print(
            Panel(
                f"[red]Invalid configuration: {e}[/red]",
                title="Error",
                border_style="red",
            )
        )
        return 1


async def show_config_status() -> str:
    """Show configuration status (for /config slash command).

    Returns:
        Status string.
    """
    config = get_config()
    config_manager = get_config_manager()

    # Create configuration table
    table = Table(title="Koder Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="cyan", width=25)
    table.add_column("Value", style="green")
    table.add_column("Source", style="dim")

    # Provider
    table.add_row("Provider", config.model.provider.upper(), "Config")

    # Model settings
    env_model = os.environ.get("KODER_MODEL")
    model_source = "ENV" if env_model else "Config"
    model_value = env_model or config.model.name
    table.add_row("Model", model_value, model_source)

    # API Key (masked) - check provider-specific env var
    provider = config.model.provider.lower()
    api_key_env_var = get_provider_api_env_var(provider)
    env_api_key = os.environ.get(api_key_env_var)
    api_key_source = "ENV" if env_api_key else ("Config" if config.model.api_key else "Not set")
    api_key_value = "****" if (env_api_key or config.model.api_key) else "Not configured"
    table.add_row("API Key", api_key_value, api_key_source)

    # CLI defaults
    table.add_row("Default Streaming", str(config.cli.stream), "Config")
    table.add_row("Default Session", config.cli.session or "(auto)", "Config")

    # MCP servers count
    mcp_count = len(config.mcp_servers)
    table.add_row("MCP Servers", str(mcp_count), "Config")

    # Config file path
    table.add_row("Config File", str(config_manager.config_path), "")

    console.print(table)
    return f"Configuration loaded from {config_manager.config_path}"
