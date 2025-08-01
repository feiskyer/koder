"""CLI handler for MCP commands."""

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .server_config import MCPAddRequest, MCPServerType
from .server_factory import MCPServerFactory
from .server_manager import MCPServerManager

console = Console()


async def handle_mcp_command(args: argparse.Namespace) -> int:
    """Handle MCP CLI commands."""
    try:
        manager = MCPServerManager()

        if args.mcp_action == "add":
            return await handle_add_command(manager, args)
        elif args.mcp_action == "list":
            return await handle_list_command(manager)
        elif args.mcp_action == "get":
            return await handle_get_command(manager, args)
        elif args.mcp_action == "remove":
            return await handle_remove_command(manager, args)
        else:
            console.print(
                Panel(
                    "[red]Unknown MCP action. Use: add, list, get, or remove[/red]",
                    title="❌ Error",
                    border_style="red",
                )
            )
            return 1

    except Exception as e:
        console.print(Panel(f"[red]Error: {e}[/red]", title="❌ Error", border_style="red"))
        return 1


async def handle_add_command(manager: MCPServerManager, args: argparse.Namespace) -> int:
    """Handle adding an MCP server."""
    try:
        # Check if server already exists
        if await manager.server_exists(args.name):
            console.print(
                Panel(
                    f"[yellow]Server '{args.name}' already exists[/yellow]",
                    title="⚠️ Warning",
                    border_style="yellow",
                )
            )
            return 1

        # Parse environment variables
        env_vars = {}
        if args.env:
            for env_str in args.env:
                if "=" not in env_str:
                    console.print(
                        Panel(
                            f"[red]Invalid environment variable format: {env_str}. Use KEY=VALUE[/red]",
                            title="❌ Error",
                            border_style="red",
                        )
                    )
                    return 1
                key, value = env_str.split("=", 1)
                env_vars[key] = value

        # Parse headers
        headers = {}
        if args.header:
            for header_str in args.header:
                if ":" not in header_str:
                    console.print(
                        Panel(
                            f"[red]Invalid header format: {header_str}. Use 'Key: Value'[/red]",
                            title="❌ Error",
                            border_style="red",
                        )
                    )
                    return 1
                key, value = header_str.split(":", 1)
                headers[key.strip()] = value.strip()

        # Create request object
        transport_type = MCPServerType(args.transport)

        request = MCPAddRequest(
            name=args.name,
            transport_type=transport_type,
            command=args.command_or_url if transport_type == MCPServerType.STDIO else None,
            args=args.args if transport_type == MCPServerType.STDIO else [],
            env_vars=env_vars,
            url=args.command_or_url if transport_type != MCPServerType.STDIO else None,
            headers=headers,
            cache_tools_list=args.cache_tools,
            allowed_tools=args.allow_tool,
            blocked_tools=args.block_tool,
        )

        # Convert to server config
        config = request.to_server_config()

        # Validate configuration
        error = MCPServerFactory.validate_config(config)
        if error:
            console.print(
                Panel(
                    f"[red]Configuration error: {error}[/red]", title="❌ Error", border_style="red"
                )
            )
            return 1

        # Add server
        await manager.add_server(config)

        console.print(
            Panel(
                f"[green]Successfully added MCP server '{args.name}'[/green]",
                title="✅ Success",
                border_style="green",
            )
        )
        return 0

    except Exception as e:
        console.print(
            Panel(f"[red]Failed to add server: {e}[/red]", title="❌ Error", border_style="red")
        )
        return 1


async def handle_list_command(manager: MCPServerManager) -> int:
    """Handle listing MCP servers."""
    try:
        servers = await manager.list_servers()

        if not servers:
            console.print(
                Panel(
                    "[yellow]No MCP servers configured[/yellow]",
                    title="📋 MCP Servers",
                    border_style="yellow",
                )
            )
            return 0

        # Create table
        table = Table(title="MCP Servers", show_header=True, header_style="bold cyan")
        table.add_column("Name", style="cyan", width=20)
        table.add_column("Type", style="green", width=10)
        table.add_column("Configuration", style="white")
        table.add_column("Tools", style="yellow", width=15)

        for server in servers:
            # Format configuration
            if server.transport_type == MCPServerType.STDIO:
                config_str = f"{server.command} {' '.join(server.args or [])}"
            else:
                config_str = server.url or ""

            # Format tools info
            tools_info = []
            if server.allowed_tools:
                tools_info.append(f"Allow: {len(server.allowed_tools)}")
            if server.blocked_tools:
                tools_info.append(f"Block: {len(server.blocked_tools)}")
            if server.cache_tools_list:
                tools_info.append("Cached")
            tools_str = ", ".join(tools_info) if tools_info else "Default"

            table.add_row(
                server.name,
                server.transport_type.value.upper(),
                config_str[:50] + "..." if len(config_str) > 50 else config_str,
                tools_str,
            )

        console.print(table)
        return 0

    except Exception as e:
        console.print(
            Panel(f"[red]Failed to list servers: {e}[/red]", title="❌ Error", border_style="red")
        )
        return 1


async def handle_get_command(manager: MCPServerManager, args: argparse.Namespace) -> int:
    """Handle getting details for a specific MCP server."""
    try:
        server = await manager.get_server(args.name)

        if not server:
            console.print(
                Panel(
                    f"[red]Server '{args.name}' not found[/red]",
                    title="❌ Error",
                    border_style="red",
                )
            )
            return 1

        # Create detailed view
        table = Table(title=f"MCP Server: {server.name}", show_header=False)
        table.add_column("Property", style="cyan", width=20)
        table.add_column("Value", style="white")

        table.add_row("Name", server.name)
        table.add_row("Transport Type", server.transport_type.value.upper())

        if server.transport_type == MCPServerType.STDIO:
            table.add_row("Command", server.command or "")
            if server.args:
                table.add_row("Arguments", ", ".join(server.args))
            if server.env_vars:
                env_str = ", ".join(f"{k}={v}" for k, v in server.env_vars.items())
                table.add_row("Environment", env_str)
        else:
            table.add_row("URL", server.url or "")
            if server.headers:
                headers_str = ", ".join(f"{k}: {v}" for k, v in server.headers.items())
                table.add_row("Headers", headers_str)

        table.add_row("Cache Tools", "Yes" if server.cache_tools_list else "No")

        if server.allowed_tools:
            table.add_row("Allowed Tools", ", ".join(server.allowed_tools))

        if server.blocked_tools:
            table.add_row("Blocked Tools", ", ".join(server.blocked_tools))

        console.print(table)
        return 0

    except Exception as e:
        console.print(
            Panel(
                f"[red]Failed to get server details: {e}[/red]",
                title="❌ Error",
                border_style="red",
            )
        )
        return 1


async def handle_remove_command(manager: MCPServerManager, args: argparse.Namespace) -> int:
    """Handle removing an MCP server."""
    try:
        # Check if server exists
        if not await manager.server_exists(args.name):
            console.print(
                Panel(
                    f"[red]Server '{args.name}' not found[/red]",
                    title="❌ Error",
                    border_style="red",
                )
            )
            return 1

        # Remove server
        success = await manager.remove_server(args.name)

        if success:
            console.print(
                Panel(
                    f"[green]Successfully removed MCP server '{args.name}'[/green]",
                    title="✅ Success",
                    border_style="green",
                )
            )
            return 0
        else:
            console.print(
                Panel(
                    f"[red]Failed to remove server '{args.name}'[/red]",
                    title="❌ Error",
                    border_style="red",
                )
            )
            return 1

    except Exception as e:
        console.print(
            Panel(f"[red]Failed to remove server: {e}[/red]", title="❌ Error", border_style="red")
        )
        return 1
