"""Permission management for tool execution."""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.panel import Panel

console = Console()


class Permission(str, Enum):
    """Permission levels for tool execution."""

    ASK = "ask"
    ALWAYS_ALLOW = "always_allow"
    DENY = "deny"


class PermissionManager:
    """Manages tool execution permissions."""

    def __init__(self, project_root: Optional[str] = None):
        """Initialize permission manager.

        Args:
            project_root: Root directory of the project. If None, uses current directory.
        """
        self.project_root = Path(project_root or os.getcwd())
        self.config_path = self.project_root / ".vscode" / "koder.json"
        self.permissions: Dict[str, Dict[str, str]] = {"tools": {}, "mcp_tools": {}}
        self._load_permissions()

    def _load_permissions(self) -> None:
        """Load permissions from configuration file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.permissions = config.get("permissions", {"tools": {}, "mcp_tools": {}})
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load permissions: {e}[/yellow]")

    def _save_permissions(self) -> None:
        """Save permissions to configuration file."""
        # Create .vscode directory if it doesn't exist
        self.config_path.parent.mkdir(exist_ok=True)

        # Load existing config or create new one
        config = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Update permissions
        config["permissions"] = self.permissions

        # Save config
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def get_permission(self, tool_name: str, is_mcp: bool = False) -> Permission:
        """Get permission for a tool.

        Args:
            tool_name: Name of the tool
            is_mcp: Whether this is an MCP tool

        Returns:
            Permission level for the tool
        """
        category = "mcp_tools" if is_mcp else "tools"
        permission_str = self.permissions.get(category, {}).get(tool_name, Permission.ASK.value)
        return Permission(permission_str)

    def set_permission(self, tool_name: str, permission: Permission, is_mcp: bool = False) -> None:
        """Set permission for a tool.

        Args:
            tool_name: Name of the tool
            permission: Permission level to set
            is_mcp: Whether this is an MCP tool
        """
        category = "mcp_tools" if is_mcp else "tools"
        if category not in self.permissions:
            self.permissions[category] = {}
        self.permissions[category][tool_name] = permission.value
        self._save_permissions()

    def check_permission(self, tool_name: str, tool_args: Dict, is_mcp: bool = False) -> bool:
        """Check if tool execution is allowed.

        Args:
            tool_name: Name of the tool
            tool_args: Arguments passed to the tool
            is_mcp: Whether this is an MCP tool

        Returns:
            True if tool should be executed, False otherwise
        """
        permission = self.get_permission(tool_name, is_mcp)

        if permission == Permission.ALWAYS_ALLOW:
            return True
        elif permission == Permission.DENY:
            console.print(
                Panel(
                    f"[red]Tool '{tool_name}' is denied by policy[/red]",
                    title="❌ Tool Denied",
                    border_style="red",
                )
            )
            return False
        else:  # ASK
            return self._prompt_user(tool_name, tool_args, is_mcp)

    def _prompt_user(self, tool_name: str, tool_args: Dict, is_mcp: bool) -> bool:
        """Prompt user for permission to execute tool.

        Args:
            tool_name: Name of the tool
            tool_args: Arguments passed to the tool
            is_mcp: Whether this is an MCP tool

        Returns:
            True if user allows execution, False otherwise
        """
        # Format tool arguments for display
        if tool_args:
            args_display = "\n".join(f"  {k}: {v}" for k, v in tool_args.items())
            args_text = f"\n[bold yellow]Arguments:[/bold yellow]\n{args_display}"
        else:
            args_text = ""

        console.print(
            Panel(
                f"[bold yellow]{'MCP ' if is_mcp else ''}Tool:[/bold yellow] {tool_name}{args_text}",
                title="🔐 Tool Approval Required",
                border_style="yellow",
            )
        )

        # Use arrow key selection
        try:
            # We need to run this in a separate thread to avoid event loop issues
            import threading
            from queue import Queue

            result_queue = Queue()

            def get_user_choice():
                try:
                    from prompt_toolkit import Application
                    from prompt_toolkit.formatted_text import HTML
                    from prompt_toolkit.key_binding import KeyBindings
                    from prompt_toolkit.layout import Layout
                    from prompt_toolkit.layout.containers import HSplit, Window
                    from prompt_toolkit.layout.controls import FormattedTextControl

                    # Options with descriptions
                    options = [
                        ("allow", "Allow this time only"),
                        ("deny", "Deny this time only"),
                        ("always_allow", "Always allow this tool"),
                        ("always_deny", "Always deny this tool"),
                    ]

                    selected_index = 0

                    def get_formatted_text():
                        lines = ["\n<b>Select an option (use ↑↓ arrows, Enter to confirm):</b>\n"]
                        for i, (value, desc) in enumerate(options):
                            if i == selected_index:
                                lines.append(f"<reverse>→ {desc}</reverse>")
                            else:
                                lines.append(f"  {desc}")
                        return HTML("\n".join(lines))

                    # Key bindings
                    kb = KeyBindings()

                    @kb.add("up")
                    def move_up(event):
                        nonlocal selected_index
                        selected_index = (selected_index - 1) % len(options)

                    @kb.add("down")
                    def move_down(event):
                        nonlocal selected_index
                        selected_index = (selected_index + 1) % len(options)

                    @kb.add("enter")
                    def accept(event):
                        event.app.exit(result=options[selected_index][0])

                    @kb.add("c-c")
                    @kb.add("c-d")
                    def cancel(event):
                        event.app.exit(result="deny")

                    # Create application
                    app = Application(
                        layout=Layout(
                            HSplit(
                                [
                                    Window(
                                        content=FormattedTextControl(get_formatted_text),
                                        height=len(options) + 4,
                                    )
                                ]
                            )
                        ),
                        key_bindings=kb,
                        mouse_support=True,
                        full_screen=False,
                    )

                    # Run and get result
                    choice = app.run()
                    result_queue.put(choice)

                except Exception as e:
                    result_queue.put(("error", str(e)))

            # Run in thread to avoid event loop issues
            thread = threading.Thread(target=get_user_choice)
            thread.start()
            thread.join(timeout=60)  # 60 second timeout

            if thread.is_alive():
                console.print("\n[red]Timeout waiting for user input. Denying execution.[/red]")
                return False

            result = result_queue.get_nowait()

            if isinstance(result, tuple) and result[0] == "error":
                raise Exception(result[1])

            choice = result

        except Exception as e:
            # Fallback to simple input if arrow selection fails
            console.print(f"\n[yellow]Arrow selection failed ({e}). Using simple input.[/yellow]")
            console.print("\n[bold]Allow execution?[/bold]")
            console.print("[cyan]  1) allow    - Allow this time only[/cyan]")
            console.print("[cyan]  2) deny     - Deny this time only[/cyan]")
            console.print("[cyan]  3) always   - Always allow this tool[/cyan]")
            console.print("[cyan]  4) never    - Always deny this tool[/cyan]")
            console.print("[dim]Enter choice (1-4) or name [default: 1/allow]:[/dim]")

            try:
                console.file.flush()
                user_input = input("> ").strip().lower()

                # Map numbers to choices
                choice_map = {
                    "1": "allow",
                    "2": "deny",
                    "3": "always_allow",
                    "4": "always_deny",
                    "always": "always_allow",
                    "never": "always_deny",
                }

                choice = choice_map.get(user_input, user_input)

                if not choice:  # Empty input
                    choice = "allow"

                if choice not in ["allow", "deny", "always_allow", "always_deny"]:
                    console.print(
                        f"[yellow]Invalid choice '{user_input}'. Defaulting to 'allow'.[/yellow]"
                    )
                    choice = "allow"

            except (EOFError, KeyboardInterrupt):
                console.print("\n[red]Aborted by user. Denying execution.[/red]")
                return False

        if choice == "allow":
            return True
        elif choice == "deny":
            return False
        elif choice == "always_allow":
            self.set_permission(tool_name, Permission.ALWAYS_ALLOW, is_mcp)
            console.print(f"[green]Permission saved: {tool_name} → always allow[/green]")
            return True
        elif choice == "always_deny":
            self.set_permission(tool_name, Permission.DENY, is_mcp)
            console.print(f"[red]Permission saved: {tool_name} → always deny[/red]")
            return False
        else:
            # Default to deny for unknown choices
            return False
