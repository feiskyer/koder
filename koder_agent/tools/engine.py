"""Tool engine for managing and executing tools."""

import asyncio
from typing import Any, Dict, Type

from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel

from ..core.security import SecurityGuard

console = Console()


class ToolEngine:
    """Engine for registering and executing tools with security checks."""

    def __init__(self):
        self.registry: Dict[str, Any] = {}
        self.schemas: Dict[str, Type[BaseModel]] = {}

    def register(self, schema: Type[BaseModel]):
        """
        Decorator factory for registering tools.

        Usage:
            @tool_engine.register(MySchema)
            def my_tool(...):
                ...
        """

        def decorator(fn: Any):
            key = getattr(fn, "__name__", fn.__class__.__name__)
            self.registry[key] = fn
            self.schemas[key] = schema
            return fn

        return decorator

    async def call(self, name: str, **kw) -> str:
        """Execute a tool with security validation."""
        # Show tool input first
        console.print(
            Panel(
                f"[bold cyan]Tool:[/bold cyan] {name}\n[bold cyan]Input:[/bold cyan] {kw}",
                title="Tool Execution",
                border_style="cyan",
            )
        )

        if name not in self.registry:
            result = f"Tool {name} not found"
            console.print(Panel(f"[red]{result}[/red]", title="❌ Tool Error", border_style="red"))
            return result

        fn = self.registry[name]
        schema = self.schemas[name]

        # Validate parameters
        try:
            schema(**kw)
        except ValidationError as e:
            result = f"Param error {e}"
            console.print(
                Panel(f"[red]{result}[/red]", title="❌ Validation Error", border_style="red")
            )
            return result

        # Security checks
        if not self._validate_tool_name(name):
            result = "No permission"
            console.print(
                Panel(f"[red]{result}[/red]", title="❌ Security Error", border_style="red")
            )
            return result

        elif name == "run_shell":
            error = SecurityGuard.validate_command(kw.get("command", ""))
            if error:
                console.print(
                    Panel(f"[red]{error}[/red]", title="❌ Security Error", border_style="red")
                )
                return error

        try:
            res = await asyncio.to_thread(fn, **kw)
        except Exception as e:
            res = f"Tool err {e}"

        filtered_res = self._output_filter(str(res))

        # Show tool output
        if len(filtered_res) > 200:
            display_res = filtered_res[:200] + "..."
        else:
            display_res = filtered_res

        console.print(
            Panel(
                f"[dim blue]{display_res}[/dim blue]", title="✅ Tool Output", border_style="blue"
            )
        )

        return filtered_res

    def _validate_tool_name(self, name: str) -> bool:
        """Validate tool name against allowed list."""
        allowed_tools = {
            "read_file",
            "write_file",
            "append_file",
            "edit_file",
            "run_shell",
            "web_search",
            "glob_search",
            "grep_search",
            "list_directory",
            "todo_read",
            "todo_write",
            "web_fetch",
            "task_delegate",
            "get_skill",
            "git_command",
        }
        return name in allowed_tools

    def _output_filter(self, txt: str) -> str:
        """Filter sensitive information from output."""
        import re

        # Filter API keys and tokens
        txt = re.sub(r"sk-\w{10,}", "[TOKEN]", txt)
        txt = re.sub(
            r"(api[_-]?key|token|secret)[\s:=]+[\w-]{10,}", "[REDACTED]", txt, flags=re.IGNORECASE
        )
        return txt
