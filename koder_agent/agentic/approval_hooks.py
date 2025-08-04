"""Approval hooks for tool execution."""

from typing import Any

from agents import Agent, RunContextWrapper, RunHooks, Tool

from ..core.permissions import PermissionManager


class ToolApprovalError(Exception):
    """Raised when tool execution is denied."""


class ApprovalHooks(RunHooks):
    """RunHooks implementation that adds approval flow for tool execution."""

    def __init__(self, permission_manager: PermissionManager, wrapped_hooks: RunHooks):
        """Initialize approval hooks.

        Args:
            permission_manager: Permission manager instance
            wrapped_hooks: Optional hooks to wrap (e.g., display hooks)
        """
        self.permission_manager = permission_manager
        self.wrapped_hooks = wrapped_hooks

    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        """Called before the agent is invoked."""
        if self.wrapped_hooks:
            await self.wrapped_hooks.on_agent_start(context, agent)

    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        """Called when the agent produces a final output."""
        if self.wrapped_hooks:
            await self.wrapped_hooks.on_agent_end(context, agent, output)

    async def on_handoff(
        self, context: RunContextWrapper, from_agent: Agent, to_agent: Agent
    ) -> None:
        """Called when a handoff occurs."""
        if self.wrapped_hooks and hasattr(self.wrapped_hooks, "on_handoff"):
            await self.wrapped_hooks.on_handoff(context, from_agent, to_agent)

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        """Check permission before tool execution."""
        # Determine if this is an MCP tool
        is_mcp = tool.name.startswith("mcp__") or hasattr(tool, "is_mcp")

        # For now, we'll pass empty args since we can't easily access them from hooks
        # The permission manager will still show the tool name for approval
        tool_args = {}

        # Check permission
        if not self.permission_manager.check_permission(tool.name, tool_args, is_mcp):
            raise ToolApprovalError(f"Tool execution denied: {tool.name}")

        # Forward to wrapped hooks if available
        if self.wrapped_hooks:
            await self.wrapped_hooks.on_tool_start(context, agent, tool)

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        """Called after a tool is invoked."""
        if self.wrapped_hooks:
            await self.wrapped_hooks.on_tool_end(context, agent, tool, result)
