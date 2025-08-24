"""Simple hooks for tool execution without permission checks."""

from typing import Any

from agents import Agent, RunContextWrapper, RunHooks, Tool


class ApprovalHooks(RunHooks):
    """RunHooks implementation that simply forwards to wrapped hooks without permission checks."""

    def __init__(self, wrapped_hooks: RunHooks):
        """Initialize approval hooks.

        Args:
            wrapped_hooks: Optional hooks to wrap (e.g., display hooks)
        """
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
        """Forward to wrapped hooks for tool display."""
        if self.wrapped_hooks:
            await self.wrapped_hooks.on_tool_start(context, agent, tool)

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        """Called after a tool is invoked."""
        if self.wrapped_hooks:
            await self.wrapped_hooks.on_tool_end(context, agent, tool, result)
