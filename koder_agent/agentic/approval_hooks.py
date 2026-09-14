"""Hooks for tool execution with permission checking."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from agents import Agent, RunContextWrapper, RunHooks, Tool

from koder_agent.harness.execution_context import get_execution_cwd
from koder_agent.harness.hooks.runtime import dispatch_command_hooks_async
from koder_agent.tools.permission_context import GUARDED_TOOLS

if TYPE_CHECKING:
    from koder_agent.harness.permissions.service import PermissionService

logger = logging.getLogger(__name__)


class ToolPermissionError(PermissionError):
    """Raised by :meth:`ApprovalHooks.on_tool_start` when a tool is denied.

    A dedicated subclass lets callers (e.g. the scheduler) classify and present
    permission denials distinctly from arbitrary ``PermissionError``s raised by
    tool bodies, without corrupting any run state.
    """

    def __init__(self, tool_name: str, reason: str | None):
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Permission denied for {tool_name}: {reason}")


class ApprovalHooks(RunHooks):
    """RunHooks implementation that checks permissions before tool execution."""

    def __init__(
        self,
        wrapped_hooks: RunHooks,
        permission_service: PermissionService | None = None,
    ):
        """Initialize approval hooks.

        Args:
            wrapped_hooks: Optional hooks to wrap (e.g., display hooks)
            permission_service: Optional permission service for tool-level checks.
                When provided, tool calls are evaluated against the active
                permission mode before execution.
        """
        self.wrapped_hooks = wrapped_hooks
        self._permission_service = permission_service
        # Tracks whether the previous agent completion was blocked by a Stop
        # hook. Initialized here (and reset in on_agent_start) so a one-time
        # block cannot permanently wedge the scheduler for its entire lifetime.
        self._stop_hook_active = False

    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        """Called before the agent is invoked."""
        # Reset the Stop-hook flag at the start of every agent run so a block
        # from a prior run does not persist across independent runs.
        self._stop_hook_active = False
        if self.wrapped_hooks:
            await self.wrapped_hooks.on_agent_start(context, agent)

    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        """Called when the agent produces a final output."""
        if self.wrapped_hooks:
            await self.wrapped_hooks.on_agent_end(context, agent, output)
        # Await the off-loop variant so a slow Stop hook cannot freeze the
        # event loop (streaming UI, concurrent subagents, cron).
        result = await dispatch_command_hooks_async(
            cwd=get_execution_cwd(),
            event_name="Stop",
            match_value=None,
            payload={
                "event": "Stop",
                "agent_type": getattr(agent, "name", None),
                "last_assistant_message": str(output),
                "stop_hook_active": self._stop_hook_active,
            },
        )
        if result.blocked:
            # A Stop hook is *meant* to block completion, so we keep raising to
            # halt the turn. We record the block so a re-entrant Stop hook can
            # see stop_hook_active=True; on_agent_start resets it for the next
            # independent run so this cannot permanently wedge the scheduler.
            self._stop_hook_active = True
            raise RuntimeError(result.block_reason or "Blocked by Stop hook")

    async def on_handoff(
        self, context: RunContextWrapper, from_agent: Agent, to_agent: Agent
    ) -> None:
        """Called when a handoff occurs."""
        if self.wrapped_hooks and hasattr(self.wrapped_hooks, "on_handoff"):
            await self.wrapped_hooks.on_handoff(context, from_agent, to_agent)

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        """Check permissions, then forward to wrapped hooks for tool display.

        The openai-agents ``RunHooks.on_tool_start`` contract returns ``None``
        and provides no mechanism to substitute a tool result from inside the
        hook: the runner awaits ``on_tool_start`` (via ``asyncio.gather``)
        *before* invoking the tool, and only the tool body — not the hook — is
        wrapped in the try/except that converts failures into a tool message.
        The only SDK-supported way to turn a denial into a tool result is the
        approval flow (``needs_approval`` / ``ToolApprovalItem``), which is
        driven by the tool/run-config and runs *before* hooks, not by a
        ``RunHooks`` object. Therefore raising is the only supported way to
        stop a denied tool from here; we raise a dedicated, classified
        exception and hold no mutable state on this path, so a denial cannot
        corrupt scheduler state.

        Raises:
            ToolPermissionError: When the permission service denies the tool.
        """
        # on_tool_start receives the Tool but NOT its arguments, so a check here
        # can only be name-level. We split on GUARDED_TOOLS:
        #
        # - Guarded tools (run_shell/run_powershell/git_command/write_file/
        #   edit_file/append_file) are re-evaluated with their FULL arguments in
        #   Phase 2 (enforce_tool_permission in the compat wrapper), which returns
        #   a graceful model-visible denial. We SKIP the name-level evaluate/raise
        #   for them here: an arguments={} evaluation would deny the whole turn
        #   (e.g. in plan mode) and can never match a target-scoped allow rule.
        # - Non-guarded tools (e.g. todo_write/task_delegate) have no Phase-2
        #   path, so this name-level check is their only plan-mode enforcement;
        #   keep the evaluate/raise for them.
        if self._permission_service is not None and tool.name not in GUARDED_TOOLS:
            result = self._permission_service.evaluate_tool_call(
                tool_name=tool.name,
                arguments={},
            )
            if not result.allowed and not result.requires_approval:
                logger.warning("Permission denied for tool %s: %s", tool.name, result.reason)
                raise ToolPermissionError(tool.name, result.reason)
            if result.requires_approval:
                # In non-interactive contexts approval cannot be obtained;
                # log and allow so the existing interactive approval flow
                # (if any) still works.
                logger.info("Tool %s requires approval: %s", tool.name, result.reason)

        if self.wrapped_hooks:
            await self.wrapped_hooks.on_tool_start(context, agent, tool)

    async def on_tool_end(
        self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str
    ) -> None:
        """Called after a tool is invoked."""
        if self.wrapped_hooks:
            await self.wrapped_hooks.on_tool_end(context, agent, tool, result)

        # The SDK's ToolContext (a RunContextWrapper subclass) carries
        # tool_arguments as a raw JSON string.  We parse it to pass structured
        # input to PostToolUse hooks so they can write rules based on what the
        # tool was actually called with.  If the context is not a ToolContext or
        # parsing fails, we fall back to an empty dict gracefully.
        tool_input: dict[str, Any] = {}
        raw_args = getattr(context, "tool_arguments", None)
        if raw_args:
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    tool_input = parsed
            except (json.JSONDecodeError, TypeError):
                pass

        # Await the off-loop variant so a slow PostToolUse hook cannot freeze
        # the event loop (streaming UI, concurrent subagents, cron).
        await dispatch_command_hooks_async(
            cwd=get_execution_cwd(),
            event_name="PostToolUse",
            match_value=tool.name,
            payload={
                "event": "PostToolUse",
                "tool_name": tool.name,
                "tool_input": tool_input,
                "result": str(result),
            },
        )
