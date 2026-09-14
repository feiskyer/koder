"""Runtime hook support for subagent lifecycle and tool events."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents import Agent, RunContextWrapper, RunHooks, Tool

from ..hooks.command_process import HookCommandCancelledError, run_command
from ..hooks.runtime import (
    _bounded_timeout,
    _cap_output,
    _extract_block,
    dispatch_command_hooks,
    dispatch_command_hooks_async,
)
from .definitions import AgentDefinition

if TYPE_CHECKING:
    from ..permissions.service import PermissionService

logger = logging.getLogger(__name__)


def _matching_rules(
    hook_config: dict[str, Any], event_name: str, match_value: str | None
) -> list[dict]:
    rules = hook_config.get(event_name) or []
    if not isinstance(rules, list):
        return []
    matched: list[dict] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        matcher = rule.get("matcher")
        if matcher is None:
            matched.append(rule)
            continue
        if match_value is None:
            continue
        try:
            if re.search(str(matcher), match_value):
                matched.append(rule)
        except re.error:
            if str(matcher) == match_value:
                matched.append(rule)
    return matched


def _run_command_hooks(
    rules: list[dict],
    payload: dict[str, Any],
    cwd: str | Path,
    *,
    cancel_event: threading.Event | None = None,
) -> None:
    """Run definition-local commands in order, stopping on cancellation or denial."""
    payload_text = json.dumps(payload)
    for rule in rules:
        hooks = rule.get("hooks") or []
        if not isinstance(hooks, list):
            continue
        for hook in hooks:
            if cancel_event is not None and cancel_event.is_set():
                raise HookCommandCancelledError()
            if not isinstance(hook, dict) or hook.get("type") != "command":
                continue
            command = hook.get("command")
            if not isinstance(command, str) or not command.strip():
                continue
            try:
                result = run_command(
                    command,
                    input=payload_text,
                    cwd=str(cwd),
                    shell=True,
                    env=os.environ.copy(),
                    timeout=_bounded_timeout(hook.get("timeout")),
                    cancel_event=cancel_event,
                )
            except subprocess.TimeoutExpired as exc:
                raise PermissionError("Subagent frontmatter hook timed out (fail-closed)") from exc
            block_reason = _extract_block(result.stdout)
            if result.returncode == 2 or block_reason:
                if result.returncode == 2:
                    block_reason = result.stderr.strip() or block_reason or "Blocked by hook"
                raise PermissionError(_cap_output(block_reason))


async def _run_command_hooks_async(
    rules: list[dict], payload: dict[str, Any], cwd: str | Path
) -> None:
    """Bridge task cancellation to the canonical command runner's stop signal.

    Unlike settings dispatch this worker only runs cancellable local commands,
    so join its cleanup before propagating cancellation. No HTTP/model worker
    or project/skill scope is introduced for definition-local hooks.
    """
    if not rules:
        return
    cancellation = threading.Event()
    task = asyncio.create_task(
        asyncio.to_thread(_run_command_hooks, rules, payload, cwd, cancel_event=cancellation)
    )
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        cancellation.set()
        # A second cancel must not abandon the thread or its running process.
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break  # Retrieve the worker outcome below; caller cancellation wins.
        if not task.cancelled():
            task.exception()
        raise


def dispatch_project_hook_event(
    *,
    cwd: str | Path,
    event_name: str,
    match_value: str | None,
    payload: dict[str, Any],
) -> Any:
    """Dispatch a non-subagent project hook event through the shared hook runner."""

    return dispatch_command_hooks(
        cwd=cwd,
        event_name=event_name,
        match_value=match_value,
        payload=payload,
    )


class SubagentLifecycleHooks(RunHooks):
    """Lifecycle hooks for subagent runs."""

    def __init__(
        self,
        *,
        agent_definition: AgentDefinition,
        cwd: str | Path,
        wrapped_hooks: RunHooks | None = None,
        permission_service: "PermissionService | None" = None,
    ):
        self.agent_definition = agent_definition
        self.cwd = Path(cwd)
        self.wrapped_hooks = wrapped_hooks
        self._permission_service = permission_service
        self.frontmatter_hooks = agent_definition.hooks or {}

    async def on_agent_start(self, context: RunContextWrapper, agent: Agent) -> None:
        if self.wrapped_hooks:
            await self.wrapped_hooks.on_agent_start(context, agent)
        payload = {"event": "SubagentStart", "agent_type": self.agent_definition.agent_type}
        # Full hook runner: picks up project, local, user, and managed settings
        # (not just the project scope the old mini-runner read).
        result = await dispatch_command_hooks_async(
            cwd=self.cwd,
            event_name="SubagentStart",
            match_value=self.agent_definition.agent_type,
            payload=payload,
        )
        if result.blocked:
            raise PermissionError(result.block_reason or "Blocked by SubagentStart hook")

    async def on_agent_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        if self.wrapped_hooks:
            await self.wrapped_hooks.on_agent_end(context, agent, output)
        payload = {
            "event": "SubagentStop",
            "agent_type": self.agent_definition.agent_type,
            "output": str(output),
            "stop_hook_active": getattr(self, "_stop_hook_active", False),
        }
        result = await dispatch_command_hooks_async(
            cwd=self.cwd,
            event_name="SubagentStop",
            match_value=self.agent_definition.agent_type,
            payload=payload,
        )
        if result.blocked:
            raise PermissionError(result.block_reason or "Blocked by SubagentStop hook")
        # Agent frontmatter "Stop" rules remain a per-definition concern and
        # keep using the local mini-runner (they are not settings-backed).
        stop_rules = _matching_rules(
            self.frontmatter_hooks, "Stop", self.agent_definition.agent_type
        )
        await _run_command_hooks_async(stop_rules, payload, self.cwd)

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        from ...tools.permission_context import GUARDED_TOOLS

        # Permission check before forwarding. Guarded tools are re-evaluated with
        # their full arguments in Phase 2 (enforce_tool_permission in the compat
        # wrapper); a name-level arguments={} check here would deny the whole turn
        # and can never match a target-scoped allow rule, so skip them. Non-guarded
        # tools have no Phase-2 path, so keep the name-level evaluate/raise.
        if self._permission_service is not None and tool.name not in GUARDED_TOOLS:
            result = self._permission_service.evaluate_tool_call(
                tool_name=tool.name,
                arguments={},
            )
            # These tools have no argument-level approval path and subagents
            # cannot prompt: pending approval is not permission to proceed.
            if not result.allowed or result.requires_approval:
                logger.warning("Permission denied for tool %s: %s", tool.name, result.reason)
                raise PermissionError(f"Permission denied for {tool.name}: {result.reason}")

        if self.wrapped_hooks:
            await self.wrapped_hooks.on_tool_start(context, agent, tool)
        payload = {
            "event": "PreToolUse",
            "agent_type": self.agent_definition.agent_type,
            "tool_name": tool.name,
        }
        rules = _matching_rules(self.frontmatter_hooks, "PreToolUse", tool.name)
        await _run_command_hooks_async(rules, payload, self.cwd)

    async def on_tool_end(
        self,
        context: RunContextWrapper,
        agent: Agent,
        tool: Tool,
        result: str,
    ) -> None:
        if self.wrapped_hooks:
            await self.wrapped_hooks.on_tool_end(context, agent, tool, result)
        payload = {
            "event": "PostToolUse",
            "agent_type": self.agent_definition.agent_type,
            "tool_name": tool.name,
            "result": str(result),
        }
        rules = _matching_rules(self.frontmatter_hooks, "PostToolUse", tool.name)
        await _run_command_hooks_async(rules, payload, self.cwd)
