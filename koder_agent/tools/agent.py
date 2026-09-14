"""Agent tool for spawning sub-agents programmatically.

Context inheritance: When spawning agents, use the AgentService's seed_items
parameter to pass parent conversation history. The ForkContext utility in
tools/fork_agent.py can build filtered message lists from the parent session
for prompt cache sharing and context continuity.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import replace as _replace
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel

from koder_agent.harness.agents.runtime_context import (
    get_runtime_agent_service,
    get_runtime_agent_session,
)
from koder_agent.harness.agents.teams.tool_runtime import get_effective_team_context
from koder_agent.harness.execution_context import get_execution_cwd
from koder_agent.utils.async_tasks import await_owned_task

from .compat import function_tool

logger = logging.getLogger(__name__)

# Fork context available via tools/fork_agent.py build_fork_context()
# Pass result via AgentService seed_items parameter for context inheritance


class AgentToolModel(BaseModel):
    description: str
    prompt: str
    subagent_type: Optional[str] = None
    model: Optional[str] = None
    run_in_background: Optional[bool] = None
    name: Optional[str] = None
    team_name: Optional[str] = None
    mode: Optional[str] = None
    isolation: Optional[str] = None
    context: Optional[str] = None
    resume: Optional[str] = None


async def _agent_tool_impl(
    description: str,
    prompt: str,
    subagent_type: str | None = None,
    model: str | None = None,
    run_in_background: bool | None = None,
    name: str | None = None,
    team_name: str | None = None,
    mode: str | None = None,
    isolation: str | None = None,
    context: str | None = None,
    resume: str | None = None,
) -> str:
    """Core implementation for the agent tool."""
    from koder_agent.harness.agents.definitions import (
        get_agent_definitions,
        resolve_agent_model,
    )

    cwd = get_execution_cwd()
    if team_name is not None and not team_name.strip():
        return json.dumps({"status": "error", "error": "Invalid team name"})
    service = None
    resumed_record = None
    if resume:
        service = get_runtime_agent_service()
        agent_id = service.resolve_agent_id(resume)
        if agent_id is None:
            return json.dumps({"status": "error", "error": f"Unknown agent to resume: {resume}"})
        resumed_record = service.get(agent_id)
    definitions = get_agent_definitions(cwd=cwd)

    # Resolve agent type
    effective_type = subagent_type or (
        resumed_record.profile if resumed_record else "general-purpose"
    )
    selected = next(
        (a for a in definitions.active_agents if a.agent_type == effective_type),
        None,
    )
    if selected is None:
        available = [a.agent_type for a in definitions.active_agents]
        return json.dumps(
            {
                "status": "error",
                "error": f"Unknown agent type: {effective_type}",
                "available_agents": available,
            }
        )

    # Apply model override
    if model:
        model_override = model
    elif resumed_record and (resumed_record.model_config or {}).get("model_override") not in (
        None,
        "",
        "inherit",
    ):
        model_override = resumed_record.model_config["model_override"]
    else:
        model_override = resolve_agent_model(selected)

    # Apply isolation from parameter or definition
    effective_isolation = isolation or (
        "worktree" if resumed_record and resumed_record.worktree_branch else selected.isolation
    )

    # Build agent definition with overrides
    agent_def = selected
    if model_override and model_override != selected.model:
        agent_def = _replace(agent_def, model=model_override)
    if effective_isolation and effective_isolation != selected.isolation:
        agent_def = _replace(agent_def, isolation=effective_isolation)

    service = service or get_runtime_agent_service()
    if name and (not name.strip() or name == "*"):
        return json.dumps({"status": "error", "error": "Invalid agent name"})
    if (
        name
        and service.name_is_registered(name)
        and (resumed_record is None or service.resolve_agent_id(name) != resumed_record.id)
    ):
        return json.dumps(
            {"status": "error", "error": f"Agent name already in use: {name}; resume explicitly"}
        )
    if resumed_record is not None:
        if context:
            return json.dumps({"status": "error", "error": "Resume uses its existing history"})
        if mode is not None and mode != resumed_record.permission_mode:
            return json.dumps(
                {"status": "error", "error": "Resume preserves the agent permission mode"}
            )
        team_runtime = service.team_tool_runtime
        if team_name is not None or (
            team_runtime is not None and team_runtime.manages(resumed_record.id)
        ):
            try:
                from koder_agent.harness.agents.service import agent_definition_matches_record

                if not agent_definition_matches_record(resumed_record, agent_def):
                    raise ValueError("background agent definition provenance does not match")
                team_runtime = service.get_team_tool_runtime()
                team_context = (
                    team_runtime.select(team_name, get_runtime_agent_session())
                    if team_name is not None
                    else get_effective_team_context()
                )
                if team_context is None or not team_runtime.manages(resumed_record.id):
                    raise ValueError("Resume requires a live teammate in the current team")
                members = team_context.team_service.member_records(team_context.team_id)
                member = next(
                    (member for member in members if member.agent_id == resumed_record.id), None
                )
                if member is None:
                    raise ValueError("Agent is not a member of the selected team")
                if name is not None and name != member.name:
                    raise ValueError("Resume preserves the teammate name")
                team_context.team_service.route(
                    team_context.team_id,
                    prompt,
                    recipient=resumed_record.id,
                    sender=team_context.sender_name,
                )
            except (KeyError, ValueError, RuntimeError, OSError) as exc:
                return json.dumps({"status": "error", "error": str(exc)})
            return json.dumps(
                {
                    "status": "async_resumed",
                    "agent_id": resumed_record.id,
                    "agent_type": agent_def.agent_type,
                    "output_file": resumed_record.output_path,
                    "team_id": team_context.team_id,
                    "delivery": "queued",
                }
            )
        record = await service.resume_background(
            agent_id=resumed_record.id,
            agent_definition=agent_def,
            prompt=prompt,
            cwd=cwd,
        )
        if name:
            service.register_name(name, record.id)
        return json.dumps(
            {
                "status": "async_resumed",
                "agent_id": record.id,
                "agent_type": agent_def.agent_type,
                "output_file": record.output_path,
            }
        )

    # Build fork context if requested
    seed_items = None
    if context == "fork":
        from koder_agent.tools.fork_agent import build_fork_context

        parent_session = get_runtime_agent_session()
        if parent_session is None:
            return json.dumps(
                {"status": "error", "error": "Fork requires an active parent session"}
            )
        try:
            parent_messages = await parent_session.get_items()
            if get_runtime_agent_session() is not parent_session:
                return json.dumps(
                    {"status": "error", "error": "Parent session ended before the fork completed"}
                )
            seed_items = build_fork_context(parent_messages).to_messages()
        except Exception as exc:
            logger.debug("Failed to read fork parent (%s)", type(exc).__name__)
            return json.dumps(
                {"status": "error", "error": "Unable to read parent conversation for fork"}
            )

    # Check if background tasks are disabled via env var
    bg_disabled = os.environ.get("KODER_DISABLE_BACKGROUND_TASKS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    # background=true in agent definition forces async execution
    should_run_async = (run_in_background or (agent_def.background is True)) and not bg_disabled

    if team_name is not None:
        if bg_disabled:
            return json.dumps(
                {"status": "error", "error": "Team members require background tasks to be enabled"}
            )
        try:
            runtime = service.get_team_tool_runtime()
            team_context = runtime.select(team_name, get_runtime_agent_session())
            spawned = await runtime.runner.spawn_teammate(
                team_id=team_context.team_id,
                name=name or f"teammate-{uuid4().hex[:8]}",
                agent_definition=agent_def,
                prompt=prompt,
                cwd=cwd,
                plan_mode_required=mode == "plan",
                permission_mode=mode,
                seed_items=seed_items,
                parent_context=team_context,
            )
            record = service.get(spawned.agent_id)
            response = {
                "status": "async_launched",
                "agent_id": record.id,
                "agent_type": agent_def.agent_type,
                "description": description,
                "output_file": record.output_path,
                "team_id": spawned.team_id,
                "name": spawned.name,
            }
            if not should_run_async:
                try:
                    await runtime.runner.wait(record.id)
                except BaseException:
                    await await_owned_task(asyncio.create_task(runtime.runner.terminate(record.id)))
                    raise
                record = service.get(record.id)
                response["status"] = "completed" if record.state == "completed" else "error"
                response["result"] = service._read_output(record) or ""
                if record.error:
                    response["error"] = record.error
            return json.dumps(response)
        except (KeyError, ValueError, RuntimeError, OSError) as exc:
            return json.dumps({"status": "error", "error": str(exc)})

    # Async (background) mode
    if should_run_async:
        record = await service.launch_background(
            agent_definition=agent_def,
            prompt=prompt,
            description=description,
            seed_items=seed_items,
            cwd=cwd,
            permission_mode=mode,
        )
        # Register name for SendMessage routing
        if name:
            service.register_name(name, record.id)
        return json.dumps(
            {
                "status": "async_launched",
                "agent_id": record.id,
                "agent_type": agent_def.agent_type,
                "description": description,
                "output_file": record.output_path,
            }
        )

    # Sync (blocking) mode
    result = await service.run_sync(
        agent_definition=agent_def,
        prompt=prompt,
        seed_items=seed_items,
        cwd=cwd,
        display_label=description,
        permission_mode=mode,
    )
    return json.dumps(
        {
            "status": "completed",
            "agent_type": agent_def.agent_type,
            "result": result,
        }
    )


@function_tool
async def agent_tool(
    description: str,
    prompt: str,
    subagent_type: str | None = None,
    model: str | None = None,
    run_in_background: bool | None = None,
    name: str | None = None,
    team_name: str | None = None,
    mode: str | None = None,
    isolation: str | None = None,
    context: str | None = None,
    resume: str | None = None,
) -> str:
    """Launch a new agent to handle complex, multi-step tasks autonomously.

    Args:
        description: A short (3-5 word) description of the task.
        prompt: The task for the agent to perform.
        subagent_type: The type of specialized agent to use (e.g. 'Explore', 'Plan').
        model: Optional model override ('sonnet', 'opus', 'haiku').
        run_in_background: Set to true to run this agent in the background.
        name: Name for the spawned agent. Makes it addressable via send_message.
        team_name: Team name for spawning within a team context.
        mode: Permission mode for spawned teammate (e.g. 'plan').
        isolation: Isolation mode ('worktree' for isolated git worktree).
        context: Context inheritance mode ('fork' to share parent conversation history).
        resume: Existing agent ID or registered name to continue asynchronously.
    """
    return await _agent_tool_impl(
        description=description,
        prompt=prompt,
        subagent_type=subagent_type,
        model=model,
        run_in_background=run_in_background,
        name=name,
        team_name=team_name,
        mode=mode,
        isolation=isolation,
        context=context,
        resume=resume,
    )
