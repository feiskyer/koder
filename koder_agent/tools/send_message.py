"""SendMessage tool for inter-agent communication."""

from __future__ import annotations

import json
from typing import Optional

from pydantic import BaseModel

from koder_agent.harness.agents.teams.context import TeamToolContext, get_team_tool_context
from koder_agent.harness.agents.teams.runtime import TEAM_LEAD_NAME
from koder_agent.harness.agents.teams.tool_runtime import get_effective_team_context

from .compat import function_tool


class SendMessageModel(BaseModel):
    to: str
    message: str
    summary: Optional[str] = None


def _resolve_team_recipient(context: TeamToolContext, to: str) -> str | None:
    if to == TEAM_LEAD_NAME:
        return TEAM_LEAD_NAME
    for member in context.team_service.member_records(context.team_id):
        if to == member.name:
            return member.name
        if to == member.agent_id:
            return member.agent_id
    return None


def _broadcast_recipients(context: TeamToolContext) -> list[str]:
    recipients: list[str] = []
    for member in context.team_service.member_records(context.team_id):
        if not member.is_active:
            continue
        if context.sender_name != TEAM_LEAD_NAME and member.agent_id == context.sender_agent_id:
            continue
        recipients.append(member.name)
    return recipients


async def _send_team_message(context: TeamToolContext, to: str, message: str) -> str:
    try:
        if to == "*":
            recipients = _broadcast_recipients(context)
            for recipient in recipients:
                context.team_service.route(
                    context.team_id,
                    message,
                    recipient=recipient,
                    sender=context.sender_name,
                )
            return json.dumps(
                {
                    "status": "sent",
                    "routing": "team_mailbox",
                    "broadcast": True,
                    "team": context.team_id,
                    "sender": context.sender_name,
                    "recipients": recipients,
                }
            )

        recipient = _resolve_team_recipient(context, to)
        if recipient is None:
            return json.dumps(
                {
                    "status": "error",
                    "routing": "team_mailbox",
                    "team": context.team_id,
                    "sender": context.sender_name,
                    "error": f"Unknown team recipient: {to}.",
                }
            )

        context.team_service.route(
            context.team_id,
            message,
            recipient=recipient,
            sender=context.sender_name,
        )
        return json.dumps(
            {
                "status": "sent",
                "routing": "team_mailbox",
                "recipient": recipient,
                "team": context.team_id,
                "sender": context.sender_name,
            }
        )
    except Exception as exc:
        return json.dumps(
            {
                "status": "error",
                "routing": "team_mailbox",
                "team": context.team_id,
                "sender": context.sender_name,
                "error": f"Team routing failed: {exc}",
            }
        )


async def _send_message_impl(
    to: str,
    message: str,
    summary: str | None = None,
    *,
    _agent_service=None,
    _team_service=None,
    _team_name: str | None = None,
) -> str:
    """Core implementation for send_message tool.

    Routes messages through either TeamService (when team context is provided)
    or AgentService mailboxes. Supports broadcast via ``to="*"``.
    """

    context = get_team_tool_context()
    if _agent_service is None and _team_service is None:
        context = get_effective_team_context()
    if context is None and _team_service is not None and _team_name is not None:
        context = TeamToolContext(
            team_id=_team_name,
            sender_name=TEAM_LEAD_NAME,
            sender_agent_id=TEAM_LEAD_NAME,
            team_service=_team_service,
        )
    if context is not None:
        return await _send_team_message(context, to, message)

    # Fall back to agent service routing
    if _agent_service is None:
        from koder_agent.harness.agents.runtime_context import get_runtime_agent_service

        _agent_service = get_runtime_agent_service()

    # Resolve recipient
    agent_id = _agent_service.resolve_agent_id(to)
    if agent_id is None:
        return json.dumps(
            {
                "status": "error",
                "error": f"Unknown recipient: {to}. Agent not found by name or ID.",
            }
        )

    try:
        _agent_service.send(agent_id, message)
    except (KeyError, ValueError, RuntimeError, OSError) as exc:
        return json.dumps({"status": "error", "error": str(exc)})

    # Detect stopped agents so callers know a resume/re-spawn may be needed
    try:
        record = _agent_service.get(agent_id)
        agent_state = record.state
    except (KeyError, Exception):
        agent_state = "unknown"

    response = {
        "status": "sent",
        "routing": "agent_mailbox",
        "recipient": to,
        "agent_id": agent_id,
        "delivery": "queued",
    }
    if agent_state in {"completed", "failed", "cancelled", "delayed", "ready"}:
        response["agent_stopped"] = True
        response["note"] = (
            "Message is queued. Use agent_tool(resume=agent_id) to continue this agent."
        )
    return json.dumps(response)


@function_tool
async def send_message(
    to: str,
    message: str,
    summary: str | None = None,
) -> str:
    """Send a message to another agent or teammate.

    Args:
        to: Recipient name or agent_id. Use '*' for broadcast to all teammates.
        message: Plain text message or JSON structured message
            (shutdown_request, shutdown_response, plan_approval_response).
        summary: A 5-10 word summary shown in UI (required for plain text messages).
    """
    return await _send_message_impl(to=to, message=message, summary=summary)
