"""Models for runtime-managed teams."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TeamRecord:
    """A runtime-managed team entry."""

    id: str
    name: str
    description: str | None
    lead_agent_id: str
    lead_session_id: str | None
    config_path: str
    created_at: str
    generation: str = ""

    @classmethod
    def create(
        cls,
        *,
        team_id: str,
        name: str,
        description: str | None = None,
        lead_agent_id: str,
        lead_session_id: str | None,
        config_path: str,
    ) -> "TeamRecord":
        return cls(
            id=team_id,
            name=name,
            description=description,
            lead_agent_id=lead_agent_id,
            lead_session_id=lead_session_id,
            config_path=config_path,
            created_at=_utc_now_iso(),
            generation=uuid4().hex,
        )


@dataclass(frozen=True)
class TeamMemberRecord:
    """A persisted team member entry."""

    agent_id: str
    name: str
    agent_type: str | None
    model: str | None
    prompt: str | None
    color: str | None
    plan_mode_required: bool
    cwd: str
    worktree_path: str | None
    session_id: str | None
    mode: str | None
    is_active: bool
    joined_at: str
    generation: str = ""

    @classmethod
    def create(
        cls,
        *,
        agent_id: str,
        name: str,
        cwd: str,
        agent_type: str | None = None,
        model: str | None = None,
        prompt: str | None = None,
        color: str | None = None,
        plan_mode_required: bool = False,
        worktree_path: str | None = None,
        session_id: str | None = None,
        mode: str | None = None,
        is_active: bool = True,
    ) -> "TeamMemberRecord":
        return cls(
            agent_id=agent_id,
            name=name,
            agent_type=agent_type,
            model=model,
            prompt=prompt,
            color=color,
            plan_mode_required=plan_mode_required,
            cwd=cwd,
            worktree_path=worktree_path,
            session_id=session_id,
            mode=mode,
            is_active=is_active,
            joined_at=_utc_now_iso(),
            generation=uuid4().hex,
        )


@dataclass(frozen=True)
class TeamMailboxMessage:
    """A persisted mailbox entry for team-scoped communication."""

    agent_id: str
    content: str
    created_at: str
    sender: str
    recipient: str
    read: bool


@dataclass(frozen=True)
class TeamHistoryEntry:
    """A chronological event in a team's discussion history."""

    event: str
    created_at: str
    sender: str | None
    recipient: str | None
    content: str | None
    read: bool | None = None
    agent_id: str | None = None
    member_name: str | None = None
    state: str | None = None
    source: str | None = None
