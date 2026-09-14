"""Models for runtime-managed agents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AgentRecord:
    """A single runtime-managed agent entry."""

    id: str
    profile: str
    session_id: str
    description: str | None
    prompt: str | None
    output_path: str | None
    worktree_path: str | None
    worktree_branch: str | None
    permission_mode: str | None
    state: str
    error: str | None
    created_at: str
    updated_at: str
    summary: str | None = None
    summary_updated_at: str | None = None
    model_config: dict[str, Any] | None = None
    origin_cwd: str | None = None
    definition_provenance: dict[str, Any] | None = None
    pending_messages: tuple[str, ...] = ()
    inflight_messages: tuple[str, ...] = ()
    runtime_owner: str | None = None
    execution_id: str | None = None

    @classmethod
    def create(
        cls,
        *,
        agent_id: str,
        profile: str,
        session_id: str | None = None,
        description: str | None = None,
        prompt: str | None = None,
        output_path: str | None = None,
        worktree_path: str | None = None,
        worktree_branch: str | None = None,
        permission_mode: str | None = None,
        state: str = "ready",
        error: str | None = None,
        summary: str | None = None,
        summary_updated_at: str | None = None,
        model_config: dict[str, Any] | None = None,
        origin_cwd: str | None = None,
        definition_provenance: dict[str, Any] | None = None,
    ) -> "AgentRecord":
        timestamp = _utc_now_iso()
        return cls(
            id=agent_id,
            profile=profile,
            session_id=session_id or f"subagent-{agent_id}",
            description=description,
            prompt=prompt,
            output_path=output_path,
            worktree_path=worktree_path,
            worktree_branch=worktree_branch,
            permission_mode=permission_mode,
            state=state,
            error=error,
            created_at=timestamp,
            updated_at=timestamp,
            summary=summary,
            summary_updated_at=summary_updated_at or (timestamp if summary else None),
            model_config=model_config,
            origin_cwd=origin_cwd,
            definition_provenance=definition_provenance,
        )


@dataclass(frozen=True)
class DelayedWorkerResult:
    """Result of marking an agent as delayed."""

    agent_id: str
    state_preserved: bool
