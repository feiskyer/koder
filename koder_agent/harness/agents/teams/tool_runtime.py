"""Owner-managed team state shared by separate SDK tool-call tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from koder_agent.harness.execution_context import get_execution_cwd

from .context import TeamToolContext, get_team_tool_context
from .runtime import TEAM_LEAD_NAME
from .service import TeamService

if TYPE_CHECKING:
    from ..service import AgentService
    from .in_process import InProcessTeammateRunner


class TeamToolRuntime:
    """Retain a leader's selection without mutating a child task's ContextVar."""

    def __init__(self, agent_service: AgentService):
        self.agent_service = agent_service
        root = agent_service.output_root.parent
        self.team_service = TeamService(
            teams_root=root / "teams",
            tasks_root=root / "tasks",
            cwd=get_execution_cwd(),
        )
        self._runner: InProcessTeammateRunner | None = None
        # Retain the owner while its selection exists, so object IDs cannot be
        # recycled into another session's context. Delete/close releases it.
        self._leaders: dict[int, tuple[object | None, TeamToolContext]] = {}

    def configure(self, *, team_service=None, permission_bridge=None, local_prompt_executor=None):
        """Reuse the same runner for interactive commands and model tools."""
        if team_service is not None and team_service is not self.team_service:
            if self._runner is not None or self._leaders:
                raise RuntimeError("Cannot replace an initialized team runtime")
            self.team_service = team_service
        if self._runner is None:
            from .in_process import InProcessTeammateRunner

            self._runner = InProcessTeammateRunner(
                agent_service=self.agent_service,
                team_service=self.team_service,
                permission_bridge=permission_bridge,
                local_prompt_executor=local_prompt_executor,
            )
        return self._runner

    @property
    def runner(self) -> InProcessTeammateRunner:
        return self.configure()

    @property
    def has_live_members(self) -> bool:
        return self._runner is not None and self._runner.has_live_members

    def manages(self, agent_id: str) -> bool:
        return self._runner is not None and self._runner.manages(agent_id)

    def current(self, session: object | None) -> TeamToolContext | None:
        entry = self._leaders.get(id(session))
        return entry[1] if entry is not None and entry[0] is session else None

    def select(self, team_name: str, session: object | None) -> TeamToolContext:
        """An explicit team name selects a real, generation-fenced team."""
        team = self.team_service.get(self.team_service._team_id(team_name))
        if team_name not in {team.name, team.id}:
            raise ValueError(f"Unknown team: {team_name}")
        member_context = get_team_tool_context()
        if member_context is not None:
            if member_context.team_id != team.id:
                raise ValueError("A teammate cannot switch to another team")
            member_context.team_service.get(team.id)
            return member_context
        context = TeamToolContext(
            team_id=team.id,
            sender_name=TEAM_LEAD_NAME,
            sender_agent_id=team.lead_agent_id,
            team_service=self.team_service.bind_team(team.id),
        )
        self._leaders[id(session)] = (session, context)
        return context

    def create(self, name: str, description: str | None, session: object | None):
        if get_team_tool_context() is not None:
            raise ValueError("Only a team leader can create a team")
        team_id = self.team_service.create_team(
            name,
            description=description,
            lead_session_id=getattr(session, "session_id", None),
        )
        self.select(name, session)
        return self.team_service.get(team_id)

    def clear(self, session: object | None, team_id: str) -> None:
        context = self.current(session)
        if context is not None and context.team_id == team_id:
            self._leaders.pop(id(session), None)

    async def aclose(self) -> None:
        if self._runner is not None:
            await self._runner.aclose()
        self._leaders.clear()

    def close(self) -> None:
        if self.has_live_members:
            raise RuntimeError("Live teammates require await service.aclose()")
        self._leaders.clear()


def get_effective_team_context() -> TeamToolContext | None:
    """Prefer the running member's identity over its leader's saved selection."""
    from ..runtime_context import (
        get_bound_agent_service,
        get_runtime_agent_service,
        get_runtime_agent_session,
    )

    context = get_team_tool_context()
    if context is not None:
        owner = get_bound_agent_service()
        return None if owner is not None and owner.is_closed else context

    owner = get_runtime_agent_service()
    if owner.is_closed:
        return None
    runtime = owner.team_tool_runtime
    return runtime.current(get_runtime_agent_session()) if runtime is not None else None
