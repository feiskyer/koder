"""In-process teammate runner -- executes teammates as asyncio tasks within the same process."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from koder_agent.harness.agents.definitions import AgentDefinition, resolve_agent_model
from koder_agent.harness.agents.service import AgentService
from koder_agent.harness.agents.teams.context import TeamToolContext
from koder_agent.harness.agents.teams.permission_bridge import PermissionBridge
from koder_agent.harness.agents.teams.service import TeamService
from koder_agent.harness.agents.teams.task_service import TERMINAL_STATES, TeamTaskRecord

logger = logging.getLogger(__name__)
POLL_INTERVAL_SECONDS = 0.1
TEAM_LEAD_NAME = "team-lead"
LocalPromptExecutor = Callable[[str, TeamToolContext | None], Awaitable[str]]


@dataclass
class TeammateSpawnResult:
    """Result of spawning an in-process teammate."""

    agent_id: str
    name: str
    team_id: str


@dataclass
class _PendingTeammateWork:
    prompt: str
    source: str
    task: TeamTaskRecord | None = None


@dataclass
class _TeammateRuntime:
    team_id: str
    agent_id: str
    name: str
    agent_definition: AgentDefinition
    current_run: asyncio.Task | None
    stop_event: asyncio.Event
    idle_event: asyncio.Event
    team_service: TeamService


class InProcessTeammateRunner:
    """Runs teammates as asyncio tasks within the current process.

    This is the 'in-process' backend for agent teams. Teammates share the same
    event loop and process as the leader, communicating via service-layer
    mailboxes and the shared task list.
    """

    def __init__(
        self,
        *,
        agent_service: AgentService,
        team_service: TeamService,
        permission_bridge: PermissionBridge | None = None,
        local_prompt_executor: LocalPromptExecutor | None = None,
    ):
        self._agent_service = agent_service
        self._team_service = team_service
        self._permission_bridge = permission_bridge
        self._local_prompt_executor = local_prompt_executor
        self._tasks: dict[str, asyncio.Task] = {}
        self._runtimes: dict[str, _TeammateRuntime] = {}
        self._closed = False

    def _local_executor_for(self, prompt: str):
        if self._local_prompt_executor is None or not prompt.lstrip().startswith("/"):
            return None

        async def execute_local(**kwargs: Any) -> str:
            return await self._local_prompt_executor(
                kwargs["prompt"],
                kwargs.get("team_context"),
            )

        return execute_local

    async def spawn_teammate(
        self,
        *,
        team_id: str,
        name: str,
        agent_definition: AgentDefinition,
        prompt: str,
        cwd: str | Path,
        plan_mode_required: bool = False,
        model: str | None = None,
        seed_items: list[dict[str, Any]] | None = None,
        permission_mode: str | None = None,
        parent_context: TeamToolContext | None = None,
    ) -> TeammateSpawnResult:
        """Spawn a teammate as an in-process background task.

        Registers the teammate as a team member, launches the agent
        via AgentService, and tracks the asyncio task for lifecycle mgmt.
        """
        if self._closed:
            raise RuntimeError("Teammate runner is closed")
        admission = parent_context.team_service if parent_context else self._team_service
        admission.validate_member_name(team_id, name)
        generation = admission.get(team_id).generation
        permission_mode = (
            "plan"
            if plan_mode_required
            else (permission_mode or agent_definition.permission_mode or "default")
        )

        effective_model = model or resolve_agent_model(agent_definition) or agent_definition.model

        member_service = None

        def register_member(record):
            nonlocal member_service
            if self._closed:
                raise RuntimeError("Teammate runner is closed")
            # AgentService calls this builder before scheduling the executor.
            with admission.lifecycle(team_id, generation):
                self._team_service.add_member(
                    team_id,
                    record.id,
                    name=name,
                    agent_type=agent_definition.agent_type,
                    model=effective_model,
                    prompt=prompt,
                    plan_mode_required=plan_mode_required,
                    cwd=str(cwd),
                    worktree_path=record.worktree_path,
                    session_id=record.session_id,
                    mode=permission_mode,
                    is_active=True,
                )
                member_service = self._team_service.bind_member(team_id, record.id)
            return TeamToolContext(
                team_id=team_id,
                sender_name=name,
                sender_agent_id=record.id,
                team_service=member_service,
                source="spawn",
            )

        record = await self._agent_service.launch_background(
            agent_definition=agent_definition,
            prompt=prompt,
            description=f"Teammate: {name}",
            cwd=cwd,
            permission_mode=permission_mode,
            seed_items=seed_items,
            team_context_builder=register_member,
            executor=self._guarded_executor(prompt),
        )
        assert member_service is not None

        # Register name for SendMessage routing
        self._agent_service.register_name(name, record.id)

        runtime = _TeammateRuntime(
            team_id=team_id,
            agent_id=record.id,
            name=name,
            agent_definition=agent_definition,
            current_run=self._agent_service._tasks.get(record.id),
            stop_event=asyncio.Event(),
            idle_event=asyncio.Event(),
            team_service=member_service,
        )
        if runtime.current_run is None:
            runtime.idle_event.set()
        self._runtimes[record.id] = runtime
        loop_task = asyncio.create_task(self._run_teammate_loop(runtime))
        self._tasks[record.id] = loop_task
        loop_task.add_done_callback(lambda finished: self._observe_consumer(runtime, finished))

        return TeammateSpawnResult(
            agent_id=record.id,
            name=name,
            team_id=team_id,
        )

    @staticmethod
    def _observe_consumer(runtime: _TeammateRuntime, task: asyncio.Task) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("Teammate consumer %s stopped (%s)", runtime.name, type(error).__name__)

    def _guarded_executor(self, prompt: str, task: TeamTaskRecord | None = None):
        """Revalidate at actual execution entry, after any async launch delay."""
        local = self._local_executor_for(prompt)

        async def execute(**kwargs):
            context = kwargs["team_context"]
            with context.team_service.lifecycle(context.team_id):
                if task is not None:
                    current = context.team_service.task_service(context.team_id).get_task(task.id)
                    if current is None or (current.status, current.owner, current.claim_id) != (
                        "in_progress",
                        task.owner,
                        task.claim_id,
                    ):
                        raise asyncio.CancelledError
            if local is not None:
                return await local(**kwargs)
            from koder_agent.harness.agents import service as agent_service_module

            return await agent_service_module._execute_agent_run(**kwargs)

        return execute

    def _notify_teammate_idle(self, runtime: _TeammateRuntime) -> None:
        """Dispatch idle hooks and notify the lead that the teammate is available again."""
        team_id, agent_id, name = runtime.team_id, runtime.agent_id, runtime.name
        try:
            runtime.team_service.notify_member_idle(team_id, agent_id)
        except Exception:
            logger.debug("Failed to notify idle transition for %s", agent_id, exc_info=True)

        try:
            runtime.team_service.route(
                team_id,
                f"Teammate '{name}' has finished and is now idle.",
                recipient=TEAM_LEAD_NAME,
                sender=name,
            )
        except Exception:
            logger.debug("Failed to send idle notification for %s", agent_id, exc_info=True)

    def _sync_permission_mode(self, runtime: _TeammateRuntime) -> None:
        members = runtime.team_service.member_records(runtime.team_id)
        member = next((item for item in members if item.agent_id == runtime.agent_id), None)
        desired_mode = member.mode if member is not None else None
        if desired_mode is None:
            desired_mode = runtime.agent_definition.permission_mode or "default"
        record = self._agent_service.get(runtime.agent_id)
        if record.permission_mode != desired_mode:
            self._agent_service.update_permission_mode(runtime.agent_id, desired_mode)

    def _claim_next_task(self, runtime: _TeammateRuntime) -> TeamTaskRecord | None:
        task_service = runtime.team_service.task_service(runtime.team_id)
        for task in task_service.list_tasks():
            if task.status in TERMINAL_STATES:
                continue
            claimed = task_service.claim_task(
                task.id,
                runtime.agent_id,
                check_agent_busy=True,
            )
            if claimed.success and claimed.task is not None:
                return claimed.task
        return None

    async def _wait_for_next_work(self, runtime: _TeammateRuntime) -> _PendingTeammateWork | None:
        while not runtime.stop_event.is_set():
            try:
                members = runtime.team_service.member_records(runtime.team_id)
            except KeyError:
                return None
            if not any(
                member.agent_id == runtime.agent_id and member.is_active for member in members
            ):
                return None
            for recipient in dict.fromkeys((runtime.name, runtime.agent_id)):
                mailbox_entry = runtime.team_service.consume_next_mailbox_entry(
                    runtime.team_id,
                    recipient=recipient,
                )
                if mailbox_entry is not None:
                    return _PendingTeammateWork(
                        prompt=mailbox_entry.content,
                        source="mailbox",
                    )

            task = self._claim_next_task(runtime)
            if task is not None:
                return _PendingTeammateWork(
                    prompt=task.active_form or task.subject,
                    source="task",
                    task=task,
                )

            try:
                await asyncio.wait_for(runtime.stop_event.wait(), timeout=POLL_INTERVAL_SECONDS)
            except asyncio.TimeoutError:
                continue
        return None

    def _member_is_current(self, runtime: _TeammateRuntime) -> bool:
        try:
            runtime.team_service.get(runtime.team_id)
        except KeyError:
            return False
        return True

    async def _await_run(self, runtime: _TeammateRuntime) -> None:
        while runtime.current_run is not None and not runtime.current_run.done():
            if runtime.stop_event.is_set() or not self._member_is_current(runtime):
                runtime.stop_event.set()
                await self._agent_service.cancel_background(runtime.agent_id)
                break
            await asyncio.wait({runtime.current_run}, timeout=POLL_INTERVAL_SECONDS)
        if runtime.current_run is not None:
            try:
                await runtime.current_run
            except asyncio.CancelledError:
                # A cancelled unit of work is an outcome, not a reason to lose
                # its claim or disable this member's future automatic pickup.
                pass

    def _finish_task(
        self, runtime: _TeammateRuntime, task: TeamTaskRecord | None, state: str
    ) -> None:
        if task is None or task.claim_id is None or not self._member_is_current(runtime):
            return
        tasks = runtime.team_service.task_service(runtime.team_id)
        try:
            tasks.finish_claim(task.id, runtime.agent_id, task.claim_id, state)
        except KeyError:
            return
        except RuntimeError:
            # A completion hook can reject the result. Preserve that outcome as
            # failed, but still compare the original claim so a newer owner wins.
            try:
                current = tasks.get_task(task.id)
            except KeyError:
                return
            if current is not None and (current.status, current.owner, current.claim_id) == (
                "in_progress",
                runtime.agent_id,
                task.claim_id,
            ):
                tasks.finish_claim(task.id, runtime.agent_id, task.claim_id, "failed")
                logger.warning("Completion rejected for team task %s; marked failed", task.id)

    async def _run_teammate_loop(self, runtime: _TeammateRuntime) -> None:
        """Keep an in-process teammate alive so it can accept follow-up work."""
        claimed_task: TeamTaskRecord | None = None
        try:
            while not runtime.stop_event.is_set():
                if runtime.current_run is not None:
                    await self._await_run(runtime)
                    runtime.current_run = None

                record = self._agent_service.get(runtime.agent_id)

                self._finish_task(runtime, claimed_task, record.state)
                claimed_task = None
                runtime.idle_event.set()

                if runtime.stop_event.is_set() or not self._member_is_current(runtime):
                    break

                if record.state in TERMINAL_STATES:
                    self._notify_teammate_idle(runtime)

                next_work = await self._wait_for_next_work(runtime)
                if next_work is None:
                    break

                claimed_task = next_work.task
                try:
                    self._sync_permission_mode(runtime)
                    await self._agent_service.resume_background(
                        agent_id=runtime.agent_id,
                        agent_definition=runtime.agent_definition,
                        prompt=next_work.prompt,
                        team_context=TeamToolContext(
                            team_id=runtime.team_id,
                            sender_name=runtime.name,
                            sender_agent_id=runtime.agent_id,
                            team_service=runtime.team_service,
                            source=next_work.source,
                        ),
                        executor=self._guarded_executor(next_work.prompt, claimed_task),
                    )
                except Exception as error:
                    self._finish_task(runtime, claimed_task, "failed")
                    claimed_task = None
                    if next_work.task is not None:
                        logger.warning(
                            "Could not launch team task %s; marked failed",
                            next_work.task.id,
                            exc_info=True,
                        )
                        continue
                    # Queue acceptance is not execution success. Retain the
                    # failed mailbox work and tell the leader why this member
                    # stopped before its consumer releases the live identity.
                    try:
                        diagnostic = f"Could not start teammate work ({type(error).__name__})"
                        runtime.team_service.record_run(
                            runtime.team_id,
                            agent_id=runtime.agent_id,
                            member_name=runtime.name,
                            prompt=next_work.prompt,
                            output=diagnostic,
                            state="failed",
                            source=next_work.source,
                        )
                        runtime.team_service.route(
                            runtime.team_id,
                            diagnostic,
                            recipient=TEAM_LEAD_NAME,
                            sender=runtime.name,
                        )
                    except Exception:
                        logger.warning("Could not record teammate startup failure", exc_info=True)
                    raise
                runtime.current_run = self._agent_service._tasks.get(runtime.agent_id)
                runtime.idle_event.clear()
        finally:
            try:
                if runtime.current_run is not None and not runtime.current_run.done():
                    await self._agent_service.cancel_background(runtime.agent_id)
                self._finish_task(runtime, claimed_task, "cancelled")
            finally:
                try:
                    with runtime.team_service.lifecycle(runtime.team_id):
                        self._team_service.set_member_active(
                            runtime.team_id, runtime.agent_id, False
                        )
                except Exception:
                    logger.debug(
                        "Failed to deactivate teammate %s", runtime.agent_id, exc_info=True
                    )
                runtime.idle_event.set()
                self._tasks.pop(runtime.agent_id, None)
                self._runtimes.pop(runtime.agent_id, None)

    async def wait(self, agent_id: str) -> None:
        """Wait for the teammate's current unit of work to finish."""
        runtime = self._runtimes.get(agent_id)
        if runtime is None:
            await self._agent_service.wait(agent_id)
            return
        await runtime.idle_event.wait()

    async def terminate(self, agent_id: str) -> bool:
        """Terminate a running or idle teammate loop."""
        runtime = self._runtimes.get(agent_id)
        if runtime is None:
            try:
                await self._agent_service.cancel_background(agent_id)
                return True
            except KeyError:
                return False

        runtime.stop_event.set()
        try:
            await self._agent_service.cancel_background(agent_id)
        except KeyError:
            return False
        loop_task = self._tasks.get(agent_id)
        if loop_task is not None:
            await loop_task
        return True

    async def aclose(self) -> None:
        """Stop/join only this runner's members before closing its AgentService."""
        self._closed = True
        results = await asyncio.gather(
            *(self.terminate(agent_id) for agent_id in tuple(self._runtimes)),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    @property
    def has_live_members(self) -> bool:
        """Idle consumers are owned live work even after their model turn ends."""
        return bool(self._runtimes)

    def is_active(self, agent_id: str) -> bool:
        """Check if a teammate is currently busy processing work."""
        runtime = self._runtimes.get(agent_id)
        if runtime is None:
            return False
        return runtime.current_run is not None and not runtime.current_run.done()

    def manages(self, agent_id: str) -> bool:
        """Return whether this runner owns the given teammate lifecycle."""
        return agent_id in self._runtimes
