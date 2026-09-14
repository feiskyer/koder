"""In-memory runtime agent lifecycle service."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import uuid
from collections.abc import Coroutine
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from agents import RunConfig, Runner
from filelock import FileLock, Timeout

from koder_agent.agentic import create_dev_agent, get_display_hooks, get_subagent_display_hooks
from koder_agent.core.constants import get_max_turns
from koder_agent.core.display_context import (
    SubagentDisplayIdentity,
    current_tool_display_call,
    detached_display_context,
    has_subagent_display_sink,
)
from koder_agent.core.session import EnhancedSQLiteSession
from koder_agent.harness.agents.hooks import SubagentLifecycleHooks
from koder_agent.harness.execution_context import (
    get_execution_cwd,
    reset_execution_cwd,
    set_execution_cwd,
)
from koder_agent.harness.paths import worktrees_dir
from koder_agent.harness.plan.mode import PlanModeService
from koder_agent.harness.worktree.service import WorktreeService
from koder_agent.tools import get_all_tools
from koder_agent.tools.permission_context import (
    get_tool_permission_context,
    reset_tool_permission_context,
    subagent_permission_scope,
)
from koder_agent.tools.plan_mode import plan_service_scope
from koder_agent.tools.skill_context import skill_run_scope
from koder_agent.tools.todo import (
    TodoRuntimeIdentity,
    TodoStore,
    reset_todo_context,
    set_todo_context,
)
from koder_agent.utils.async_tasks import await_owned_task, run_sync_owned
from koder_agent.utils.atomic_file import write_text_atomic
from koder_agent.utils.client import get_model_client_snapshot

from .definitions import (
    AgentDefinition,
    build_agent_system_prompt,
    filter_tools_for_agent_definition,
    resolve_agent_mcp_server_configs,
    resolve_agent_model,
)
from .messages import AgentMessage
from .models import AgentRecord, DelayedWorkerResult
from .runtime_context import agent_service_scope, agent_session_scope, set_agent_session
from .summary import summarize_agent_record
from .teams.context import TeamToolContext, team_tool_context

logger = logging.getLogger(__name__)

AgentRunExecutor = Callable[..., Awaitable[str]]


async def _deny_approver(tool_name, arguments, decision):
    """Always-deny approver used for subagents.

    A subagent has no interactive terminal, so an approval-gated call must fail
    CLOSED. Passing approver=None instead would hit enforce_tool_permission's
    TTY-aware fallback, which fails OPEN in an interactive session. Returning
    "deny" is the verdict enforce_tool_permission understands as a block.
    """
    return "deny"


def _create_detached_task(coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
    """Create a background task without retaining parent display routing."""

    context = detached_display_context()
    context.run(set_agent_session, None)
    return context.run(asyncio.create_task, coroutine)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_provider_from_snapshot(snapshot: dict[str, Any]) -> str | None:
    kwargs = snapshot.get("litellm_kwargs") or {}
    litellm_model = str(kwargs.get("model") or "")
    if litellm_model:
        return litellm_model.split("/", 1)[0]
    model_name = str(snapshot.get("model_name") or "")
    if model_name.startswith("litellm/"):
        remainder = model_name[len("litellm/") :]
        return remainder.split("/", 1)[0] if remainder else None
    if snapshot.get("native_openai"):
        return "openai"
    return None


def _redacted_model_config_snapshot(agent_definition: AgentDefinition) -> dict[str, Any]:
    """Return safe model config evidence for runtime agent records."""

    model_override = resolve_agent_model(agent_definition)
    try:
        snapshot = get_model_client_snapshot(model_override)
    except Exception as exc:  # pragma: no cover - defensive config path
        return {
            "model_override": model_override or "inherit",
            "error": str(exc),
        }
    kwargs = snapshot.get("litellm_kwargs") or {}
    extra_headers = kwargs.get("extra_headers") or {}
    return {
        "model_override": model_override or "inherit",
        "model_name": snapshot.get("model_name"),
        "provider": _model_provider_from_snapshot(snapshot),
        "base_url": snapshot.get("base_url") or kwargs.get("base_url"),
        "native_openai": bool(snapshot.get("native_openai")),
        "api_key_present": bool(snapshot.get("api_key") or kwargs.get("api_key")),
        "reasoning_effort": snapshot.get("reasoning_effort"),
        "litellm_model": kwargs.get("model"),
        "oauth_provider": extra_headers.get("x-oauth-provider"),
        "oauth_headers_present": bool(extra_headers),
    }


def agent_definition_provenance(agent_definition: AgentDefinition) -> dict[str, Any]:
    """Return stable, non-secret evidence identifying an agent definition."""
    serialized = json.dumps(
        asdict(agent_definition),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return {
        "source": agent_definition.source,
        "filename": agent_definition.filename,
        "base_dir": agent_definition.base_dir,
        "plugin": agent_definition.plugin,
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def agent_definition_matches_record(record: AgentRecord, agent_definition: AgentDefinition) -> bool:
    """Return whether a definition is exactly the one recorded for a run."""
    provenance = record.definition_provenance
    return isinstance(provenance, dict) and provenance == agent_definition_provenance(
        agent_definition
    )


def resolve_agent_record_origin(record: AgentRecord) -> Path:
    """Resolve a persisted origin cwd, rejecting missing or unsafe values."""
    raw = record.origin_cwd
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("background agent record has no originating directory")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError("background agent originating directory must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("background agent originating directory is unavailable") from exc
    if not resolved.is_dir():
        raise ValueError("background agent originating directory is not a directory")
    return resolved


def resolve_agent_record_execution_cwd(record: AgentRecord) -> Path:
    """Resolve the persisted worktree or originating cwd used for execution."""
    origin = resolve_agent_record_origin(record)
    if not record.worktree_path:
        return origin
    worktree = Path(record.worktree_path).expanduser()
    if not worktree.is_absolute():
        raise ValueError("background agent worktree directory must be absolute")
    try:
        resolved = worktree.resolve(strict=True)
    except OSError as exc:
        raise ValueError("background agent worktree directory is unavailable") from exc
    if not resolved.is_dir():
        raise ValueError("background agent worktree directory is not a directory")
    return resolved


async def _cleanup_agent_mcp_servers(
    agent: Any,
    *,
    propagate_cancellation: bool = True,
) -> None:
    from koder_agent.mcp import close_mcp_servers, detach_mcp_server_owner

    owner = detach_mcp_server_owner(agent)
    await close_mcp_servers(
        owner,
        propagate_cancellation=propagate_cancellation,
    )


def _sync_display_identity(
    agent_definition: AgentDefinition,
    session_id: str,
    display_label: str | None,
) -> SubagentDisplayIdentity | None:
    """Build parent-managed display identity for a synchronous agent tool run."""

    parent_call = current_tool_display_call()
    if (
        not has_subagent_display_sink()
        or parent_call is None
        or parent_call.tool_name != "agent_tool"
    ):
        return None

    label = agent_definition.agent_type
    if display_label:
        label = f"{label} · {display_label}"
    return SubagentDisplayIdentity(
        group_id=f"agent-tool-{uuid.uuid4().hex}",
        agent_id=session_id,
        label=label,
        parent_call_id=parent_call.call_id,
        order=0,
    )


async def _execute_agent_run(
    *,
    agent_definition: AgentDefinition,
    prompt: str,
    session_id: str,
    seed_items: list[dict[str, Any]] | None,
    cwd: str | None,
    team_context: TeamToolContext | None = None,
    permission_service: Any = None,
    todo_store: TodoStore | None = None,
    display_identity: SubagentDisplayIdentity | None = None,
    direct_display: bool = False,
) -> str:
    if todo_store is None:
        todo_store = TodoStore(
            TodoRuntimeIdentity(
                session_id=session_id,
                agent_id=agent_definition.agent_type,
                run_id=session_id,
            )
        )

    tools = filter_tools_for_agent_definition(agent_definition, get_all_tools())
    # Subagents cannot spawn other subagents
    tools = [tool for tool in tools if tool.name not in {"task_delegate", "agent_tool"}]
    agent = None
    session = None
    perm_token = None
    todo_token = None
    if display_identity is not None:
        display_hooks = get_subagent_display_hooks(
            group_id=display_identity.group_id,
            agent_id=display_identity.agent_id,
            label=display_identity.label,
            parent_call_id=display_identity.parent_call_id,
            order=display_identity.order,
        )
    else:
        display_hooks = get_display_hooks()
        if hasattr(display_hooks, "streaming_mode"):
            display_hooks.streaming_mode = not direct_display
    display_status = "failed"
    display_detail = None
    primary_error: BaseException | None = None
    directory_token = set_execution_cwd(cwd)
    try:
        agent = await create_dev_agent(
            tools,
            name=agent_definition.agent_type,
            instructions_override=build_agent_system_prompt(
                agent_definition, cwd=get_execution_cwd()
            ),
            model_override=resolve_agent_model(agent_definition),
            extra_mcp_server_configs=resolve_agent_mcp_server_configs(agent_definition),
        )
        session = EnhancedSQLiteSession(session_id=session_id)
        if seed_items:
            existing_items = await session.get_items()
            if not existing_items:
                await session.add_items(seed_items)

        if todo_store.identity.session_id != session_id:
            raise ValueError("todo store session identity does not match subagent session")
        perm_token = subagent_permission_scope(permission_service, deny_approver=_deny_approver)
        todo_token = set_todo_context(todo_store)
        perm_ctx = get_tool_permission_context()
        effective_service = perm_ctx.permission_service if perm_ctx else None
        with team_tool_context(team_context), agent_session_scope(session):
            lifecycle_hooks = SubagentLifecycleHooks(
                agent_definition=agent_definition,
                cwd=get_execution_cwd(),
                wrapped_hooks=display_hooks,
                permission_service=effective_service,
            )
            with skill_run_scope(lifecycle_hooks) as run_hooks:
                result = await Runner.run(
                    agent,
                    prompt,
                    session=session,
                    run_config=RunConfig(),
                    hooks=run_hooks,
                    max_turns=agent_definition.max_turns or get_max_turns(),
                )
        display_status = "completed"
    except BaseException as exc:
        primary_error = exc
        display_status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        display_detail = str(exc)
        raise
    finally:
        try:
            if todo_token is not None:
                reset_todo_context(todo_token)
            if perm_token is not None:
                reset_tool_permission_context(perm_token)
            try:
                if agent is not None:
                    await _cleanup_agent_mcp_servers(
                        agent,
                        propagate_cancellation=primary_error is None,
                    )
            finally:
                if session is not None:
                    await run_sync_owned(session.close)
        except BaseException as exc:
            if primary_error is None:
                display_status = (
                    "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
                )
                display_detail = str(exc)
                raise
            logger.debug(
                "Suppressed subagent cleanup failure while preserving %s",
                type(primary_error).__name__,
                exc_info=True,
            )
        finally:
            reset_execution_cwd(directory_token)
            finish_display = getattr(display_hooks, "finish", None)
            if callable(finish_display):
                finish_display(display_status, display_detail)
    return str(result.final_output)


class AgentService:
    """Stable service for spawning agents and routing mailbox messages."""

    def __init__(
        self,
        *,
        output_root: Path | None = None,
        permission_service: Any = None,
        retain_completed_tasks: bool = True,
    ):
        self._agents: dict[str, AgentRecord] = {}
        self._mailboxes: dict[str, list[AgentMessage]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._todo_stores: dict[TodoRuntimeIdentity, TodoStore] = {}
        self._todo_identities_by_agent: dict[str, TodoRuntimeIdentity] = {}
        self._name_registry: dict[str, set[str]] = {}
        self._owned_agents: set[str] = set()
        self._state_lock = threading.RLock()
        self._sync_tasks: set[asyncio.Task] = set()
        self._closing = False
        self._close_task: asyncio.Task | None = None
        self.team_tool_runtime = None
        self._retain_completed_tasks = retain_completed_tasks
        self._run_claims: dict[str, FileLock] = {}
        self._runtime_id = uuid.uuid4().hex
        self._owned_temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._permission_service = permission_service
        self.output_root = (
            (output_root or (Path.home() / ".koder" / "agents")).expanduser().resolve()
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._load_records()

    @classmethod
    def for_test(cls, root: Path | None = None) -> "AgentService":
        if root is not None:
            return cls(output_root=root / "agent-output")
        temp_dir = tempfile.TemporaryDirectory(prefix="koder-agent-output-")
        service = cls(output_root=Path(temp_dir.name) / "agent-output")
        service._owned_temp_dir = temp_dir
        return service

    def close(self) -> None:
        if any(not task.done() for task in (*self._tasks.values(), *self._sync_tasks)):
            raise RuntimeError("Active agent work requires await service.aclose()")
        if self.team_tool_runtime is not None:
            self.team_tool_runtime.close()
        self._closing = True
        self._todo_stores.clear()
        self._todo_identities_by_agent.clear()
        if self._owned_temp_dir is not None:
            self._owned_temp_dir.cleanup()
            self._owned_temp_dir = None

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, RuntimeError):
            # Active work must be joined by its runtime owner, not by finalizers.
            pass

    @property
    def is_closed(self) -> bool:
        return self._closing

    def _ensure_open(self) -> None:
        if self._closing:
            raise RuntimeError("Agent service is closed")

    def get_team_tool_runtime(self):
        """Lazily create team resources owned by this service, not by one tool call."""
        self._ensure_open()
        if self.team_tool_runtime is None:
            from .teams.tool_runtime import TeamToolRuntime

            self.team_tool_runtime = TeamToolRuntime(self)
        return self.team_tool_runtime

    async def aclose(self) -> None:
        """Stop admission, cancel and join only this service's owned work."""
        current = asyncio.current_task()
        if current in self._tasks.values() or current in self._sync_tasks:
            raise RuntimeError("An owned agent run cannot close its own service")
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close_owned_work())
        await await_owned_task(self._close_task)

    async def _close_owned_work(self) -> None:
        errors = []
        if self.team_tool_runtime is not None:
            try:
                await self.team_tool_runtime.aclose()
            except BaseException as exc:
                errors.append(exc)
        tasks = list({*self._tasks.values(), *self._sync_tasks})
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._sync_tasks.clear()
        try:
            self.close()
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise BaseExceptionGroup("Agent resource cleanup failed", errors)

    def _track_background(self, agent_id: str, coroutine) -> None:
        claim = self._run_claims.get(agent_id)
        execution_id = self.get(agent_id).execution_id
        task = _create_detached_task(coroutine)
        self._tasks[agent_id] = task

        def settled(finished):
            try:
                if finished.cancelled() and self.get(agent_id).state == "in_progress":
                    self._finish_record(
                        agent_id,
                        "cancelled",
                        "Cancelled before execution",
                        expected_execution=execution_id,
                    )
                if not finished.cancelled():
                    finished.exception()
            except Exception as exc:
                logger.debug("Agent settlement failed (%s)", type(exc).__name__)
            finally:
                if claim is not None:
                    if self._run_claims.get(agent_id) is claim:
                        self._run_claims.pop(agent_id, None)
                    claim.release()
                if not self._retain_completed_tasks and self._tasks.get(agent_id) is finished:
                    self._tasks.pop(agent_id, None)

        task.add_done_callback(settled)

    @contextmanager
    def _reserve_run(self, agent_id: str):
        self._record_path(agent_id)  # Validate before constructing a lock path.
        claim = FileLock(str(self.output_root / f".{agent_id}.run.lock"), mode=0o600)
        try:
            claim.acquire(timeout=0)
        except Timeout as exc:
            raise RuntimeError("Agent is still executing in another runtime") from exc
        self._run_claims[agent_id] = claim
        previous_task = self._tasks.get(agent_id)
        previous = self._agents.get(agent_id)
        previous_execution = previous.execution_id if previous is not None else None
        try:
            yield
        except BaseException as exc:
            record = self._agents.get(agent_id)
            if (
                self._tasks.get(agent_id) is previous_task
                and record is not None
                and record.execution_id != previous_execution
                and record.runtime_owner == self._runtime_id
                and record.state == "in_progress"
            ):
                try:
                    removed = self._cleanup_clean_worktree(record)
                    self._finish_record(
                        agent_id,
                        "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                        "Agent setup did not complete",
                        error=f"Agent setup failed ({type(exc).__name__})",
                        expected_execution=record.execution_id,
                        worktree_path=None if removed else record.worktree_path,
                    )
                except Exception:
                    logger.debug("Failed to record aborted agent setup", exc_info=True)
            raise
        finally:
            task = self._tasks.get(agent_id)
            if task is None or task.done():
                if self._run_claims.get(agent_id) is claim:
                    self._run_claims.pop(agent_id, None)
                claim.release()

    @contextmanager
    def _record_transaction(self, agent_id: str):
        self._record_path(agent_id)
        with (
            self._state_lock,
            FileLock(str(self.output_root / f".{agent_id}.state.lock"), timeout=5, mode=0o600),
        ):
            record = self._read_record(self._record_path(agent_id))
            if record is None:
                raise KeyError(agent_id)
            self._agents[agent_id] = record
            yield record

    def _finish_record(
        self,
        agent_id: str,
        state: str,
        output: str,
        *,
        error=None,
        expected_execution: str | None = None,
        **fields,
    ):
        with self._record_transaction(agent_id) as current:
            if current.runtime_owner not in (None, self._runtime_id):
                raise RuntimeError("Agent ownership changed before completion")
            if expected_execution is not None and current.execution_id != expected_execution:
                raise RuntimeError("Agent execution changed before completion")
            timestamp = _utc_now_iso()
            if state == "completed" and current.pending_messages:
                state = "delayed"
            updated = replace(current, state=state, error=error, updated_at=timestamp, **fields)
            updated = self._with_summary(
                updated, output_text=output, summary_timestamp=timestamp, record_timestamp=timestamp
            )
            self._save_record(updated)
            self._agents[agent_id] = updated
            return updated

    def spawn(self, profile: str) -> str:
        self._ensure_open()
        agent_id = f"agent-{uuid.uuid4().hex}"
        record = replace(
            AgentRecord.create(agent_id=agent_id, profile=profile), runtime_owner=self._runtime_id
        )
        self._agents[agent_id] = self._with_summary(record)
        self._mailboxes[agent_id] = []
        self._owned_agents.add(agent_id)
        self._save_record(self._agents[agent_id])
        return agent_id

    def get(self, agent_id: str) -> AgentRecord:
        return self._agents[agent_id]

    def list_records(self) -> list[AgentRecord]:
        return sorted(self._agents.values(), key=lambda item: item.updated_at, reverse=True)

    def refresh_summary(self, agent_id: str) -> AgentRecord:
        with self._record_transaction(agent_id) as record:
            updated = self._with_summary(record, output_text=self._read_output(record))
            if agent_id in self._owned_agents and record.runtime_owner in (None, self._runtime_id):
                self._save_record(updated)
            self._agents[agent_id] = updated
            return updated

    def send(self, agent_id: str, content: str) -> AgentMessage:
        self._ensure_open()
        with self._record_transaction(agent_id) as record:
            if agent_id not in self._owned_agents:
                raise ValueError(
                    "Recipient is historical or owned by another runtime; resume explicitly"
                )
            if record.runtime_owner not in (None, self._runtime_id):
                raise ValueError("Recipient has moved to another runtime; resume explicitly")
            updated = replace(
                record,
                pending_messages=(*record.pending_messages, content),
                updated_at=_utc_now_iso(),
            )
            self._save_record(updated)
            self._agents[agent_id] = updated
            message = AgentMessage.create(agent_id=agent_id, content=content)
            self._mailboxes[agent_id].append(message)
            return message

    def _claim_message_prompt(self, agent_id: str, initial: str | None = None) -> str | None:
        with self._record_transaction(agent_id) as record:
            if record.runtime_owner != self._runtime_id:
                raise RuntimeError("Agent ownership changed before message delivery")
            messages = (*record.inflight_messages, *record.pending_messages)
            if not messages:
                return initial
            content = "Agent mailbox messages (user-level input):\n" + json.dumps(
                list(messages), ensure_ascii=False
            )
            prompt = f"{initial}\n\n{content}" if initial else content
            updated = replace(
                record,
                prompt=prompt,
                pending_messages=(),
                inflight_messages=messages,
                updated_at=_utc_now_iso(),
            )
            # Assignment to the next attempted run is durable before dequeuing.
            self._save_record(updated)
            self._agents[agent_id] = updated
            return prompt

    def _acknowledge_messages(self, agent_id: str, execution_id: str | None) -> None:
        with self._record_transaction(agent_id) as record:
            if record.runtime_owner != self._runtime_id or record.execution_id != execution_id:
                raise RuntimeError("Agent execution changed before message acknowledgement")
            if record.inflight_messages:
                updated = replace(record, inflight_messages=(), updated_at=_utc_now_iso())
                self._save_record(updated)
                self._agents[agent_id] = updated

    def read_mailbox(self, agent_id: str) -> list[AgentMessage]:
        return list(self._mailboxes[agent_id])

    def mark_worker_delayed(self, agent_id: str) -> DelayedWorkerResult:
        with self._record_transaction(agent_id) as agent:
            if agent.runtime_owner not in (None, self._runtime_id):
                raise RuntimeError("Agent is owned by another runtime")
            updated = replace(agent, state="delayed", updated_at=_utc_now_iso())
            self._save_record(updated)
            self._agents[agent_id] = updated
        return DelayedWorkerResult(agent_id=agent_id, state_preserved=True)

    async def launch_background(
        self,
        *,
        agent_definition: AgentDefinition,
        prompt: str,
        description: str,
        seed_items: list[dict[str, Any]] | None = None,
        cwd: str | Path | None = None,
        permission_mode: str | None = None,
        team_context_builder: Callable[[AgentRecord], TeamToolContext | None] | None = None,
        executor: AgentRunExecutor | None = None,
    ) -> AgentRecord:
        self._ensure_open()
        agent_id = f"agent-{uuid.uuid4().hex}"
        with self._reserve_run(agent_id):
            if self._record_path(agent_id).exists():
                raise RuntimeError("Generated agent ID already exists")
            return await self._launch_background_reserved(
                agent_id=agent_id,
                agent_definition=agent_definition,
                prompt=prompt,
                description=description,
                seed_items=seed_items,
                cwd=cwd,
                permission_mode=permission_mode,
                team_context_builder=team_context_builder,
                executor=executor,
            )

    async def _launch_background_reserved(
        self,
        *,
        agent_id: str,
        agent_definition: AgentDefinition,
        prompt: str,
        description: str,
        seed_items: list[dict[str, Any]] | None,
        cwd: str | Path | None,
        permission_mode: str | None,
        team_context_builder: Callable[[AgentRecord], TeamToolContext | None] | None,
        executor: AgentRunExecutor | None,
    ) -> AgentRecord:
        session_id = f"subagent-{agent_id}"
        output_path = self.output_root / f"{agent_id}.output"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(output_path, "")
        worktree_path = None
        worktree_branch = None
        origin_cwd = Path(cwd) if cwd is not None else Path.cwd()
        origin_cwd = origin_cwd.expanduser().resolve(strict=True)
        if not origin_cwd.is_dir():
            raise ValueError("background agent cwd must be a directory")
        effective_cwd = str(origin_cwd)
        if agent_definition.isolation == "worktree" and cwd is not None:
            cwd_path = origin_cwd
            service = WorktreeService(worktrees_dir(cwd_path), repo_root=cwd_path)
            created = service.create(f"agent/{agent_id}")
            worktree_path = str(created.path)
            worktree_branch = created.branch
            effective_cwd = worktree_path

        record = AgentRecord.create(
            agent_id=agent_id,
            profile=agent_definition.agent_type,
            session_id=session_id,
            description=description,
            prompt=prompt,
            output_path=str(output_path),
            worktree_path=worktree_path,
            worktree_branch=worktree_branch,
            permission_mode=permission_mode or agent_definition.permission_mode,
            state="in_progress",
            model_config=_redacted_model_config_snapshot(agent_definition),
            origin_cwd=str(origin_cwd),
            definition_provenance=agent_definition_provenance(agent_definition),
        )
        record = self._with_summary(
            replace(record, runtime_owner=self._runtime_id, execution_id=uuid.uuid4().hex)
        )
        self._agents[agent_id] = record
        self._mailboxes[agent_id] = []
        self._owned_agents.add(agent_id)
        self._save_record(record)
        team_context = team_context_builder(record) if team_context_builder is not None else None
        self._track_background(
            agent_id,
            self._run_background(
                agent_id=agent_id,
                agent_definition=agent_definition,
                prompt=prompt,
                session_id=session_id,
                seed_items=seed_items,
                cwd=effective_cwd,
                team_context=team_context,
                executor=executor,
                todo_store=self._get_or_create_todo_store(agent_id, session_id),
            ),
        )
        return record

    async def run_sync(
        self,
        *,
        agent_definition: AgentDefinition,
        prompt: str,
        seed_items: list[dict[str, Any]] | None = None,
        cwd: str | Path | None = None,
        permission_mode: str | None = None,
        display_label: str | None = None,
    ) -> str:
        self._ensure_open()
        with agent_session_scope(None):
            task = asyncio.create_task(
                self._run_sync(
                    agent_definition=agent_definition,
                    prompt=prompt,
                    seed_items=seed_items,
                    cwd=cwd,
                    permission_mode=permission_mode,
                    display_label=display_label,
                )
            )
        self._sync_tasks.add(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            try:
                await await_owned_task(task)
            except BaseException:
                pass  # Preserve the caller's cancellation after joining cleanup.
            raise
        finally:
            self._sync_tasks.discard(task)

    async def _run_sync(
        self,
        *,
        agent_definition: AgentDefinition,
        prompt: str,
        seed_items: list[dict[str, Any]] | None,
        cwd: str | Path | None,
        permission_mode: str | None,
        display_label: str | None,
    ) -> str:
        effective_cwd = str(cwd) if cwd is not None else None
        worktree_service: WorktreeService | None = None
        worktree_created = None
        if agent_definition.isolation == "worktree" and cwd is not None:
            cwd_path = Path(cwd).resolve()
            worktree_service = WorktreeService(worktrees_dir(cwd_path), repo_root=cwd_path)
            worktree_created = worktree_service.create(f"sync-agent/{uuid.uuid4().hex[:8]}")
            effective_cwd = str(worktree_created.path)
        try:
            scoped_service = PlanModeService()
            effective_permission_mode = (
                permission_mode or agent_definition.permission_mode or "default"
            )
            if effective_permission_mode == "plan":
                scoped_service.enter_plan_mode(permission_mode="plan")
            with plan_service_scope(scoped_service), agent_service_scope(self):
                session_id = f"subagent-sync-{uuid.uuid4().hex[:8]}"
                display_identity = _sync_display_identity(
                    agent_definition,
                    session_id,
                    display_label,
                )
                return await _execute_agent_run(
                    agent_definition=agent_definition,
                    prompt=prompt,
                    session_id=session_id,
                    seed_items=seed_items,
                    cwd=effective_cwd,
                    permission_service=self._permission_service,
                    display_identity=display_identity,
                    direct_display=display_identity is None,
                    todo_store=TodoStore(
                        TodoRuntimeIdentity(
                            session_id=session_id,
                            agent_id=f"sync-{agent_definition.agent_type}",
                            run_id=session_id,
                        )
                    ),
                )
        finally:
            # Dirty worktrees are kept so the user can inspect or merge them.
            if worktree_service is not None and worktree_created is not None:
                try:
                    worktree_service.remove_if_clean(
                        worktree_created.path, branch=worktree_created.branch
                    )
                except Exception:
                    logger.debug(
                        "Worktree cleanup failed for %s", worktree_created.path, exc_info=True
                    )

    async def resume_background(
        self,
        *,
        agent_id: str,
        agent_definition: AgentDefinition,
        prompt: str,
        cwd: str | Path | None = None,
        team_context: TeamToolContext | None = None,
        executor: AgentRunExecutor | None = None,
    ) -> AgentRecord:
        self._ensure_open()
        if agent_id in self._tasks and not self._tasks[agent_id].done():
            raise RuntimeError(f"Agent is still running: {agent_id}")
        with self._reserve_run(agent_id):
            return await self._resume_background_reserved(
                agent_id=agent_id,
                agent_definition=agent_definition,
                prompt=prompt,
                cwd=cwd,
                team_context=team_context,
                executor=executor,
            )

    async def _resume_background_reserved(
        self,
        *,
        agent_id: str,
        agent_definition: AgentDefinition,
        prompt: str,
        cwd: str | Path | None,
        team_context: TeamToolContext | None,
        executor: AgentRunExecutor | None,
    ) -> AgentRecord:
        # The execution lease excludes another runner; reload current metadata
        # rather than using a historical service object's stale worktree/queue.
        with self._record_transaction(agent_id) as current:
            record = current
        if not agent_definition_matches_record(record, agent_definition):
            raise ValueError("background agent definition provenance does not match")
        execution_cwd = resolve_agent_record_execution_cwd(record)
        if cwd is not None:
            requested_cwd = Path(cwd).expanduser().resolve(strict=True)
            if requested_cwd != execution_cwd:
                raise ValueError("background agent resume cwd does not match persisted origin")
        worktree_path = record.worktree_path
        worktree_branch = record.worktree_branch
        if agent_definition.isolation == "worktree" and worktree_path is None:
            # A previous clean run may have removed its checkout. Resuming must
            # recreate isolation rather than execute in the parent repository.
            origin = resolve_agent_record_origin(record)
            service = WorktreeService(worktrees_dir(origin), repo_root=origin)
            created = service.create(f"agent/{agent_id}-{uuid.uuid4().hex[:8]}")
            execution_cwd = created.path
            worktree_path = str(created.path)
            worktree_branch = created.branch
        timestamp = _utc_now_iso()
        with self._record_transaction(agent_id) as latest:
            updated = replace(
                latest,
                prompt=prompt,
                worktree_path=worktree_path,
                worktree_branch=worktree_branch,
                state="in_progress",
                error=None,
                model_config=_redacted_model_config_snapshot(agent_definition),
                updated_at=timestamp,
                runtime_owner=self._runtime_id,
                execution_id=uuid.uuid4().hex,
            )
            self._agents[agent_id] = self._with_summary(
                updated, summary_timestamp=timestamp, record_timestamp=timestamp
            )
            self._save_record(self._agents[agent_id])
        self._owned_agents.add(agent_id)
        self._track_background(
            agent_id,
            self._run_background(
                agent_id=agent_id,
                agent_definition=agent_definition,
                prompt=prompt,
                session_id=record.session_id,
                seed_items=None,
                cwd=str(execution_cwd),
                team_context=team_context,
                executor=executor,
                todo_store=self._get_or_create_todo_store(agent_id, record.session_id),
            ),
        )
        return self._agents[agent_id]

    async def wait(self, agent_id: str) -> AgentRecord:
        task = self._tasks.get(agent_id)
        if task is not None:
            await asyncio.shield(task)
        return self.get(agent_id)

    async def cancel_background(self, agent_id: str) -> AgentRecord:
        task = self._tasks.get(agent_id)
        if task is None or task.done():
            with self._record_transaction(agent_id) as record:
                if record.state in {"completed", "failed", "cancelled"}:
                    return record
                if agent_id not in self._owned_agents or record.runtime_owner not in (
                    None,
                    self._runtime_id,
                ):
                    raise ValueError("Cannot cancel an agent owned by another runtime")
        if task is not None and not task.done():
            task.cancel()
            if task is asyncio.current_task():
                raise asyncio.CancelledError
            try:
                await await_owned_task(task)
            except asyncio.CancelledError:
                pass
            return self.get(agent_id)
        updated = self._finish_record(agent_id, "cancelled", "Cancelled")
        if updated.output_path:
            write_text_atomic(Path(updated.output_path), "Cancelled")
        return updated

    def _cleanup_clean_worktree(self, record: AgentRecord) -> bool:
        """Remove a completed agent's isolation worktree when it has no changes.

        Dispatches ``WorktreeRemove`` hooks via WorktreeService.exit(). Dirty
        worktrees are kept so the user can inspect or merge the agent's work.
        Returns True when the worktree was removed.
        """
        if not record.worktree_path:
            return False
        path = Path(record.worktree_path)
        repo_root = None
        # The worktree lives under <repo>/.koder/worktrees/<slug>; walk up to
        # find the owning repository root.
        for parent in path.parents:
            if (parent / ".git").exists():
                repo_root = parent
                break
        try:
            service = WorktreeService(path.parent, repo_root=repo_root)
            return service.remove_if_clean(path, branch=record.worktree_branch)
        except Exception:
            logger.debug("Worktree cleanup failed for %s", record.worktree_path, exc_info=True)
            return False

    async def _run_background(
        self,
        *,
        agent_id: str,
        agent_definition: AgentDefinition,
        prompt: str,
        session_id: str,
        seed_items: list[dict[str, Any]] | None,
        cwd: str | None,
        team_context: TeamToolContext | None = None,
        executor: AgentRunExecutor | None = None,
        todo_store: TodoStore,
    ) -> None:
        record = self.get(agent_id)
        output_path = Path(record.output_path) if record.output_path else None
        try:
            scoped_service = PlanModeService()
            effective_permission_mode = (
                record.permission_mode or agent_definition.permission_mode or "default"
            )
            if effective_permission_mode == "plan":
                scoped_service.enter_plan_mode(permission_mode="plan")
            outputs: list[str] = []
            with plan_service_scope(scoped_service), agent_service_scope(self):
                initial_prompt: str | None = prompt
                for _ in range(32):
                    if self._closing:
                        raise asyncio.CancelledError
                    next_prompt = self._claim_message_prompt(agent_id, initial_prompt)
                    if next_prompt is None:
                        break
                    execute_kwargs: dict[str, Any] = {
                        "agent_definition": agent_definition,
                        "prompt": next_prompt,
                        "session_id": session_id,
                        "seed_items": seed_items if initial_prompt is not None else None,
                        "cwd": cwd,
                        "permission_service": self._permission_service,
                        "todo_store": todo_store,
                    }
                    if team_context is not None:
                        execute_kwargs["team_context"] = team_context
                    todo_token = set_todo_context(todo_store)
                    try:
                        execute = _execute_agent_run if executor is None else executor
                        result = await execute(**execute_kwargs)
                    finally:
                        reset_todo_context(todo_token)
                    outputs.append(result)
                    self._acknowledge_messages(agent_id, record.execution_id)
                    self._record_team_run_history(
                        team_context=team_context,
                        prompt=next_prompt,
                        output=result,
                        state="completed",
                    )
                    initial_prompt = None
            result = "\n\n".join(outputs)
            if output_path is not None:
                write_text_atomic(output_path, result)
            worktree_removed = self._cleanup_clean_worktree(record)
            self._finish_record(
                agent_id,
                "completed",
                result,
                expected_execution=record.execution_id,
                worktree_path=None if worktree_removed else record.worktree_path,
            )
        except asyncio.CancelledError:
            if output_path is not None:
                write_text_atomic(output_path, "Cancelled")
            self._record_team_run_history(
                team_context=team_context,
                prompt=prompt,
                output="Cancelled",
                state="cancelled",
            )
            self._finish_record(
                agent_id, "cancelled", "Cancelled", expected_execution=record.execution_id
            )
            raise
        except Exception as exc:  # pragma: no cover - defensive runtime path
            output_text = f"Error: {exc}"
            if output_path is not None:
                write_text_atomic(output_path, output_text)
            self._record_team_run_history(
                team_context=team_context,
                prompt=prompt,
                output=output_text,
                state="failed",
            )
            self._finish_record(
                agent_id,
                "failed",
                output_text,
                error=str(exc),
                expected_execution=record.execution_id,
            )

    def _record_team_run_history(
        self,
        *,
        team_context: TeamToolContext | None,
        prompt: str,
        output: str,
        state: str,
    ) -> None:
        if team_context is None:
            return
        try:
            team_context.team_service.record_run(
                team_context.team_id,
                agent_id=team_context.sender_agent_id,
                member_name=team_context.sender_name,
                prompt=prompt,
                output=output,
                state=state,
                source=team_context.source,
            )
        except Exception:
            logger.debug("Failed to record team run history", exc_info=True)

    def register_name(self, name: str, agent_id: str) -> None:
        """Register a human-readable name for an agent, making it addressable."""
        if not isinstance(name, str) or not name.strip() or name == "*":
            raise ValueError("Agent name must be non-empty and cannot be '*'")
        if agent_id not in self._agents:
            raise ValueError("Cannot name an unknown agent")
        if name in self._agents and name != agent_id:
            raise ValueError("Agent name conflicts with an existing agent ID")
        self._name_registry.setdefault(name, set()).add(agent_id)

    def name_is_registered(self, name: str) -> bool:
        return bool(self._name_registry.get(name)) or name in self._agents

    def name_is_ambiguous(self, name: str) -> bool:
        return len(self._name_registry.get(name, ())) > 1

    def release_agent(self, agent_id: str) -> None:
        """Release in-memory state owned by an explicitly cleaned-up agent."""
        task = self._tasks.get(agent_id)
        if task is not None and not task.done():
            raise RuntimeError(f"Agent is still running: {agent_id}")
        identity = self._todo_identities_by_agent.pop(agent_id, None)
        if identity is not None:
            self._todo_stores.pop(identity, None)

    def _get_or_create_todo_store(self, agent_id: str, session_id: str) -> TodoStore:
        identity = TodoRuntimeIdentity(
            session_id=session_id,
            agent_id=agent_id,
            run_id=f"agent-conversation:{agent_id}",
        )
        previous = self._todo_identities_by_agent.get(agent_id)
        if previous is not None and previous != identity:
            self._todo_stores.pop(previous, None)
        self._todo_identities_by_agent[agent_id] = identity
        store = self._todo_stores.get(identity)
        if store is None:
            store = TodoStore(identity)
            self._todo_stores[identity] = store
        return store

    def update_permission_mode(self, agent_id: str, permission_mode: str) -> AgentRecord:
        """Persist an updated permission mode for an existing agent record."""

        with self._record_transaction(agent_id) as record:
            if record.runtime_owner not in (None, self._runtime_id):
                raise RuntimeError("Agent is owned by another runtime")
            updated = replace(record, permission_mode=permission_mode, updated_at=_utc_now_iso())
            self._save_record(updated)
            self._agents[agent_id] = updated
            return updated

    def get_by_name(self, name: str) -> AgentRecord | None:
        """Look up an agent by registered name."""
        matches = self._name_registry.get(name, set())
        if len(matches) != 1:
            return None
        agent_id = next(iter(matches))
        try:
            return self.get(agent_id)
        except KeyError:
            return None

    def resolve_agent_id(self, name_or_id: str) -> str | None:
        """Resolve a name or agent_id to an agent_id."""
        if name_or_id in self._agents:
            return name_or_id
        matches = self._name_registry.get(name_or_id, set())
        return next(iter(matches)) if len(matches) == 1 else None

    def _record_path(self, agent_id: str) -> Path:
        if not isinstance(agent_id, str) or not re.fullmatch(r"agent-[A-Za-z0-9_-]+", agent_id):
            raise ValueError("Invalid agent record ID")
        return self.output_root / f"{agent_id}.json"

    def _save_record(self, record: AgentRecord) -> None:
        write_text_atomic(
            self._record_path(record.id), json.dumps(asdict(record), ensure_ascii=False)
        )

    def _output_path(self, record: AgentRecord) -> Path | None:
        if not record.output_path:
            return None
        expected = self.output_root / f"{record.id}.output"
        if Path(os.path.abspath(record.output_path)) != expected or expected.is_symlink():
            return None
        return expected

    def _read_output(self, record: AgentRecord) -> str | None:
        output = self._output_path(record)
        if output is None:
            return None
        try:
            descriptor = os.open(output, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                return handle.read(32768)  # Only used to derive a bounded summary.
        except OSError:
            return None

    def _with_summary(
        self,
        record: AgentRecord,
        *,
        output_text: str | None = None,
        summary_timestamp: str | None = None,
        record_timestamp: str | None = None,
    ) -> AgentRecord:
        timestamp = summary_timestamp or _utc_now_iso()
        return replace(
            record,
            summary=summarize_agent_record(record, output_text=output_text),
            summary_updated_at=timestamp,
            updated_at=record_timestamp or record.updated_at,
        )

    def _read_record(self, path: Path) -> AgentRecord | None:
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            # Remaining optional fields use the dataclass's migration defaults.
            data.setdefault("permission_mode", None)
            for field in ("pending_messages", "inflight_messages"):
                messages = data.get(field, ())
                if not isinstance(messages, (list, tuple)) or not all(
                    isinstance(x, str) for x in messages
                ):
                    raise ValueError("Invalid agent mailbox messages")
                data[field] = tuple(messages)
            record = AgentRecord(**data)
            if self._record_path(record.id) != path:
                raise ValueError("Agent record identity does not match its file")
            if record.output_path and self._output_path(record) is None:
                raise ValueError("Agent output is outside its owned path")
            return record
        except Exception:
            logger.debug("Failed to parse agent record from file", exc_info=True)
            return None

    def _load_records(self) -> None:
        for path in sorted(self.output_root.glob("agent-*.json")):
            record = self._read_record(path)
            if record is None:
                continue
            if not record.summary:
                record = self._with_summary(
                    record,
                    output_text=self._read_output(record),
                    summary_timestamp=record.updated_at,
                    record_timestamp=record.updated_at,
                )
            self._agents[record.id] = record
            self._mailboxes.setdefault(record.id, [])
