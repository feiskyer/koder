"""Harness session flow used by interactive and prompt runtime modes."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from rich.panel import Panel
from rich.text import Text

from koder_agent.config import get_config, get_config_manager
from koder_agent.core.terminal_reflow import CLEAR_VIEWPORT, get_reflow_buffer, print_reflowable
from koder_agent.harness.agents.definitions import (
    extract_agent_mention,
    get_agent_definitions,
    get_configured_agent_name,
)
from koder_agent.harness.commands.interactive import HarnessInteractiveCommandHandler
from koder_agent.harness.hooks.runtime import (
    dispatch_command_hooks,
    poll_file_change_hooks,
    update_watch_paths,
)
from koder_agent.harness.plugins.session_root import build_session_plugin_root, default_plugin_root
from koder_agent.harness.session_env import load_session_env
from koder_agent.harness.tools.shell_executor import execute_shell_command
from koder_agent.litellm_cost_map import get_litellm_cost_map_debug_lines
from koder_agent.utils.async_tasks import run_sync_owned
from koder_agent.utils.client import get_model_name
from koder_agent.utils.terminal_theme import get_adaptive_console

logger = logging.getLogger(__name__)
console = get_adaptive_console()
DIRECT_COMMAND_PASSTHROUGHS = {"commit"}


class _UnsuccessfulTurnError(RuntimeError):
    def __init__(self, *, cancelled: bool):
        self.cancelled = cancelled
        super().__init__("Scheduled turn did not complete successfully")


@dataclass(frozen=True)
class _SchedulerBuilder:
    """Immutable constructor state shared by initial and switched sessions."""

    scheduler_type: type
    streaming: bool
    agent_definition: Any
    instructions_override: str | None
    instructions_append: str | None
    permission_service: Any
    approver: Any
    agent_definitions: Any
    plugin_root: Path | None = None
    cli_agents_json: Any = None
    wire_approver: Callable[[Any], None] | None = None
    agent_service: Any = None

    def build(self, session_id: str, *, target: "_SessionSwitchTarget | None" = None):
        if target is None:
            project_root = Path.cwd()
            agent_definition = self.agent_definition
            agent_definitions = self.agent_definitions
            todo_store = None
        else:
            project_root = target.cwd or Path.cwd()
            agent_definition = target.agent_definition
            agent_definitions = target.agent_definitions
            todo_store = target.todo_store
        constructor_kwargs = {
            "session_id": session_id,
            "streaming": self.streaming,
            "agent_definition": agent_definition,
            "instructions_override": self.instructions_override,
            "instructions_append": self.instructions_append,
            "permission_service": self.permission_service,
            "approver": self.approver,
            "todo_store": todo_store,
            "project_root": project_root,
            "plugin_root": self.plugin_root,
            "agent_service": self.agent_service,
        }
        signature = inspect.signature(self.scheduler_type)
        if not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            constructor_kwargs = {
                key: value
                for key, value in constructor_kwargs.items()
                if key in signature.parameters
            }
        scheduler = self.scheduler_type(**constructor_kwargs)
        if self.wire_approver is not None:
            self.wire_approver(scheduler)
        scheduler.agent_definitions = agent_definitions
        return scheduler

    def load_agent_definitions(self, cwd: Path) -> Any:
        """Load definitions from the target project without mutating process cwd."""
        return get_agent_definitions(
            cwd=cwd,
            plugin_root=self.plugin_root,
            cli_agents_json=self.cli_agents_json,
        )


@dataclass(frozen=True)
class _SessionSwitchTarget:
    """Cancellable target metadata resolved before replacement construction."""

    cwd: Path | None
    agent_name: str | None
    agent_definition: Any
    agent_definitions: Any = None
    todo_store: Any = None
    todo_key: tuple[str, str] | None = None


@dataclass(frozen=True)
class _SessionSwitchCommit:
    """Fully prepared values applied synchronously at the switch linearization point."""

    target: _SessionSwitchTarget
    display_name: str | None


class _SchedulerState:
    """Own scheduler dispatch, replacement, and exact-once retirement."""

    def __init__(self, builder: _SchedulerBuilder, scheduler: Any):
        self.builder = builder
        self._scheduler = scheduler
        self._lifecycle_lock = asyncio.Lock()
        self._cleanup_tasks: dict[int, tuple[Any, asyncio.Task[None]]] = {}
        self._closed = False
        self.agent_definitions = getattr(
            scheduler,
            "agent_definitions",
            getattr(builder, "agent_definitions", None),
        )
        self.selected_agent = getattr(
            scheduler,
            "agent_definition",
            getattr(builder, "agent_definition", None),
        )
        self.todo_stores_by_identity: dict[tuple[str, str], Any] = {}
        todo_store = getattr(scheduler, "todo_store", None)
        identity = getattr(todo_store, "identity", None)
        if identity is not None:
            self.todo_stores_by_identity[(identity.session_id, identity.agent_id)] = todo_store

    @classmethod
    def create(cls, builder: _SchedulerBuilder, session_id: str) -> "_SchedulerState":
        return cls(builder=builder, scheduler=builder.build(session_id))

    @property
    def scheduler(self) -> Any:
        """Return the current scheduler for non-turn UI and command plumbing."""
        return self._scheduler

    def __getattr__(self, name: str) -> Any:
        """Forward non-lifecycle scheduler APIs to the current scheduler."""
        return getattr(self._scheduler, name)

    @property
    def session_id(self) -> str:
        return self._scheduler.session.session_id

    async def dispatch_handle(self, prompt: str, *, require_success: bool = False, **kwargs) -> str:
        """Run one producer turn against the scheduler active after lock acquisition."""
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Scheduler state is closed")
            response = await self._scheduler.handle(prompt, **kwargs)
            # Interactive callers render errors as text. Durable cron callers
            # must not acknowledge that text as success and delete a one-shot job.
            if require_success and (
                getattr(self._scheduler, "_last_turn_errored", False)
                or getattr(self._scheduler, "_last_turn_cancelled", False)
            ):
                raise _UnsuccessfulTurnError(
                    cancelled=bool(getattr(self._scheduler, "_last_turn_cancelled", False))
                )
            return response

    async def dispatch_stream_json(self, prompt: str, **kwargs) -> str:
        """Run one stream-json turn under the shared lifecycle lock."""
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Scheduler state is closed")
            return await self._scheduler.handle_stream_json(prompt, **kwargs)

    async def handle(self, prompt: str, **kwargs) -> str:
        """Scheduler-compatible entry point for interactive command producers."""
        return await self.dispatch_handle(prompt, **kwargs)

    async def handle_stream_json(self, prompt: str, **kwargs) -> str:
        return await self.dispatch_stream_json(prompt, **kwargs)

    async def switch(
        self,
        session_id: str,
        *,
        prepare_target: Callable[[], Awaitable[_SessionSwitchTarget]] | None = None,
        prepare_commit: Callable[[Any, _SessionSwitchTarget], Awaitable[Any]] | None = None,
        commit: Callable[[Any, Any], None] | None = None,
        post_commit: Callable[[Any, Any], None] | None = None,
    ):
        """Build and commit a replacement while producer admission is closed.

        ``commit`` owns every external state update that must become visible with
        the replacement (cwd, CLI args, prompt/status-line state). ``post_commit``
        runs synchronously only after the replacement is authoritative, so caller
        cancellation cannot skip committed event delivery or expose it for an
        aborted switch. The old scheduler is retired only after that transaction
        commits, and retirement failures are observed separately rather than
        changing the switch result.
        """
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Scheduler state is closed")
            previous = self._scheduler
            replacement = None
            try:
                if prepare_target is None:
                    target = _SessionSwitchTarget(
                        cwd=Path.cwd(),
                        agent_name=None,
                        agent_definition=getattr(self.builder, "agent_definition", None),
                        agent_definitions=getattr(self.builder, "agent_definitions", None),
                    )
                else:
                    target = await prepare_target()
                if isinstance(self.builder, _SchedulerBuilder):
                    replacement = self.builder.build(session_id, target=target)
                else:
                    replacement = self.builder.build(session_id)
                prepared = (
                    await prepare_commit(replacement, target)
                    if prepare_commit is not None
                    else target
                )
                if commit is not None:
                    # This is the switch linearization point. It deliberately
                    # contains no await: scheduler, CLI, prompt, history, status,
                    # and cwd become visible as one lifecycle-lock transaction.
                    commit(replacement, prepared)
                self._scheduler = replacement

                # Snapshot resources owned by the old scheduler before the
                # lifecycle lock admits any work against the replacement. This
                # prevents delayed retirement from selecting resources created
                # by the new session after the switch commits.
                prepare_retirement = getattr(previous, "prepare_retirement", None)
                if callable(prepare_retirement):
                    try:
                        prepare_retirement()
                    except BaseException:
                        logger.warning(
                            "Failed to snapshot retired scheduler resources",
                            exc_info=True,
                        )

                if post_commit is not None:
                    try:
                        post_commit(replacement, prepared)
                    except BaseException:
                        # The switch is already committed. Event delivery is
                        # best-effort and must never roll the authoritative state
                        # back after externally visible hooks may have fired.
                        logger.warning(
                            "Failed to deliver committed session switch event",
                            exc_info=True,
                        )
            except BaseException:
                if replacement is not None:
                    prepare_uncommitted_cleanup = getattr(
                        replacement,
                        "prepare_uncommitted_cleanup",
                        None,
                    )
                    if callable(prepare_uncommitted_cleanup):
                        prepare_uncommitted_cleanup()
                    self._start_cleanup(replacement, reason="aborted session replacement")
                raise
            self._start_cleanup(previous, reason="retired session scheduler")
            return replacement

    async def cleanup(self) -> None:
        async with self._lifecycle_lock:
            self._closed = True
            self._start_cleanup(self._scheduler, reason="active session scheduler")
            cleanup_tasks = [task for _scheduler, task in self._cleanup_tasks.values()]
        for task in cleanup_tasks:
            await _await_task_resiliently(task)

    def _start_cleanup(self, scheduler: Any, *, reason: str) -> asyncio.Task[None]:
        key = id(scheduler)
        owned = self._cleanup_tasks.get(key)
        if owned is None:
            task = asyncio.create_task(self._retire_scheduler(scheduler, reason=reason))
            self._cleanup_tasks[key] = (scheduler, task)
        else:
            task = owned[1]
        return task

    @staticmethod
    async def _retire_scheduler(scheduler: Any, *, reason: str) -> None:
        try:
            await scheduler.cleanup()
        except BaseException:
            logger.warning("Failed to clean %s", reason, exc_info=True)


async def _await_task_resiliently(task: asyncio.Task[Any]) -> Any:
    """Wait through repeated caller cancellation, then preserve cancellation."""
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True

    try:
        result = task.result()
    except BaseException:
        if cancelled:
            raise asyncio.CancelledError from None
        raise
    if cancelled:
        raise asyncio.CancelledError
    return result


class _SessionCleanupOwner:
    """Own all session resources and finish cleanup despite repeated cancellation."""

    def __init__(
        self,
        scheduler_state: _SchedulerState,
        *,
        bare_mode: bool,
        previous_simple: str | None,
    ) -> None:
        self.scheduler_state = scheduler_state
        self.bare_mode = bare_mode
        self.previous_simple = previous_simple
        self.pr_poller: Any = None
        self.channel_task: asyncio.Task[Any] | None = None
        self.channel_inbox: Any = None
        self.unregister_channel_callback: Any = None
        self.cron_prompt_runner: Any = None
        self.agent_service: Any = None
        self.teammate_runner: Any = None
        self.skip_auto_dream = False
        self._task: asyncio.Task[None] | None = None

    async def finish(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        await _await_task_resiliently(self._task)

    async def _run(self) -> None:
        try:
            await self._guard("PR status poller shutdown", self._stop_pr_poller)
            await self._guard("Channel callback unregister", self._unregister_channel_callback)
            await self._guard("Channel task cancellation", self._stop_channel_task)
            await self._guard("Channel inbox finalization", self._close_channel_inbox)
            await self._guard("Cron prompt runner shutdown", self._stop_cron_runner)
            await self._guard("Teammate runner shutdown", self._stop_teammates)
            await self._guard("Agent service shutdown", self._stop_agents)
            await self._guard("SessionEnd hook dispatch", self._dispatch_session_end)
            await self._guard("AutoDream memory consolidation", self._run_auto_dream)
            await self._guard("Scheduler cleanup", self.scheduler_state.cleanup)
        finally:
            self._restore_bare_mode()

    async def _guard(self, label: str, action) -> None:
        try:
            result = action()
            if inspect.isawaitable(result):
                await result
        except BaseException:
            logger.debug("%s failed", label, exc_info=True)

    def _stop_pr_poller(self) -> None:
        if self.pr_poller is not None:
            self.pr_poller.stop()

    def _unregister_channel_callback(self) -> None:
        if self.unregister_channel_callback is not None:
            self.unregister_channel_callback()
            self.unregister_channel_callback = None

    async def _stop_channel_task(self) -> None:
        if self.channel_task is None:
            return
        self.channel_task.cancel()
        results = await asyncio.gather(self.channel_task, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                logger.debug(
                    "Channel task cancellation failed",
                    exc_info=(type(result), result, result.__traceback__),
                )
        self.channel_task = None

    async def _close_channel_inbox(self) -> None:
        if self.channel_inbox is not None:
            await self.channel_inbox.aclose()
            self.channel_inbox = None

    async def _stop_cron_runner(self) -> None:
        if self.cron_prompt_runner is not None:
            await self.cron_prompt_runner.stop()

    async def _stop_teammates(self) -> None:
        if self.teammate_runner is not None:
            await self.teammate_runner.aclose()

    async def _stop_agents(self) -> None:
        if self.agent_service is not None:
            await self.agent_service.aclose()

    def _dispatch_session_end(self) -> None:
        dispatch_command_hooks(
            cwd=Path.cwd(),
            event_name="SessionEnd",
            match_value="other",
            payload={
                "event": "SessionEnd",
                "reason": "other",
                "session_id": self.scheduler_state.session_id,
            },
        )

    async def _run_auto_dream(self) -> None:
        from koder_agent.harness.memory.auto_dream import (
            AutoDreamManager,
            DreamConfig,
            default_auto_dream_task_storage,
            run_auto_dream_from_messages,
        )

        harness_config = getattr(get_config(), "harness", None)
        write_mode = getattr(harness_config, "auto_dream_write_mode", "review")
        dream_manager = AutoDreamManager(
            config=DreamConfig(write_mode=write_mode),
            state_path=Path.home() / ".koder" / "auto_dream_state.json",
        )
        dream_manager.record_session()
        try:
            if self.skip_auto_dream or not dream_manager.should_dream():
                dream_manager.save()
                return
            result = await asyncio.wait_for(
                run_auto_dream_from_messages(
                    await self.scheduler_state.scheduler.session.get_items(),
                    manager=dream_manager,
                    origin_project_root=Path.cwd(),
                    origin_session_id=self.scheduler_state.session_id,
                    task_storage=default_auto_dream_task_storage(),
                ),
                timeout=AUTO_DREAM_SHUTDOWN_TIMEOUT_SECONDS,
            )
            logger.info(
                "AutoDream completed: memories=%d saved=%s errors=%d",
                result.memories_written,
                str(result.saved_path) if result.saved_path else "none",
                len(result.errors),
            )
        except BaseException:
            try:
                dream_manager.save()
            except BaseException:
                logger.debug("Failed to save dream manager state", exc_info=True)
            raise

    def _restore_bare_mode(self) -> None:
        if not self.bare_mode:
            return
        if self.previous_simple is None:
            os.environ.pop("KODER_SIMPLE", None)
        else:
            os.environ["KODER_SIMPLE"] = self.previous_simple


def _parse_session_switch(slash_response: str) -> tuple[str, bool] | None:
    if slash_response.startswith("session_switch_clear:"):
        return slash_response.split(":", 1)[1], True
    if slash_response.startswith("session_switch:"):
        return slash_response.split(":", 1)[1], False
    return None


async def _close_session_probe(session: Any) -> None:
    """Close a temporary SQLite session without letting cancellation skip ownership."""
    close = getattr(session, "close", None)
    if not callable(close):
        return

    async def close_owned() -> None:
        if inspect.iscoroutinefunction(close):
            await close()
        else:
            error: list[BaseException] = []

            def worker() -> None:
                try:
                    close()
                except BaseException as exc:
                    error.append(exc)

            thread = threading.Thread(
                target=worker,
                name="koder-session-probe-close",
                daemon=False,
            )
            thread.start()
            while thread.is_alive():
                await asyncio.sleep(0.01)
            thread.join()
            if error:
                raise error[0]

    task = asyncio.create_task(close_owned())
    await _await_task_resiliently(task)


def _resolve_hook_watch_paths(
    paths: list[str] | None,
    *,
    base_dir: Path,
) -> list[str]:
    """Resolve hook watch paths relative to the directory that dispatched them."""
    resolved_paths: list[str] = []
    for raw in paths or []:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        resolved_paths.append(str(candidate.resolve()))
    return resolved_paths


def _change_working_directory(target: Path) -> str | None:
    """Commit one cwd change, then deliver its hook from the old trust root."""
    target = target.expanduser().resolve()
    previous = Path.cwd().resolve()
    if not target.is_dir() or target == previous:
        return None

    os.chdir(target)
    hook_result = dispatch_command_hooks(
        cwd=previous,
        event_name="CwdChanged",
        match_value=None,
        payload={
            "event": "CwdChanged",
            "old_cwd": str(previous),
            "cwd": str(target),
        },
    )
    update_watch_paths(
        _resolve_hook_watch_paths(
            hook_result.watch_paths,
            base_dir=previous,
        )
    )
    return str(target)


async def restore_session_cwd(session_id: str) -> str | None:
    """Restore a recorded cwd with exact temporary-session cleanup."""
    from koder_agent.core.session import EnhancedSQLiteSession

    probe = EnhancedSQLiteSession(session_id=session_id)
    try:
        recorded = await probe.get_cwd()
    finally:
        await _close_session_probe(probe)
    if not recorded:
        return None
    target = Path(recorded).expanduser()
    if not target.is_absolute() or not target.is_dir():
        return None
    return _change_working_directory(target)


def _todo_store_matches(todo_store, *, session_id: str, agent_id: str) -> bool:
    identity = getattr(todo_store, "identity", None)
    return (
        identity is not None and identity.session_id == session_id and identity.agent_id == agent_id
    )


async def _prepare_session_switch_target(
    builder: _SchedulerBuilder,
    session_id: str,
    *,
    todo_stores_by_identity: dict[tuple[str, str], Any] | None = None,
) -> _SessionSwitchTarget:
    """Resolve target cwd and agent without changing visible state."""
    from koder_agent.core.session import EnhancedSQLiteSession

    probe = EnhancedSQLiteSession(session_id=session_id)
    try:
        recorded_cwd = await probe.get_cwd()
        get_agent = getattr(probe, "get_agent", None)
        if callable(get_agent):
            agent_name = await get_agent()
        else:
            get_session_agent = getattr(EnhancedSQLiteSession, "get_session_agent", None)
            agent_name = (
                await get_session_agent(session_id) if callable(get_session_agent) else None
            )
    finally:
        await _close_session_probe(probe)

    if recorded_cwd:
        target_cwd = Path(recorded_cwd).expanduser()
        if not target_cwd.is_absolute() or not target_cwd.is_dir():
            raise ValueError(
                "Cannot switch sessions because its recorded directory is unavailable: "
                f"{recorded_cwd}"
            )
        target_cwd = target_cwd.resolve()
    else:
        target_cwd = Path.cwd().resolve()

    load_definitions = getattr(builder, "load_agent_definitions", None)
    if callable(load_definitions):
        agent_definitions = load_definitions(target_cwd)
    else:
        agent_definitions = getattr(builder, "agent_definitions", None)

    agent_definition = None
    if agent_name:
        if agent_definitions is None:
            raise ValueError(
                f"Session agent '{agent_name}' cannot be resolved without target definitions."
            )
        agent_definition = next(
            (
                definition
                for definition in agent_definitions.active_agents
                if definition.agent_type == agent_name
            ),
            None,
        )
        if agent_definition is None:
            raise ValueError(
                "Cannot switch sessions because its recorded agent is unavailable: "
                f"{agent_name} (target project: {target_cwd})."
            )

    agent_id = agent_definition.agent_type if agent_definition is not None else "main"
    todo_key = (session_id, agent_id)
    todo_store = (todo_stores_by_identity or {}).get(todo_key)
    if todo_store is not None and not _todo_store_matches(
        todo_store,
        session_id=session_id,
        agent_id=agent_id,
    ):
        raise ValueError("Target TodoStore identity does not match its session and agent")
    return _SessionSwitchTarget(
        cwd=target_cwd,
        agent_name=agent_name,
        agent_definition=agent_definition,
        agent_definitions=agent_definitions,
        todo_store=todo_store,
        todo_key=todo_key,
    )


async def _switch_active_session(
    scheduler_state: _SchedulerState,
    args: Any,
    session_id: str,
    *,
    interactive_prompt: Any = None,
    clear_viewport: bool = False,
) -> Any:
    """Switch the owned scheduler and update common CLI/UI session state."""
    committed_cwd: str | None = None
    committed_previous_cwd: Path | None = None

    async def prepare_target() -> _SessionSwitchTarget:
        return await _prepare_session_switch_target(
            scheduler_state.builder,
            session_id,
            todo_stores_by_identity=scheduler_state.todo_stores_by_identity,
        )

    async def prepare_commit(
        scheduler: Any,
        target: _SessionSwitchTarget,
    ) -> _SessionSwitchCommit:
        display_name = None
        if interactive_prompt is not None and interactive_prompt.status_line:
            try:
                display_name = await scheduler.session.get_display_name()
            except Exception:
                logger.debug(
                    "Failed to prepare display name for session switch",
                    exc_info=True,
                )
        return _SessionSwitchCommit(target=target, display_name=display_name)

    def commit(scheduler: Any, prepared: _SessionSwitchCommit) -> None:
        nonlocal committed_cwd, committed_previous_cwd
        missing = object()
        previous_cwd = Path.cwd()
        previous_args_session = getattr(args, "session", missing)
        previous_args_agent = getattr(args, "agent", missing)
        status_line = getattr(interactive_prompt, "status_line", None)
        previous_history = getattr(interactive_prompt, "history", missing)
        auto_suggest = getattr(interactive_prompt, "auto_suggest", None)
        previous_auto_history = (
            list(auto_suggest._history) if hasattr(auto_suggest, "_history") else missing
        )
        previous_speculative = getattr(auto_suggest, "_speculative_suggestion", missing)
        previous_status_session = getattr(status_line, "session_id", missing)
        previous_status_display = getattr(status_line, "_display_name", missing)
        previous_usage_tracker = getattr(status_line, "usage_tracker", missing)
        previous_agent_definitions = scheduler_state.agent_definitions
        previous_selected_agent = scheduler_state.selected_agent
        previous_target_todo = (
            scheduler_state.todo_stores_by_identity.get(prepared.target.todo_key)
            if prepared.target.todo_key is not None
            else missing
        )
        at_completer = getattr(interactive_prompt, "at_completer", None)
        previous_agent_names = (
            list(getattr(at_completer, "_agent_names", [])) if at_completer is not None else missing
        )
        previous_file_index = (
            getattr(at_completer, "_file_index", missing) if at_completer is not None else missing
        )

        try:
            replacement_agent_id = (
                prepared.target.agent_definition.agent_type
                if prepared.target.agent_definition is not None
                else "main"
            )
            if not _todo_store_matches(
                scheduler.todo_store,
                session_id=session_id,
                agent_id=replacement_agent_id,
            ):
                raise ValueError("Replacement TodoStore identity is invalid")
            if prepared.target.cwd is not None and prepared.target.cwd != previous_cwd:
                os.chdir(prepared.target.cwd)
                committed_cwd = str(prepared.target.cwd)
                committed_previous_cwd = previous_cwd
            if interactive_prompt is not None:
                interactive_prompt.update_session(session_id)
                interactive_prompt.reset_history()
                if status_line is not None:
                    status_line.usage_tracker = scheduler.usage_tracker
                    if prepared.display_name is not None:
                        status_line.update_display_name(prepared.display_name)
                if at_completer is not None:
                    from koder_agent.core.file_index import ProjectFileIndex

                    at_completer._file_index = ProjectFileIndex(
                        cwd=prepared.target.cwd or previous_cwd
                    )
                    at_completer.update_agents(
                        [
                            (definition.agent_type, (definition.when_to_use or "")[:60])
                            for definition in prepared.target.agent_definitions.active_agents
                        ]
                    )
            args.session = session_id
            if hasattr(args, "agent"):
                args.agent = prepared.target.agent_name
            scheduler.agent_definitions = prepared.target.agent_definitions
            scheduler_state.agent_definitions = prepared.target.agent_definitions
            scheduler_state.selected_agent = prepared.target.agent_definition
            if prepared.target.todo_key is not None:
                scheduler_state.todo_stores_by_identity[prepared.target.todo_key] = (
                    scheduler.todo_store
                )
        except BaseException:
            if Path.cwd() != previous_cwd:
                os.chdir(previous_cwd)
            if previous_args_session is missing:
                vars(args).pop("session", None)
            else:
                args.session = previous_args_session
            if previous_args_agent is missing:
                vars(args).pop("agent", None)
            else:
                args.agent = previous_args_agent
            if interactive_prompt is not None and previous_history is not missing:
                interactive_prompt.history = previous_history
            if auto_suggest is not None and previous_auto_history is not missing:
                auto_suggest._history[:] = previous_auto_history
            if auto_suggest is not None and previous_speculative is not missing:
                auto_suggest._speculative_suggestion = previous_speculative
            if status_line is not None:
                if previous_status_session is not missing:
                    status_line.session_id = previous_status_session
                if previous_status_display is not missing:
                    status_line._display_name = previous_status_display
                if previous_usage_tracker is not missing:
                    status_line.usage_tracker = previous_usage_tracker
            if at_completer is not None and previous_agent_names is not missing:
                at_completer.update_agents(previous_agent_names)
            if at_completer is not None and previous_file_index is not missing:
                at_completer._file_index = previous_file_index
            scheduler_state.agent_definitions = previous_agent_definitions
            scheduler_state.selected_agent = previous_selected_agent
            if prepared.target.todo_key is not None:
                if previous_target_todo is missing:
                    scheduler_state.todo_stores_by_identity.pop(prepared.target.todo_key, None)
                else:
                    scheduler_state.todo_stores_by_identity[prepared.target.todo_key] = (
                        previous_target_todo
                    )
            committed_cwd = None
            committed_previous_cwd = None
            raise

    def post_commit(_scheduler: Any, prepared: _SessionSwitchCommit) -> None:
        if prepared.target.cwd is None or committed_previous_cwd is None:
            return
        hook_result = dispatch_command_hooks(
            cwd=committed_previous_cwd,
            event_name="CwdChanged",
            match_value=None,
            payload={
                "event": "CwdChanged",
                "old_cwd": str(committed_previous_cwd),
                "cwd": str(prepared.target.cwd),
            },
        )
        update_watch_paths(
            _resolve_hook_watch_paths(
                hook_result.watch_paths,
                base_dir=committed_previous_cwd,
            )
        )

    scheduler = await scheduler_state.switch(
        session_id,
        prepare_target=prepare_target,
        prepare_commit=prepare_commit,
        commit=commit,
        post_commit=post_commit,
    )

    if interactive_prompt is not None:
        try:
            if clear_viewport:
                _clear_interactive_viewport()
            if committed_cwd:
                print_reflowable(
                    console,
                    f"[dim]Working directory restored: {committed_cwd}[/dim]",
                )
        except Exception:
            logger.debug("Failed to render committed session switch", exc_info=True)

    return scheduler


def _clear_interactive_viewport() -> None:
    get_reflow_buffer().clear()
    console.file.write(CLEAR_VIEWPORT)
    console.file.flush()


async def _list_sessions_for_cwd_ids() -> set[str]:
    """Return session ids that were last recorded in the current directory."""
    from koder_agent.core.session import EnhancedSQLiteSession

    ids: set[str] = set()
    try:
        all_sessions = await EnhancedSQLiteSession.list_sessions_with_titles()
        cwd = os.getcwd()
        for sid, _title in all_sessions:
            try:
                session_cwd = await EnhancedSQLiteSession(session_id=sid).get_cwd()
            except Exception:
                session_cwd = None
            if session_cwd == cwd:
                ids.add(sid)
    except Exception:
        logger.debug("Failed to scope sessions to cwd", exc_info=True)
    return ids


async def prompt_select_session(
    current_session_id: str | None = None, *, all_dirs: bool = True
) -> Optional[str]:
    from koder_agent.core.session import EnhancedSQLiteSession
    from koder_agent.utils import parse_session_dt, picker_arrows_with_titles

    sessions_with_titles = await EnhancedSQLiteSession.list_sessions_with_titles()
    if current_session_id is not None:
        sessions_with_titles = [
            (sid, title) for sid, title in sessions_with_titles if sid != current_session_id
        ]
    if not all_dirs:
        cwd_ids = await _list_sessions_for_cwd_ids()
        scoped = [(sid, title) for sid, title in sessions_with_titles if sid in cwd_ids]
        # Fall back to all sessions when nothing is scoped to this directory.
        if scoped:
            sessions_with_titles = scoped
    if not sessions_with_titles:
        console.print(Panel("No sessions found.", title="Sessions", border_style="yellow"))
        return None

    sessions_with_titles.sort(
        key=lambda x: (parse_session_dt(x[0])[0], parse_session_dt(x[0])[1] or None),
        reverse=True,
    )
    return picker_arrows_with_titles(sessions_with_titles)


async def _resolve_resume_value(candidate: str) -> Optional[str]:
    """Resolve a `--resume <value>` argument to a session id.

    Returns the value directly when it matches a known session id. Otherwise
    falls back to a unique title match. Returns None when no match is found or
    when the title is ambiguous.
    """
    from koder_agent.core.session import EnhancedSQLiteSession

    sessions = await EnhancedSQLiteSession.list_sessions_with_titles()
    session_ids = {sid for sid, _title in sessions}
    if candidate in session_ids:
        return candidate
    title_matches = [sid for sid, title in sessions if title == candidate]
    if len(title_matches) == 1:
        return title_matches[0]
    return None


async def load_context() -> str:
    context_info = [f"Working directory: {os.getcwd()}"]
    agents_md_path = Path(os.getcwd()) / "AGENTS.md"
    if agents_md_path.exists():
        try:
            agents_content = agents_md_path.read_text("utf-8", errors="ignore")
            hook_result = dispatch_command_hooks(
                cwd=Path.cwd(),
                event_name="InstructionsLoaded",
                match_value="session_start",
                payload={
                    "event": "InstructionsLoaded",
                    "reason": "session_start",
                    "file_path": str(agents_md_path.resolve()),
                },
            )
            update_watch_paths(hook_result.watch_paths)
            if not hook_result.blocked:
                context_info.append(f"AGENTS.md content:\n{agents_content}")
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            context_info.append(f"Error reading AGENTS.md: {exc}")
    return "\n\n".join(context_info)


def _read_piped_stdin() -> Optional[str]:
    if sys.stdin.isatty():
        return None

    stdin_text = sys.stdin.read()
    if stdin_text == "":
        return None
    return stdin_text


def _read_stream_json_messages() -> list[dict]:
    if sys.stdin.isatty():
        return []

    stdin_text = sys.stdin.read()
    if stdin_text == "":
        return []

    messages: list[dict] = []
    for line in stdin_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --input-format stream-json payload: {exc.msg}.") from exc
        if payload.get("type") != "user":
            continue
        message = payload.get("message")
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        messages.append(payload)
    return messages


def _build_prompt_text(
    *, prompt_text: Optional[list[str]], stdin_text: Optional[str]
) -> Optional[str]:
    prompt = None
    if prompt_text:
        prompt = " ".join(prompt_text).strip() or None

    if prompt and stdin_text:
        return f"{prompt}\n\nStdin content:\n{stdin_text}"
    if prompt:
        return prompt
    return stdin_text


def _build_stream_json_prompt(messages: list[dict]) -> Optional[str]:
    if not messages:
        return None

    parts: list[str] = []
    for payload in messages:
        content = payload.get("message", {}).get("content")
        if isinstance(content, str):
            parts.append(content)
        elif content is not None:
            parts.append(json.dumps(content, ensure_ascii=False))

    combined = "\n\n".join(part for part in parts if part)
    return combined or None


def _load_json_schema(schema_source: Optional[str]) -> Optional[dict]:
    if not schema_source:
        return None

    raw_source = schema_source.strip()
    schema_path = Path(raw_source).expanduser()
    if schema_path.exists() and schema_path.is_file():
        schema_text = schema_path.read_text("utf-8")
    else:
        schema_text = raw_source

    try:
        schema = json.loads(schema_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --json-schema value: {exc.msg}.") from exc

    if not isinstance(schema, dict):
        raise ValueError("--json-schema must decode to a JSON object.")
    return schema


def _load_prompt_text(
    text: Optional[str], file_path: Optional[str], *, label: str
) -> Optional[str]:
    if text is not None:
        return text
    if file_path is None:
        return None
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label} file does not exist: {path}")
    return path.read_text(encoding="utf-8")


def _augment_prompt_for_json_schema(prompt: str, schema: dict) -> str:
    schema_text = json.dumps(schema, ensure_ascii=False)
    return (
        f"{prompt}\n\n"
        "Return only valid JSON that matches this schema exactly.\n"
        "Do not wrap the JSON in markdown fences or add extra commentary.\n"
        f"JSON Schema:\n{schema_text}"
    )


def _extract_structured_output(result: str, schema: dict):
    from jsonschema import ValidationError, validate

    decoder = json.JSONDecoder()
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    stripped = text.strip()
    candidates: list[str] = []

    fence_matches = re.findall(
        r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE
    )
    candidates.extend(match.strip() for match in fence_matches if match.strip())
    if stripped:
        candidates.append(stripped)

    parsed = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError:
            pass

        for marker in ("{", "["):
            start = candidate.find(marker)
            if start == -1:
                continue
            try:
                parsed, end = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if candidate[start + end :].strip():
                parsed = None
                continue
            break
        if parsed is not None:
            break

    if parsed is None:
        raise ValueError("Response did not contain valid JSON for --json-schema.")

    try:
        validate(parsed, schema)
    except ValidationError as exc:
        raise ValueError(f"Response did not match --json-schema: {exc.message}") from exc
    return parsed


def _build_usage_payload(scheduler) -> Optional[dict]:
    usage_tracker = getattr(scheduler, "usage_tracker", None)
    if usage_tracker is None:
        return None

    usage_payload: dict[str, object] = {}
    model = getattr(usage_tracker, "model", None)
    if model:
        usage_payload["model"] = model

    session_usage = getattr(usage_tracker, "session_usage", None)
    if session_usage is not None:
        usage_payload["requests"] = getattr(session_usage, "request_count", 0)
        usage_payload["input_tokens"] = getattr(session_usage, "input_tokens", 0)
        usage_payload["output_tokens"] = getattr(session_usage, "output_tokens", 0)
        usage_payload["context_tokens"] = getattr(session_usage, "current_context_tokens", 0)
        usage_payload["cost_usd"] = getattr(session_usage, "total_cost", 0.0)

    return usage_payload or None


def _write_json_line(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


async def _replay_stream_json_messages(*, scheduler, messages: list[dict]) -> None:
    session_id = scheduler.session.session_id
    for payload in messages:
        replay_payload = {
            "type": "user",
            "message": payload["message"],
            "session_id": session_id,
            "parent_tool_use_id": payload.get("parent_tool_use_id"),
            "isReplay": True,
        }
        if payload.get("uuid") is not None:
            replay_payload["uuid"] = payload["uuid"]
        if payload.get("timestamp") is not None:
            replay_payload["timestamp"] = payload["timestamp"]
        _write_json_line(replay_payload)


def _turn_exit_code(scheduler) -> int:
    """Snapshot turn status without interpreting the rendered response text."""
    if getattr(scheduler, "_last_turn_cancelled", False) is True:
        return 130
    if getattr(scheduler, "_last_turn_errored", False) is True:
        return 1
    return 0


async def _print_json_output(
    *, scheduler, result: str, output_format: str, structured_output=None, exit_code: int = 0
) -> int:
    payload = {
        "output_format": output_format,
        "session_id": scheduler.session.session_id,
        "display_name": await scheduler.session.get_display_name(),
        "result": result,
    }
    usage_payload = _build_usage_payload(scheduler)
    if usage_payload is not None:
        payload["usage"] = usage_payload
    if structured_output is not None:
        payload["structured_output"] = structured_output
    if exit_code:
        payload["is_error"] = True
        payload["exit_code"] = exit_code
    if output_format == "stream-json":
        payload["type"] = "result"
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return exit_code


AUTO_DREAM_SHUTDOWN_TIMEOUT_SECONDS = 30


async def run_harness_session_flow(
    *,
    first_arg: Optional[str],
    argv: list[str],
    permission_service=None,
) -> int:
    from koder_agent.core.session import EnhancedSQLiteSession

    if first_arg in DIRECT_COMMAND_PASSTHROUGHS and argv == [first_arg]:
        command_handler = HarnessInteractiveCommandHandler(emit_console=False)
        result = await command_handler.handle_slash_input(f"/{first_arg}", scheduler=None)
        if result:
            sys.stdout.write(result + "\n")
        return 0

    from koder_agent.cli import _build_cli_parser

    parser = _build_cli_parser(first_arg)
    args = parser.parse_args(argv)
    if getattr(args, "command", None) == "config":
        from koder_agent.harness.config.commands import handle_config_subcommand

        return await handle_config_subcommand(args)

    is_subcommand = first_arg in {
        "config",
        "auth",
        "mcp",
        "agents",
        "plugin",
        "plugins",
        "doctor",
        "review",
        "completion",
        "upgrade",
    }
    if not is_subcommand:
        from koder_agent.utils import setup_openai_client

        try:
            await run_sync_owned(setup_openai_client)
        except ValueError as exc:
            console.print(Panel(f"[red]{exc}[/red]", title="Error", border_style="red"))
            return 1

    config_manager = get_config_manager()
    config = get_config()
    try:
        from koder_agent.harness.memory.auto_dream import (
            default_auto_dream_task_storage,
            reconcile_stale_auto_dream_tasks,
        )

        reconcile_stale_auto_dream_tasks(default_auto_dream_task_storage())
    except Exception:
        logger.debug("AutoDream startup reconciliation failed", exc_info=True)

    bare_mode = bool(getattr(args, "bare", False))
    previous_simple = os.environ.get("KODER_SIMPLE")
    if bare_mode:
        os.environ["KODER_SIMPLE"] = "1"

    cli_agents_json = None
    raw_agents = getattr(args, "agents", None)
    if raw_agents:
        try:
            cli_agents_json = json.loads(raw_agents)
        except json.JSONDecodeError as exc:
            console.print(
                Panel(
                    f"Invalid --agents JSON: {exc.msg}.",
                    title="Error",
                    border_style="red",
                )
            )
            return 1

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled")
        logging.debug("%s", "\n".join(get_litellm_cost_map_debug_lines()))

    if not hasattr(args, "command"):
        args.command = None

    plugin_dirs = list(getattr(args, "plugin_dir", []) or [])
    effective_plugin_root = default_plugin_root()
    if plugin_dirs:
        try:
            effective_plugin_root = build_session_plugin_root(
                getattr(args, "session", None) or first_arg or "adhoc",
                plugin_dirs,
            )
        except ValueError as exc:
            console.print(Panel(str(exc), title="Error", border_style="red"))
            return 1

    try:
        system_prompt_override = _load_prompt_text(
            getattr(args, "system_prompt", None),
            getattr(args, "system_prompt_file", None),
            label="System prompt",
        )
        if getattr(args, "system_prompt", None) and getattr(args, "system_prompt_file", None):
            raise ValueError("--system-prompt and --system-prompt-file are mutually exclusive.")
        append_segments = [
            _load_prompt_text(
                getattr(args, "append_system_prompt", None),
                None,
                label="Append system prompt",
            ),
            _load_prompt_text(
                None,
                getattr(args, "append_system_prompt_file", None),
                label="Append system prompt",
            ),
        ]
    except ValueError as exc:
        console.print(Panel(str(exc), title="Error", border_style="red"))
        return 1
    system_prompt_append = "\n\n".join(segment.strip() for segment in append_segments if segment)
    if not system_prompt_append:
        system_prompt_append = None

    if args.command == "auth":
        from koder_agent.harness.auth.commands import handle_auth_subcommand

        return await handle_auth_subcommand(args)

    if args.command == "agents":
        from koder_agent.harness.agents.commands import handle_agents_subcommand

        return await handle_agents_subcommand(args)

    if args.command in {"plugin", "plugins"}:
        from koder_agent.harness.plugins.commands import handle_plugin_subcommand

        return await handle_plugin_subcommand(args)

    if args.command == "doctor":
        from koder_agent.harness.cli.headless import handle_doctor_command

        return await handle_doctor_command(args)

    if args.command == "review":
        from koder_agent.harness.cli.headless import handle_review_command

        return await handle_review_command(args)

    if args.command == "completion":
        from koder_agent.harness.cli.headless import handle_completion_command

        return handle_completion_command(args)

    if args.command == "upgrade":
        from koder_agent.harness.cli.headless import handle_upgrade_command

        return await handle_upgrade_command(args)

    if getattr(args, "continue_session", False):
        continued_session_id = await EnhancedSQLiteSession.get_most_recent_session_for_cwd(
            os.getcwd()
        )
        if not continued_session_id:
            console.print(
                Panel(
                    "No saved session found for the current directory.",
                    title="Error",
                    border_style="red",
                )
            )
            return 1
        args.session = continued_session_id

    resume_value = getattr(args, "resume", None)
    if isinstance(resume_value, str) and resume_value.strip():
        candidate = resume_value.strip()
        resolved = await _resolve_resume_value(candidate)
        if resolved is None:
            console.print(
                Panel(
                    f"No session found matching '{candidate}' (by id or title).",
                    title="Error",
                    border_style="red",
                )
            )
            return 1
        args.session = resolved
    elif resume_value:
        if not getattr(args, "session", None) and (
            not sys.stdin.isatty() or not sys.stdout.isatty()
        ):
            console.print(
                Panel(
                    "--resume requires an interactive terminal. "
                    "Use --session <id> to target a known session non-interactively.",
                    title="Error",
                    border_style="red",
                )
            )
            return 1
        from koder_agent.utils import default_session_local_ms

        selected = await prompt_select_session(all_dirs=bool(getattr(args, "resume_all", False)))
        if selected:
            args.session = selected
        elif not getattr(args, "session", None):
            args.session = (
                config_manager.get_effective_value(config.cli.session, None)
                or default_session_local_ms()
            )

    if not getattr(args, "session", None):
        from koder_agent.utils import default_session_local_ms

        args.session = (
            config_manager.get_effective_value(config.cli.session, None)
            or default_session_local_ms()
        )

    startup_cwd = Path.cwd().resolve()
    startup_probe = EnhancedSQLiteSession(session_id=args.session)
    try:
        recorded_cwd = await startup_probe.get_cwd()
        get_agent = getattr(startup_probe, "get_agent", None)
        if callable(get_agent):
            recorded_agent_name = await get_agent()
        else:
            get_session_agent = getattr(EnhancedSQLiteSession, "get_session_agent", None)
            recorded_agent_name = (
                await get_session_agent(args.session) if callable(get_session_agent) else None
            )
    finally:
        await _close_session_probe(startup_probe)

    target_cwd = startup_cwd
    if recorded_cwd:
        recorded_path = Path(recorded_cwd).expanduser()
        if not recorded_path.is_absolute() or not recorded_path.is_dir():
            console.print(
                Panel(
                    f"Cannot resume session because its recorded directory is unavailable: "
                    f"{recorded_cwd}",
                    title="Error",
                    border_style="red",
                )
            )
            return 1
        target_cwd = recorded_path.resolve()

    selected_agent_name = getattr(args, "agent", None)
    if not selected_agent_name:
        selected_agent_name = recorded_agent_name
    if not selected_agent_name:
        selected_agent_name = get_configured_agent_name(str(target_cwd))

    selected_agent = None
    agent_definitions = get_agent_definitions(
        cwd=target_cwd,
        plugin_root=effective_plugin_root,
        cli_agents_json=cli_agents_json,
    )
    if selected_agent_name:
        selected_agent = next(
            (
                agent
                for agent in agent_definitions.active_agents
                if agent.agent_type == selected_agent_name
            ),
            None,
        )
        if selected_agent is None:
            console.print(
                Panel(
                    f"Unknown --agent value: {selected_agent_name}.",
                    title="Error",
                    border_style="red",
                )
            )
            return 1

    if target_cwd != startup_cwd:
        _change_working_directory(target_cwd)
    if selected_agent_name:
        await EnhancedSQLiteSession.record_session_agent(args.session, selected_agent_name)

    # Metadata is updated only after the target project and recorded agent have
    # both resolved successfully.  A failed resume therefore cannot rewrite
    # the saved cwd to the directory from which Koder was invoked.
    await EnhancedSQLiteSession.record_session_cwd(args.session, os.getcwd())

    # --- Channel setup ---
    from koder_agent.harness.channels.state import (
        reset_channel_state,
        set_allowed_channels,
        set_has_dev_channels,
    )
    from koder_agent.harness.channels.types import parse_channel_entries

    reset_channel_state()
    channel_entries = []

    raw_channels = getattr(args, "channels", None)
    if raw_channels:
        entries_raw = [e.strip() for e in raw_channels.split(",") if e.strip()]
        channel_entries = parse_channel_entries(entries_raw, "--channels")

    raw_dev = getattr(args, "dev_channels", None)
    if raw_dev:
        from dataclasses import replace

        dev_raw = [e.strip() for e in raw_dev.split(",") if e.strip()]
        dev_entries = [
            replace(e, dev=True)
            for e in parse_channel_entries(dev_raw, "--dangerously-load-development-channels")
        ]
        channel_entries = channel_entries + dev_entries
        set_has_dev_channels(True)

    if channel_entries:
        set_allowed_channels(channel_entries)

    if args.command == "mcp":
        from koder_agent.harness.mcp.commands import handle_mcp_subcommand

        return await handle_mcp_subcommand(args)

    context = "" if bare_mode else await load_context()

    # Check onboarding state (best-effort, non-blocking)
    if not bare_mode:
        try:
            from koder_agent.harness.onboarding import check_onboarding_state, get_onboarding_steps

            effective_env = os.environ.copy()
            effective_env.update(load_session_env(args.session))
            onboarding_state = check_onboarding_state(Path.cwd(), env=effective_env)
            if not onboarding_state.completed:
                missing_steps = get_onboarding_steps(onboarding_state)
                if missing_steps:
                    dispatch_command_hooks(
                        cwd=Path.cwd(),
                        event_name="Setup",
                        match_value=None,
                        payload={
                            "event": "Setup",
                            "missing_steps": missing_steps,
                        },
                    )
                    # Show onboarding panel to user
                    console.print(
                        Panel(
                            "\n".join(f"  • {step}" for step in missing_steps),
                            title="[yellow]Setup Recommended[/yellow]",
                            border_style="yellow",
                        )
                    )
        except Exception as error:
            logger.debug(
                "Startup onboarding check failed exception_type=%s",
                type(error).__name__,
            )

    streaming = config.cli.stream and not args.no_stream
    input_format = getattr(args, "input_format", "text")
    if input_format == "stream-json" and getattr(args, "output_format", "text") != "stream-json":
        console.print(
            Panel(
                "--input-format=stream-json requires --output-format=stream-json.",
                title="Error",
                border_style="red",
            )
        )
        return 1
    if getattr(args, "replay_user_messages", False):
        if input_format != "stream-json" or getattr(args, "output_format", "text") != "stream-json":
            console.print(
                Panel(
                    "--replay-user-messages requires both --input-format=stream-json and --output-format=stream-json.",
                    title="Error",
                    border_style="red",
                )
            )
            return 1
        if not getattr(args, "verbose", False):
            console.print(
                Panel(
                    "--replay-user-messages requires --verbose.",
                    title="Error",
                    border_style="red",
                )
            )
            return 1

    stdin_text = None
    stream_json_messages: list[dict] = []
    if input_format == "stream-json":
        try:
            stream_json_messages = _read_stream_json_messages()
        except ValueError as exc:
            console.print(Panel(str(exc), title="Error", border_style="red"))
            return 1
    else:
        stdin_text = _read_piped_stdin()

    from koder_agent.core.file_index import ProjectFileIndex
    from koder_agent.core.interactive import InteractivePrompt
    from koder_agent.core.scheduler import AgentScheduler
    from koder_agent.harness.permissions.interactive_approver import build_interactive_approver

    interactive_stdin = False
    try:
        interactive_stdin = sys.stdin.isatty()
    except Exception:
        interactive_stdin = False

    def _wire_interactive_approver(active_scheduler) -> None:
        if not interactive_stdin:
            return
        queued_input = getattr(active_scheduler, "queued_input", None)
        approval_broker = getattr(queued_input, "approval_broker", None)
        active_scheduler.approver = build_interactive_approver(approval_broker=approval_broker)

    from koder_agent.harness.agents.service import AgentService

    app_agent_service = AgentService(permission_service=permission_service)
    scheduler_builder = _SchedulerBuilder(
        scheduler_type=AgentScheduler,
        streaming=streaming,
        agent_definition=selected_agent,
        instructions_override=system_prompt_override,
        instructions_append=system_prompt_append,
        permission_service=permission_service,
        approver=None,
        agent_definitions=agent_definitions,
        plugin_root=effective_plugin_root,
        cli_agents_json=cli_agents_json,
        wire_approver=_wire_interactive_approver,
        agent_service=app_agent_service,
    )
    scheduler_state = _SchedulerState.create(scheduler_builder, args.session)
    scheduler = scheduler_state.scheduler
    cleanup_owner = _SessionCleanupOwner(
        scheduler_state,
        bare_mode=bare_mode,
        previous_simple=previous_simple,
    )
    cleanup_owner.agent_service = app_agent_service

    # Wire channel messages into scheduler if channels are active.
    # The router is created eagerly here so the callback is ready BEFORE
    # load_mcp_servers() runs (which happens lazily on first agent call).
    # load_mcp_servers() will pick up the existing router from the handler.
    try:
        if channel_entries:
            from koder_agent.harness.channels.inbox import (
                ChannelInbox,
                ChannelInboxRejectedError,
                ChannelInboxStorageError,
            )
            from koder_agent.harness.channels.notification import (
                ChannelNotificationRouter,
                wrap_channel_message,
            )
            from koder_agent.harness.channels.state import get_channel_state
            from koder_agent.mcp.notifications import get_notification_handler

            _notif_handler = get_notification_handler()
            if _notif_handler.channel_router is None:
                _notif_handler.set_channel_router(ChannelNotificationRouter())

            inbox = ChannelInbox()
            cleanup_owner.channel_inbox = inbox
            await inbox.open()
            get_channel_state().inbox = inbox

            async def _on_channel_msg(server_name: str, content: str, meta: dict | None) -> None:
                wrapped = wrap_channel_message(server_name, content, meta)
                try:
                    identifier = await inbox.put(server_name, wrapped)
                except (ChannelInboxRejectedError, ChannelInboxStorageError) as exc:
                    # Preserve every rejection in the counters/retained manifest
                    # without turning a flood into an unbounded warning stream.
                    rejected = inbox.rejected
                    if rejected <= 1 or rejected & (rejected - 1) == 0:
                        logger.warning(
                            "Channel notification not accepted from '%s': %s (rejected=%d)",
                            server_name,
                            exc,
                            rejected,
                        )
                    return
                logging.getLogger(__name__).info(
                    "Channel message staged from '%s' (entry=%d)", server_name, identifier
                )

            cleanup_owner.unregister_channel_callback = (
                _notif_handler.channel_router.on_channel_message(_on_channel_msg)
            )

            async def _channel_consumer() -> None:
                """Background task that drains channel messages into the active scheduler."""
                while True:
                    try:
                        message = await inbox.get()
                    except ChannelInboxStorageError:
                        logger.error(
                            "Channel consumer stopped; inspect inbox at %s", inbox.directory
                        )
                        return
                    if message is None:
                        return
                    outcome, reason = "completed", None
                    interrupted = None
                    try:
                        logging.getLogger(__name__).info("Processing channel message...")
                        await scheduler_state.dispatch_handle(message.content, require_success=True)
                    except asyncio.CancelledError as exc:
                        task = asyncio.current_task()
                        if task is None or task.cancelling():
                            outcome, interrupted = "interrupted", exc
                        else:
                            outcome = "cancelled"
                            # A downstream turn may be cancelled without the
                            # runtime owner stopping this long-lived consumer.
                            logger.info("Channel turn cancelled; continuing to listen")
                    except _UnsuccessfulTurnError as exc:
                        outcome = "cancelled" if exc.cancelled else "failed"
                        reason = f"scheduler_{outcome}"
                        logger.warning("Channel turn did not complete (%s)", outcome)
                    except Exception as exc:
                        outcome, reason = "failed", type(exc).__name__
                        logging.getLogger(__name__).error("Channel message error: %s", exc)
                    try:
                        await inbox.finish(message.id, outcome, reason=reason)
                    except ChannelInboxStorageError:
                        logger.error(
                            "Channel outcome not finalized; inbox retained at %s", inbox.directory
                        )
                        if interrupted is not None:
                            raise interrupted
                        return
                    if interrupted is not None:
                        raise interrupted

            # Bootstrap MCP servers eagerly so channels start receiving
            # immediately — don't wait for the first user prompt.
            await scheduler._ensure_agent_initialized()

            cleanup_owner.channel_task = asyncio.create_task(_channel_consumer())

        if getattr(args, "name", None):
            await scheduler.session.set_title(args.name)
        command_handler = HarnessInteractiveCommandHandler(
            cli_agents_json=cli_agents_json,
            plugin_root=effective_plugin_root,
            teammate_mode=getattr(args, "teammate_mode", None),
            emit_console=getattr(args, "output_format", "text") == "text",
            permission_service=permission_service,
            mcp_owner_provider=lambda: getattr(scheduler_state.scheduler, "_mcp_servers", None),
            agent_service=app_agent_service,
        )
        cleanup_owner.teammate_runner = getattr(command_handler, "in_process_teammate_runner", None)
        if not bare_mode:
            session_start_source = "resume" if resume_value else "startup"
            session_start_result = dispatch_command_hooks(
                cwd=Path.cwd(),
                event_name="SessionStart",
                match_value=session_start_source,
                payload={
                    "event": "SessionStart",
                    "source": session_start_source,
                    "session_id": scheduler_state.session_id,
                },
            )
            if session_start_result.blocked:
                console.print(
                    Panel(
                        session_start_result.block_reason or "Blocked by SessionStart hook.",
                        title="Error",
                        border_style="red",
                    )
                )
                return 1
            update_watch_paths(session_start_result.watch_paths)
        existing_session_items = await scheduler.session.get_items()
        initial_agent_prompt = (
            scheduler_state.selected_agent.initial_prompt.strip()
            if scheduler_state.selected_agent is not None
            and scheduler_state.selected_agent.initial_prompt
            and not existing_session_items
            else None
        )

        async def _run_explicit_agent(agent_name: str, agent_prompt: str) -> str:
            agent = next(
                (
                    candidate
                    for candidate in scheduler_state.agent_definitions.active_agents
                    if candidate.agent_type == agent_name
                ),
                None,
            )
            if agent is None:
                return f"Unknown agent mention: {agent_name}"
            if agent.background:
                record = await command_handler.agent_service.launch_background(
                    agent_definition=agent,
                    prompt=agent_prompt,
                    description=agent_prompt[:80],
                    seed_items=None,
                    cwd=Path.cwd(),
                )
                return (
                    "fork: launched background subagent\n"
                    f"forked_agent_id: {record.id}\n"
                    f"agent_type: {agent.agent_type}\n"
                    "status: background\n"
                    f"session_id: {record.session_id}\n"
                    f"output_file: {record.output_path}"
                )
            return await command_handler.agent_service.run_sync(
                agent_definition=agent,
                prompt=agent_prompt,
                seed_items=None,
                cwd=Path.cwd(),
            )

        command_list = command_handler.get_command_list()
        commands_dict = {name: desc for name, desc in command_list}

        file_index = ProjectFileIndex(cwd=Path.cwd())
        agent_list = [
            (a.agent_type, (a.when_to_use or "")[:60])
            for a in scheduler_state.agent_definitions.active_agents
        ]
        interactive_prompt = InteractivePrompt(
            commands_dict,
            usage_tracker=scheduler.usage_tracker,
            session_id=args.session,
            file_index=file_index,
            agents=agent_list,
            config_service=command_handler.config_service,
        )

        # Wire interactive_prompt to command_handler for vim mode and other integrations
        command_handler.interactive_prompt = interactive_prompt

        prompt_text = getattr(args, "print_prompt", None)
        has_stream_json_input = bool(stream_json_messages)
        if (
            prompt_text is not None
            and len(prompt_text) == 0
            and stdin_text is None
            and not has_stream_json_input
        ):
            console.print(
                Panel(
                    "--print requires a prompt argument.",
                    title="Error",
                    border_style="red",
                )
            )
            return 1
        if getattr(args, "output_format", "text") != "text" and prompt_text is None:
            console.print(
                Panel(
                    "--output-format requires --print.",
                    title="Error",
                    border_style="red",
                )
            )
            return 1
        if getattr(args, "include_partial_messages", False):
            if getattr(args, "output_format", "text") != "stream-json":
                console.print(
                    Panel(
                        "--include-partial-messages requires --output-format stream-json.",
                        title="Error",
                        border_style="red",
                    )
                )
                return 1
            if not getattr(args, "verbose", False):
                console.print(
                    Panel(
                        "--include-partial-messages requires --verbose.",
                        title="Error",
                        border_style="red",
                    )
                )
                return 1
        json_schema = None
        if getattr(args, "json_schema", None):
            if getattr(args, "output_format", "text") != "json":
                console.print(
                    Panel(
                        "--json-schema requires --output-format json.",
                        title="Error",
                        border_style="red",
                    )
                )
                return 1
            try:
                json_schema = _load_json_schema(args.json_schema)
            except ValueError as exc:
                console.print(Panel(str(exc), title="Error", border_style="red"))
                return 1
        if prompt_text is None:
            prompt_text = getattr(args, "prompt", None)
        if input_format == "stream-json":
            prompt = _build_stream_json_prompt(stream_json_messages)
            if prompt is None:
                prompt = _build_prompt_text(prompt_text=prompt_text, stdin_text=None)
        else:
            prompt = _build_prompt_text(prompt_text=prompt_text, stdin_text=stdin_text)
        if initial_agent_prompt:
            if command_handler.is_slash_command(initial_agent_prompt):
                await command_handler.handle_slash_input(initial_agent_prompt, scheduler_state)
            else:
                seeded_prompt = initial_agent_prompt
                if context:
                    seeded_prompt = f"Context:\n{context}\n\nUser request: {seeded_prompt}"
                await scheduler_state.dispatch_handle(
                    seeded_prompt,
                    render_output=getattr(args, "output_format", "text") == "text",
                )
        if prompt:
            mention = (
                None if command_handler.is_slash_command(prompt) else extract_agent_mention(prompt)
            )
            if mention:
                agent_result = await _run_explicit_agent(*mention)
                if getattr(args, "output_format", "text") in {"json", "stream-json"}:
                    return await _print_json_output(
                        scheduler=scheduler,
                        result=agent_result,
                        output_format=getattr(args, "output_format", "text"),
                    )
                print_reflowable(console, agent_result)
                return 0
            if command_handler.is_slash_command(prompt):
                if json_schema is not None:
                    console.print(
                        Panel(
                            "--json-schema is only supported for non-slash prompts in print mode.",
                            title="Error",
                            border_style="red",
                        )
                    )
                    return 1
                slash_response = await command_handler.handle_slash_input(prompt, scheduler_state)
                if slash_response:
                    if slash_response == "__EXIT__":
                        return 0
                    session_switch = _parse_session_switch(slash_response)
                    if session_switch:
                        new_session_id, _clear_viewport = session_switch
                        try:
                            scheduler = await _switch_active_session(
                                scheduler_state,
                                args,
                                new_session_id,
                            )
                        except ValueError as exc:
                            if getattr(args, "output_format", "text") in {"json", "stream-json"}:
                                return await _print_json_output(
                                    scheduler=scheduler,
                                    result=str(exc),
                                    output_format=getattr(args, "output_format", "text"),
                                    exit_code=1,
                                )
                            print_reflowable(
                                console,
                                Panel(str(exc), title="Error", border_style="red"),
                            )
                            return 1
                        if getattr(args, "output_format", "text") in {"json", "stream-json"}:
                            return await _print_json_output(
                                scheduler=scheduler,
                                result=f"Switched to session: {new_session_id}",
                                output_format=getattr(args, "output_format", "text"),
                            )
                        print_reflowable(
                            console, f"[dim]Switched to session: {new_session_id}[/dim]"
                        )
                    else:
                        if getattr(args, "output_format", "text") in {"json", "stream-json"}:
                            return await _print_json_output(
                                scheduler=scheduler,
                                result=slash_response,
                                output_format=getattr(args, "output_format", "text"),
                            )
                        print_reflowable(
                            console,
                            Panel(
                                Text(slash_response, style="bold green"),
                                title="Command Response",
                                border_style="green",
                            ),
                        )
            elif prompt.lstrip().startswith("!"):
                shell_command = prompt.lstrip()[1:].strip()
                run_in_background = False
                if shell_command.endswith("&"):
                    shell_command = shell_command[:-1].rstrip()
                    run_in_background = True
                if not shell_command:
                    shell_result = "Usage: !<command>"
                    exit_code = 1
                else:
                    execution = await execute_shell_command(
                        shell_command,
                        run_in_background=run_in_background,
                        session_id=scheduler.session.session_id,
                    )
                    shell_result = execution.output
                    exit_code = execution.exit_code
                    if exit_code is None or (exit_code == 0 and execution.status == "error"):
                        exit_code = 1 if execution.status == "error" else 0
                    elif exit_code < 0:
                        exit_code = 128 - exit_code
                if getattr(args, "output_format", "text") in {"json", "stream-json"}:
                    return await _print_json_output(
                        scheduler=scheduler,
                        result=shell_result,
                        output_format=getattr(args, "output_format", "text"),
                        exit_code=exit_code,
                    )
                print_reflowable(
                    console,
                    Panel(
                        Text(shell_result, style="bold green"),
                        title="Shell Mode",
                        border_style="green",
                    ),
                )
                return exit_code
            else:
                if json_schema is not None:
                    prompt = _augment_prompt_for_json_schema(prompt, json_schema)
                if context:
                    prompt = f"Context:\n{context}\n\nUser request: {prompt}"
                poll_file_change_hooks(Path.cwd())
                submit_result = dispatch_command_hooks(
                    cwd=Path.cwd(),
                    event_name="UserPromptSubmit",
                    match_value=None,
                    payload={
                        "event": "UserPromptSubmit",
                        "prompt": prompt,
                        "session_id": args.session,
                    },
                )
                if submit_result.blocked:
                    console.print(
                        Panel(
                            submit_result.block_reason or "Blocked by UserPromptSubmit hook.",
                            title="Error",
                            border_style="red",
                        )
                    )
                    return 1
                if getattr(args, "output_format", "text") == "stream-json" and getattr(
                    args, "verbose", False
                ):
                    if getattr(args, "replay_user_messages", False):
                        await _replay_stream_json_messages(
                            scheduler=scheduler,
                            messages=stream_json_messages,
                        )
                    response = await scheduler_state.dispatch_stream_json(
                        prompt,
                        on_event=_write_json_line,
                        include_partial_messages=getattr(args, "include_partial_messages", False),
                    )
                else:
                    try:
                        from koder_agent.core.at_mentions import (
                            async_process_at_mentions,
                        )

                        prompt = await async_process_at_mentions(
                            prompt,
                            cwd=Path.cwd(),
                            active_agent_names={
                                a.agent_type
                                for a in scheduler_state.agent_definitions.active_agents
                            },
                            mcp_servers=getattr(scheduler, "_mcp_servers", []),
                        )
                        # Attach any -i/--image files to this first (one-shot)
                        # turn as multimodal input. The plain `prompt` string is
                        # still passed to scheduler.handle for all bookkeeping;
                        # only the model input carries the image blocks.
                        multimodal_input = None
                        images = getattr(args, "image", []) or []
                        if images:
                            from koder_agent.utils.image_input import (
                                ImageInputError,
                                build_multimodal_input,
                                model_supports_vision,
                            )

                            if not model_supports_vision(get_model_name()):
                                console.print(
                                    Panel(
                                        f"Model '{get_model_name()}' is not known to support "
                                        "image input; sending anyway and letting the provider "
                                        "decide.",
                                        title="Warning",
                                        border_style="yellow",
                                    )
                                )
                            try:
                                built = build_multimodal_input(prompt, images)
                            except ImageInputError as exc:
                                console.print(Panel(str(exc), title="Error", border_style="red"))
                                return 1
                            # build_multimodal_input returns a list only when at
                            # least one image was attached; otherwise it echoes
                            # the plain text (leave the plain-text path intact).
                            if isinstance(built, list):
                                multimodal_input = built
                        response = await scheduler_state.dispatch_handle(
                            prompt,
                            render_output=getattr(args, "output_format", "text") == "text",
                            multimodal_input=multimodal_input,
                        )
                    except Exception as exc:
                        dispatch_command_hooks(
                            cwd=Path.cwd(),
                            event_name="StopFailure",
                            match_value="unknown",
                            payload={
                                "event": "StopFailure",
                                "last_assistant_message": str(exc),
                                "session_id": args.session,
                            },
                        )
                        raise
                exit_code = _turn_exit_code(scheduler_state.scheduler)
                structured_output = None
                if json_schema is not None and exit_code == 0:
                    try:
                        structured_output = _extract_structured_output(response, json_schema)
                    except ValueError as exc:
                        return await _print_json_output(
                            scheduler=scheduler,
                            result=str(exc),
                            output_format="json",
                            exit_code=1,
                        )
                if getattr(args, "output_format", "text") in {"json", "stream-json"}:
                    return await _print_json_output(
                        scheduler=scheduler,
                        result=response,
                        output_format=getattr(args, "output_format", "text"),
                        structured_output=structured_output,
                        exit_code=exit_code,
                    )
                return exit_code
        else:
            # Discover MCP resources for autocomplete (best-effort).
            # The agent/MCP servers may already be initialized (channels) or
            # will be initialized lazily on the first prompt.  Attempt eagerly
            # so autocomplete has resources from the start.
            try:
                await scheduler._ensure_agent_initialized()
                commands_dict.clear()
                commands_dict.update(command_handler.get_command_list())
                if scheduler._mcp_servers and interactive_prompt.at_completer is not None:
                    from koder_agent.mcp import discover_mcp_resources

                    _mcp_res = await discover_mcp_resources(scheduler._mcp_servers)
                    if _mcp_res:
                        interactive_prompt.at_completer.update_mcp_resources(_mcp_res)
            except Exception:
                pass  # Resource discovery is best-effort

            # Start PR status background poller for the interactive loop.
            from koder_agent.harness.pr_status import PrStatusPoller

            _pr_poller = PrStatusPoller()
            cleanup_owner.pr_poller = _pr_poller
            if interactive_prompt.status_line is not None:
                interactive_prompt.status_line.pr_poller = _pr_poller
            _pr_poller.start()

            from koder_agent.harness.cron.runtime import CronPromptRunner

            cleanup_owner.cron_prompt_runner = CronPromptRunner(
                partial(scheduler_state.dispatch_handle, require_success=True)
            )
            cleanup_owner.cron_prompt_runner.start()

            if args.debug:
                print_reflowable(
                    console,
                    Panel(
                        "\n".join(get_litellm_cost_map_debug_lines()),
                        title="Debug: LiteLLM Cost Data",
                        border_style="cyan",
                    ),
                )

            # Opt-in startup version check (interactive-only, never in CI/headless).
            if not bare_mode:
                try:
                    from koder_agent.harness.version_info import check_for_update

                    update_message = check_for_update(interactive=True)
                    if update_message:
                        print_reflowable(
                            console,
                            Panel(
                                update_message,
                                title="[yellow]Update Available[/yellow]",
                                border_style="yellow",
                            ),
                        )
                except Exception:
                    logger.debug("Startup version check failed", exc_info=True)

            while True:
                try:
                    user_input = await interactive_prompt.get_input()
                    if not user_input and not os.isatty(0):
                        break
                except EOFError:
                    break
                except KeyboardInterrupt:
                    cleanup_owner.skip_auto_dream = True
                    break
                _pr_poller.touch()
                poll_file_change_hooks(Path.cwd())

                if user_input.lower() in {"exit", "quit"}:
                    break

                if scheduler._title_generation_task and scheduler._title_generation_task.done():
                    try:
                        display_name = await scheduler.session.get_display_name()
                        if interactive_prompt.status_line:
                            interactive_prompt.status_line.update_display_name(display_name)
                    except Exception:
                        logger.debug("Failed to update display name from title task", exc_info=True)
                    scheduler._title_generation_task = None

                if user_input:
                    if command_handler.is_slash_command(user_input):
                        # Record skill usage for recently-used sorting
                        cmd_parts = user_input.strip().lstrip("/").split()
                        cmd_name = cmd_parts[0] if cmd_parts else ""
                        if cmd_name and hasattr(interactive_prompt, "usage_tracker_skills"):
                            interactive_prompt.usage_tracker_skills.record(cmd_name)
                        slash_response = await command_handler.handle_slash_input(
                            user_input, scheduler_state
                        )
                        if slash_response:
                            if slash_response == "__EXIT__":
                                break
                            session_switch = _parse_session_switch(slash_response)
                            if session_switch:
                                new_session_id, clear_viewport = session_switch
                                try:
                                    scheduler = await _switch_active_session(
                                        scheduler_state,
                                        args,
                                        new_session_id,
                                        interactive_prompt=interactive_prompt,
                                        clear_viewport=clear_viewport,
                                    )
                                except ValueError as exc:
                                    print_reflowable(
                                        console,
                                        Panel(str(exc), title="Error", border_style="red"),
                                    )
                                    continue
                                print_reflowable(
                                    console,
                                    f"[dim]Switched to session: {new_session_id}[/dim]",
                                )
                            else:
                                print_reflowable(
                                    console,
                                    Panel(
                                        Text(slash_response, style="bold green"),
                                        title="Command Response",
                                        border_style="green",
                                    ),
                                )
                                pending_input = command_handler.consume_pending_input_text()
                                if pending_input:
                                    interactive_prompt.set_next_input_text(pending_input)
                                await interactive_prompt.refresh_prompt_suggestion(
                                    user_input,
                                    slash_response,
                                )
                    else:
                        if user_input.lstrip().startswith("!"):
                            shell_command = user_input.lstrip()[1:].strip()
                            run_in_background = False
                            if shell_command.endswith("&"):
                                shell_command = shell_command[:-1].rstrip()
                                run_in_background = True
                            if not shell_command:
                                shell_result = "Usage: !<command>"
                            else:
                                shell_result = (
                                    await execute_shell_command(
                                        shell_command,
                                        run_in_background=run_in_background,
                                        session_id=scheduler.session.session_id,
                                    )
                                ).output
                            print_reflowable(
                                console,
                                Panel(
                                    Text(shell_result, style="bold green"),
                                    title="Shell Mode",
                                    border_style="green",
                                ),
                            )
                            await interactive_prompt.refresh_prompt_suggestion(
                                user_input,
                                shell_result,
                            )
                            continue
                        mention = extract_agent_mention(user_input)
                        if mention:
                            result = await _run_explicit_agent(*mention)
                            print_reflowable(console, result)
                            await interactive_prompt.refresh_prompt_suggestion(user_input, result)
                        else:
                            submit_result = dispatch_command_hooks(
                                cwd=Path.cwd(),
                                event_name="UserPromptSubmit",
                                match_value=None,
                                payload={
                                    "event": "UserPromptSubmit",
                                    "prompt": user_input,
                                    "session_id": args.session,
                                },
                            )
                            if submit_result.blocked:
                                print_reflowable(
                                    console,
                                    Panel(
                                        submit_result.block_reason
                                        or "Blocked by UserPromptSubmit hook.",
                                        title="Command Response",
                                        border_style="red",
                                    ),
                                )
                                continue
                            try:
                                from koder_agent.core.at_mentions import (
                                    async_process_at_mentions,
                                )

                                processed_input = await async_process_at_mentions(
                                    user_input,
                                    cwd=Path.cwd(),
                                    active_agent_names={
                                        a.agent_type
                                        for a in scheduler_state.agent_definitions.active_agents
                                    },
                                    mcp_servers=getattr(scheduler, "_mcp_servers", []),
                                )
                                # Mark response start for timing
                                interactive_prompt.mark_response_start()
                                if streaming:
                                    async with interactive_prompt.capture_queued_input(
                                        scheduler.queued_input
                                    ) as streaming_ui:
                                        response = await scheduler_state.dispatch_handle(
                                            processed_input,
                                            streaming_ui=streaming_ui,
                                        )
                                else:
                                    response = await scheduler_state.dispatch_handle(
                                        processed_input
                                    )
                                # Mark response complete, show tip, send notification if long-running
                                interactive_prompt.mark_response_complete(
                                    show_tip=True,
                                    context={
                                        "in_vim_mode": interactive_prompt.vim_mode_manager.enabled,
                                        "model": get_model_name(),
                                    },
                                )
                                leftover_queued_input = (
                                    scheduler.queued_input.drain_for_tool_result()
                                )
                                if leftover_queued_input:
                                    interactive_prompt.set_next_input_text(
                                        "\n\n".join(leftover_queued_input)
                                    )
                                await interactive_prompt.refresh_prompt_suggestion(
                                    user_input,
                                    response,
                                )
                            except Exception as exc:
                                dispatch_command_hooks(
                                    cwd=Path.cwd(),
                                    event_name="StopFailure",
                                    match_value="unknown",
                                    payload={
                                        "event": "StopFailure",
                                        "last_assistant_message": str(exc),
                                        "session_id": args.session,
                                    },
                                )
                                print_reflowable(
                                    console,
                                    Panel(
                                        str(exc),
                                        title="Command Response",
                                        border_style="red",
                                    ),
                                )
                                continue

                        if (
                            scheduler._title_generation_task
                            and scheduler._title_generation_task.done()
                        ):
                            try:
                                display_name = await scheduler.session.get_display_name()
                                if interactive_prompt.status_line:
                                    interactive_prompt.status_line.update_display_name(display_name)
                            except Exception:
                                logger.debug(
                                    "Failed to update display name from title task", exc_info=True
                                )
                            scheduler._title_generation_task = None
    finally:
        await cleanup_owner.finish()

    return 0
