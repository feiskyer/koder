"""Shared runtime for koder command hooks."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import threading
import urllib.error
import urllib.request
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar, Token, copy_context
from copy import deepcopy
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Callable, Iterator

from koder_agent.harness.managed_settings import load_managed_settings, managed_settings_path
from koder_agent.harness.paths import harness_home_dir, settings_path
from koder_agent.harness.permissions.tool_arguments import (
    ToolArgumentError,
    extract_canonical_tool_target,
    normalize_tool_arguments,
)
from koder_agent.harness.session_env import (
    apply_session_env_file_to_process,
    session_env_file,
)

from .command_process import HookCommandCancelledError, run_command

logger = logging.getLogger(__name__)

# Maximum characters of hook output injected into model context.
_MAX_HOOK_OUTPUT_CHARS = 10_000

# Thread-local guard to prevent reentrant dispatch (e.g., an agent-type hook
# triggering its own PreToolUse check which would cause infinite recursion).
_dispatch_guard = threading.local()
_foreground_cancellation: ContextVar[threading.Event | None] = ContextVar(
    "koder_hook_foreground_cancellation", default=None
)

# Default hook timeout (seconds) when a hook config omits ``timeout``.
# A hook must never run unbounded: ``subprocess.run(timeout=None)`` /
# ``urlopen(timeout=None)`` would block the dispatching thread forever.
_DEFAULT_HOOK_TIMEOUT_SECONDS = 60
# Command cleanup normally completes within a polling interval. Other hook
# types may be blocked in non-cancellable I/O; never join those indefinitely.
_CANCELLATION_JOIN_TIMEOUT_SECONDS = 2.0

# Events that receive KODER_ENV_FILE.
_ENV_FILE_EVENTS = frozenset({"SessionStart", "CwdChanged", "FileChanged"})

# ---------------------------------------------------------------------------
# Hook event types & typed payloads
# ---------------------------------------------------------------------------

# Complete set of hook events.
HOOK_EVENTS = {
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "Notification",
    "UserPromptSubmit",
    "SessionStart",
    "SessionEnd",
    "Stop",
    "StopFailure",
    "SubagentStart",
    "SubagentStop",
    "PreCompact",
    "PostCompact",
    "PermissionRequest",
    "PermissionDenied",
    "Setup",
    "ConfigChange",
    "TaskCreated",
    "TaskCompleted",
    "TeammateIdle",
    "WorktreeCreate",
    "WorktreeRemove",
    "CwdChanged",
    "InstructionsLoaded",
    "FileChanged",
    "Elicitation",
    "ElicitationResult",
}


@dataclass(frozen=True)
class HookPayload:
    """Typed payload for hook dispatch."""

    event: str
    data: dict[str, Any]

    @staticmethod
    def pre_tool_use(tool_name: str, tool_input: dict) -> "HookPayload":
        return HookPayload("PreToolUse", {"tool_name": tool_name, "tool_input": tool_input})

    @staticmethod
    def post_tool_use(tool_name: str, tool_input: dict, tool_output: str) -> "HookPayload":
        return HookPayload(
            "PostToolUse",
            {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_output,
            },
        )

    @staticmethod
    def post_tool_use_failure(tool_name: str, tool_input: dict, error: str) -> "HookPayload":
        return HookPayload(
            "PostToolUseFailure",
            {"tool_name": tool_name, "tool_input": tool_input, "error": error},
        )

    @staticmethod
    def user_prompt_submit(prompt: str, session_id: str = "") -> "HookPayload":
        return HookPayload("UserPromptSubmit", {"prompt": prompt, "session_id": session_id})

    @staticmethod
    def pre_compact(token_count: int, threshold: int) -> "HookPayload":
        return HookPayload("PreCompact", {"token_count": token_count, "threshold": threshold})

    @staticmethod
    def post_compact(old_count: int = 0, new_count: int = 0) -> "HookPayload":
        return HookPayload(
            "PostCompact",
            {"old_message_count": old_count, "new_message_count": new_count},
        )

    @staticmethod
    def permission_denied(tool_name: str, tool_input: dict, reason: str) -> "HookPayload":
        return HookPayload(
            "PermissionDenied",
            {"tool_name": tool_name, "tool_input": tool_input, "reason": reason},
        )

    @staticmethod
    def permission_request(tool_name: str, tool_input: dict) -> "HookPayload":
        return HookPayload(
            "PermissionRequest",
            {"tool_name": tool_name, "tool_input": tool_input},
        )

    @staticmethod
    def cwd_changed(old_cwd: str, new_cwd: str) -> "HookPayload":
        return HookPayload("CwdChanged", {"old_cwd": old_cwd, "new_cwd": new_cwd})

    @staticmethod
    def instructions_loaded(paths: list[str]) -> "HookPayload":
        return HookPayload("InstructionsLoaded", {"paths": paths})

    @staticmethod
    def session_start(source: str = "startup", model: str = "") -> "HookPayload":
        return HookPayload("SessionStart", {"source": source, "model": model})

    @staticmethod
    def session_end(session_id: str = "") -> "HookPayload":
        return HookPayload("SessionEnd", {"session_id": session_id})

    @staticmethod
    def stop(last_assistant_message: str = "") -> "HookPayload":
        return HookPayload("Stop", {"last_assistant_message": last_assistant_message})

    @staticmethod
    def stop_failure(error: str) -> "HookPayload":
        return HookPayload("StopFailure", {"error": error})

    @staticmethod
    def subagent_start(agent_id: str, agent_type: str) -> "HookPayload":
        return HookPayload("SubagentStart", {"agent_id": agent_id, "agent_type": agent_type})

    @staticmethod
    def subagent_stop(
        agent_id: str,
        agent_type: str,
        last_assistant_message: str = "",
    ) -> "HookPayload":
        return HookPayload(
            "SubagentStop",
            {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "last_assistant_message": last_assistant_message,
            },
        )

    @staticmethod
    def config_change(key: str, old_value: Any = None, new_value: Any = None) -> "HookPayload":
        return HookPayload(
            "ConfigChange",
            {"key": key, "old_value": old_value, "new_value": new_value},
        )

    @staticmethod
    def task_created(task_id: str, subject: str) -> "HookPayload":
        return HookPayload("TaskCreated", {"task_id": task_id, "subject": subject})

    @staticmethod
    def task_completed(task_id: str, subject: str) -> "HookPayload":
        return HookPayload("TaskCompleted", {"task_id": task_id, "subject": subject})

    @staticmethod
    def worktree_create(path: str, branch: str) -> "HookPayload":
        return HookPayload("WorktreeCreate", {"path": path, "branch": branch})

    @staticmethod
    def worktree_remove(path: str) -> "HookPayload":
        return HookPayload("WorktreeRemove", {"path": path})

    @staticmethod
    def teammate_idle(agent_name: str) -> "HookPayload":
        return HookPayload("TeammateIdle", {"agent_name": agent_name})

    @staticmethod
    def setup() -> "HookPayload":
        return HookPayload("Setup", {})

    @staticmethod
    def notification(message: str, level: str = "info") -> "HookPayload":
        return HookPayload("Notification", {"message": message, "level": level})

    @staticmethod
    def file_changed(file_path: str, event: str = "change") -> "HookPayload":
        return HookPayload("FileChanged", {"file_path": file_path, "event": event})

    @staticmethod
    def elicitation(server_name: str, message: str) -> "HookPayload":
        return HookPayload("Elicitation", {"server_name": server_name, "message": message})

    @staticmethod
    def elicitation_result(server_name: str, result: dict) -> "HookPayload":
        return HookPayload(
            "ElicitationResult",
            {"server_name": server_name, "result": result},
        )


@dataclass(frozen=True)
class HookScope:
    source: str
    file_path: Path
    hooks: dict[str, Any]
    skill_root: Path | None = None
    settings_data: dict[str, Any] | None = None
    disable_all_hooks: bool = False
    plugin_name: str | None = None


@dataclass
class CommandHookSnapshot:
    """Loaded hook definitions; usable only inside ``snapshot_command_hooks``.

    Plugin resources remain owned by the context. Definitions, including project
    trust payloads and disable flags, are detached from mutable skill registries.
    Referenced command files and external approval state are not frozen.
    """

    cwd: Path
    _scopes: list[HookScope]
    _active: bool = True


@dataclass(frozen=True)
class HookListing:
    event: str
    matcher: str | None
    hook_type: str
    command: str | None
    source: str
    file_path: str
    scope_root: str | None = None


@dataclass(frozen=True)
class HookDispatchResult:
    matched_hooks: int
    blocked: bool = False
    block_reason: str | None = None
    permission_request_result: dict[str, Any] | None = None
    watch_paths: list[str] | None = None
    worktree_path: str | None = None
    elicitation_action: str | None = None
    elicitation_content: dict[str, Any] | None = None
    retry: bool = False


@dataclass
class _DynamicSkillHookRegistry:
    """Task-shared skill hook scopes for one logical scheduler turn."""

    scopes: list[HookScope] = dataclass_field(default_factory=list)
    keys: set[tuple[str, str]] = dataclass_field(default_factory=set)
    lock: threading.Lock = dataclass_field(default_factory=threading.Lock)

    def add(self, skill_name: str, scope: HookScope) -> None:
        key = (skill_name, str(scope.file_path))
        with self.lock:
            if key in self.keys:
                return
            self.keys.add(key)
            self.scopes.append(scope)

    def snapshot(self) -> list[HookScope]:
        with self.lock:
            return list(self.scopes)


_active_skill_hook_scopes: ContextVar[_DynamicSkillHookRegistry | None] = ContextVar(
    "active_skill_hook_scopes",
    default=None,
)
_watched_paths: dict[str, float | None] = {}

# Tracks hooks marked ``once: true`` that have already fired.
# Key is ``(source, event_name, hook_identity)`` where hook_identity
# is the command/url/prompt text that uniquely identifies the hook.
_once_fired: set[tuple[str, str, str]] = set()
_once_fired_lock = threading.Lock()


@dataclass
class HookConfig:
    """Configuration for a single hook instance."""

    type: str  # "command" | "http" | "prompt" | "agent"
    command: str = ""
    url: str = ""
    prompt: str = ""
    timeout: int = _DEFAULT_HOOK_TIMEOUT_SECONDS
    shell: str = ""  # "" means default, or "bash"/"zsh"/"sh"
    headers: dict[str, str] | None = None
    allowed_env_vars: list[str] | None = None
    async_hook: bool = False
    once: bool = False
    matcher: str = ""
    if_condition: str = ""
    model: str = ""


def _expand_env_vars(value: str, allowed_vars: list[str]) -> str:
    """Expand ``${VAR}`` in *value*, only for allowed env var names."""
    for var in allowed_vars:
        placeholder = f"${{{var}}}"
        if placeholder in value:
            value = value.replace(placeholder, os.environ.get(var, ""))
    return value


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _hooks_disabled_level(
    cwd: str | Path,
    scopes: list[HookScope] | None = None,
) -> str | None:
    """Return the scope level where ``disableAllHooks`` is set, or None.

    Returns ``"managed"`` when set in managed-settings.json (disables everything),
    or ``"user"`` when set in user/project/local settings (managed hooks still run).
    """
    if scopes is not None:
        if any(scope.source == "policy_settings" and scope.disable_all_hooks for scope in scopes):
            return "managed"
        if any(
            scope.source in {"user_settings", "project_settings", "local_settings"}
            and scope.disable_all_hooks
            for scope in scopes
        ):
            return "user"
        return None

    managed = load_managed_settings()
    if managed.exists and managed.valid:
        if managed.data.get("disableAllHooks") is True:
            return "managed"
    user = harness_home_dir() / "settings.json"
    if user.exists():
        data = _load_json_file(user)
        if data.get("disableAllHooks") is True:
            return "user"
    for project_path in _project_settings_paths(cwd):
        data = _load_json_file(project_path)
        if data.get("disableAllHooks") is True:
            return "user"
    return None


def _project_search_roots(cwd: str | Path, home: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    current = Path(cwd).resolve()
    home = (home if home is not None else Path.home()).resolve()
    while True:
        roots.append(current)
        if current == home or current.parent == current:
            break
        if (current / ".git").exists():
            break
        current = current.parent
    return roots


def resolve_hook_project_root(cwd: str | Path) -> Path:
    """Resolve the nearest project root that owns hook runtime settings."""
    project_paths = _project_settings_paths(cwd)
    if project_paths:
        return project_paths[0].parent.parent.resolve()
    search_roots = _project_search_roots(cwd)
    for root in search_roots:
        if (root / ".git").exists():
            return root
    return Path(cwd).resolve()


def _project_settings_paths(cwd: str | Path, home: Path | None = None) -> list[Path]:
    paths: list[Path] = []
    for current in _project_search_roots(cwd, home):
        candidate = settings_path(current)
        if candidate.exists():
            paths.append(candidate)
        local = current / ".koder" / "settings.local.json"
        if local.exists():
            paths.append(local)
    return paths


@contextmanager
def load_hook_scopes(
    cwd: str | Path, *, home: str | Path | None = None
) -> Iterator[list[HookScope]]:
    """Load scopes, optionally for an explicit destination user profile."""
    home_dir = Path(home).expanduser().resolve() if home is not None else None
    with ExitStack() as plugin_snapshots:
        scopes: list[HookScope] = []
        user_root = home_dir / ".koder" if home_dir is not None else harness_home_dir()
        user_settings = user_root / "settings.json"
        if user_settings.exists():
            settings = _load_json_file(user_settings)
            hooks = settings.get("hooks")
            if isinstance(hooks, dict) or settings.get("disableAllHooks") is True:
                scopes.append(
                    HookScope(
                        source="user_settings",
                        file_path=user_settings,
                        hooks=hooks if isinstance(hooks, dict) else {},
                        settings_data=settings,
                        disable_all_hooks=settings.get("disableAllHooks") is True,
                    )
                )
        policy_settings = (
            user_root / "managed-settings.json" if home_dir is not None else managed_settings_path()
        )
        if policy_settings.exists():
            settings = load_managed_settings(policy_settings).data
            hooks = settings.get("hooks")
            if isinstance(hooks, dict) or settings.get("disableAllHooks") is True:
                scopes.append(
                    HookScope(
                        source="policy_settings",
                        file_path=policy_settings,
                        hooks=hooks if isinstance(hooks, dict) else {},
                        settings_data=settings,
                        disable_all_hooks=settings.get("disableAllHooks") is True,
                    )
                )
        for path in _project_settings_paths(cwd, home_dir):
            settings = _load_json_file(path)
            hooks = settings.get("hooks")
            source = "local_settings" if path.name == "settings.local.json" else "project_settings"
            scopes.append(
                HookScope(
                    source=source,
                    file_path=path,
                    hooks=hooks if isinstance(hooks, dict) else {},
                    settings_data=settings,
                    disable_all_hooks=settings.get("disableAllHooks") is True,
                )
            )
        from koder_agent.harness.plugins.context import get_plugin_root

        plugin_root = user_root / "plugins" if home_dir is not None else get_plugin_root()
        if plugin_root.exists():
            try:
                from koder_agent.harness.plugins.lifecycle import PluginLifecycleService
                from koder_agent.harness.plugins.path_safety import (
                    PluginPathError,
                    open_plugin_component,
                    snapshot_plugin_tree,
                )

                lifecycle = PluginLifecycleService(plugin_root)
                for manifest, state in lifecycle.installed_plugins():
                    if not state.enabled:
                        continue
                    try:
                        plugin_dir = plugin_snapshots.enter_context(
                            snapshot_plugin_tree(lifecycle.resolve_plugin_target(manifest.name))
                        )
                    except PluginPathError:
                        continue
                    hooks_source_path = plugin_dir.joinpath(
                        *((manifest.hooks or "hooks/hooks.json").split("/"))
                    )
                    try:
                        with open_plugin_component(
                            plugin_dir,
                            manifest.hooks,
                            default="hooks/hooks.json",
                            field_name="hooks",
                            expect="file",
                        ) as hooks_path:
                            if hooks_path is None:
                                continue
                            hooks = _load_json_file(hooks_path).get("hooks")
                    except PluginPathError:
                        continue
                    if isinstance(hooks, dict):
                        scopes.append(
                            HookScope(
                                source="plugin",
                                file_path=hooks_source_path,
                                hooks=hooks,
                                skill_root=plugin_dir,
                                plugin_name=manifest.name,
                            )
                        )
            except (OSError, ValueError):
                logger.debug("Plugin hook discovery skipped", exc_info=True)
        dynamic_registry = _active_skill_hook_scopes.get()
        if dynamic_registry is not None:
            scopes.extend(dynamic_registry.snapshot())
        yield scopes


@contextmanager
def snapshot_command_hooks(
    *, cwd: str | Path, home: str | Path | None = None
) -> Iterator[CommandHookSnapshot]:
    """Capture hooks before a transaction and retain resources through dispatch.

    Pass the yielded snapshot to ``dispatch_command_hooks`` for each event.
    This does not execute hooks or persist approvals. Existing dispatch rules
    (trust checks, deduplication, once, async and cancellation) remain in force.
    """
    with load_hook_scopes(cwd, home=home) as scopes:
        snapshot = CommandHookSnapshot(Path(cwd).resolve(), deepcopy(scopes))
        try:
            yield snapshot
        finally:
            snapshot._active = False


def list_configured_hooks(cwd: str | Path) -> list[HookListing]:
    with load_hook_scopes(cwd) as scopes:
        return _list_configured_hooks(scopes)


def _list_configured_hooks(scopes: list[HookScope]) -> list[HookListing]:
    listings: list[HookListing] = []
    for scope in scopes:
        for event_name, groups in scope.hooks.items():
            if not isinstance(groups, list):
                continue
            for group in groups:
                if not isinstance(group, dict):
                    continue
                matcher = group.get("matcher")
                for hook in group.get("hooks") or []:
                    if not isinstance(hook, dict):
                        continue
                    listings.append(
                        HookListing(
                            event=event_name,
                            matcher=str(matcher) if matcher is not None else None,
                            hook_type=str(hook.get("type") or "unknown"),
                            command=(
                                str(hook.get("command"))
                                if hook.get("command") is not None
                                else None
                            ),
                            source=scope.source,
                            file_path=str(scope.file_path),
                            scope_root=str(scope.skill_root) if scope.skill_root else None,
                        )
                    )
    return listings


def _matches_matcher(matcher: str | None, match_value: str | None) -> bool:
    if matcher in (None, "", "*"):
        return True
    if match_value is None:
        return False
    try:
        return re.search(str(matcher), match_value) is not None
    except re.error:
        return str(matcher) == match_value


def _payload_target(payload: dict[str, Any]) -> str:
    tool_input = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
    tool_name = str(payload.get("tool_name") or "")
    try:
        normalized_input = normalize_tool_arguments(tool_name, tool_input)
    except ToolArgumentError:
        return ""
    target = extract_canonical_tool_target(tool_name, normalized_input)
    if target:
        return target
    # Fallback: check all string values in tool_input so tools with non-standard
    # parameter names (e.g. "query", "pattern", "selector") can still match `if`
    # conditions rather than silently failing open.
    for value in normalized_input.values():
        if isinstance(value, str) and value:
            return value
    return ""


def _matches_if(condition: str | None, payload: dict[str, Any]) -> bool:
    if not condition:
        return True
    match = re.fullmatch(r"([A-Za-z0-9_*|.-]+)\((.*)\)", condition.strip())
    if match is None:
        return True
    tool_pattern = match.group(1).strip()
    arg_pattern = match.group(2).strip()
    tool_name = str(payload.get("tool_name") or "")
    if tool_pattern not in {"", "*"}:
        tool_patterns = [part.strip() for part in tool_pattern.split("|") if part.strip()]
        if not any(
            re.fullmatch(pattern.replace("*", ".*"), tool_name) for pattern in tool_patterns
        ):
            return False
    if arg_pattern in {"", "*"}:
        return True
    target = _payload_target(payload)
    regex = "^" + re.escape(arg_pattern).replace(r"\*", ".*") + "$"
    return re.fullmatch(regex, target) is not None


def _hook_identity(hook: dict[str, Any]) -> str:
    """Return a string that uniquely identifies a hook for deduplication."""
    hook_type = hook.get("type", "")
    if hook_type == "command":
        return f"command:{hook.get('command', '')}"
    if hook_type == "http":
        return f"http:{hook.get('url', '')}"
    if hook_type in ("prompt", "agent"):
        return f"{hook_type}:{hook.get('prompt', '')}"
    return f"unknown:{id(hook)}"


def _bounded_timeout(raw: Any) -> int | float:
    """Return a positive timeout in seconds, defaulting when unset/invalid.

    Guards against ``None`` (key omitted), zero, negative, and non-numeric
    values — any of which would otherwise disable the timeout and let a hung
    hook block forever.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return _DEFAULT_HOOK_TIMEOUT_SECONDS
    if raw <= 0:
        return _DEFAULT_HOOK_TIMEOUT_SECONDS
    return raw


def _cap_output(text: str | None) -> str | None:
    """Cap hook output to _MAX_HOOK_OUTPUT_CHARS."""
    if text is None:
        return None
    if len(text) <= _MAX_HOOK_OUTPUT_CHARS:
        return text
    return text[:_MAX_HOOK_OUTPUT_CHARS] + f"\n... (truncated, {len(text)} chars total)"


def _run_command_hook(
    *,
    command: str,
    payload_text: str,
    cwd: Path,
    env: dict[str, str],
    timeout: int | float | None = None,
    shell: str = "",
) -> tuple[int, str, str]:
    if shell:
        # Explicit shell selection: run as [shell, "-c", command].
        cmd: str | list[str] = [shell, "-c", command]
        use_shell = False
    else:
        cmd = command
        use_shell = True
    try:
        result = run_command(
            cmd,
            input=payload_text,
            cwd=str(cwd),
            shell=use_shell,
            env=env,
            # Never run unbounded: timeout=None would block forever.
            timeout=_bounded_timeout(timeout),
            cancel_event=_foreground_cancellation.get(),
        )
    except subprocess.TimeoutExpired:
        return 2, "", "Hook timed out (fail-closed)"
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def _run_http_hook(
    *,
    url: str,
    payload_text: str,
    timeout: int | float | None,
    headers: dict[str, str] | None = None,
    allowed_env_vars: list[str] | None = None,
) -> tuple[int, str, str]:
    merged_headers: dict[str, str] = {"Content-Type": "application/json"}
    if headers:
        env_vars = allowed_env_vars or []
        for key, value in headers.items():
            merged_headers[key] = _expand_env_vars(value, env_vars) if env_vars else value
    request = urllib.request.Request(
        url,
        data=payload_text.encode("utf-8"),
        headers=merged_headers,
        method="POST",
    )
    try:
        # Never run unbounded: timeout=None would block forever.
        with urllib.request.urlopen(request, timeout=_bounded_timeout(timeout)) as response:
            body = response.read().decode("utf-8")
            return response.status, body.strip(), ""
    except urllib.error.HTTPError as exc:
        diagnostic = str(exc)
        try:
            exc.close()
        except Exception as cleanup_error:
            logger.debug("HTTP hook response cleanup failed (%s)", type(cleanup_error).__name__)
        return 500, "", diagnostic
    except Exception as exc:  # pragma: no cover - defensive
        return 500, "", str(exc)


def _run_prompt_hook(
    *,
    prompt_text: str,
    payload_text: str,
    model: str | None,
) -> str:
    from agents import RunConfig, Runner

    from koder_agent.agentic.agent import create_dev_agent
    from koder_agent.harness.agents.service import _cleanup_agent_mcp_servers

    async def _run() -> str:
        hook_prompt = (
            "You are evaluating a hook condition. "
            "Return only JSON. "
            f"Hook prompt:\n{prompt_text}\n\n"
            f"Hook input JSON:\n{payload_text}"
        )
        agent = await create_dev_agent([], name="HookPrompt", model_override=model)
        try:
            result = await Runner.run(agent, hook_prompt, run_config=RunConfig(), max_turns=5)
            return str(result.final_output or "")
        finally:
            await _cleanup_agent_mcp_servers(agent)

    return _run_coroutine_sync(_run())


def _run_agent_hook(
    *,
    prompt_text: str,
    payload_text: str,
    model: str | None,
) -> str:
    from agents import RunConfig, Runner

    from koder_agent.agentic.agent import create_dev_agent
    from koder_agent.harness.agents.service import _cleanup_agent_mcp_servers
    from koder_agent.tools import get_all_tools
    from koder_agent.tools.skill_context import skill_run_scope

    async def _run() -> str:
        hook_prompt = (
            "You are an agent-based hook evaluator. "
            "Return only JSON. "
            f"Hook prompt:\n{prompt_text}\n\n"
            f"Hook input JSON:\n{payload_text}"
        )
        tools = [
            tool
            for tool in get_all_tools()
            if tool.name not in {"task_delegate", "todo_read", "todo_write"}
        ]
        agent = await create_dev_agent(tools, name="HookAgent", model_override=model)
        try:
            with skill_run_scope() as run_hooks:
                result = await Runner.run(
                    agent,
                    hook_prompt,
                    run_config=RunConfig(),
                    hooks=run_hooks,
                    max_turns=10,
                )
            return str(result.final_output or "")
        finally:
            await _cleanup_agent_mcp_servers(agent)

    return _run_coroutine_sync(_run())


def _run_coroutine_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    holder: dict[str, Any] = {}
    context = copy_context()

    def _target():
        try:
            holder["result"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive
            holder["error"] = exc

    thread = threading.Thread(target=lambda: context.run(_target))
    thread.start()
    thread.join()
    if "error" in holder:
        raise holder["error"]
    return holder.get("result", "")


def _extract_block(stdout: str) -> str | None:
    if not stdout:
        return None
    try:
        data = json.loads(stdout)
    except Exception:
        return None
    if isinstance(data, dict):
        hook_specific = data.get("hookSpecificOutput")
        if isinstance(hook_specific, dict):
            decision = hook_specific.get("permissionDecision") or hook_specific.get("decision")
            if isinstance(decision, dict):
                behavior = decision.get("behavior")
                if behavior in {"deny", "block"}:
                    return str(
                        decision.get("message")
                        or hook_specific.get("permissionDecisionReason")
                        or hook_specific.get("reason")
                        or "Blocked by hook"
                    )
            elif decision in {"deny", "block"}:
                return str(
                    hook_specific.get("permissionDecisionReason")
                    or hook_specific.get("reason")
                    or "Blocked by hook"
                )
        if data.get("decision") == "block":
            return str(data.get("reason") or "Blocked by hook")
    return None


def _parse_hook_output(stdout: str) -> dict[str, Any]:
    if not stdout:
        return {}
    try:
        data = json.loads(stdout)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _merge_dispatch_result(
    current: HookDispatchResult, parsed: dict[str, Any]
) -> HookDispatchResult:
    hook_specific = (
        parsed.get("hookSpecificOutput")
        if isinstance(parsed.get("hookSpecificOutput"), dict)
        else {}
    )
    # Extract new values from the hook output.
    new_permission = (
        hook_specific.get("decision") if isinstance(hook_specific.get("decision"), dict) else None
    )
    new_watch_paths = (
        hook_specific.get("watchPaths")
        if isinstance(hook_specific.get("watchPaths"), list)
        else None
    )
    new_worktree_path = (
        hook_specific.get("worktreePath")
        if isinstance(hook_specific.get("worktreePath"), str)
        else None
    )
    new_elicitation_action = (
        hook_specific.get("action") if isinstance(hook_specific.get("action"), str) else None
    )
    new_elicitation_content = (
        hook_specific.get("content") if isinstance(hook_specific.get("content"), dict) else None
    )
    new_retry = hook_specific.get("retry") is True

    # Merge: only overwrite fields from the current result when the new value
    # is non-None, so earlier hooks' values are preserved (not clobbered to None).
    return HookDispatchResult(
        matched_hooks=current.matched_hooks,
        blocked=current.blocked,
        block_reason=current.block_reason,
        permission_request_result=(
            new_permission if new_permission is not None else current.permission_request_result
        ),
        watch_paths=new_watch_paths if new_watch_paths is not None else current.watch_paths,
        worktree_path=(
            new_worktree_path if new_worktree_path is not None else current.worktree_path
        ),
        elicitation_action=(
            new_elicitation_action
            if new_elicitation_action is not None
            else current.elicitation_action
        ),
        elicitation_content=(
            new_elicitation_content
            if new_elicitation_content is not None
            else current.elicitation_content
        ),
        retry=new_retry or current.retry,
    )


def update_watch_paths(paths: list[str] | None) -> None:
    if not paths:
        return
    for raw in paths:
        path = str(Path(raw).expanduser().resolve())
        current = _watched_paths.get(path)
        if current is not None:
            continue
        resolved = Path(path)
        if resolved.exists():
            _watched_paths[path] = resolved.stat().st_mtime
        else:
            _watched_paths[path] = None


def poll_file_change_hooks(cwd: str | Path) -> int:
    fired = 0
    for path_str, previous_mtime in list(_watched_paths.items()):
        path = Path(path_str)
        current_mtime = path.stat().st_mtime if path.exists() else None
        if current_mtime == previous_mtime:
            continue
        _watched_paths[path_str] = current_mtime
        result = dispatch_command_hooks(
            cwd=cwd,
            event_name="FileChanged",
            match_value=path.name,
            payload={
                "event": "FileChanged",
                "file_path": str(path.resolve()),
            },
        )
        update_watch_paths(result.watch_paths)
        fired += 1
    return fired


def _build_hook_env(hook: dict, scope: "HookScope") -> dict[str, str]:
    """Build scrubbed env for hook execution.

    User-settings hooks get full env by default (trusted).
    Project/plugin hooks get scrubbed env (untrusted).
    Any hook with "passFullEnv": true gets full env (opt-in).
    """
    from koder_agent.harness.session_env import SANDBOX_ENV_ALLOWLIST, is_probably_secret_env_name

    # User-level hooks or explicit opt-in: full env
    if scope.source == "user_settings" or hook.get("passFullEnv"):
        return os.environ.copy()

    # All other sources (project, plugin, skill): scrubbed env
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in SANDBOX_ENV_ALLOWLIST or key.startswith("LC_"):
            env[key] = value
        elif not is_probably_secret_env_name(key):
            env[key] = value
    return env


def _run_async_command(
    *,
    command: str,
    payload_text: str,
    cwd: Path,
    env: dict[str, str],
    shell: str = "",
    timeout: int | float | None = None,
    on_complete: Callable[[], None] | None = None,
) -> None:
    # Use bounded timeout to prevent indefinite thread blocking.
    effective_timeout = _bounded_timeout(timeout)

    def _target():
        try:
            code, _stdout, _stderr = _run_command_hook(
                command=command,
                payload_text=payload_text,
                cwd=cwd,
                env=env,
                shell=shell,
                timeout=effective_timeout,
            )
            if code == 2:
                logger.debug(
                    "Async hook failed or timed out after at most %s seconds", effective_timeout
                )
        finally:
            if on_complete is not None:
                on_complete()

    thread = threading.Thread(target=_target, daemon=True)
    try:
        thread.start()
    except Exception:
        if on_complete is not None:
            on_complete()
        raise


def _check_foreground_cancellation() -> None:
    cancellation = _foreground_cancellation.get()
    if cancellation is not None and cancellation.is_set():
        raise HookCommandCancelledError()


def dispatch_command_hooks(
    *,
    cwd: str | Path,
    event_name: str,
    payload: dict[str, Any],
    match_value: str | None = None,
    snapshot: CommandHookSnapshot | None = None,
) -> HookDispatchResult:
    _check_foreground_cancellation()
    # Reentrancy guard: prevent infinite recursion when an agent-type hook
    # triggers tool calls that would dispatch back into this function.
    if getattr(_dispatch_guard, "in_dispatch", False):
        return HookDispatchResult(matched_hooks=0)
    _dispatch_guard.in_dispatch = True
    try:
        return _dispatch_command_hooks_inner(
            cwd=cwd,
            event_name=event_name,
            payload=payload,
            match_value=match_value,
            snapshot=snapshot,
        )
    finally:
        _dispatch_guard.in_dispatch = False


def _dispatch_command_hooks_inner(
    *,
    cwd: str | Path,
    event_name: str,
    payload: dict[str, Any],
    match_value: str | None = None,
    snapshot: CommandHookSnapshot | None = None,
) -> HookDispatchResult:
    if snapshot is not None and (not snapshot._active or snapshot.cwd != Path(cwd).resolve()):
        raise ValueError("Hook snapshot is closed or belongs to a different cwd")
    scope_context = nullcontext(snapshot._scopes) if snapshot is not None else load_hook_scopes(cwd)
    with scope_context as scopes:
        # Parse settings once so trust validation and execution share one immutable
        # dispatch snapshot even if a settings file changes concurrently. The same
        # scope also keeps plugin snapshots alive through synchronous execution.
        disabled_level = _hooks_disabled_level(cwd, scopes=scopes)
        if disabled_level == "managed":
            return HookDispatchResult(matched_hooks=0)
        return _dispatch_loaded_hook_scopes(
            cwd=cwd,
            event_name=event_name,
            payload=payload,
            match_value=match_value,
            disabled_level=disabled_level,
            scopes=scopes,
        )


def _dispatch_loaded_hook_scopes(
    *,
    cwd: str | Path,
    event_name: str,
    payload: dict[str, Any],
    match_value: str | None,
    disabled_level: str | None,
    scopes: list[HookScope],
) -> HookDispatchResult:
    if disabled_level == "user":
        scopes = [s for s in scopes if s.source == "policy_settings"]
    payload_text = json.dumps(payload)
    result = HookDispatchResult(matched_hooks=0)
    seen_hooks: set[str] = set()
    project_settings_by_root: dict[Path, dict[str, dict[str, Any]]] = {}
    for project_scope in scopes:
        if project_scope.source not in ("project_settings", "local_settings"):
            continue
        project_root = project_scope.file_path.parent.parent.resolve()
        project_settings_by_root.setdefault(project_root, {})[project_scope.source] = (
            project_scope.settings_data or {}
        )
    project_approval_errors: dict[Path, str | None] = {}
    for scope in scopes:
        # C1: Project-level hook trust gate — skip untrusted project hooks.
        if scope.source in ("project_settings", "local_settings"):
            from .project_approval import (
                is_project_hooks_allowed,
                project_hooks_approval_error,
            )

            # Project root is the parent of the .koder/ directory containing settings.
            project_root = scope.file_path.parent.parent.resolve()
            if project_root not in project_approval_errors:
                settings_snapshot = project_settings_by_root[project_root]
                project_approval_errors[project_root] = (
                    None
                    if is_project_hooks_allowed(project_root, settings_snapshot)
                    else project_hooks_approval_error(project_root, settings_snapshot)
                )
            approval_error = project_approval_errors[project_root]
            if approval_error is not None:
                logger.warning(
                    "Skipping project hooks from %s: %s.",
                    project_root,
                    approval_error,
                )
                continue

        groups = scope.hooks.get(event_name)
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            if not _matches_matcher(group.get("matcher"), match_value):
                continue
            for hook in group.get("hooks") or []:
                _check_foreground_cancellation()
                if not isinstance(hook, dict):
                    continue
                if not _matches_if(hook.get("if"), payload):
                    continue

                # Deduplication: skip identical hooks within a dispatch.
                identity = _hook_identity(hook)
                if identity in seen_hooks:
                    continue
                seen_hooks.add(identity)

                # Once-only: skip hooks that have already fired.
                if hook.get("once") is True:
                    once_key = (scope.source, event_name, identity)
                    with _once_fired_lock:
                        if once_key in _once_fired:
                            continue
                        _once_fired.add(once_key)

                command = hook.get("command")
                # C4: Scrub secrets from hook env for untrusted sources.
                env = _build_hook_env(hook, scope)
                env["KODER_PROJECT_DIR"] = str(Path(cwd).resolve())
                if scope.skill_root is not None:
                    env["KODER_SKILL_DIR"] = str(scope.skill_root)
                if (
                    scope.source == "plugin"
                    and scope.skill_root is not None
                    and scope.plugin_name is not None
                ):
                    from koder_agent.harness.plugins.env import plugin_env_vars

                    try:
                        env.update(plugin_env_vars(scope.plugin_name, scope.skill_root))
                    except Exception:
                        logger.debug("Failed to load plugin env vars", exc_info=True)
                if isinstance(payload.get("session_id"), str):
                    env["KODER_SESSION_ID"] = str(payload["session_id"])
                    if event_name in _ENV_FILE_EVENTS:
                        env["KODER_ENV_FILE"] = str(session_env_file(str(payload["session_id"])))
                hook_type = hook.get("type")
                hook_timeout = _bounded_timeout(hook.get("timeout"))
                hook_shell = hook.get("shell") or ""
                hook_headers = (
                    hook.get("headers") if isinstance(hook.get("headers"), dict) else None
                )
                hook_allowed_env = (
                    hook.get("allowedEnvVars")
                    if isinstance(hook.get("allowedEnvVars"), list)
                    else None
                )
                stdout = ""
                stderr = ""
                code = 0
                if hook_type == "command":
                    if not isinstance(command, str) or not command.strip():
                        continue
                    if hook.get("async") is True:
                        async_cwd = scope.skill_root or Path(cwd).resolve()
                        async_env = env
                        async_cleanup: Callable[[], None] | None = None
                        if scope.source == "plugin" and scope.skill_root is not None:
                            from koder_agent.harness.plugins.env import plugin_env_vars
                            from koder_agent.harness.plugins.path_safety import snapshot_plugin_tree

                            snapshot_context = snapshot_plugin_tree(scope.skill_root)
                            async_cwd = snapshot_context.__enter__()
                            async_env = dict(env)
                            async_env["KODER_SKILL_DIR"] = str(async_cwd)
                            if scope.plugin_name:
                                async_env.update(plugin_env_vars(scope.plugin_name, async_cwd))

                            def close_async_snapshot(
                                context=snapshot_context,
                            ) -> None:
                                context.__exit__(None, None, None)

                            async_cleanup = close_async_snapshot
                        result = replace(result, matched_hooks=result.matched_hooks + 1)
                        _run_async_command(
                            command=command,
                            payload_text=payload_text,
                            cwd=async_cwd,
                            env=async_env,
                            shell=hook_shell,
                            timeout=hook_timeout,
                            on_complete=async_cleanup,
                        )
                        continue
                    result = replace(result, matched_hooks=result.matched_hooks + 1)
                    code, stdout, stderr = _run_command_hook(
                        command=command,
                        payload_text=payload_text,
                        cwd=scope.skill_root or Path(cwd).resolve(),
                        env=env,
                        timeout=hook_timeout,
                        shell=hook_shell,
                    )
                elif hook_type == "http":
                    url = hook.get("url")
                    if not isinstance(url, str) or not url.strip():
                        continue
                    result = replace(result, matched_hooks=result.matched_hooks + 1)
                    code, stdout, stderr = _run_http_hook(
                        url=url,
                        payload_text=payload_text,
                        timeout=hook_timeout,
                        headers=hook_headers,
                        allowed_env_vars=hook_allowed_env,
                    )
                elif hook_type == "prompt":
                    prompt_text = hook.get("prompt")
                    if not isinstance(prompt_text, str) or not prompt_text.strip():
                        continue
                    result = replace(result, matched_hooks=result.matched_hooks + 1)
                    stdout = _run_prompt_hook(
                        prompt_text=prompt_text,
                        payload_text=payload_text,
                        model=hook.get("model") if isinstance(hook.get("model"), str) else None,
                    )
                elif hook_type == "agent":
                    prompt_text = hook.get("prompt")
                    if not isinstance(prompt_text, str) or not prompt_text.strip():
                        continue
                    result = replace(result, matched_hooks=result.matched_hooks + 1)
                    stdout = _run_agent_hook(
                        prompt_text=prompt_text,
                        payload_text=payload_text,
                        model=hook.get("model") if isinstance(hook.get("model"), str) else None,
                    )
                else:
                    continue

                parsed = _parse_hook_output(stdout)
                result = _merge_dispatch_result(result, parsed)
                block_reason = _extract_block(stdout)
                if code == 2 or block_reason:
                    if code == 2:
                        block_reason = stderr or block_reason or "Blocked by hook"
                    return HookDispatchResult(
                        matched_hooks=result.matched_hooks,
                        blocked=True,
                        # Cap the reason injected into model context so a runaway
                        # hook cannot flood the conversation.
                        block_reason=_cap_output(block_reason),
                        permission_request_result=result.permission_request_result,
                        watch_paths=result.watch_paths,
                        worktree_path=result.worktree_path,
                        elicitation_action=result.elicitation_action,
                        elicitation_content=result.elicitation_content,
                    )
    _check_foreground_cancellation()
    if event_name == "SessionStart" and isinstance(payload.get("session_id"), str):
        session_id = str(payload["session_id"])
        session_env_file(session_id)
        apply_session_env_file_to_process(session_id)
    return result


async def dispatch_command_hooks_async(
    *,
    cwd: str | Path,
    event_name: str,
    payload: dict[str, Any],
    match_value: str | None = None,
) -> HookDispatchResult:
    """Run :func:`dispatch_command_hooks` off the event loop.

    ``dispatch_command_hooks`` performs blocking I/O (command subprocesses,
    ``urllib.request.urlopen``). Async callers must use this entrypoint so a
    slow hook cannot freeze the event loop (streaming UI, subagents, cron).
    ``asyncio.to_thread`` copies the current context, so contextvars such as
    the active skill hook scopes propagate into the worker thread.
    """
    cancellation = threading.Event()

    def dispatch() -> HookDispatchResult:
        token = _foreground_cancellation.set(cancellation)
        try:
            return dispatch_command_hooks(
                cwd=cwd,
                event_name=event_name,
                payload=payload,
                match_value=match_value,
            )
        finally:
            _foreground_cancellation.reset(token)

    task = asyncio.create_task(asyncio.to_thread(dispatch))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancellation.set()
        # Cancelling a to_thread waiter cannot stop its worker. Signal the
        # command loop and allow bounded cleanup before propagating cancellation.
        # HTTP/model hooks may remain in uninterruptible I/O; their worker still
        # owns its scopes and must not continue to later hooks once it returns.
        deadline = asyncio.get_running_loop().time() + _CANCELLATION_JOIN_TIMEOUT_SECONDS
        while not task.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait({task}, timeout=remaining)
            except asyncio.CancelledError:
                continue

        def discard_outcome(completed: asyncio.Task[HookDispatchResult]) -> None:
            if not completed.cancelled():
                completed.exception()  # cancellation wins over a worker error

        task.add_done_callback(discard_outcome)
        raise


@contextmanager
def active_skill_hooks(
    skill_name: str, hooks: dict[str, Any] | None, skill_root: Path | None
) -> Iterator[None]:
    token = None
    if _active_skill_hook_scopes.get() is None:
        token = begin_skill_hook_scope()
    register_skill_hooks(skill_name, hooks, skill_root)
    try:
        yield
    finally:
        if token is not None:
            reset_skill_hook_scope(token)


def begin_skill_hook_scope() -> Token:
    """Seed a mutable hook registry shared by SDK child tasks."""
    current = _active_skill_hook_scopes.get()
    registry = _DynamicSkillHookRegistry()
    if current is not None:
        for scope in current.snapshot():
            registry.add(scope.source, scope)
    return _active_skill_hook_scopes.set(registry)


def reset_skill_hook_scope(token: Token) -> None:
    """Restore the dynamic skill-hook registry captured before a turn."""
    try:
        _active_skill_hook_scopes.reset(token)
    except (ValueError, LookupError):
        _active_skill_hook_scopes.set(None)


def register_skill_hooks(
    skill_name: str, hooks: dict[str, Any] | None, skill_root: Path | None
) -> None:
    """Install a skill's hooks for the remainder of the logical turn."""
    if not hooks:
        return
    scope = HookScope(
        source="skills",
        file_path=(skill_root / "SKILL.md") if skill_root is not None else Path(f"<{skill_name}>"),
        hooks=hooks,
        skill_root=skill_root,
    )
    registry = _active_skill_hook_scopes.get()
    if registry is None:
        registry = _DynamicSkillHookRegistry()
        _active_skill_hook_scopes.set(registry)
    registry.add(skill_name, scope)
