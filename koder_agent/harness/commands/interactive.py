"""Interactive slash-command handler backed by harness runtime services."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from koder_agent.auth.client_integration import map_provider_to_oauth
from koder_agent.config import get_config, get_config_manager
from koder_agent.core.keybindings import DEFAULT_KEYBINDINGS, KeybindingManager
from koder_agent.core.session import EnhancedSQLiteSession, migrate_legacy_sessions
from koder_agent.core.terminal_reflow import print_reflowable
from koder_agent.core.vim_mode import VimModeManager
from koder_agent.harness.add_dir_validation import (
    add_dir_help_message,
    validate_directory_for_workspace,
)
from koder_agent.harness.agents.definitions import get_agent_definitions, resolve_agent_model
from koder_agent.harness.agents.service import (
    AgentService,
    agent_definition_matches_record,
    resolve_agent_record_execution_cwd,
    resolve_agent_record_origin,
)
from koder_agent.harness.agents.teams.context import TeamToolContext
from koder_agent.harness.agents.teams.runtime import (
    resolve_teammate_execution_mode,
    resolve_teammate_mode,
)
from koder_agent.harness.agents.teams.service import TeamService
from koder_agent.harness.commands.advisor import run_advisor_review
from koder_agent.harness.commands.agents_view import (
    render_agent_details,
    render_agent_runtime_summaries,
    render_agent_runtime_summary,
    render_agents_overview,
)
from koder_agent.harness.commands.brief import run_brief
from koder_agent.harness.commands.buddy import run_buddy
from koder_agent.harness.commands.pr_comments import run_pr_comments
from koder_agent.harness.commands.registry import CommandRegistry
from koder_agent.harness.commands.review_context import session_transcript_from_items
from koder_agent.harness.commands.security_review import run_security_review
from koder_agent.harness.commands.workflow_helpers import (
    current_branch,
    diff_stat,
    recent_commits,
    remote_url,
    staged_diff_stat,
    status_short,
)
from koder_agent.harness.config.service import RuntimeConfigService
from koder_agent.harness.hooks.runtime import (
    dispatch_command_hooks,
    list_configured_hooks,
    resolve_hook_project_root,
)
from koder_agent.harness.memory.budget import estimate_messages_tokens, estimate_text_tokens
from koder_agent.harness.memory.compact import (
    build_compacted_session_items,
    compactable_session_items,
    llm_compact_messages,
)
from koder_agent.harness.output_styles import (
    discover_output_styles,
    find_output_style,
    load_active_output_style_name,
    save_active_output_style_name,
)
from koder_agent.harness.paths import (
    harness_home_dir,
    project_agents_dir,
    user_agents_dir,
)
from koder_agent.harness.permissions.modes import PermissionMode
from koder_agent.harness.permissions.service import PermissionService
from koder_agent.harness.permissions.tool_arguments import (
    ToolArgumentError,
    extract_canonical_tool_target,
    normalize_tool_arguments,
    permission_arguments_for_target,
)
from koder_agent.harness.plan.mode import PlanModeService
from koder_agent.harness.plugins.lifecycle import PluginLifecycleService
from koder_agent.harness.plugins.registry import PluginRegistry
from koder_agent.harness.reasoning_display import (
    VALID_REASONING_DISPLAY_MODES,
    normalize_reasoning_display_mode,
)
from koder_agent.harness.release_notes import (
    fetch_and_store_changelog,
    format_release_notes,
    get_all_release_notes,
    get_recent_release_note_groups,
    get_stored_changelog,
)
from koder_agent.harness.sandbox_settings import (
    add_excluded_command,
    resolve_sandbox_settings,
    update_local_sandbox_settings,
)
from koder_agent.harness.session_env import (
    clear_session_env,
    delete_session_env_var,
    is_valid_env_name,
    load_session_env,
    set_session_env_var,
)
from koder_agent.harness.skills.bundled import is_trusted_bundled_skill
from koder_agent.harness.statusline_settings import (
    resolve_statusline_config,
    update_user_statusline_config,
)
from koder_agent.harness.statusline_setup import auto_configure_statusline_from_shell_prompt
from koder_agent.harness.tasks.service import TaskService
from koder_agent.harness.version_info import render_command_version, resolve_runtime_version
from koder_agent.harness.voice.service import (
    SUPPORTED_VOICE_PROVIDERS,
    VoiceDictationError,
    resolve_voice_credentials,
    resolve_voice_provider,
)
from koder_agent.mcp.server_manager import MCPServerManager
from koder_agent.tools.skill import SKILL_ADDITIONAL_DIRS_ENV, Skill, discover_merged_skills
from koder_agent.tools.skill_context import skill_invocation_scope
from koder_agent.utils import parse_session_dt
from koder_agent.utils.client import get_model_name, get_provider_api_env_var, llm_completion
from koder_agent.utils.model_info import get_context_window_size, resolve_model_alias

console = Console()
VALID_OUTPUT_THEMES = ("adaptive", "dark", "light")


def _output_style_settings_path() -> Path:
    return harness_home_dir() / "settings.json"


def _load_output_theme() -> str:
    settings_path = _output_style_settings_path()
    try:
        loaded = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        return "adaptive"
    if not isinstance(loaded, dict):
        return "adaptive"
    output_style = loaded.get("outputStyle")
    theme = output_style.get("theme") if isinstance(output_style, dict) else None
    return theme if theme in VALID_OUTPUT_THEMES else "adaptive"


def _save_output_theme(theme: str) -> Path:
    settings_path = _output_style_settings_path()
    try:
        loaded = json.loads(settings_path.read_text(encoding="utf-8"))
    except Exception:
        loaded = {}
    if not isinstance(loaded, dict):
        loaded = {}
    output_style = loaded.get("outputStyle")
    if not isinstance(output_style, dict):
        output_style = {}
    output_style["theme"] = theme
    loaded["outputStyle"] = output_style
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(loaded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return settings_path


def _vim_state_path() -> Path:
    return harness_home_dir() / "vim_state.json"


def _load_vim_enabled() -> bool:
    manager = VimModeManager(state_path=_vim_state_path())
    manager.load()
    return manager.enabled


def _save_vim_enabled(enabled: bool) -> Path:
    state_path = _vim_state_path()
    manager = VimModeManager(state_path=state_path)
    if enabled:
        manager.enable()
    else:
        manager.disable()
    manager.save()
    return state_path


def _redact_sensitive_debug_text(text: object) -> str:
    if text is None:
        return ""
    redacted = str(text)
    for key, value in os.environ.items():
        key_lower = key.lower()
        if not value or len(value) < 6:
            continue
        if any(marker in key_lower for marker in ("api_key", "apikey", "token", "secret")):
            redacted = redacted.replace(value, "[REDACTED]")
    redacted = re.sub(r"sk-[A-Za-z0-9_-]{10,}", "[TOKEN]", redacted)
    redacted = re.sub(
        r"(?i)(api[_-]?key|token|secret)(\s*[=:]\s*)[^\s,\"'}]+",
        r"\1\2[REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)(authorization\s*[=:]\s*bearer\s+)[^\s,\"'}]+",
        r"\1[REDACTED]",
        redacted,
    )
    return redacted


InteractiveCommand = Callable[[object, list[str]], Awaitable[str]]

AGENT_COLORS = ("red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan")
RESET_COLOR_ALIASES = {"default", "reset", "none", "gray", "grey"}


def _split_model_selection(model: str) -> tuple[str | None, str]:
    selected = model.strip()
    if selected.startswith("litellm/"):
        selected = selected[len("litellm/") :]
    if "/" not in selected:
        return None, selected
    provider, name = selected.split("/", 1)
    return provider.lower(), name


def _infer_provider_for_model(model_name: str, current_provider: str | None) -> str:
    provider = (current_provider or "openai").lower()
    lower = model_name.lower()
    if lower.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
        return "openai"
    if lower.startswith("claude-"):
        return provider if provider in {"anthropic", "claude"} else "anthropic"
    if lower.startswith("gemini-") or lower.startswith("models/gemini"):
        return provider if provider in {"google", "gemini"} else "google"
    return provider


def _normalize_model_selection(
    selected: str, *, current_provider: str | None
) -> tuple[str, str, str]:
    resolved = resolve_model_alias(selected.strip())
    explicit_provider, model_name = _split_model_selection(resolved)
    provider = explicit_provider or _infer_provider_for_model(model_name, current_provider)
    env_model = f"{provider}/{model_name}" if explicit_provider else model_name
    return provider, model_name, env_model


class HarnessInteractiveCommandHandler:
    """Slash-command handler that routes through harness-owned services."""

    def __init__(
        self,
        *,
        registry: Optional[CommandRegistry] = None,
        task_service: Optional[TaskService] = None,
        permission_service: Optional[PermissionService] = None,
        plugin_root: Optional[Path] = None,
        cli_agents_json: Optional[dict] = None,
        agent_service: Optional[AgentService] = None,
        team_service: Optional[TeamService] = None,
        config_service: Optional[RuntimeConfigService] = None,
        teammate_mode: str | None = None,
        emit_console: bool = True,
        interactive_prompt=None,
        mcp_owner_provider: Callable[[], object | None] | None = None,
    ):
        self.registry = registry or CommandRegistry.with_all_commands()
        self.task_service = task_service or TaskService.in_memory()
        self.permission_service = permission_service or PermissionService.default()
        self.plugin_root = plugin_root or (Path.home() / ".koder" / "plugins")
        self.cli_agents_json = cli_agents_json
        self.agent_service = agent_service or AgentService(
            permission_service=self.permission_service
        )
        team_runtime = self.agent_service.get_team_tool_runtime()
        self.team_service = team_service or team_runtime.team_service

        # Create permission bridge for in-process teammates
        from koder_agent.harness.agents.teams.permission_bridge import (
            PermissionBridge,
            PermissionRequest,
            PermissionResponse,
        )

        async def leader_permission_handler(req: PermissionRequest) -> PermissionResponse:
            # For now, auto-approve (the leader's own permission service will re-check)
            return PermissionResponse(
                request_id=req.request_id, approved=True, reason="leader approved"
            )

        permission_bridge = PermissionBridge(handler=leader_permission_handler)

        self.in_process_teammate_runner = team_runtime.configure(
            team_service=self.team_service,
            permission_bridge=permission_bridge,
            local_prompt_executor=self._execute_teammate_local_prompt,
        )
        self.config_service = config_service or RuntimeConfigService()
        self.plan_mode_service = PlanModeService()
        self._pre_plan_permission_mode = self.permission_service.mode
        self.teammate_mode_override = teammate_mode
        self.additional_skill_dirs: list[Path] = []
        self.command_aliases: dict[str, str] = {
            "reset": "clear",
            "new": "clear",
            "settings": "config",
            "quit": "exit",
            "bug": "feedback",
            "allowed-tools": "permissions",
            "continue": "resume",
            "checkpoint": "rewind",
            "pr_comments": "pr-comments",
            "magic_docs": "magic-docs",
        }
        self.current_color = "default"
        self.current_theme = _load_output_theme()
        self.vim_enabled = _load_vim_enabled()
        config = get_config()
        self.current_model = config.model.name
        self.current_model_provider = config.model.provider
        self.emit_console = emit_console
        self._pending_input_text: str | None = None
        self.interactive_prompt = interactive_prompt
        self._mcp_owner_provider = mcp_owner_provider
        self.commands: Dict[str, InteractiveCommand] = {
            "help": self._execute_help,
            "init": self._execute_init,
            "clear": self._execute_clear,
            "status": self._execute_status,
            "config": self._execute_config,
            "model": self._execute_model,
            "channels": self._execute_channels,
            "mcp": self._execute_mcp,
            "session": self._execute_session,
            "resume": self._execute_resume,
            "rename": self._execute_rename,
            "skills": self._execute_skills,
            "plugin": self._execute_plugin,
            "reload-plugins": self._execute_reload_plugins,
            "files": self._execute_files,
            "goal": self._execute_goal,
            "magic-docs": self._execute_magic_docs,
            "diff": self._execute_diff,
            "context": self._execute_context,
            "cost": self._execute_cost,
            "doctor": self._execute_doctor,
            "memory": self._execute_memory,
            "assistant": self._execute_assistant,
            "init-verifiers": self._execute_init_verifiers,
            "thinkback": self._execute_thinkback,
            "thinkback-play": self._execute_thinkback_play,
            "tasks": self._execute_tasks,
            "permissions": self._execute_permissions,
            "theme": self._execute_theme,
            "keybindings": self._execute_keybindings,
            "output-style": self._execute_output_style,
            "usage": self._execute_usage,
            "effort": self._execute_effort,
            "reasoning": self._execute_reasoning,
            "export": self._execute_export,
            "commit": self._execute_commit,
            "commit-push-pr": self._execute_commit_push_pr,
            "review": self._execute_review,
            "advisor": self._execute_advisor,
            "brief": self._execute_brief,
            "buddy": self._execute_buddy,
            "compact": self._execute_compact,
            "branch": self._execute_branch,
            "rewind": self._execute_rewind,
            "exit": self._execute_exit,
            "plan": self._execute_plan,
            "hooks": self._execute_hooks,
            "vim": self._execute_vim,
            "release-notes": self._execute_release_notes,
            "version": self._execute_version,
            "env": self._execute_env,
            "add-dir": self._execute_add_dir,
            "agents": self._execute_agents,
            "peers": self._execute_peers,
            "feedback": self._execute_feedback,
            "statusline": self._execute_statusline,
            "color": self._execute_color,
            "btw": self._execute_btw,
            "insights": self._execute_insights,
            "sandbox": self._execute_sandbox,
            "loop": self._execute_loop,
            "schedule": self._execute_schedule,
            "torch": self._execute_torch,
            "ultraplan": self._execute_ultraplan,
            "fork": self._execute_fork,
            "issue": self._execute_issue,
            "pr_comments": self._execute_pr_comments,
            "pr-comments": self._execute_pr_comments,
            "voice": self._execute_voice,
            "ctx_viz": self._execute_ctx_viz,
            "security-review": self._execute_security_review,
            "summary": self._execute_summary,
            "onboarding": self._execute_onboarding,
            "autofix-pr": self._execute_autofix_pr,
            "subscribe-pr": self._execute_subscribe_pr,
            "oauth-refresh": self._execute_oauth_refresh,
            "backfill-sessions": self._execute_backfill_sessions,
            "bughunter": self._execute_bughunter,
            "debug-tool-call": self._execute_debug_tool_call,
        }
        self._register_static_command_messages(
            {
                "btw": "btw: btw workflow is registered in the harness runtime.",
                "commit": "commit: use /commit for commit readiness and git status details.",
                "commit-push-pr": "commit-push-pr: use /commit-push-pr for branch and diff readiness details.",
                "release-notes": "release-notes: use /release-notes to inspect recent commits in this workspace.",
            }
        )
        for name in self.registry.list_names():
            self.commands.setdefault(name, self._make_fallback_handler(name))

    def consume_pending_input_text(self) -> str | None:
        """Return and clear any input text restored by a slash command."""
        pending = self._pending_input_text
        self._pending_input_text = None
        return pending

    @staticmethod
    def _user_message_text(item: object) -> str | None:
        if not isinstance(item, dict):
            return None
        if item.get("role") != "user":
            return None

        content = item.get("content")
        if isinstance(content, str):
            text = content.strip()
            return text or None
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    raw_text = block.get("text")
                    if isinstance(raw_text, str) and raw_text.strip():
                        text_parts.append(raw_text.strip())
            if text_parts:
                return "\n".join(text_parts)
        return None

    @staticmethod
    def _truncate_prompt_preview(text: str, limit: int = 72) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def get_command_list(self) -> List[Tuple[str, str]]:
        command_list: list[tuple[str, str]] = []
        for name in self.commands:
            spec = self.registry.get(name)
            description = spec.help_text if spec else f"Execute /{name}"
            command_list.append((name, description))
        for skill in self._user_visible_skills().values():
            if skill.name in self.commands:
                continue
            description = skill.description or "Execute skill"
            if skill.argument_hint:
                description = f"{description} {skill.argument_hint}"
            command_list.append((skill.name, description))
        # Add standalone .koder/commands prompt commands
        for prompt_command in self._prompt_commands().values():
            if prompt_command.name in self.commands:
                continue
            description = prompt_command.description or "Prompt command"
            if prompt_command.argument_hint:
                description = f"{description} {prompt_command.argument_hint}"
            command_list.append((prompt_command.name, description))
        # Add MCP prompt commands
        from koder_agent.mcp.prompts import get_prompt_registry

        owner = self._mcp_owner_provider() if self._mcp_owner_provider is not None else None
        prompts = get_prompt_registry(owner).list_prompts() if owner is not None else []
        for prompt in prompts:
            command_list.append(
                (
                    prompt.command_name,
                    prompt.description or f"MCP prompt from {prompt.server_name}",
                )
            )
        return sorted(command_list, key=lambda item: item[0])

    def is_slash_command(self, user_input: str) -> bool:
        return user_input.strip().startswith("/")

    def _with_teammate_sender(
        self,
        user_input: str,
        team_context: TeamToolContext | None,
    ) -> str:
        if team_context is None:
            return user_input
        parts = user_input.strip().split()
        if len(parts) < 5 or parts[:2] != ["/peers", "send"]:
            return user_input
        if "--from" in parts[4:]:
            return user_input
        return " ".join(parts[:4] + ["--from", team_context.sender_name] + parts[4:])

    async def _execute_teammate_local_prompt(
        self,
        user_input: str,
        team_context: TeamToolContext | None,
    ) -> str:
        user_input = self._with_teammate_sender(user_input, team_context)
        response = await self.handle_slash_input(user_input, scheduler=None)
        return response or ""

    async def handle_slash_input(self, user_input: str, scheduler) -> Optional[str]:
        if not self.is_slash_command(user_input):
            return None

        if user_input.strip() == "/":
            return await self._show_command_selection()

        parts = user_input[1:].split()
        if not parts:
            return await self._show_command_selection()

        raw_command_name = parts[0]
        exact_mcp_prompt = None
        prompt_registry = None
        if raw_command_name.startswith("mcp__"):
            from koder_agent.mcp.prompts import get_prompt_registry

            owner = getattr(scheduler, "_mcp_servers", None) if scheduler else None
            if owner is not None:
                prompt_registry = get_prompt_registry(owner)
                exact_mcp_prompt = prompt_registry.get(raw_command_name)
        command_name = (
            raw_command_name
            if raw_command_name in self.commands or exact_mcp_prompt is not None
            else raw_command_name.lower()
        )
        if command_name not in self.commands and exact_mcp_prompt is None:
            command_name = self.command_aliases.get(command_name, command_name)
        args = parts[1:]

        if command_name not in self.commands:
            if exact_mcp_prompt is not None:
                return await self._execute_mcp_prompt(exact_mcp_prompt, scheduler, args)
            skills = self._user_visible_skills()
            if command_name in skills:
                return await self._execute_dynamic_skill(skills[command_name], scheduler, args)
            # Check standalone .koder/commands prompt commands
            prompt_commands = self._prompt_commands()
            if command_name in prompt_commands:
                return await self._execute_prompt_command(
                    prompt_commands[command_name], scheduler, args
                )
            # Check MCP prompt commands
            if command_name.startswith("mcp__"):
                from koder_agent.mcp.prompts import get_prompt_registry

                owner = getattr(scheduler, "_mcp_servers", None) if scheduler else None
                if prompt_registry is None and owner is not None:
                    prompt_registry = get_prompt_registry(owner)
                prompt = prompt_registry.get(command_name) if prompt_registry is not None else None
                if prompt is not None:
                    return await self._execute_mcp_prompt(prompt, scheduler, args)
            available = ", ".join(self.commands.keys())
            return f"❌ Unknown command '{command_name}'. Available commands: {available}"

        try:
            self._refresh_scheduler_usage(scheduler)
            return await self.commands[command_name](scheduler, args)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            return f"❌ Error executing command '{command_name}': {exc}"

    def _refresh_scheduler_usage(self, scheduler) -> None:
        usage_tracker = getattr(scheduler, "usage_tracker", None)
        usage_path = getattr(scheduler, "usage_path", None)
        if usage_tracker is None or usage_path is None:
            return
        path = Path(usage_path)
        if not path.exists():
            return
        try:
            usage_tracker.load(path)
        except Exception:
            return

    async def _show_command_selection(self) -> str:
        command_lines = ["Available Slash Commands"]
        for name, description in self.get_command_list():
            command_lines.append(f"/{name}: {description}")
        plain_text = "\n".join(command_lines)

        if not self.emit_console:
            return plain_text

        command_list = self.get_command_list()
        table = Table(
            title="📋 Available Slash Commands",
            show_header=True,
            header_style="bold cyan",
            caption=(
                f"Showing {min(len(command_list), 24)} of {len(command_list)} commands. "
                "Keep typing after / to filter."
            ),
        )
        table.add_column("Command", style="cyan", width=16)
        table.add_column("Description", style="white")
        for name, description in command_list[:24]:
            table.add_row(f"/{name}", description)
        print_reflowable(console, table)
        return ""

    async def _execute_help(self, _scheduler, _args: list[str]) -> str:
        catalog_lines = [
            f"/{name:<20} {self.registry.get(name).help_text}"
            for name in sorted(self.registry.list_names())
            if self.registry.get(name) is not None
        ]
        if _args:
            command_name = _args[0].lstrip("/")
            spec = self.registry.get(command_name)
            if spec is None:
                return f"help: unknown command /{command_name}\nUse /help to list commands."
            aliases = (
                f"\naliases: {', '.join('/' + alias for alias in spec.aliases)}"
                if spec.aliases
                else ""
            )
            return f"/{spec.name}: {spec.help_text}{aliases}"

        command_catalog = "\n".join(catalog_lines)
        usage_text = f"""[bold cyan]Koder - Harness Runtime Commands[/bold cyan]  [dim]general   commands   skills[/dim]

Koder understands your codebase, edits files with your permission, and runs local commands from this terminal.

[bold]Shortcuts[/bold]
! for shell mode              Ctrl+C to cancel or exit          Ctrl+R for history search
/ for commands                Tab to cycle completions          Right/Tab to accept ghost text
@ for file paths              Shift+Enter for newline           Ctrl+J/Alt+Enter fallback

[bold]Available Slash Commands[/bold]
/help      command guide        /status    runtime state        /skills    available skills
/doctor    environment checks   /diff      pending changes      /model     active model

[bold]Command Catalog[/bold]
{command_catalog}

[dim]Type / to browse commands. Keep typing after / to filter the short completion list.[/dim]
"""
        if not self.emit_console:
            return (
                "Koder - Harness Runtime Commands\n\n"
                "Shortcuts:\n"
                "- ! for shell mode\n"
                "- / for commands\n"
                "- @ for file paths\n"
                "- Ctrl+R for history search\n\n"
                "Available Slash Commands:\n"
                "/help, /status, /skills, /doctor, /diff, /model\n\n"
                "Command Catalog:\n"
                f"{command_catalog}\n"
            )
        print_reflowable(console, Panel(usage_text, title="📖 Help", border_style="cyan"))
        return ""

    def _detect_init_commands(self, cwd: Path) -> list[str]:
        commands: list[str] = []
        if (cwd / "pyproject.toml").exists() or (cwd / "uv.lock").exists():
            commands.extend(
                [
                    "uv sync",
                    "uv run pytest",
                    "uv run ruff check",
                    "uv run black .",
                ]
            )

        package_json = cwd / "package.json"
        if package_json.exists():
            commands.append("npm install")
            try:
                package_data = json.loads(package_json.read_text(encoding="utf-8"))
            except Exception:
                package_data = {}
            scripts = package_data.get("scripts", {}) if isinstance(package_data, dict) else {}
            if isinstance(scripts, dict):
                for script_name in ("test", "lint", "typecheck", "build", "dev"):
                    if script_name in scripts:
                        commands.append(f"npm run {script_name}")

        makefile = cwd / "Makefile"
        if makefile.exists():
            try:
                makefile_text = makefile.read_text(encoding="utf-8")
            except Exception:
                makefile_text = ""
            for target in ("test", "lint", "format", "build"):
                if re.search(rf"^{re.escape(target)}\s*:", makefile_text, re.MULTILINE):
                    commands.append(f"make {target}")

        deduped: list[str] = []
        for command in commands:
            if command not in deduped:
                deduped.append(command)
        return deduped

    def _render_default_agents_md(self, cwd: Path) -> tuple[str, int]:
        commands = self._detect_init_commands(cwd)
        command_lines = [f"- `{command}`" for command in commands]
        if not command_lines:
            command_lines = [
                "- Inspect project manifests such as `pyproject.toml`, `package.json`, and `Makefile` before running build or test commands.",
                "- Run the narrowest relevant test first, then broader checks when the change touches shared behavior.",
            ]

        content = [
            "# AGENTS.md",
            "",
            "This file provides guidance to Koder and other AI agents when working with this repository.",
            "",
            "## Project Context",
            "",
            f"- Repository root: `{cwd.name}`",
            "- Read nearby code and existing documentation before changing behavior.",
            "- Keep edits scoped to the requested task and avoid unrelated refactors.",
            "",
            "## Commands",
            "",
            *command_lines,
            "",
            "## Working Guidelines",
            "",
            "- Prefer established project patterns over new abstractions.",
            "- Use structured parsers or project helpers when they are available.",
            "- Add or update focused tests for behavior changes.",
            "- Do not overwrite user changes or generated state that is unrelated to the task.",
            "",
        ]
        return "\n".join(content), len(commands)

    async def _execute_init(self, scheduler, _args: list[str]) -> str:
        if _args:
            return "Usage: /init"

        koder_md_path = Path(os.getcwd()) / "AGENTS.md"
        if koder_md_path.exists():
            return "AGENTS.md already exists. Run /init-explore to improve it from the codebase."

        content, command_count = self._render_default_agents_md(Path.cwd())
        koder_md_path.write_text(content, encoding="utf-8")

        # After generating AGENTS.md, scan for and report any existing magic docs
        from koder_agent.harness.magic_docs import find_magic_docs

        lines = [
            "AGENTS.md generated.",
            f"path: {koder_md_path}",
            f"commands_detected: {command_count}",
        ]
        try:
            magic_docs = find_magic_docs(Path.cwd())
            if magic_docs:
                lines.append(f"Found {len(magic_docs)} magic doc(s):")
                lines.extend(
                    f"  - {doc.path.relative_to(Path.cwd())}: {doc.title}" for doc in magic_docs
                )
        except Exception:
            pass  # Don't fail init if magic doc scanning fails

        lines.append(
            "tip: run /init-explore to have Koder explore the codebase and enrich AGENTS.md."
        )
        return "\n".join(lines)

    async def _execute_magic_docs(self, _scheduler, args: list[str]) -> str:
        from koder_agent.harness.magic_docs import (
            format_magic_docs_status,
            refresh_tracked_magic_docs,
        )

        if args and args[0] not in {"status", "list", "refresh"}:
            return "Usage: /magic-docs [status|refresh]"

        if args and args[0] == "refresh":
            results = refresh_tracked_magic_docs(
                "Manual /magic-docs refresh",
                "Koder refreshed tracked Magic Docs from the current TUI session.",
                cwd=Path.cwd(),
                include_untracked=True,
            )
            changed = sum(1 for result in results if result.changed)
            lines = [
                "magic_docs: refresh",
                f"  checked: {len(results)}",
                f"  updated: {changed}",
            ]
            if results:
                lines.append("  docs:")
                for result in results:
                    try:
                        display_path = result.path.relative_to(Path.cwd())
                    except ValueError:
                        display_path = result.path
                    lines.append(f"    - {display_path}: {result.status} ({result.message})")
            return "\n".join(lines)

        return format_magic_docs_status(Path.cwd())

    async def _execute_clear(self, _scheduler, _args: list[str]) -> str:
        from koder_agent.utils import default_session_local_ms

        return f"session_switch_clear:{default_session_local_ms()}"

    async def _execute_status(self, scheduler, _args: list[str]) -> str:
        model = get_model_name()
        session_id = scheduler.session.session_id if scheduler is not None else "unknown"
        command_count = len(self.get_command_list())
        resolved = resolve_runtime_version()
        return (
            f"version: {resolved}\n"
            f"Model: {model}\n"
            f"Session: {session_id}\n"
            f"provider: {self.current_model_provider}\n"
            "account: local-runtime\n"
            "connectivity: local\n"
            f"Runtime slash commands: {command_count}\n"
            f"Working directory: {os.getcwd()}"
        )

    async def _execute_config(self, _scheduler, _args: list[str]) -> str:
        config = get_config()
        rendered = yaml.safe_dump(
            config.model_dump(exclude_none=False),
            sort_keys=False,
            allow_unicode=True,
        ).strip()
        return rendered or "No config loaded."

    @staticmethod
    async def _reset_scheduler_agent(scheduler) -> bool:
        if scheduler is None:
            return False
        reset_agent = getattr(scheduler, "reset_agent", None)
        if callable(reset_agent):
            await reset_agent()
            return True
        if hasattr(scheduler, "dev_agent") or hasattr(scheduler, "_agent_initialized"):
            try:
                setattr(scheduler, "dev_agent", None)
                setattr(scheduler, "_agent_initialized", False)
                return True
            except Exception:
                return False
        return False

    async def _execute_model(self, _scheduler, _args: list[str]) -> str:
        manager = get_config_manager()
        config = manager.load()
        if _args:
            selected = _args[0].strip()
            provider, model_name, env_model = _normalize_model_selection(
                selected,
                current_provider=config.model.provider or self.current_model_provider,
            )
            config.model.name = model_name
            config.model.provider = provider
            manager.save(config)
            os.environ["KODER_MODEL"] = env_model
            self.current_model = model_name
            self.current_model_provider = provider
            agent_reloaded = await self._reset_scheduler_agent(_scheduler)
            return (
                f"model: {self.current_model}\n"
                f"provider: {self.current_model_provider}\n"
                f"effective_model: {get_model_name()}\n"
                f"settings_path: {manager.config_path}\n"
                f"agent_reloaded: {agent_reloaded}"
            )

        env_model = os.environ.get("KODER_MODEL")
        if env_model:
            provider, model_name, _ = _normalize_model_selection(
                env_model,
                current_provider=config.model.provider or self.current_model_provider,
            )
        else:
            provider = config.model.provider
            model_name = config.model.name
        self.current_model = model_name
        self.current_model_provider = provider
        return (
            f"model: {self.current_model}\n"
            f"provider: {self.current_model_provider}\n"
            f"effective_model: {get_model_name()}"
        )

    async def _execute_mcp(self, _scheduler, _args: list[str]) -> str:
        if _args:
            return "Usage: /mcp"
        manager = MCPServerManager()
        servers = await manager.list_servers(cwd=os.getcwd())
        if not servers:
            return "No MCP servers configured."
        lines = []
        for server in servers:
            target = server.url or f"{server.command} {' '.join(server.args or [])}".strip()
            scope = getattr(server.scope, "value", server.scope) or "unknown"
            lines.append(
                f"- {server.name} [{scope}]: {server.transport_type.value} {target}".strip()
            )
        return "\n".join(lines)

    async def _get_session_color(self, scheduler) -> Optional[str]:
        session = getattr(scheduler, "session", None) if scheduler is not None else None
        if session is None:
            return None
        if hasattr(session, "get_color"):
            return await session.get_color()
        return None

    async def _set_session_color(self, scheduler, color: Optional[str]) -> None:
        session = getattr(scheduler, "session", None) if scheduler is not None else None
        if session is None:
            return
        if hasattr(session, "set_color"):
            await session.set_color(color)

    async def _get_session_cwd(self, scheduler) -> Optional[str]:
        session = getattr(scheduler, "session", None) if scheduler is not None else None
        if session is None:
            return None
        if hasattr(session, "get_cwd"):
            return await session.get_cwd()
        return None

    async def _execute_session(self, scheduler, _args: list[str]) -> str:
        if scheduler is None:
            return "No active session."
        session = scheduler.session
        display_name = await session.get_display_name()
        lines = [f"session_id: {session.session_id}\ndisplay_name: {display_name}"]
        if hasattr(session, "get_title"):
            title = await session.get_title()
            if title:
                lines.append(f"title: {title}")
        current_tag = None
        if hasattr(session, "get_tag"):
            current_tag = await session.get_tag()
        if current_tag:
            lines.append(f"tag: {current_tag}")
        current_color = await self._get_session_color(scheduler)
        if current_color:
            lines.append(f"color: {current_color}")
        current_cwd = await self._get_session_cwd(scheduler)
        if current_cwd:
            lines.append(f"cwd: {current_cwd}")
        if getattr(scheduler, "agent_definition", None) is not None:
            lines.append(f"agent: {scheduler.agent_definition.agent_type}")
        lines.append("hint: use /resume to switch sessions")
        return "\n".join(lines)

    async def _list_resume_candidates(
        self, *, current_session_id: str | None = None
    ) -> list[tuple[str, str | None]]:
        from koder_agent.core.session import EnhancedSQLiteSession

        sessions = await EnhancedSQLiteSession.list_sessions_with_titles()
        if current_session_id is not None:
            sessions = [(sid, title) for sid, title in sessions if sid != current_session_id]
        sessions.sort(
            key=lambda item: (parse_session_dt(item[0])[0], parse_session_dt(item[0])[1] or None),
            reverse=True,
        )
        return sessions

    async def _resolve_resume_target(
        self, target: str, *, current_session_id: str | None = None
    ) -> str:
        candidate = target.strip()
        sessions = await self._list_resume_candidates(current_session_id=current_session_id)
        exact_session_match = next((sid for sid, _title in sessions if sid == candidate), None)
        if exact_session_match is not None:
            return f"session_switch:{exact_session_match}"

        title_matches = [(sid, title) for sid, title in sessions if title == candidate]
        if len(title_matches) == 1:
            return f"session_switch:{title_matches[0][0]}"
        if len(title_matches) > 1:
            return (
                f"Found {len(title_matches)} sessions matching {candidate}. "
                "Please use /resume to pick a specific session."
            )
        return f"Session {candidate} was not found."

    async def _execute_resume(self, _scheduler, _args: list[str]) -> str:
        if _args:
            current_session_id = (
                _scheduler.session.session_id
                if _scheduler is not None and hasattr(_scheduler, "session")
                else None
            )
            return await self._resolve_resume_target(
                " ".join(_args).strip(),
                current_session_id=current_session_id,
            )

        from koder_agent.harness.session_flow import prompt_select_session

        current_session_id = (
            _scheduler.session.session_id
            if _scheduler is not None and hasattr(_scheduler, "session")
            else None
        )
        selected = await prompt_select_session(current_session_id=current_session_id)
        if not selected:
            return "Resume cancelled"
        return f"session_switch:{selected}"

    @staticmethod
    def _flatten_session_text(content: object) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                else:
                    text = HarnessInteractiveCommandHandler._flatten_session_text(item)
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text.strip()
        return ""

    @staticmethod
    def _slugify_session_name(text: str) -> str | None:
        words = re.findall(r"[a-z0-9]+", text.lower())
        if not words:
            return None
        stop_words = {
            "a",
            "an",
            "and",
            "callback",
            "for",
            "help",
            "i",
            "in",
            "me",
            "my",
            "of",
            "on",
            "our",
            "please",
            "the",
            "this",
            "to",
            "with",
        }
        filtered = [word for word in words if word not in stop_words]
        selected = filtered or words
        slug = "-".join(selected[:4]).strip("-")
        return slug[:60] if slug else None

    async def _generate_session_name_from_context(self, scheduler) -> str | None:
        if scheduler is None or not hasattr(scheduler, "session"):
            return None
        if not hasattr(scheduler.session, "get_items"):
            return None
        items = await scheduler.session.get_items()
        if not items:
            return None

        primary_text = ""
        transcript_lines: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            text = self._flatten_session_text(item.get("content"))
            if not text or role not in {"user", "assistant"}:
                continue
            if not primary_text and role == "user":
                primary_text = text
            transcript_lines.append(f"{role}: {text}")

        if not primary_text:
            primary_text = next(
                (
                    self._flatten_session_text(item.get("content"))
                    for item in items
                    if isinstance(item, dict) and self._flatten_session_text(item.get("content"))
                ),
                "",
            )
        if not primary_text:
            return None

        conversation_text = "\n".join(transcript_lines[-8:])[:2000]
        try:
            generated = await llm_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate a short kebab-case session name with 2-4 words. "
                            "Return only the name."
                        ),
                    },
                    {"role": "user", "content": conversation_text},
                ]
            )
            normalized = self._slugify_session_name(generated)
            if normalized:
                return normalized
        except Exception:
            pass

        return self._slugify_session_name(primary_text)

    async def _execute_rename(self, scheduler, args: list[str]) -> str:
        if scheduler is None:
            return "No active session to rename."
        if not args:
            new_title = await self._generate_session_name_from_context(scheduler)
            if not new_title:
                return (
                    "Could not generate a name: no conversation context yet. "
                    "Usage: /rename <new title>"
                )
        else:
            new_title = " ".join(args).strip()
        await scheduler.session.set_title(new_title)
        return f"Session renamed to: {new_title}"

    async def _execute_skills(self, _scheduler, _args: list[str]) -> str:
        skills = sorted(
            self._user_visible_skills().values(),
            key=lambda item: (item.source, item.name),
        )
        if not skills:
            return "No skills available."
        lines = []
        for skill in skills:
            suffix: list[str] = []
            if skill.argument_hint:
                suffix.append(skill.argument_hint)
            if skill.disable_model_invocation:
                suffix.append("manual-only")
            if skill.execution_context == "fork":
                suffix.append(f"fork:{skill.agent or 'general-purpose'}")
            details = f" ({', '.join(suffix)})" if suffix else ""
            lines.append(
                f"- [{skill.source}] {skill.name}: {skill.description or '(no description)'}{details}"
            )
        return "\n".join(lines)

    async def _execute_plugin(self, _scheduler, args: list[str]) -> str:
        lifecycle = PluginLifecycleService(self.plugin_root)

        if not args or args[0] == "list":
            registry = PluginRegistry.from_lifecycle(lifecycle, include_disabled=True)
            plugins = registry.list_plugins()
            if not plugins:
                return "No installed plugins."
            lines = []
            for p in plugins:
                status = "enabled" if p.enabled else "disabled"
                comps = ", ".join(p.components) if p.components else ""
                line = f"- {p.name} v{p.version} [{p.scope}] ({status})"
                if comps:
                    line += f" [{comps}]"
                lines.append(line)
            return "\n".join(lines)

        action = args[0]
        if action == "install" and len(args) >= 2:
            plugin_ref = args[1]
            scope = args[2] if len(args) > 2 else "user"

            # Support name@marketplace install
            if "@" in plugin_ref and not Path(plugin_ref).exists():
                from koder_agent.harness.plugins.marketplace import MarketplaceStore

                store = MarketplaceStore.default()
                found = store.find_plugin(plugin_ref)
                if found is None:
                    return f"Plugin '{plugin_ref}' not found in any marketplace"
                plugin_dir = Path(found.path)
            elif Path(plugin_ref).is_dir():
                plugin_dir = Path(plugin_ref).resolve()
            else:
                # Try bare name across all marketplaces
                from koder_agent.harness.plugins.marketplace import MarketplaceStore

                store = MarketplaceStore.default()
                found = store.find_plugin(plugin_ref)
                if found is not None:
                    plugin_dir = Path(found.path)
                else:
                    return f"'{plugin_ref}' is not a directory and not found in any marketplace"

            result = lifecycle.install_from_dir(plugin_dir, scope=scope)
            return (
                result.message
                if result.message
                else (f"Installed {result.plugin_name}" if result.success else "Install failed")
            )

        if action == "uninstall" and len(args) >= 2:
            result = lifecycle.uninstall(args[1])
            return result.message

        if action == "enable" and len(args) >= 2:
            result = lifecycle.enable(args[1])
            return result.message

        if action == "disable" and len(args) >= 2:
            result = lifecycle.disable(args[1])
            return result.message

        return "Usage: /plugin [list|install <path>|uninstall <name>|enable <name>|disable <name>]"

    async def _execute_channels(self, _scheduler, args: list[str]) -> str:
        if args and args[0] in {"help", "-h", "--help"}:
            return (
                "Usage: /channels\n"
                "Enable channels when starting Koder with --channels server:<name> "
                "or --channels plugin:<name>@<marketplace>."
            )
        if args:
            return "Usage: /channels"

        from koder_agent.harness.channels.state import (
            get_allowed_channels,
            get_channel_state,
            get_has_dev_channels,
        )
        from koder_agent.harness.channels.types import (
            ChannelEntryPlugin,
            ChannelEntryServer,
        )

        entries = get_allowed_channels()
        lines = [
            "channels:",
            f"enabled: {'true' if entries else 'false'}",
            f"configured: {len(entries)}",
            f"development_channels: {'true' if get_has_dev_channels() else 'false'}",
            "runtime: MCP servers that declare channel capability can deliver messages into the active session.",
            "usage: uv run koder --channels server:<name>",
            "plugin_usage: uv run koder --channels plugin:<name>@<marketplace>",
        ]
        if entries:
            lines.append("entries:")
            for entry in entries:
                marker = " [development]" if entry.dev else ""
                if isinstance(entry, ChannelEntryServer):
                    lines.append(f"- server:{entry.name}{marker}")
                elif isinstance(entry, ChannelEntryPlugin):
                    lines.append(f"- plugin:{entry.name}@{entry.marketplace}{marker}")
                else:
                    lines.append(f"- {entry}{marker}")
        inbox = get_channel_state().inbox
        if inbox is not None:
            snapshot = inbox.snapshot()
            lines.extend(
                [
                    f"inbox_status: {snapshot['status']}",
                    "inbox_counts: "
                    f"staging={snapshot['admitting']} "
                    f"pending={snapshot['pending']} running={snapshot['running']} "
                    f"failed={snapshot['failed']} cancelled={snapshot['cancelled']} "
                    f"interrupted={snapshot['interrupted']} completed={snapshot['completed']}",
                    "inbox_limits: "
                    f"messages={snapshot['retained_messages'] + snapshot['admitting']}/{inbox.limits.max_messages} "
                    f"bytes={snapshot['retained_bytes'] + snapshot['admitting_bytes']}/{inbox.limits.max_bytes} "
                    f"max_message={inbox.limits.max_message_bytes}",
                    f"inbox_rejected: {snapshot['rejected']}",
                    f"inbox_path: {snapshot['path']}",
                ]
            )
        return "\n".join(lines)

    async def _execute_reload_plugins(self, _scheduler, _args: list[str]) -> str:
        lifecycle = PluginLifecycleService(self.plugin_root)
        registry = PluginRegistry.from_lifecycle(lifecycle, include_disabled=True)
        return f"Reloaded {len(registry.list_names())} plugins."

    @staticmethod
    def _extract_tool_argument_path(tool_name: str, payload: object) -> str | None:
        """Return a persisted call's canonical path, or ignore it fail-closed.

        Transcript entries are untrusted persisted data. Malformed JSON,
        non-object arguments, and ambiguous/wrong aliases contribute no path,
        preventing a decoy alias from being displayed or read for token estimates.
        """
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(payload, dict):
            return None
        try:
            normalized = normalize_tool_arguments(tool_name, payload)
        except ToolArgumentError:
            return None
        target = extract_canonical_tool_target(tool_name, normalized)
        return target.strip() if isinstance(target, str) and target.strip() else None

    def _collect_context_file_paths_from_item(self, item: object) -> list[str]:
        if not isinstance(item, dict):
            return []

        tracked_tools = {"read_file", "write_file", "edit_file", "notebook_edit"}
        collected: list[str] = []

        def collect_path(tool_name: str, payload: object) -> None:
            path = self._extract_tool_argument_path(tool_name, payload)
            if path:
                collected.append(path)

        item_type = item.get("type")
        item_name = item.get("name")
        if (
            item_type in {"function_call", "tool_called"}
            and isinstance(item_name, str)
            and item_name in tracked_tools
        ):
            collect_path(
                item_name,
                item["arguments"] if "arguments" in item else item.get("tool_input"),
            )

        tool_name = item.get("tool_name")
        if isinstance(tool_name, str) and tool_name in tracked_tools:
            collect_path(
                tool_name,
                item["tool_input"] if "tool_input" in item else item.get("arguments"),
            )

        tool_calls = item.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function_payload = call.get("function")
                if isinstance(function_payload, dict):
                    function_name = function_payload.get("name")
                    if isinstance(function_name, str) and function_name in tracked_tools:
                        collect_path(function_name, function_payload.get("arguments"))
                call_name = call.get("name")
                if isinstance(call_name, str) and call_name in tracked_tools:
                    collect_path(
                        call_name,
                        call["arguments"] if "arguments" in call else call.get("tool_input"),
                    )

        return collected

    def _format_context_file_path(self, raw_path: str) -> str:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        try:
            return os.path.relpath(path, Path.cwd())
        except ValueError:
            return str(path)

    @staticmethod
    def _format_token_count(value: int) -> str:
        return f"{value:,}"

    def _collect_context_file_paths(self, items: list[object]) -> list[str]:
        files: list[str] = []
        seen: set[str] = set()
        for item in items:
            for raw_path in self._collect_context_file_paths_from_item(item):
                display_path = self._format_context_file_path(raw_path)
                if display_path in seen:
                    continue
                seen.add(display_path)
                files.append(display_path)
        files.sort(key=str.lower)
        return files

    def _estimate_context_file_tokens(self, file_paths: list[str]) -> int:
        total = 0
        for display_path in file_paths:
            absolute_path = (Path.cwd() / display_path).resolve()
            try:
                text = absolute_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = display_path
            if len(text) > 20000:
                text = text[:20000]
            total += estimate_text_tokens(text)
        return total

    def _estimate_instruction_tokens(self) -> tuple[int, list[str]]:
        instruction_files: list[str] = []
        total = 0
        agents_md = Path.cwd() / "AGENTS.md"
        if agents_md.exists():
            try:
                text = agents_md.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = agents_md.name
            total += estimate_text_tokens(text[:20000])
            instruction_files.append("AGENTS.md")
        return total, instruction_files

    @staticmethod
    def _coerce_json_dict(value: object) -> dict | None:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    @staticmethod
    def _count_diff_lines(lines: list[object]) -> tuple[int, int]:
        added = 0
        removed = 0
        for line in lines:
            if not isinstance(line, str):
                continue
            if line.startswith("+"):
                added += 1
            elif line.startswith("-"):
                removed += 1
        return added, removed

    @staticmethod
    def _count_text_lines(text: str) -> int:
        if not text:
            return 0
        return len(text.splitlines()) or 1

    def _extract_file_edit_result(self, item: object) -> dict | None:
        if not isinstance(item, dict):
            return None
        candidates = [
            item.get("toolUseResult"),
            item.get("tool_use_result"),
            (
                item.get("output")
                if item.get("type") in {"tool_output", "function_call_output"}
                else None
            ),
        ]
        if isinstance(item.get("filePath"), str):
            candidates.append(item)
        for candidate in candidates:
            payload = self._coerce_json_dict(candidate)
            if payload is None:
                continue
            if isinstance(payload.get("filePath"), str) and (
                isinstance(payload.get("structuredPatch"), list)
                or payload.get("type") in {"create", "update"}
            ):
                return payload
        return None

    def _collect_conversation_diffs(self, items: list[object]) -> list[dict]:
        turns: list[dict] = []
        current_turn: dict | None = None
        turn_index = 0
        for item in items:
            if (
                isinstance(item, dict)
                and item.get("role") == "user"
                and not any(key in item for key in ("toolUseResult", "tool_use_result"))
            ):
                if current_turn is not None and current_turn["files"]:
                    turns.append(current_turn)
                turn_index += 1
                preview = self._flatten_session_text(item.get("content"))
                current_turn = {
                    "turn_index": turn_index,
                    "preview": preview[:60] if preview else "",
                    "files": {},
                }
                continue

            if current_turn is None:
                continue

            result = self._extract_file_edit_result(item)
            if result is None:
                continue

            file_path = result["filePath"]
            entry = current_turn["files"].setdefault(
                file_path,
                {"file_path": file_path, "added": 0, "removed": 0, "is_new_file": False},
            )
            structured_patch = result.get("structuredPatch")
            if isinstance(structured_patch, list) and structured_patch:
                for hunk in structured_patch:
                    if isinstance(hunk, dict):
                        added, removed = self._count_diff_lines(hunk.get("lines", []))
                        entry["added"] += added
                        entry["removed"] += removed
            elif result.get("type") == "create" and isinstance(result.get("content"), str):
                entry["added"] += self._count_text_lines(result["content"])
                entry["is_new_file"] = True
            if result.get("type") == "create":
                entry["is_new_file"] = True

        if current_turn is not None and current_turn["files"]:
            turns.append(current_turn)
        return turns

    @staticmethod
    def _format_change_summary(added: int, removed: int) -> str:
        return f"(+{added} -{removed})"

    def _collect_git_diff(self) -> tuple[dict[str, int], list[dict[str, object]]]:
        proc = subprocess.run(
            ["git", "diff", "HEAD", "--numstat"],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            return {"files": 0, "added": 0, "removed": 0}, []
        files: list[dict[str, object]] = []
        total_added = 0
        total_removed = 0
        for raw_line in proc.stdout.splitlines():
            if not raw_line.strip():
                continue
            parts = raw_line.split("\t")
            if len(parts) < 3:
                continue
            add_str, remove_str = parts[0], parts[1]
            path = "\t".join(parts[2:])
            added = 0 if add_str == "-" else int(add_str or "0")
            removed = 0 if remove_str == "-" else int(remove_str or "0")
            total_added += added
            total_removed += removed
            files.append({"path": path, "added": added, "removed": removed})
        return {"files": len(files), "added": total_added, "removed": total_removed}, files

    @staticmethod
    def _detect_installation_type() -> str:
        from koder_agent.harness.diagnostics import detect_installation_type

        return detect_installation_type()

    @staticmethod
    def _detect_invoked_binary() -> str:
        from koder_agent.harness.diagnostics import detect_invoked_binary

        return detect_invoked_binary()

    @staticmethod
    def _detect_ripgrep_status() -> tuple[bool, str, str]:
        from koder_agent.harness.diagnostics import detect_ripgrep_status

        return detect_ripgrep_status()

    async def _execute_files(self, _scheduler, _args: list[str]) -> str:
        if _scheduler is None or not hasattr(_scheduler, "session"):
            return "No files in context"
        if not hasattr(_scheduler.session, "get_items"):
            return "No files in context"

        items = await _scheduler.session.get_items()
        files = self._collect_context_file_paths(items)
        if not files:
            return "No files in context"
        lines = ["Files in context:"]
        for display_path in files:
            absolute_path = (Path.cwd() / display_path).resolve()
            status = "exists" if absolute_path.exists() else "missing"
            lines.append(f"- {display_path} ({status})")
        return "\n".join(lines)

    @staticmethod
    def _parse_goal_objective_args(args: list[str]) -> tuple[str, str | None, int | None]:
        """Split ``/goal`` args into (objective, error, token_budget).

        Supports an optional trailing ``--budget N`` since the text UI has no
        other way to attach a budget when creating a goal.
        """
        budget: int | None = None
        if "--budget" in args:
            idx = args.index("--budget")
            budget_args = args[idx + 1 :]
            if len(budget_args) != 1:
                return "", "Usage: /goal <objective> [--budget <tokens>]", None
            try:
                budget = int(budget_args[0].replace(",", "").replace("_", ""))
            except ValueError:
                return "", f"Invalid token budget: {budget_args[0]}", None
            args = args[:idx]
        objective = " ".join(args).strip()
        return objective, None, budget

    async def _start_goal_turn(self, scheduler, goal) -> None:
        """Kick off the goal continuation loop for a freshly activated goal."""
        from koder_agent.core.goal_prompts import GOAL_CONTEXT_MARKER, continuation_prompt

        await scheduler.handle(
            f"{GOAL_CONTEXT_MARKER}\n\n{continuation_prompt(goal)}",
            render_output=self.emit_console,
        )

    async def _execute_goal(self, scheduler, args: list[str]) -> str:
        from koder_agent.core.goal_display import (
            GOAL_USAGE,
            GOAL_USAGE_HINT,
            goal_status_label,
            goal_summary_text,
            goal_usage_summary,
            should_confirm_before_replacing_goal,
        )
        from koder_agent.core.goals import (
            GoalStatus,
            GoalUpdate,
            validate_goal_budget,
            validate_goal_objective,
        )

        goal_store = getattr(scheduler, "goal_store", None)
        session = getattr(scheduler, "session", None)
        if goal_store is None or session is None:
            return (
                "Goals need a saved session. This session is temporary.\n"
                "Start Koder normally to use goals."
            )
        session_id = str(session.session_id)

        if not args:
            goal = await goal_store.get_goal(session_id)
            if goal is None:
                return f"{GOAL_USAGE}\nNo goal is currently set.\n{GOAL_USAGE_HINT}"
            return goal_summary_text(goal)

        subcommand = args[0].lower()

        if subcommand == "clear" and len(args) == 1:
            cleared = await goal_store.delete_goal(session_id)
            if cleared is None:
                return "No goal to clear\nThis thread does not currently have a goal."
            return "Goal cleared"

        if subcommand == "pause" and len(args) == 1:
            goal = await goal_store.pause_active_goal(session_id)
            if goal is None:
                current = await goal_store.get_goal(session_id)
                if current is None:
                    return f"No goal is currently set.\n{GOAL_USAGE}"
                return (
                    f"Goal is {goal_status_label(current.status)}; only active goals can be paused."
                )
            return f"Goal {goal_status_label(goal.status)}\n{goal_usage_summary(goal)}"

        if subcommand == "resume" and len(args) == 1:
            current = await goal_store.get_goal(session_id)
            if current is None:
                return f"No goal is currently set.\n{GOAL_USAGE}"
            goal = await goal_store.update_goal(
                session_id,
                GoalUpdate(status=GoalStatus.ACTIVE, expected_goal_id=current.goal_id),
            )
            if goal is None:
                return "Failed to resume goal: the goal changed while resuming."
            header = f"Goal {goal_status_label(goal.status)}\n{goal_usage_summary(goal)}"
            if goal.status is GoalStatus.ACTIVE:
                await self._start_goal_turn(scheduler, goal)
            return header

        if subcommand == "budget":
            if len(args) != 2:
                return "Usage: /goal budget <tokens>"
            try:
                budget = int(args[1].replace(",", "").replace("_", ""))
            except ValueError:
                return f"Invalid token budget: {args[1]}"
            try:
                validate_goal_budget(budget)
            except ValueError as exc:
                return str(exc)
            current = await goal_store.get_goal(session_id)
            if current is None:
                return f"No goal is currently set.\n{GOAL_USAGE}"
            goal = await goal_store.update_goal(
                session_id,
                GoalUpdate(token_budget=budget, expected_goal_id=current.goal_id),
            )
            if goal is None:
                return "Failed to update goal budget: the goal changed while updating."
            return f"Goal {goal_status_label(goal.status)}\n{goal_usage_summary(goal)}"

        if subcommand == "edit":
            objective, error, budget = self._parse_goal_objective_args(args[1:])
            if error:
                return error
            current = await goal_store.get_goal(session_id)
            if current is None:
                return f"No goal is currently set.\nCreate a goal before editing it.\n{GOAL_USAGE}"
            if not objective:
                return f"Usage: /goal edit <objective>\nCurrent objective: {current.objective}"
            try:
                validate_goal_objective(objective)
                validate_goal_budget(budget)
            except ValueError as exc:
                return str(exc)
            # Editing a budget-limited/complete goal reactivates it; stopped
            # statuses (paused/blocked/usage-limited) are preserved.
            if current.status in (GoalStatus.BUDGET_LIMITED, GoalStatus.COMPLETE):
                new_status = GoalStatus.ACTIVE
            else:
                new_status = current.status
            update = GoalUpdate(
                objective=objective,
                status=new_status,
                expected_goal_id=current.goal_id,
            )
            if budget is not None:
                update.token_budget = budget
            goal = await goal_store.update_goal(session_id, update)
            if goal is None:
                return "Failed to edit goal: the goal changed while editing."
            header = f"Goal {goal_status_label(goal.status)}\n{goal_usage_summary(goal)}"
            if goal.status is GoalStatus.ACTIVE:
                await self._start_goal_turn(scheduler, goal)
            return header

        replace_requested = subcommand == "replace"
        objective_args = args[1:] if replace_requested else args
        objective, error, budget = self._parse_goal_objective_args(objective_args)
        if error:
            return error
        if not objective:
            return GOAL_USAGE
        try:
            validate_goal_objective(objective)
            validate_goal_budget(budget)
        except ValueError as exc:
            return str(exc)

        existing = await goal_store.get_goal(session_id)
        if (
            existing is not None
            and should_confirm_before_replacing_goal(existing)
            and not replace_requested
        ):
            return (
                "Replace goal?\n"
                f"Current objective: {existing.objective}\n"
                f"New objective: {objective}\n"
                "Run /goal replace <objective> to replace the current goal, "
                "or /goal to view it."
            )

        goal = await goal_store.replace_goal(session_id, objective, GoalStatus.ACTIVE, budget)
        header = f"Goal {goal_status_label(goal.status)}\n{goal_usage_summary(goal)}"
        if goal.status is GoalStatus.ACTIVE:
            await self._start_goal_turn(scheduler, goal)
        return header

    async def _execute_diff(self, _scheduler, _args: list[str]) -> str:
        git_stats, git_files = self._collect_git_diff()
        conversation_turns = []
        if (
            _scheduler is not None
            and hasattr(_scheduler, "session")
            and hasattr(_scheduler.session, "get_items")
        ):
            items = await _scheduler.session.get_items()
            conversation_turns = self._collect_conversation_diffs(items)

        lines = ["## Diff", "", "### Uncommitted changes"]
        if git_files:
            lines.append(
                f"{git_stats['files']} file(s) changed, {git_stats['added']} insertion(s), {git_stats['removed']} deletion(s)"
            )
            lines.extend(
                [
                    f"- {entry['path']} {self._format_change_summary(int(entry['added']), int(entry['removed']))}"
                    for entry in git_files
                ]
            )
        else:
            lines.append("No uncommitted changes.")

        lines.extend(["", "### Conversation edits"])
        if conversation_turns:
            for turn in conversation_turns:
                preview = f': "{turn["preview"]}"' if turn["preview"] else ""
                lines.append(f"Turn {turn['turn_index']}{preview}")
                for file_entry in turn["files"].values():
                    suffix = " [new file]" if file_entry["is_new_file"] else ""
                    lines.append(
                        f"- {file_entry['file_path']} {self._format_change_summary(file_entry['added'], file_entry['removed'])}{suffix}"
                    )
        else:
            lines.append("No conversation edits recorded.")
        return "\n".join(lines)

    async def _execute_context(self, _scheduler, _args: list[str]) -> str:
        if (
            _scheduler is not None
            and hasattr(_scheduler, "usage_tracker")
            and hasattr(_scheduler, "session")
        ):
            usage = getattr(_scheduler.usage_tracker, "session_usage", None)
            model = getattr(_scheduler.usage_tracker, "model", "unknown")
            items = (
                await _scheduler.session.get_items()
                if hasattr(_scheduler.session, "get_items")
                else []
            )
            conversation_items = [
                item
                for item in items
                if isinstance(item, dict)
                and item.get("role") in {"user", "assistant", "system", "tool"}
            ]
            conversation_tokens = estimate_messages_tokens(conversation_items)
            context_files = self._collect_context_file_paths(items)
            file_tokens = self._estimate_context_file_tokens(context_files)
            instruction_tokens, instruction_files = self._estimate_instruction_tokens()
            estimated_total = conversation_tokens + file_tokens + instruction_tokens
            tracked_total = getattr(usage, "current_context_tokens", 0) if usage else 0
            total_tokens = max(estimated_total, tracked_total)
            max_context = get_context_window_size(model)
            percentage = (total_tokens / max_context * 100) if max_context else 0.0
            rows = [
                ("Conversation", conversation_tokens),
                ("Files", file_tokens),
                ("Instructions", instruction_tokens),
            ]
            lines = [
                "## Context Usage",
                "",
                f"**Model:** {model}  ",
                f"**Tokens:** {self._format_token_count(total_tokens)} / {self._format_token_count(max_context)} ({percentage:.1f}%)",
                "",
                "### Estimated usage by category",
                "",
                "| Category | Tokens | Percentage |",
                "|----------|--------|------------|",
            ]
            for label, token_count in rows:
                if token_count <= 0:
                    continue
                lines.append(
                    f"| {label} | {self._format_token_count(token_count)} | {(token_count / max_context * 100 if max_context else 0.0):.1f}% |"
                )
            free_space = max(0, max_context - total_tokens)
            lines.append(
                f"| Free space | {self._format_token_count(free_space)} | {(free_space / max_context * 100 if max_context else 0.0):.1f}% |"
            )
            if context_files:
                lines.extend(
                    [
                        "",
                        "### Files in context",
                        "",
                        *[f"- {path}" for path in context_files],
                    ]
                )
            if instruction_files:
                lines.extend(
                    [
                        "",
                        "### Instructions",
                        "",
                        *[f"- {path}" for path in instruction_files],
                    ]
                )
            return "\n".join(lines)

        from koder_agent.harness.session_flow import load_context

        return await load_context()

    async def _execute_cost(self, scheduler, _args: list[str]) -> str:
        tracker = getattr(scheduler, "usage_tracker", None) if scheduler else None
        if tracker is None:
            return (
                "requests: 0\ninput_tokens: 0\noutput_tokens: 0\n"
                "cache_read_tokens: 0\ncache_write_tokens: 0\ncontext_tokens: 0\ncost: 0.0000"
            )
        # Prefer the summary() API which splits cache-read tokens and flags cost
        # as unavailable (subscription/OAuth) instead of printing a misleading $0.
        summary = tracker.summary() if hasattr(tracker, "summary") else None
        if summary is not None:
            cost_line = (
                "cost: unavailable (pricing unknown for this model)"
                if getattr(summary, "cost_unavailable", False)
                else f"cost: {summary.total_cost:.4f}"
            )
            return (
                f"requests: {summary.request_count}\n"
                f"input_tokens: {summary.input_tokens}\n"
                f"output_tokens: {summary.output_tokens}\n"
                f"cache_read_tokens: {summary.cache_read_tokens}\n"
                f"cache_write_tokens: {summary.cache_write_tokens}\n"
                f"context_tokens: {summary.context_tokens}\n"
                f"{cost_line}"
            )
        usage = tracker.session_usage
        if usage is None:
            return (
                "requests: 0\ninput_tokens: 0\noutput_tokens: 0\n"
                "cache_read_tokens: 0\ncache_write_tokens: 0\ncontext_tokens: 0\ncost: 0.0000"
            )
        return (
            f"requests: {getattr(usage, 'request_count', 0)}\n"
            f"input_tokens: {getattr(usage, 'input_tokens', 0)}\n"
            f"output_tokens: {getattr(usage, 'output_tokens', 0)}\n"
            f"cache_read_tokens: {getattr(usage, 'cache_read_tokens', 0)}\n"
            f"cache_write_tokens: {getattr(usage, 'cache_write_tokens', 0)}\n"
            f"context_tokens: {getattr(usage, 'current_context_tokens', 0)}\n"
            f"cost: {getattr(usage, 'total_cost', 0.0):.4f}"
        )

    async def _execute_doctor(self, _scheduler, _args: list[str]) -> str:
        from koder_agent.harness.diagnostics import collect_doctor_report, render_doctor_text

        report = await collect_doctor_report()
        return render_doctor_text(report)

    async def _execute_memory(self, _scheduler, _args: list[str]) -> str:
        from pathlib import Path

        from koder_agent.harness.memory.candidates import (
            approve_candidate,
            default_memory_candidate_store,
            default_skill_candidate_store,
            reject_candidate,
        )
        from koder_agent.harness.memory.governance import sanitize_text
        from koder_agent.harness.memory.memory_files import parse_memory_file

        candidate_args = list(_args)
        candidate_command = bool(candidate_args) and candidate_args[0] in {
            "candidates",
            "list",
            "show",
            "approve",
            "reject",
        }
        if candidate_command:
            if candidate_args[0] == "candidates":
                candidate_args = candidate_args[1:]
            memory_store = default_memory_candidate_store()
            skill_store = default_skill_candidate_store()
            action = candidate_args[0] if candidate_args else "list"
            if action == "list":
                candidates = memory_store.list() + skill_store.list()
                if not candidates:
                    return "memory candidates: none pending"
                lines = [f"memory candidates: {len(candidates)} pending"]
                for candidate in sorted(candidates, key=lambda item: (item.created_at, item.id)):
                    description = sanitize_text(candidate.payload.get("description") or "")
                    lines.append(
                        f"- {candidate.id} kind={candidate.kind} state={candidate.state} "
                        f"scope={candidate.storage_scope} "
                        f"origin_project={sanitize_text(candidate.origin_project_root)} "
                        f"origin_session={sanitize_text(candidate.origin_session_id)} "
                        f"description={description}"
                    )
                return "\n".join(lines)
            if len(candidate_args) != 2:
                return "Usage: /memory candidates | /memory show|approve|reject <candidate-id>"
            candidate_id = candidate_args[1]
            try:
                if action == "show":
                    candidate = memory_store.get(candidate_id) or skill_store.get(candidate_id)
                    if candidate is None:
                        return f"memory candidate not found: {candidate_id}"
                    return (
                        f"memory candidate {candidate.id}\n"
                        f"kind: {candidate.kind}\n"
                        f"state: {candidate.state}\n"
                        f"storage_scope: {candidate.storage_scope}\n"
                        f"origin_project_root: {sanitize_text(candidate.origin_project_root)}\n"
                        f"origin_session_id: {sanitize_text(candidate.origin_session_id)}\n"
                        + json.dumps(
                            candidate.payload, indent=2, sort_keys=True, ensure_ascii=False
                        )
                    )
                if action == "approve":
                    result = approve_candidate(
                        candidate_id,
                        memory_store=memory_store,
                        skill_store=skill_store,
                        skill_draft_dir=Path.home() / ".koder" / "skill-drafts",
                    )
                    noun = "memory candidate" if result.kind == "memory" else "skill draft"
                    return f"{noun} approved\nstatus: {result.status}\npath: {result.output_path}"
                if reject_candidate(
                    candidate_id,
                    memory_store=memory_store,
                    skill_store=skill_store,
                ):
                    return f"candidate rejected: {candidate_id}"
                return f"memory candidate not pending: {candidate_id}"
            except (KeyError, ValueError) as exc:
                return f"memory candidate error: {exc}"

        if _args:
            return "Usage: /memory [candidates|show|approve|reject <candidate-id>]"

        memory_dirs = [
            Path.cwd() / ".koder" / "memory",
            Path.home() / ".koder" / "memory",
        ]

        files_found = []
        for mem_dir in memory_dirs:
            if not mem_dir.exists():
                continue
            for md_file in sorted(mem_dir.glob("*.md")):
                if md_file.name == "MEMORY.md":
                    continue
                try:
                    parsed = parse_memory_file(md_file.read_text(encoding="utf-8"))
                    files_found.append(
                        {
                            "file": md_file.name,
                            "type": parsed.memory_type or "unknown",
                            "description": parsed.description or "",
                            "dir": str(mem_dir),
                        }
                    )
                except Exception:
                    continue

        if not files_found:
            return "No memories stored yet. Use the /remember skill to save memories."

        lines = [f"Found {len(files_found)} memory files:\n"]
        for f in files_found:
            lines.append(f"  [{f['type']}] {f['file']}: {f['description']}")
        return "\n".join(lines)

    @staticmethod
    def _slugify_memory_name(text: str) -> str:
        words = re.findall(r"[a-z0-9]+", text.lower())
        slug = "-".join(words[:8]).strip("-")
        return slug[:80] or "memory"

    async def _execute_remember_skill(self, arguments_text: str) -> str:
        from koder_agent.harness.memory.memory_files import save_memory_file

        normalized = " ".join(arguments_text.split())
        if not normalized:
            return "Usage: /remember <what to remember>"

        description = normalized[:120]
        memory_dir = Path.cwd() / ".koder" / "memory"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        memory_path = memory_dir / f"{timestamp}-{self._slugify_memory_name(normalized)}.md"
        save_memory_file(
            memory_path,
            memory_type="project",
            description=description,
            body=normalized,
        )

        index_path = memory_dir / "MEMORY.md"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            relative_memory_path = memory_path.relative_to(Path.cwd())
            relative_index_path = index_path.relative_to(Path.cwd())
        except ValueError:
            relative_memory_path = memory_path
            relative_index_path = index_path
        with index_path.open("a", encoding="utf-8") as index_file:
            index_file.write(f"- {description}: {relative_memory_path}\n")

        return (
            "remember: saved\n"
            f"path: {relative_memory_path}\n"
            "type: project\n"
            f"description: {description}\n"
            f"index: {relative_index_path}"
        )

    async def _get_session_items(self, scheduler) -> list[dict]:
        if scheduler is None or not hasattr(scheduler, "session"):
            return []
        session = scheduler.session
        if not hasattr(session, "get_items"):
            return []
        try:
            items = await session.get_items()
        except Exception:
            return []
        return [item for item in items if isinstance(item, dict)]

    def _session_text_turns(self, items: list[dict]) -> list[tuple[str, str]]:
        turns: list[tuple[str, str]] = []
        for item in items:
            role = str(item.get("role", "")).strip().lower()
            if role not in {"user", "assistant", "system", "tool"}:
                continue
            text = self._flatten_session_text(item.get("content"))
            if text:
                turns.append((role, text))
        return turns

    @staticmethod
    def _parse_positive_limit(args: list[str], default: int, maximum: int) -> int | None:
        if not args:
            return default
        try:
            value = int(args[0])
        except ValueError:
            return None
        if value < 1:
            return None
        return min(value, maximum)

    async def _execute_assistant(self, scheduler, args: list[str]) -> str:
        definitions = get_agent_definitions(
            cwd=Path.cwd(),
            plugin_root=self.plugin_root,
            cli_agents_json=self.cli_agents_json,
        )
        if args and args[0] in {"help", "-h", "--help"}:
            return "Usage: /assistant [list|show <agent-name>]"
        if args and args[0] in {"list", "profiles"}:
            lines = [f"assistant_profiles: {len(definitions.all_agents)}"]
            for agent in sorted(definitions.all_agents, key=lambda item: item.agent_type.lower()):
                model = f" model={agent.model}" if agent.model else ""
                lines.append(f"- {agent.agent_type} [{agent.source}]{model}")
            return "\n".join(lines)
        if args and args[0] == "show":
            if len(args) != 2:
                return "Usage: /assistant show <agent-name>"
            requested = args[1]
            matches = [agent for agent in definitions.all_agents if agent.agent_type == requested]
            if not matches:
                return f"assistant: profile not found {requested}"
            return render_agent_details(matches, requested_name=requested)
        if args:
            return "Usage: /assistant [list|show <agent-name>]"

        current_agent = getattr(scheduler, "agent_definition", None) if scheduler else None
        current_agent_name = getattr(current_agent, "agent_type", None) or "general-purpose"
        session_id = getattr(getattr(scheduler, "session", None), "session_id", None) or "none"
        active_count = len(definitions.active_agents)
        return (
            "assistant:\n"
            f"active_profile: {current_agent_name}\n"
            f"model: {self.current_model}\n"
            f"provider: {self.current_model_provider}\n"
            f"session_id: {session_id}\n"
            f"active_profiles: {active_count}\n"
            f"all_profiles: {len(definitions.all_agents)}\n"
            f"project_agents_dir: {project_agents_dir(Path.cwd())}\n"
            f"user_agents_dir: {user_agents_dir()}\n"
            "related_commands: /agents, /model, /session, /skills"
        )

    @staticmethod
    def _slugify_verifier_name(raw_name: str) -> str:
        slug = re.sub(r"[^a-z0-9-]+", "-", raw_name.lower()).strip("-")
        slug = re.sub(r"-+", "-", slug)
        if not slug:
            slug = "verifier-cli"
        if "verifier" not in slug:
            slug = f"verifier-{slug}"
        return slug

    @staticmethod
    def _infer_verifier_kind_from_project(cwd: Path) -> str:
        package_json = cwd / "package.json"
        if package_json.exists():
            try:
                package = json.loads(package_json.read_text(encoding="utf-8"))
                text = json.dumps(package).lower()
                if any(marker in text for marker in ("next", "vite", "react", "vue", "svelte")):
                    return "playwright"
            except Exception:
                return "playwright"
        api_markers = ("fastapi", "flask", "express", "django")
        for candidate in (cwd / "pyproject.toml", cwd / "requirements.txt", cwd / "package.json"):
            if candidate.exists():
                try:
                    if any(
                        marker in candidate.read_text(encoding="utf-8").lower()
                        for marker in api_markers
                    ):
                        return "api"
                except Exception:
                    continue
        return "cli"

    @staticmethod
    def _infer_verifier_kind_from_name(name: str, fallback: str) -> str:
        lowered = name.lower()
        if any(marker in lowered for marker in ("web", "ui", "playwright", "browser")):
            return "playwright"
        if any(marker in lowered for marker in ("api", "http", "server")):
            return "api"
        if any(marker in lowered for marker in ("cli", "tmux", "terminal", "tui")):
            return "cli"
        return fallback

    @staticmethod
    def _render_verifier_skill(skill_name: str, kind: str, cwd: Path) -> str:
        title = skill_name.replace("-", " ").title()
        if kind == "playwright":
            description = "Verify web UI behavior with browser automation and functional assertions"
            allowed_tools = [
                "run_shell:npm *",
                "run_shell:yarn *",
                "run_shell:pnpm *",
                "run_shell:bun *",
                "mcp__playwright__*",
                "read_file",
                "glob_search",
                "grep_search",
            ]
            setup = [
                "1. Start the app with the project dev-server command.",
                "2. Wait for the configured local URL to respond.",
                "3. Use browser automation to execute the verification plan.",
            ]
        elif kind == "api":
            description = "Verify API behavior with local server startup and HTTP assertions"
            allowed_tools = [
                "run_shell:curl *",
                "run_shell:http *",
                "run_shell:uv *",
                "run_shell:npm *",
                "read_file",
                "glob_search",
                "grep_search",
            ]
            setup = [
                "1. Start the API server command for this project.",
                "2. Wait for the health endpoint or base URL to respond.",
                "3. Execute the requested HTTP assertions with explicit status/body checks.",
            ]
        else:
            description = "Verify CLI and TUI behavior with tmux and multi-turn assertions"
            allowed_tools = [
                "run_shell:tmux *",
                "run_shell:uv *",
                "run_shell:asciinema *",
                "read_file",
                "glob_search",
                "grep_search",
            ]
            setup = [
                "1. Start the CLI or TUI in a fresh tmux session from the project root.",
                "2. Send each verification input as a separate interactive turn.",
                "3. Capture the pane after each turn and assert on stable output text.",
            ]
        allowed_yaml = "\n".join(f"  - {tool}" for tool in allowed_tools)
        setup_text = "\n".join(setup)
        return f"""---
name: {skill_name}
description: {description}
allowed-tools:
{allowed_yaml}
---

# {title}

You are a Koder verification executor. Execute the provided verification plan exactly as written.

## Project Context

- Project root: {cwd}
- Verification type: {kind}
- Prefer functional proof over superficial smoke checks.

## Setup Instructions

{setup_text}

## Reporting

Report PASS or FAIL for each verification step, include the command or tmux turn that produced the evidence, and include the exact output marker that supports the result.

## Cleanup

Stop services and tmux sessions started during verification, then report a concise final summary.

## Self-Update

If verification fails because this skill's instructions are stale, ask before editing this file and then make the smallest targeted correction.
"""

    async def _execute_init_verifiers(self, _scheduler, args: list[str]) -> str:
        if args and args[0] in {"help", "-h", "--help"}:
            return "Usage: /init-verifiers [cli|tmux|playwright|web|api|verifier-name]"

        kind_aliases = {
            "api": "api",
            "http": "api",
            "cli": "cli",
            "terminal": "cli",
            "tmux": "cli",
            "tui": "cli",
            "playwright": "playwright",
            "web": "playwright",
            "ui": "playwright",
            "browser": "playwright",
        }
        cwd = Path.cwd()
        inferred_kind = self._infer_verifier_kind_from_project(cwd)
        if args:
            first = args[0].strip()
            kind = kind_aliases.get(first.lower())
            if kind:
                raw_name = args[1] if len(args) > 1 else f"verifier-{kind}"
            else:
                raw_name = first
                kind = self._infer_verifier_kind_from_name(raw_name, inferred_kind)
        else:
            kind = inferred_kind
            raw_name = f"verifier-{kind}"

        skill_name = self._slugify_verifier_name(raw_name)
        skill_dir = cwd / ".koder" / "skills" / skill_name
        skill_file = skill_dir / "SKILL.md"
        if skill_file.exists():
            return (
                "init-verifiers: exists\n"
                f"name: {skill_name}\n"
                f"type: {kind}\n"
                f"path: {skill_file}\n"
                "discovery: skill folders containing verifier are visible to Koder"
            )

        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_file.write_text(
            self._render_verifier_skill(skill_name=skill_name, kind=kind, cwd=cwd),
            encoding="utf-8",
        )
        return (
            "init-verifiers: created\n"
            f"name: {skill_name}\n"
            f"type: {kind}\n"
            f"path: {skill_file}\n"
            "discovery: skill folders containing verifier are visible to Koder\n"
            "next: run /skills to confirm it is loaded"
        )

    async def _execute_thinkback(self, scheduler, args: list[str]) -> str:
        limit = self._parse_positive_limit(args, default=5, maximum=20)
        if limit is None:
            return "Usage: /thinkback [recent-turn-count]"

        items = await self._get_session_items(scheduler)
        turns = self._session_text_turns(items)
        user_turns = [text for role, text in turns if role == "user"]
        assistant_turns = [text for role, text in turns if role == "assistant"]
        tool_turns = [text for role, text in turns if role == "tool"]
        title = None
        if scheduler is not None and hasattr(getattr(scheduler, "session", None), "get_title"):
            try:
                title = await scheduler.session.get_title()
            except Exception:
                title = None

        lines = ["thinkback: session review"]
        if title:
            lines.append(f"title: {title}")
        lines.extend(
            [
                f"messages: {len(turns)}",
                f"user_turns: {len(user_turns)}",
                f"assistant_turns: {len(assistant_turns)}",
                f"tool_outputs: {len(tool_turns)}",
            ]
        )
        if user_turns:
            lines.append("recent_prompts:")
            for index, prompt in enumerate(user_turns[-limit:], start=1):
                lines.append(f"{index}. {self._truncate_prompt_preview(prompt, limit=120)}")
        else:
            lines.append("recent_prompts: none")
        return "\n".join(lines)

    async def _execute_thinkback_play(self, scheduler, args: list[str]) -> str:
        limit = self._parse_positive_limit(args, default=8, maximum=40)
        if limit is None:
            return "Usage: /thinkback-play [recent-turn-count]"

        turns = self._session_text_turns(await self._get_session_items(scheduler))
        if not turns:
            return "thinkback-play: no session turns available"

        selected = turns[-limit:]
        lines = [f"thinkback-play: replaying {len(selected)} turn(s)"]
        for role, text in selected:
            lines.append(f"{role}: {self._truncate_prompt_preview(text, limit=180)}")
        return "\n".join(lines)

    async def _execute_tasks(self, _scheduler, _args: list[str]) -> str:
        tasks = self.task_service.list_tasks()
        auto_dream_rows = self._auto_dream_task_rows()
        if not tasks and not auto_dream_rows:
            return "No runtime tasks tracked."
        lines = [f"- {task.id}: {task.title} [{task.status}]" for task in tasks]
        lines.extend(auto_dream_rows)
        return "\n".join(lines)

    @staticmethod
    def _auto_dream_task_rows() -> list[str]:
        try:
            from koder_agent.harness.memory.auto_dream import list_auto_dream_tasks_with_errors
        except Exception:
            return []

        try:
            tasks, malformed = list_auto_dream_tasks_with_errors()
        except Exception:
            return []

        rows: list[str] = []
        for item in malformed:
            rows.append(f"- auto-dream/malformed: {item}")
        for task in tasks:
            metadata = task.metadata or {}
            parts = [f"- auto-dream/{task.id}: {task.title} status={task.status}"]
            memories = metadata.get("memories_written")
            if memories is not None:
                parts.append(f"memories={memories}")
            memory_candidates = metadata.get("memory_candidates_staged")
            if memory_candidates is not None:
                parts.append(f"memory_candidates={memory_candidates}")
            skill_candidates = metadata.get("skill_candidates_staged")
            if skill_candidates is not None:
                parts.append(f"skill_candidates={skill_candidates}")
            errors = metadata.get("errors")
            if isinstance(errors, list) and errors:
                parts.append(f"errors={len(errors)}")
            saved_path = metadata.get("saved_path")
            if saved_path:
                parts.append(f"saved={saved_path}")
            rows.append(" ".join(parts))
        return rows

    async def _execute_permissions(self, _scheduler, _args: list[str]) -> str:
        if _args and _args[0] == "check":
            if len(_args) < 3:
                return "permissions: usage /permissions check <tool> <target-or-command>"
            tool_name = _args[1]
            target = " ".join(_args[2:]).strip()
            if len(target) >= 2 and target[0] == target[-1] and target[0] in {'"', "'"}:
                target = target[1:-1]
            arguments = permission_arguments_for_target(tool_name, target)
            result = await self.permission_service.evaluate_tool_call_async(
                tool_name,
                arguments,
            )
            lines = [
                "permissions: check",
                f"tool: {result.tool_name}",
                f"allowed: {str(result.allowed).lower()}",
                f"requires_approval: {str(result.requires_approval).lower()}",
                f"mode: {result.mode.value}",
                f"reason: {result.reason}",
            ]
            if result.matched_rule:
                lines.append(f"matched_rule: {result.matched_rule}")
            return "\n".join(lines)

        rules = self.permission_service.export_rules()
        denial_count = len(self.permission_service.denial_log.recent())
        return (
            f"mode: {self.permission_service.mode.value}\n"
            f"rules: {sum(len(v) for tool in rules.values() for v in tool.values())}\n"
            f"denials: {denial_count}\n"
            f"working_directories: {len(self.permission_service.list_working_directories())}"
        )

    async def _execute_theme(self, _scheduler, _args: list[str]) -> str:
        if _args:
            requested = " ".join(_args).strip().lower()
            if requested not in VALID_OUTPUT_THEMES:
                return (
                    f"theme: invalid {requested or '<empty>'}\n"
                    f"valid_themes: {', '.join(VALID_OUTPUT_THEMES)}"
                )
            self.current_theme = requested
            settings_path = _save_output_theme(self.current_theme)
        else:
            self.current_theme = _load_output_theme()
            settings_path = _output_style_settings_path()
        return f"theme: {self.current_theme}\nsettings_path: {settings_path}"

    async def _execute_keybindings(self, _scheduler, args: list[str]) -> str:
        config_path = Path.home() / ".koder" / "keybindings.json"
        manager = KeybindingManager(config_path=config_path)

        if not args or args[0] in {"list", "status"}:
            bindings = manager.get_all_bindings()
            overrides = [
                action for action, key in bindings.items() if DEFAULT_KEYBINDINGS.get(action) != key
            ]
            lines = [
                "keybindings:",
                f"settings_path: {config_path}",
                f"overrides: {len(overrides)}",
                "bindings:",
            ]
            for action in sorted(bindings):
                key = bindings[action]
                key_display = "unbound" if key is None else key
                lines.append(f"- {action}: {key_display}")
            return "\n".join(lines)

        action = args[0]
        valid_actions = sorted(DEFAULT_KEYBINDINGS)
        if action == "set":
            if len(args) < 3:
                return "Usage: /keybindings set <action> <key>"
            binding_action = args[1]
            if binding_action not in DEFAULT_KEYBINDINGS:
                return "keybindings: unknown action\nvalid_actions: " + ", ".join(valid_actions)
            key = " ".join(args[2:]).strip()
            try:
                manager.set_override(binding_action, key)
            except ValueError as exc:
                return f"keybindings: invalid key\nkey: {key}\nerror: {exc}"
            manager.save()
            return (
                "keybindings: set\n"
                f"action: {binding_action}\n"
                f"key: {key}\n"
                f"settings_path: {config_path}"
            )

        if action == "unset":
            if len(args) != 2:
                return "Usage: /keybindings unset <action>"
            binding_action = args[1]
            if binding_action not in DEFAULT_KEYBINDINGS:
                return "keybindings: unknown action\nvalid_actions: " + ", ".join(valid_actions)
            manager.set_override(binding_action, None)
            manager.save()
            return f"keybindings: unset\naction: {binding_action}\nsettings_path: {config_path}"

        if action == "reset":
            if len(args) != 2:
                return "Usage: /keybindings reset <action|all>"
            binding_action = args[1]
            if binding_action == "all":
                config_path.unlink(missing_ok=True)
                return f"keybindings: reset all\nsettings_path: {config_path}"
            if binding_action not in DEFAULT_KEYBINDINGS:
                return "keybindings: unknown action\nvalid_actions: " + ", ".join(valid_actions)
            manager.reset(binding_action)
            manager.save()
            return f"keybindings: reset\naction: {binding_action}\nsettings_path: {config_path}"

        return "Usage: /keybindings [list|set <action> <key>|unset <action>|reset <action|all>]"

    async def _execute_output_style(self, scheduler, args: list[str]) -> str:
        if not args or args[0] in {"status", "list"}:
            session_color = await self._get_session_color(scheduler)
            current_color = session_color or self.current_color
            self.current_color = current_color
            if self.interactive_prompt is not None:
                self.vim_enabled = self.interactive_prompt.vim_mode_manager.enabled
            else:
                self.vim_enabled = _load_vim_enabled()
            statusline = resolve_statusline_config(Path.cwd())
            active_style = load_active_output_style_name()
            lines = [
                "output-style:",
                f"theme: {self.current_theme}",
                f"color: {current_color}",
                f"vim_mode: {str(self.vim_enabled).lower()}",
                f"style: {active_style if active_style else 'none'}",
            ]
            if statusline is None:
                lines.append("statusline: not configured")
            else:
                lines.append(f"statusline: {statusline.command}")
                lines.append(f"statusline_source: {statusline.source_path}")
            lines.append("controls: /theme, /color, /statusline, /vim, /output-style set <name>")
            return "\n".join(lines)

        action = args[0]
        if action == "theme":
            if len(args) < 2:
                return "Usage: /output-style theme <name>"
            return await self._execute_theme(scheduler, args[1:])
        if action == "color":
            if len(args) < 2:
                return "Usage: /output-style color <name>"
            return await self._execute_color(scheduler, args[1:])
        if action == "statusline":
            return await self._execute_statusline(scheduler, args[1:])
        if action == "vim":
            return await self._execute_vim(scheduler, args[1:])
        if action in {"styles", "list-styles"}:
            return self._render_output_styles_listing()
        if action == "set":
            if len(args) < 2:
                return "Usage: /output-style set <name>"
            return await self._execute_output_style_set(scheduler, " ".join(args[1:]))
        if action in {"unset", "clear"}:
            return await self._execute_output_style_set(scheduler, None)
        if action == "reset":
            self.current_theme = "adaptive"
            self.current_color = "default"
            await self._set_session_color(scheduler, None)
            self.vim_enabled = False
            if self.interactive_prompt is not None:
                self.interactive_prompt.set_vim_mode(False)
                vim_settings_path = _vim_state_path()
            else:
                vim_settings_path = _save_vim_enabled(False)
            theme_settings_path = _save_output_theme(self.current_theme)
            style_settings_path = save_active_output_style_name(None)
            await self._reset_scheduler_agent(scheduler)
            settings_path = update_user_statusline_config(None)
            return (
                "output-style: reset\n"
                "theme: adaptive\n"
                "color: default\n"
                "vim_mode: false\n"
                "style: none\n"
                f"theme_settings_path: {theme_settings_path}\n"
                f"vim_settings_path: {vim_settings_path}\n"
                f"style_settings_path: {style_settings_path}\n"
                f"statusline_settings_path: {settings_path}"
            )
        return (
            "Usage: /output-style [status|theme <name>|color <name>|statusline ...|"
            "vim [on|off]|styles|set <name>|unset|reset]"
        )

    def _render_output_styles_listing(self) -> str:
        styles = discover_output_styles(Path.cwd())
        active = load_active_output_style_name()
        active_key = active.lower() if active else None
        lines = ["output-style styles:"]
        if not styles:
            lines.append("(no output styles found)")
            lines.append(
                "hint: add markdown personas to .koder/output-styles/ or ~/.koder/output-styles/"
            )
            return "\n".join(lines)
        for key in sorted(styles):
            style = styles[key]
            marker = "* " if key == active_key else "- "
            summary = style.description or "(no description)"
            lines.append(f"{marker}{style.name} [{style.source}]: {summary}")
        lines.append(f"active: {active if active else 'none'}")
        return "\n".join(lines)

    async def _execute_output_style_set(self, scheduler, name: Optional[str]) -> str:
        if name is None:
            style_settings_path = save_active_output_style_name(None)
            agent_reloaded = await self._reset_scheduler_agent(scheduler)
            return (
                "output-style: style cleared\n"
                "style: none\n"
                f"settings_path: {style_settings_path}\n"
                f"agent_reloaded: {agent_reloaded}"
            )
        requested = name.strip()
        if not requested:
            return "Usage: /output-style set <name>"
        style = find_output_style(requested, Path.cwd())
        if style is None:
            available = sorted(discover_output_styles(Path.cwd()))
            available_display = ", ".join(available) if available else "(none)"
            return f"output-style: unknown style {requested}\navailable: {available_display}"
        style_settings_path = save_active_output_style_name(style.name)
        agent_reloaded = await self._reset_scheduler_agent(scheduler)
        return (
            f"output-style: style set to {style.name}\n"
            f"source: {style.source}\n"
            f"settings_path: {style_settings_path}\n"
            f"agent_reloaded: {agent_reloaded}"
        )

    async def _execute_usage(self, scheduler, _args: list[str]) -> str:
        usage = scheduler.usage_tracker.session_usage if scheduler else None
        if usage is None:
            return (
                "requests: 0\ninput_tokens: 0\noutput_tokens: 0\n"
                "last_input_tokens: 0\nlast_output_tokens: 0\ncontext_tokens: 0\n"
                "cost: 0.0000\n"
                "plan_usage: unavailable\nrate_limit_status: unknown"
            )
        return (
            f"requests: {getattr(usage, 'request_count', 0)}\n"
            f"input_tokens: {getattr(usage, 'input_tokens', 0)}\n"
            f"output_tokens: {getattr(usage, 'output_tokens', 0)}\n"
            f"last_input_tokens: {getattr(usage, 'last_input_tokens', 0)}\n"
            f"last_output_tokens: {getattr(usage, 'last_output_tokens', 0)}\n"
            f"context_tokens: {getattr(usage, 'current_context_tokens', 0)}\n"
            f"cost: {getattr(usage, 'total_cost', 0.0):.4f}\n"
            "plan_usage: unavailable\n"
            "rate_limit_status: unknown"
        )

    async def _execute_effort(self, _scheduler, _args: list[str]) -> str:
        manager = get_config_manager()
        config = manager.load()
        args = [arg.strip().lower() for arg in _args if arg.strip()]

        if not args or args[0] in {"status", "current"}:
            current = manager.get_effective_value(
                config.model.reasoning_effort,
                "KODER_REASONING_EFFORT",
            )
            if current is None:
                return "Effort level: auto"
            return f"Current effort level: {current}"

        if args[0] in {"help", "-h", "--help"}:
            return (
                "Usage: /effort [low|medium|high|xhigh|max|auto]\n\n"
                "Effort levels:\n"
                "- low: Quick, straightforward implementation\n"
                "- medium: Balanced approach with standard testing\n"
                "- high: Comprehensive implementation with extensive testing\n"
                "- xhigh: Extra-high reasoning for especially difficult tasks\n"
                "- max: Maximum reasoning for supported models\n"
                "- auto: Reset to the default medium effort level"
            )

        desired = args[0]
        if desired == "auto":
            config.model.reasoning_effort = "medium"
            manager.save(config)
            os.environ.pop("KODER_REASONING_EFFORT", None)
            agent_reloaded = await self._reset_scheduler_agent(_scheduler)
            return f"Effort level reset to default: medium\nsettings_path: {manager.config_path}\nagent_reloaded: {agent_reloaded}"

        if desired not in {"low", "medium", "high", "xhigh", "max"}:
            return (
                f"Invalid argument: {desired}. "
                "Valid options are: low, medium, high, xhigh, max, auto"
            )

        config.model.reasoning_effort = desired
        manager.save(config)
        os.environ["KODER_REASONING_EFFORT"] = desired
        agent_reloaded = await self._reset_scheduler_agent(_scheduler)
        return f"Set effort level to {desired}\nsettings_path: {manager.config_path}\nagent_reloaded: {agent_reloaded}"

    async def _execute_reasoning(self, _scheduler, _args: list[str]) -> str:
        manager = get_config_manager()
        config = manager.load()
        args = [arg.strip().lower() for arg in _args if arg.strip()]

        if not args or args[0] in {"status", "current"}:
            current = normalize_reasoning_display_mode(
                manager.get_effective_value(
                    config.harness.reasoning_display,
                    "KODER_REASONING_DISPLAY",
                )
            )
            return f"Reasoning display: {current}"

        if args[0] in {"help", "-h", "--help"}:
            return (
                "Usage: /reasoning [off|summary|full|status]\n\n"
                "Reasoning display modes:\n"
                "- off: Hide reasoning content\n"
                "- summary: Show model-provided reasoning summaries\n"
                "- full: Also show raw reasoning text when the provider exposes it"
            )

        if args[0] not in VALID_REASONING_DISPLAY_MODES:
            valid = ", ".join(VALID_REASONING_DISPLAY_MODES)
            return f"Invalid argument: {args[0]}. Valid options are: {valid}"

        desired = normalize_reasoning_display_mode(args[0], default="off")
        config.harness.reasoning_display = desired
        manager.save(config)
        os.environ["KODER_REASONING_DISPLAY"] = desired
        agent_reloaded = await self._reset_scheduler_agent(_scheduler)
        return (
            f"Reasoning display set to {desired}\n"
            f"settings_path: {manager.config_path}\n"
            f"agent_reloaded: {agent_reloaded}"
        )

    async def _execute_export(self, scheduler, _args: list[str]) -> str:
        if scheduler is None or not hasattr(scheduler, "session"):
            return "No active session to export."
        session = scheduler.session

        display_name = None
        if hasattr(session, "get_display_name"):
            display_name = await session.get_display_name()

        items = await session.get_items() if hasattr(session, "get_items") else []
        transcript_records: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip().lower()
            text = self._flatten_session_text(item.get("content"))
            if role not in {"user", "assistant", "system", "tool"} or not text:
                continue
            transcript_records.append({"role": role, "content": text})

        transcript_lines = [
            f"{record['role']}: {record['content']}" for record in transcript_records
        ]

        if _args:
            export_format: str
            target_arg: str
            first = _args[0].strip().lower()
            if len(_args) == 1:
                if first in {"json", "markdown", "md"}:
                    return "Usage: /export [json|markdown] <path>"
                target_arg = _args[0]
                export_format = "json" if Path(target_arg).suffix.lower() == ".json" else "markdown"
            elif len(_args) == 2 and first in {"json", "markdown", "md"}:
                export_format = "markdown" if first == "md" else first
                target_arg = _args[1]
            else:
                return "Usage: /export [json|markdown] <path>"

            target = Path(target_arg).expanduser()
            if target.exists() and target.is_dir():
                return f"export: target is a directory\npath: {target}"
            parent = target.parent if str(target.parent) else Path(".")
            if parent and not parent.exists():
                return f"export: parent directory not found\npath: {parent}"

            if export_format == "json":
                payload = {
                    "session_id": session.session_id,
                    "display_name": display_name,
                    "messages": transcript_records,
                }
                content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            else:
                content = "\n".join(
                    [
                        f"# Koder Session Export: {display_name or session.session_id}",
                        "",
                        f"session_id: {session.session_id}",
                        f"display_name: {display_name or ''}",
                        f"messages: {len(transcript_lines)}",
                        "",
                        "## Transcript",
                        *(transcript_lines or ["No textual messages available to export."]),
                    ]
                )
                content += "\n"
            target.write_text(content, encoding="utf-8")
            return (
                "export: written\n"
                f"format: {export_format}\n"
                f"path: {target}\n"
                f"session_id: {session.session_id}\n"
                f"messages: {len(transcript_lines)}"
            )

        lines = [f"export session_id: {session.session_id}"]
        if display_name:
            lines.append(f"display_name: {display_name}")
        lines.append(f"messages: {len(transcript_lines)}")
        if transcript_lines:
            lines.extend(["### Transcript", *transcript_lines])
        else:
            lines.append("No textual messages available to export.")
        return "\n".join(lines)

    async def _execute_review(self, _scheduler, _args: list[str]) -> str:
        """Review uncommitted code changes for quality, bugs, and security."""
        from koder_agent.harness.review_flow import run_review

        pr = _args[0] if _args and _args[0].startswith("#") else None
        text, _has_findings = await run_review(pr=pr)
        return text

    async def _execute_advisor(self, _scheduler, _args: list[str]) -> str:
        focus = " ".join(_args).strip() or None
        session_items = None
        if (
            _scheduler is not None
            and hasattr(_scheduler, "session")
            and hasattr(_scheduler.session, "get_items")
        ):
            try:
                session_items = await _scheduler.session.get_items()
            except Exception:
                session_items = None
        return await run_advisor_review(
            cwd=Path.cwd(),
            session_items=session_items,
            focus=focus,
            config=self.config_service.load(),
        )

    async def _execute_brief(self, _scheduler, _args: list[str]) -> str:
        if _args:
            return "Usage: /brief"
        return await run_brief(config_service=self.config_service)

    async def _execute_buddy(self, _scheduler, _args: list[str]) -> str:
        action = _args[0] if _args else None
        if len(_args) > 1:
            return "Usage: /buddy [status|mute|unmute]"
        return await run_buddy(
            config_service=self.config_service,
            action=action,
        )

    @staticmethod
    async def _replace_runtime_session_items(scheduler, session, items: list[dict]) -> None:
        """Await an exact history replacement for the passed runtime session."""
        scheduler_replace = getattr(scheduler, "_replace_session_items", None)
        if callable(scheduler_replace):
            await scheduler_replace(items)
            return

        session_replace = getattr(session, "replace_items", None)
        if callable(session_replace):
            await session_replace(items)
            return

        raise RuntimeError("Session does not support exact history replacement")

    async def _execute_compact(self, scheduler, _args: list[str]) -> str:
        if _args:
            return "Usage: /compact"
        if scheduler is None:
            return "No active conversation state to compact yet."
        session = scheduler.session
        dispatch_command_hooks(
            cwd=Path.cwd(),
            event_name="PreCompact",
            match_value="manual",
            payload={
                "event": "PreCompact",
                "trigger": "manual",
                "session_id": session.session_id,
            },
        )
        items = await session.get_items()
        context_before = (
            await scheduler.refresh_context_usage_from_session(
                [item for item in items if isinstance(item, dict)]
            )
            if hasattr(scheduler, "refresh_context_usage_from_session")
            else estimate_messages_tokens([item for item in items if isinstance(item, dict)])
        )
        if self.emit_console:
            console.print("[dim]compacting...[/dim]")
        compactable_items = compactable_session_items(items)
        result = await llm_compact_messages(compactable_items)
        persisted = False
        persistence_error = None
        final_count = len(items)
        context_after = context_before
        original_dict_items = [item for item in items if isinstance(item, dict)]
        compacted_items = build_compacted_session_items(result)
        should_persist = bool(result.summary) or compacted_items != original_dict_items
        if should_persist and hasattr(scheduler, "_active_todo_preserved_message"):
            todo_message = scheduler._active_todo_preserved_message()
            if todo_message is not None:
                instruction_count = sum(
                    item.get("role") in {"system", "developer"} for item in result.kept_messages
                )
                insert_at = instruction_count + (1 if result.summary else 0)
                compacted_items = (
                    compacted_items[:insert_at] + [todo_message] + compacted_items[insert_at:]
                )
        if should_persist:
            try:
                await self._replace_runtime_session_items(scheduler, session, compacted_items)
                final_count = len(await session.get_items())
                if hasattr(scheduler, "refresh_context_usage_from_session"):
                    context_after = await scheduler.refresh_context_usage_from_session(
                        compacted_items
                    )
                else:
                    context_after = estimate_messages_tokens(compacted_items)
                persisted = True
            except Exception as exc:
                persistence_error = str(exc)
        dispatch_command_hooks(
            cwd=Path.cwd(),
            event_name="PostCompact",
            match_value="manual",
            payload={
                "event": "PostCompact",
                "trigger": "manual",
                "session_id": session.session_id,
                "summary": result.summary or "",
                "original_count": result.original_count,
                "kept_count": len(result.kept_messages),
                "final_count": final_count,
                "persisted": persisted,
            },
        )
        if persistence_error:
            return f"compact failed: {persistence_error}"
        return (
            "compacted, context size "
            f"{self._format_token_count(context_before)} -> "
            f"{self._format_token_count(context_after)}"
        )

    async def _execute_branch(self, _scheduler, _args: list[str]) -> str:
        cwd = os.getcwd()
        git_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,
        )
        if git_root.returncode != 0:
            return "branch: no git repository"

        if not _args:
            branch = current_branch(cwd=cwd)
            status = status_short(cwd=cwd)
            dirty = str(status != "Clean working tree.").lower()
            return f"branch: {branch}\ndirty: {dirty}\nstatus:\n{status}"

        if len(_args) != 1:
            return "Usage: /branch [branch-name]"
        desired = _args[0].strip()
        valid = subprocess.run(
            ["git", "check-ref-format", "--branch", desired],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,
        )
        if valid.returncode != 0:
            return f"branch: invalid name {desired}"

        exists = subprocess.run(
            ["git", "show-ref", "--verify", f"refs/heads/{desired}"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,
        )
        action = "switched" if exists.returncode == 0 else "created"
        switch_args = ["switch", desired] if exists.returncode == 0 else ["switch", "-c", desired]
        switched = subprocess.run(
            ["git", *switch_args],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,
        )
        if switched.returncode != 0:
            error = switched.stderr.strip() or switched.stdout.strip() or "unknown git error"
            return f"branch: failed\nerror: {error}"
        status = status_short(cwd=cwd)
        dirty = str(status != "Clean working tree.").lower()
        return f"branch: {desired}\naction: {action}\ndirty: {dirty}\nstatus:\n{status}"

    _REWIND_MODES = {"conversation", "code", "both"}
    _REWIND_USAGE = "Usage: /rewind [number] [conversation|code|both]"

    async def _execute_rewind(self, scheduler, _args: list[str]) -> str:
        self._pending_input_text = None

        # Parse optional [number] and [mode] in any order.
        selection: int | None = None
        mode = "conversation"
        for arg in _args:
            token = arg.strip().lower()
            if not token:
                continue
            if token in {"help", "-h", "--help"}:
                return self._rewind_help_text()
            if token in self._REWIND_MODES:
                mode = token
                continue
            try:
                selection = int(token)
            except ValueError:
                return self._REWIND_USAGE

        if scheduler is None or not hasattr(scheduler, "session"):
            return "Rewind requires an active session."
        session = scheduler.session
        if not hasattr(session, "get_items"):
            return "Rewind requires a session with stored conversation history."

        items = await session.get_items()
        prompt_targets: list[tuple[int, str]] = []
        for index, item in enumerate(items):
            prompt_text = self._user_message_text(item)
            if prompt_text:
                prompt_targets.append((index, prompt_text))

        if not prompt_targets:
            return "Nothing to rewind to yet."

        # Map each user prompt (newest first) to a checkpoint counter value.
        # The Nth-oldest user prompt corresponds to checkpoint N; restoring
        # code "to prompt N" reverts files edited during prompt N and later.
        newest_first = list(reversed(prompt_targets))
        total_prompts = len(newest_first)

        if selection is None:
            lines = [
                "Rewind targets",
                "Choose a previous user prompt to restore the conversation before that point.",
                "Use /rewind <number> [conversation|code|both].",
                "  conversation - trim history and restore the prompt to input (default)",
                "  code         - restore tracked files to that point",
                "  both         - do both",
            ]
            for number, (item_index, prompt_text) in enumerate(newest_first, start=1):
                trimmed = len(items) - item_index
                lines.append(
                    f"{number}. {self._truncate_prompt_preview(prompt_text)}"
                    f" (removes {trimmed} newer transcript item{'s' if trimmed != 1 else ''})"
                )
            return "\n".join(lines)

        if selection < 1 or selection > total_prompts:
            return f"Rewind target must be between 1 and {total_prompts}."

        selected_index, selected_prompt = newest_first[selection - 1]
        # Checkpoint boundary: prompt #selection is the (total-selection+1)-th
        # oldest user prompt. Restoring code should undo edits made from that
        # prompt onward, i.e. revert everything after checkpoint (target-1).
        target_checkpoint = total_prompts - selection

        result_lines: list[str] = []

        if mode in {"conversation", "both"}:
            kept_items = items[:selected_index]
            removed_count = len(items) - selected_index
            await self._replace_runtime_session_items(scheduler, session, kept_items)
            self._pending_input_text = selected_prompt
            result_lines.append(f"Rewound conversation to prompt {selection}.")
            result_lines.append(f"Removed transcript items: {removed_count}")
            result_lines.append(f"Restored input: {selected_prompt}")

        if mode in {"code", "both"}:
            result_lines.extend(
                self._restore_code_checkpoint(session, target_checkpoint, selection)
            )

        return "\n".join(result_lines) if result_lines else self._REWIND_USAGE

    def _restore_code_checkpoint(
        self, session, target_checkpoint: int, selection: int
    ) -> list[str]:
        """Restore tracked files to a checkpoint; return message lines."""
        from koder_agent.harness import checkpoint as checkpoint_store

        if not checkpoint_store.checkpoints_enabled():
            return ["Code restore is disabled (file checkpointing turned off)."]

        session_id = getattr(session, "session_id", None)
        if session_id is None:
            return ["Code restore requires a session with an id."]

        try:
            restored = checkpoint_store.restore_to(str(session_id), target_checkpoint)
        except Exception as exc:  # defensive
            return [f"Code restore failed: {exc}"]

        if not restored:
            return [f"No tracked file changes to restore for prompt {selection}."]

        lines = [f"Restored {len(restored)} file(s) to before prompt {selection}:"]
        for path in restored:
            lines.append(f"  {path}")
        return lines

    def _rewind_help_text(self) -> str:
        return "\n".join(
            [
                self._REWIND_USAGE,
                "Restore the conversation and/or tracked files to a previous prompt.",
                "Modes:",
                "  conversation - trim history and restore the prompt to input (default)",
                "  code         - restore tracked files edited since that prompt",
                "  both         - conversation + code",
            ]
        )

    async def _execute_exit(self, _scheduler, _args: list[str]) -> str:
        return "__EXIT__"

    async def _execute_plan(self, _scheduler, _args: list[str]) -> str:
        if self.plan_mode_service.is_plan_mode():
            result = self.plan_mode_service.exit_plan_mode()
            self.permission_service.mode = self._pre_plan_permission_mode
            return "\n".join(
                [
                    "Exited plan mode.",
                    f"mode: {result.mode}",
                    f"permission_mode: {self.permission_service.mode.value}",
                ]
            )

        self._pre_plan_permission_mode = self.permission_service.mode
        result = self.plan_mode_service.enter_plan_mode(permission_mode="plan")
        self.permission_service.mode = PermissionMode.PLAN

        lines = ["Entered plan mode (read-only). Write operations are blocked."]
        lines.append(f"mode: {result.mode}")
        lines.append(f"permission_mode: {self.permission_service.mode.value}")
        lines.append("")
        lines.append("In plan mode you can:")
        lines.append("  - Read and analyze code")
        lines.append("  - Search the codebase")
        lines.append("  - Discuss architecture and design")
        lines.append("  - Create a plan document")
        lines.append("")
        lines.append("To exit plan mode and start implementing: /plan (toggle)")

        return "\n".join(lines)

    async def _execute_hooks(self, scheduler, _args: list[str]) -> str:
        from koder_agent.harness.hooks.project_approval import (
            approve_project_hooks,
            canonical_project_hooks_payload,
            load_project_hook_settings,
            project_hooks_approval_error,
            project_hooks_digest,
            revoke_project_hooks,
        )

        project_root = resolve_hook_project_root(Path.cwd())
        action = _args[0].lower() if _args else "list"
        if action in {"review", "approve", "revoke"}:
            settings_by_source = load_project_hook_settings(project_root)
            payload = canonical_project_hooks_payload(settings_by_source)
            payload_data = json.loads(payload)
            digest = project_hooks_digest(settings_by_source)
            approval_error = project_hooks_approval_error(project_root, settings_by_source)
            status = "approved" if approval_error is None else f"not approved: {approval_error}"
            source_paths = {
                "project_settings": project_root / ".koder" / "settings.json",
                "local_settings": project_root / ".koder" / "settings.local.json",
            }
            has_executable_payload = bool(payload_data.get("sources"))

            def review_lines(*, heading: str = "Project hook approval review") -> list[str]:
                lines = [
                    heading,
                    f"project: {project_root}",
                    f"status: {status}",
                    f"digest: {digest}",
                    "sources:",
                ]
                present_sources = [
                    source for source in source_paths if source in settings_by_source
                ]
                if present_sources:
                    lines.extend(
                        f"  - {source}: {source_paths[source]}" for source in present_sources
                    )
                else:
                    lines.append("  - none")
                lines.extend(
                    [
                        "exact executable payload:",
                        json.dumps(payload_data, ensure_ascii=False, indent=2, sort_keys=True),
                    ]
                )
                return lines

            if action == "review":
                if len(_args) != 1:
                    return "Usage: /hooks review"
                lines = review_lines()
                if has_executable_payload:
                    lines.extend(
                        [
                            "",
                            "Approve this exact payload from an interactive Koder session with:",
                            f"/hooks approve {digest}",
                            "The full digest is required; hook changes require a new review and approval.",
                        ]
                    )
                else:
                    lines.extend(["", "No project or local executable hook payload to approve."])
                return "\n".join(lines)

            if action == "approve":
                if len(_args) != 2:
                    return "Usage: /hooks approve <full-sha256-from-/hooks-review>"
                if scheduler is None or self.interactive_prompt is None:
                    return "\n".join(
                        [
                            "Hook approval denied: approval is only available in a live interactive Koder session.",
                            "Run `uv run koder`, then use `/hooks review` and `/hooks approve <full-sha256>`.",
                        ]
                    )
                if not has_executable_payload:
                    return "No project or local executable hook payload to approve."
                requested_digest = _args[1]
                if requested_digest != digest:
                    return "\n".join(
                        [
                            "Hook approval denied: the supplied digest does not match the current payload.",
                            f"current_digest: {digest}",
                            "Run /hooks review and approve the full current digest.",
                        ]
                    )
                approve_project_hooks(
                    project_root,
                    settings_by_source,
                    expected_digest=requested_digest,
                )
                lines = review_lines(heading="Project hook payload approved")
                lines[2] = "status: approved"
                lines.append("Future executable hook changes will require reapproval.")
                return "\n".join(lines)

            if len(_args) != 1:
                return "Usage: /hooks revoke"
            removed = revoke_project_hooks(project_root)
            if removed:
                return "\n".join(
                    [
                        "Project hook approval revoked.",
                        f"project: {project_root}",
                        "Project and local hooks are now blocked until explicitly reapproved.",
                    ]
                )
            return f"No project hook approval existed for {project_root}."

        if action != "list" or len(_args) > 1:
            return "Usage: /hooks [review|approve <full-sha256>|revoke]"

        listings = list_configured_hooks(Path.cwd())
        lines = []
        if scheduler is not None and hasattr(scheduler, "hooks"):
            lines.append(f"hooks: {scheduler.hooks.__class__.__name__}")
        else:
            lines.append("hooks: configured")
        if not listings:
            lines.append("No hooks configured.")
            lines.append("Use /hooks review to inspect project/local hook trust payloads.")
            return "\n".join(lines)
        lines.append(f"count: {len(listings)}")
        for listing in listings:
            suffix: list[str] = [listing.source, listing.hook_type]
            if listing.matcher:
                suffix.append(f"matcher={listing.matcher}")
            if listing.scope_root:
                suffix.append(f"scope={listing.scope_root}")
            lines.append(
                f"- {listing.event}: {listing.command or '(non-command hook)'} "
                f"[{', '.join(suffix)}]"
            )
        lines.append("Use /hooks review to inspect the exact project/local approval payload.")
        return "\n".join(lines)

    async def _execute_vim(self, _scheduler, _args: list[str]) -> str:
        state_path = _vim_state_path()
        vim_manager = None
        if self.interactive_prompt is None:
            vim_manager = VimModeManager(state_path=state_path)
            vim_manager.load()
            self.vim_enabled = vim_manager.enabled

        if _args:
            desired = _args[0].strip().lower()
            if desired not in {"on", "off"}:
                return "Usage: /vim [on|off]"
            self.vim_enabled = desired == "on"
            if self.interactive_prompt:
                self.interactive_prompt.set_vim_mode(self.vim_enabled)
            elif vim_manager is not None:
                if self.vim_enabled:
                    vim_manager.enable()
                else:
                    vim_manager.disable()
                vim_manager.save()
        else:
            self.vim_enabled = not self.vim_enabled
            if self.interactive_prompt:
                enabled = self.interactive_prompt.toggle_vim_mode()
                self.vim_enabled = enabled
            elif vim_manager is not None:
                vim_manager.toggle()
                vim_manager.save()
                self.vim_enabled = vim_manager.enabled
        return f"vim: {'enabled' if self.vim_enabled else 'disabled'}\nsettings_path: {state_path}"

    async def _execute_commit(self, _scheduler, _args: list[str]) -> str:
        cwd = os.getcwd()
        lines = []

        # Branch
        branch = current_branch(cwd=cwd)
        lines.append(f"Branch: {branch}")

        # Staged changes
        try:
            staged = subprocess.run(
                ["git", "diff", "--cached", "--stat"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=cwd,
            ).stdout.strip()
            if staged:
                lines.append(f"\nStaged changes:\n{staged}")
            else:
                lines.append("\nNo staged changes.")
        except Exception:
            lines.append("\nNo staged changes.")

        # Unstaged changes
        try:
            unstaged = subprocess.run(
                ["git", "diff", "--stat"], capture_output=True, text=True, timeout=5, cwd=cwd
            ).stdout.strip()
            if unstaged:
                lines.append(f"\nUnstaged changes:\n{unstaged}")
        except Exception:
            pass

        # Untracked files
        try:
            untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=cwd,
            ).stdout.strip()
            if untracked:
                paths = untracked.splitlines()
                lines.append(f"\n{len(paths)} untracked file(s):")
                lines.extend(f"- {path}" for path in paths[:10])
                if len(paths) > 10:
                    lines.append(f"- ... {len(paths) - 10} more")
        except Exception:
            pass

        # Guidance
        has_staged = "Staged changes:" in "\n".join(lines)
        has_unstaged = "Unstaged changes:" in "\n".join(lines)
        has_untracked = "untracked file(s)" in "\n".join(lines)

        if has_staged:
            lines.append("\nReady to commit. Ask me to 'commit these changes' with a message.")
        elif has_unstaged or has_untracked:
            lines.append("\nNo staged changes. Ask me to 'stage and commit' the changes.")
        else:
            lines.append("\nNothing to commit, working tree clean.")

        return "\n".join(lines)

    async def _execute_commit_push_pr(self, _scheduler, _args: list[str]) -> str:
        cwd = os.getcwd()
        return (
            f"branch: {current_branch(cwd=cwd)}\n"
            f"remote: {remote_url(cwd=cwd)}\n"
            f"status:\n{status_short(cwd=cwd)}\n"
            f"staged_diff:\n{staged_diff_stat(cwd=cwd)}\n"
            f"unstaged_diff:\n{diff_stat(cwd=cwd)}"
        )

    async def _execute_release_notes(self, _scheduler, _args: list[str]) -> str:
        config = self.config_service.load()
        current_version = resolve_runtime_version()
        changelog = get_stored_changelog()
        if not changelog:
            changelog = fetch_and_store_changelog(timeout_seconds=0.5)

        if changelog:
            groups = get_recent_release_note_groups(
                current_version,
                config.harness.last_release_notes_seen,
                changelog,
                max_versions=3,
            )
            if not groups:
                groups = list(reversed(get_all_release_notes(changelog)[-3:]))
            if groups:
                if config.harness.last_release_notes_seen != current_version:
                    config.harness.last_release_notes_seen = current_version
                    try:
                        self.config_service.save(config)
                    except Exception:
                        pass
                return format_release_notes(groups)

        return recent_commits(cwd=os.getcwd(), limit=10)

    async def _execute_version(self, _scheduler, _args: list[str]) -> str:
        return render_command_version()

    async def _execute_env(self, _scheduler, _args: list[str]) -> str:
        if _args:
            if _scheduler is None or not hasattr(_scheduler, "session"):
                return "env: setting session variables requires an active session."
            session_id = getattr(_scheduler.session, "session_id", None)
            if not isinstance(session_id, str) or not session_id:
                return "env: setting session variables requires an active session."

            if _args[0] in {"help", "-h", "--help"}:
                return "Usage: /env [NAME=value | unset NAME | clear]"
            if _args[0] == "clear":
                clear_session_env(session_id)
                return "env: cleared session-scoped environment variables."
            if _args[0] == "unset":
                if len(_args) != 2:
                    return "Usage: /env unset NAME"
                name = _args[1].strip()
                if not is_valid_env_name(name):
                    return f"env: invalid variable name: {name}"
                delete_session_env_var(session_id, name)
                os.environ.pop(name, None)
                return f"env: removed {name} from this session."

            if len(_args) != 1 or "=" not in _args[0]:
                return "Usage: /env [NAME=value | unset NAME | clear]"

            name, value = _args[0].split("=", 1)
            name = name.strip()
            if not is_valid_env_name(name):
                return f"env: invalid variable name: {name}"
            set_session_env_var(session_id, name, value)
            return f"env: set {name} for this session."

        interesting = [
            ("cwd", os.getcwd()),
            ("python", sys.executable),
            ("KODER_MODEL", os.environ.get("KODER_MODEL")),
            ("KODER_REASONING_EFFORT", os.environ.get("KODER_REASONING_EFFORT")),
            ("OPENAI_API_KEY", "set" if os.environ.get("OPENAI_API_KEY") else "unset"),
            ("ANTHROPIC_API_KEY", "set" if os.environ.get("ANTHROPIC_API_KEY") else "unset"),
            ("GOOGLE_API_KEY", "set" if os.environ.get("GOOGLE_API_KEY") else "unset"),
        ]
        lines = [f"{key}: {value}" for key, value in interesting]
        session_id = getattr(getattr(_scheduler, "session", None), "session_id", None)
        session_vars = (
            load_session_env(session_id) if isinstance(session_id, str) and session_id else {}
        )
        if session_vars:
            lines.append("session_env:")
            for key, value in sorted(session_vars.items()):
                lines.append(f"- {key}={value}")
        else:
            lines.append("session_env: none")
        return "\n".join(lines)

    async def _execute_add_dir(self, _scheduler, _args: list[str]) -> str:
        if _args:
            directory_path = " ".join(_args).strip()
            result = validate_directory_for_workspace(
                directory_path,
                workspace_root=self.permission_service.workspace_root,
                additional_roots=self.permission_service.additional_roots,
            )
            if result.result_type != "success":
                return add_dir_help_message(result)

            candidate = self.permission_service.add_working_directory(
                result.absolute_path or directory_path
            )
            if candidate not in self.additional_skill_dirs:
                self.additional_skill_dirs.append(candidate)
            os.environ[SKILL_ADDITIONAL_DIRS_ENV] = os.pathsep.join(
                str(path) for path in self.additional_skill_dirs
            )
            return f"Added {candidate} as a working directory for this session · /permissions to manage"
        return f"active_project_dir: {os.getcwd()}"

    async def _execute_agents(self, _scheduler, _args: list[str]) -> str:
        definitions = get_agent_definitions(
            cwd=Path.cwd(),
            plugin_root=self.plugin_root,
            cli_agents_json=self.cli_agents_json,
        )
        if _args:
            action = _args[0]
            if action == "create":
                if len(_args) < 4:
                    return "agents: usage /agents create <project|user> <name> <description>"
                scope = _args[1]
                if scope in {"personal", "user"}:
                    target_dir = user_agents_dir()
                elif scope == "project":
                    target_dir = project_agents_dir(Path.cwd())
                else:
                    return "agents: scope must be project or user"
                name = _args[2]
                description = " ".join(_args[3:]).strip()
                target_dir.mkdir(parents=True, exist_ok=True)
                target_file = target_dir / f"{name}.md"
                target_file.write_text(
                    "\n".join(
                        [
                            "---",
                            f"name: {name}",
                            f"description: {description}",
                            "---",
                            f"You are {name}. Complete the delegated task carefully.",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
                return f"agents: created\npath: {target_file}"
            if action == "delete":
                if len(_args) < 2:
                    return "agents: usage /agents delete <name>"
                name = _args[1]
                candidates = [
                    project_agents_dir(Path.cwd()) / f"{name}.md",
                    user_agents_dir() / f"{name}.md",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        candidate.unlink()
                        return f"agents: deleted\npath: {candidate}"
                return f"agents: not found {name}"
            if action in {"show", "path"}:
                if len(_args) < 2:
                    return "agents: usage /agents show <name>"
                name = _args[1]
                matches = [agent for agent in definitions.all_agents if agent.agent_type == name]
                if not matches:
                    return f"agents: not found {name}"
                return render_agent_details(matches, requested_name=name)
            if action in {"summary", "status"}:
                if len(_args) == 1:
                    records = [
                        self.agent_service.refresh_summary(item.id)
                        for item in self.agent_service.list_records()
                    ]
                    return render_agent_runtime_summaries(records)
                requested = _args[1]
                agent_id = self.agent_service.resolve_agent_id(requested)
                if agent_id is None:
                    return f"agents: runtime agent not found {requested}"
                record = self.agent_service.refresh_summary(agent_id)
                return render_agent_runtime_summary(record)
        return render_agents_overview(
            definitions,
            cwd=Path.cwd(),
        )

    async def _execute_peers(self, _scheduler, _args: list[str]) -> str:
        mode = resolve_teammate_mode(
            config_service=self.config_service,
            cli_mode=self.teammate_mode_override,
        )
        execution_mode = resolve_teammate_execution_mode(mode)
        service = self.team_service
        if _args:
            action = _args[0]
            if action == "create":
                if len(_args) < 2:
                    return "peers: usage /peers create <team-name>"
                team_id = service.create_team(_args[1])
                return (
                    "peers: created\n"
                    f"team_id: {team_id}\n"
                    f"teammate_mode: {mode}\n"
                    f"effective_teammate_mode: {execution_mode}\n"
                    f"teams_root: {service.teams_root}\n"
                    f"tasks_root: {service.tasks_root}"
                )
            if action == "show":
                if len(_args) < 2:
                    return "peers: usage /peers show <team-id>"
                team_id = _args[1]
                try:
                    record = service.get(team_id)
                except KeyError:
                    return f"peers: not found {team_id}"
                members = service.member_records(team_id)
                tasks = service.task_service(team_id).list_tasks()
                approvals = service.list_plan_approvals(team_id)
                shutdowns = service.list_shutdown_requests(team_id)

                def tmux_pane_state(pane_id: str) -> str:
                    try:
                        list_result = subprocess.run(
                            ["tmux", "list-panes", "-a", "-F", "#{pane_id}"],
                            capture_output=True,
                            text=True,
                            timeout=1,
                            check=False,
                        )
                        if list_result.returncode != 0:
                            return "unknown"
                        pane_ids = {line.strip() for line in list_result.stdout.splitlines()}
                        if pane_id not in pane_ids:
                            return "missing"
                        result = subprocess.run(
                            ["tmux", "display-message", "-p", "-t", pane_id, "#{pane_dead}"],
                            capture_output=True,
                            text=True,
                            timeout=1,
                            check=False,
                        )
                    except (OSError, subprocess.SubprocessError):
                        return "unknown"
                    if result.returncode != 0:
                        return "missing"
                    return "dead" if result.stdout.strip() == "1" else "running"

                def render_member(member) -> str:
                    base = (
                        f"- member {member.agent_id}: name={member.name} "
                        f"type={member.agent_type} mode={member.mode} active={member.is_active} "
                    )
                    if member.mode == "tmux":
                        pane_id = member.session_id or member.agent_id
                        return base + f"pane_state={tmux_pane_state(pane_id)}"
                    agent_state = (
                        self.agent_service.get(member.agent_id).state
                        if member.agent_id in self.agent_service._agents
                        else "unknown"
                    )
                    return base + f"agent_state={agent_state}"

                return (
                    f"peers: {record.name}\n"
                    f"team_id: {record.id}\n"
                    f"member_count: {len(members)}\n"
                    f"task_count: {len(tasks)}\n"
                    f"pending_plan_approvals: {len(approvals)}\n"
                    f"pending_shutdown_requests: {len(shutdowns)}\n"
                    f"config_path: {record.config_path}\n"
                    + "\n".join(render_member(member) for member in members)
                )
            if action == "memory":
                if len(_args) < 2:
                    return "peers: usage /peers memory <team-id> [status|sync]"
                team_id = _args[1]
                memory_action = _args[2] if len(_args) > 2 else "status"

                def display_path(path: Path) -> str:
                    try:
                        return str(path.resolve().relative_to(Path.cwd().resolve()))
                    except ValueError:
                        return str(path)

                if memory_action == "status":
                    try:
                        status = service.team_memory_status(team_id)
                    except KeyError:
                        return f"peers: not found {team_id}"
                    return (
                        "peers: team memory\n"
                        f"team_id: {team_id}\n"
                        f"project_dir: {display_path(status.project_dir)}\n"
                        f"runtime_dir: {status.runtime_dir}\n"
                        f"project_files: {status.project_files}\n"
                        f"runtime_files: {status.runtime_files}"
                    )
                if memory_action == "sync":
                    try:
                        result = service.sync_team_memory(team_id)
                    except KeyError:
                        return f"peers: not found {team_id}"
                    return (
                        "peers: team memory sync\n"
                        f"team_id: {team_id}\n"
                        f"project_dir: {display_path(result.project_dir)}\n"
                        f"runtime_dir: {result.runtime_dir}\n"
                        f"copied_to_project: {result.copied_to_project}\n"
                        f"copied_to_runtime: {result.copied_to_runtime}\n"
                        f"unchanged: {result.unchanged}\n"
                        f"skipped: {result.skipped}"
                    )
                return "peers: usage /peers memory <team-id> [status|sync]"
            if action == "cleanup":
                if len(_args) < 2:
                    return "peers: usage /peers cleanup <team-id>"
                team_id = _args[1]
                try:
                    service.delete_team(team_id)
                except KeyError:
                    return f"peers: not found {team_id}"
                except RuntimeError as exc:
                    return f"peers: cleanup blocked\nreason: {exc}"
                return f"peers: cleaned up\nteam_id: {team_id}"
            if action == "member":
                if len(_args) < 4 or _args[1] != "add":
                    return "peers: usage /peers member add <team-id> <agent-id> [name]"
                team_id = _args[2]
                agent_id = _args[3]
                name = _args[4] if len(_args) > 4 else agent_id
                try:
                    service.add_member(
                        team_id,
                        agent_id,
                        name=name,
                        cwd=Path.cwd(),
                        mode=self.permission_service.mode.value,
                    )
                except KeyError:
                    return f"peers: not found {team_id}"
                return (
                    f"peers: member added\nteam_id: {team_id}\nagent_id: {agent_id}\nname: {name}"
                )
            if action == "spawn":
                if len(_args) < 5:
                    return "peers: usage /peers spawn <team-id> <agent-type> <member-name> <prompt...> [--plan-mode]"
                team_id = _args[1]
                agent_type = _args[2]
                member_name = _args[3]
                raw_args = _args[4:]
                plan_mode_required = False
                filtered_args: list[str] = []
                for token in raw_args:
                    if token == "--plan-mode":
                        plan_mode_required = True
                    else:
                        filtered_args.append(token)
                prompt = " ".join(filtered_args).strip()
                if not prompt:
                    return "peers: usage /peers spawn <team-id> <agent-type> <member-name> <prompt...> [--plan-mode]"
                definitions = get_agent_definitions(
                    cwd=Path.cwd(),
                    plugin_root=self.plugin_root,
                    cli_agents_json=self.cli_agents_json,
                )
                selected = next(
                    (
                        candidate
                        for candidate in definitions.active_agents
                        if candidate.agent_type == agent_type
                    ),
                    None,
                )
                if selected is None:
                    return f"peers: unknown agent type {agent_type}"
                if selected.permission_mode == "plan":
                    plan_mode_required = True
                member_mode = "plan" if plan_mode_required else self.permission_service.mode.value
                effective_member_model = resolve_agent_model(selected) or get_model_name()
                if execution_mode == "in-process":
                    spawned = await self.in_process_teammate_runner.spawn_teammate(
                        team_id=team_id,
                        name=member_name,
                        agent_definition=selected,
                        prompt=prompt,
                        cwd=Path.cwd(),
                        plan_mode_required=plan_mode_required,
                        model=effective_member_model,
                    )
                    record = self.agent_service.get(spawned.agent_id)
                    member = next(
                        item
                        for item in service.member_records(team_id)
                        if item.agent_id == spawned.agent_id
                    )
                elif execution_mode == "tmux":
                    from koder_agent.harness.agents.teams.runtime import create_backend

                    backend = create_backend("tmux", team_id)
                    if backend is None:
                        return "peers: failed to create tmux backend"
                    pane = backend.spawn_member(
                        name=member_name,
                        prompt=prompt,
                        cwd=str(Path.cwd()),
                        model=effective_member_model,
                        env=os.environ,
                    )
                    service.add_member(
                        team_id,
                        pane.pane_id,
                        name=member_name,
                        agent_type=selected.agent_type,
                        model=effective_member_model,
                        prompt=prompt,
                        plan_mode_required=plan_mode_required,
                        cwd=Path.cwd(),
                        session_id=pane.pane_id,
                        mode="tmux",
                    )
                    return (
                        "peers: teammate spawned in tmux\n"
                        f"team_id: {team_id}\n"
                        f"agent_id: {pane.pane_id}\n"
                        f"name: {member_name}\n"
                        f"pane_id: {pane.pane_id}"
                    )
                else:
                    record = await self.agent_service.launch_background(
                        agent_definition=selected,
                        prompt=prompt,
                        description=prompt[:80],
                        cwd=Path.cwd(),
                        permission_mode=member_mode,
                    )
                    member = service.add_member(
                        team_id,
                        record.id,
                        name=member_name,
                        agent_type=selected.agent_type,
                        model=effective_member_model,
                        prompt=prompt,
                        plan_mode_required=plan_mode_required,
                        cwd=record.worktree_path or Path.cwd(),
                        worktree_path=record.worktree_path,
                        session_id=record.session_id,
                        mode=member_mode,
                    )
                return (
                    "peers: teammate spawned\n"
                    f"team_id: {team_id}\n"
                    f"agent_id: {member.agent_id}\n"
                    f"name: {member.name}\n"
                    f"agent_type: {member.agent_type}\n"
                    f"mode: {member.mode}\n"
                    f"session_id: {record.session_id}\n"
                    f"output_file: {record.output_path}"
                )
            if action == "mode":
                if len(_args) < 5 or _args[1] != "set":
                    return "peers: usage /peers mode set <team-id> <agent-id|all> <mode>"
                team_id = _args[2]
                target = _args[3]
                desired_mode = _args[4] if len(_args) > 4 else None
                if desired_mode not in {"default", "strict", "bypass", "plan"}:
                    return "peers: mode must be one of default, strict, bypass, plan"
                if target == "all":
                    updated = service.set_all_member_modes(team_id, desired_mode)
                    for member in updated:
                        try:
                            self.agent_service.update_permission_mode(member.agent_id, desired_mode)
                        except KeyError:
                            pass
                    return f"peers: mode updated\nteam_id: {team_id}\ntarget: all\ncount: {len(updated)}\nmode: {desired_mode}"
                try:
                    member = service.set_member_mode(team_id, target, desired_mode)
                except KeyError:
                    return f"peers: not found {target}"
                try:
                    self.agent_service.update_permission_mode(member.agent_id, desired_mode)
                except KeyError:
                    pass
                return f"peers: mode updated\nteam_id: {team_id}\nagent_id: {member.agent_id}\nmode: {member.mode}"
            if action == "plan":
                if len(_args) < 4:
                    return "peers: usage /peers plan <request|approve|reject> <team-id> <agent-id> [...]"
                plan_action = _args[1]
                team_id = _args[2]
                agent_id = _args[3]
                if plan_action == "request":
                    if len(_args) < 6:
                        return "peers: usage /peers plan request <team-id> <agent-id> <permission-mode> <plan>"
                    permission_mode = _args[4]
                    plan_text = " ".join(_args[5:]).strip()
                    service.request_plan_approval(
                        team_id,
                        agent_id=agent_id,
                        plan=plan_text,
                        requested_permission_mode=permission_mode,
                    )
                    return f"peers: plan approval requested\nteam_id: {team_id}\nagent_id: {agent_id}\npermission_mode: {permission_mode}"
                if plan_action == "approve":
                    permission_mode = _args[4] if len(_args) > 4 else None
                    service.respond_plan_approval(
                        team_id,
                        agent_id=agent_id,
                        approved=True,
                        permission_mode=permission_mode,
                    )
                    try:
                        self.agent_service.update_permission_mode(
                            agent_id,
                            permission_mode or "default",
                        )
                    except KeyError:
                        pass
                    return f"peers: plan approved\nteam_id: {team_id}\nagent_id: {agent_id}\nmode: {permission_mode or 'default'}"
                if plan_action == "reject":
                    feedback = " ".join(_args[4:]).strip() if len(_args) > 4 else None
                    service.respond_plan_approval(
                        team_id,
                        agent_id=agent_id,
                        approved=False,
                        feedback=feedback,
                    )
                    return f"peers: plan rejected\nteam_id: {team_id}\nagent_id: {agent_id}"
                return (
                    "peers: usage /peers plan <request|approve|reject> <team-id> <agent-id> [...]"
                )
            if action == "shutdown":
                if len(_args) < 4:
                    return "peers: usage /peers shutdown <request|approve|reject> <team-id> <agent-id> [...]"
                shutdown_action = _args[1]
                team_id = _args[2]
                agent_id = _args[3]
                if shutdown_action == "request":
                    reason = " ".join(_args[4:]).strip() if len(_args) > 4 else None
                    service.request_shutdown(team_id, agent_id=agent_id, reason=reason)
                    return f"peers: shutdown requested\nteam_id: {team_id}\nagent_id: {agent_id}"
                if shutdown_action == "approve":
                    if self.in_process_teammate_runner.manages(agent_id):
                        await self.in_process_teammate_runner.terminate(agent_id)
                    await service.respond_shutdown(
                        team_id,
                        agent_id=agent_id,
                        approved=True,
                        agent_service=(
                            None
                            if self.in_process_teammate_runner.manages(agent_id)
                            else self.agent_service
                        ),
                    )
                    return f"peers: shutdown approved\nteam_id: {team_id}\nagent_id: {agent_id}"
                if shutdown_action == "reject":
                    feedback = " ".join(_args[4:]).strip() if len(_args) > 4 else None
                    await service.respond_shutdown(
                        team_id,
                        agent_id=agent_id,
                        approved=False,
                        feedback=feedback,
                        agent_service=self.agent_service,
                    )
                    return f"peers: shutdown rejected\nteam_id: {team_id}\nagent_id: {agent_id}"
                return "peers: usage /peers shutdown <request|approve|reject> <team-id> <agent-id> [...]"
            if action == "send":
                if len(_args) < 4:
                    return (
                        "peers: usage /peers send <team-id> <recipient> [--from <sender>] <message>"
                    )
                team_id = _args[1]
                recipient = _args[2]
                sender = "team-lead"
                message_args = list(_args[3:])
                if message_args[:1] == ["--from"]:
                    if len(message_args) < 3:
                        return "peers: usage /peers send <team-id> <recipient> [--from <sender>] <message>"
                    sender = message_args[1]
                    message_args = message_args[2:]
                message = " ".join(message_args).strip()
                if not message:
                    return (
                        "peers: usage /peers send <team-id> <recipient> [--from <sender>] <message>"
                    )
                service.route(team_id, message, recipient=recipient, sender=sender)
                return (
                    "peers: message queued\n"
                    f"team_id: {team_id}\n"
                    f"recipient: {recipient}\n"
                    f"sender: {sender}"
                )
            if action == "inbox":
                if len(_args) < 2:
                    return "peers: usage /peers inbox <team-id> [recipient] [--consume]"
                team_id = _args[1]
                consume = "--consume" in _args[2:]
                recipient_args = [arg for arg in _args[2:] if arg != "--consume"]
                recipient = recipient_args[0] if recipient_args else "team-lead"
                if consume:
                    message = service.consume_next_mailbox_entry(team_id, recipient=recipient)
                    if message is None:
                        return f"peers: inbox empty\nteam_id: {team_id}\nrecipient: {recipient}"
                    return (
                        "peers: inbox consumed\n"
                        f"team_id: {team_id}\n"
                        f"recipient: {recipient}\n"
                        f"sender: {message.sender}\n"
                        "read: true\n"
                        f"message: {message.content}"
                    )
                messages = service.read_mailbox(team_id, recipient=recipient)
                if not messages:
                    return f"peers: inbox empty\nteam_id: {team_id}\nrecipient: {recipient}"
                return "\n".join(
                    [f"peers: inbox\nteam_id: {team_id}\nrecipient: {recipient}"]
                    + [f"- {message.created_at}: {message.content}" for message in messages]
                )
            if action == "history":
                if len(_args) < 2:
                    return "peers: usage /peers history <team-id> [--json]"
                team_id = _args[1]
                json_mode = "--json" in _args[2:]
                try:
                    entries = service.history_entries(team_id)
                except KeyError:
                    return f"peers: not found {team_id}"
                if json_mode:
                    return json.dumps(
                        [entry.__dict__ for entry in entries],
                        ensure_ascii=False,
                        indent=2,
                    )
                if not entries:
                    return f"peers: history empty\nteam_id: {team_id}"
                lines = [f"peers: history\nteam_id: {team_id}"]
                for entry in entries:
                    if entry.event == "message_sent":
                        lines.append(
                            f"- {entry.created_at} sent {entry.sender} -> {entry.recipient}: {entry.content}"
                        )
                    elif entry.event == "message_read":
                        lines.append(
                            f"- {entry.created_at} read {entry.recipient} <= {entry.sender}"
                        )
                    elif entry.event == "run_completed":
                        label = entry.member_name or entry.agent_id or "teammate"
                        source = f" source={entry.source}" if entry.source else ""
                        lines.append(
                            f"- {entry.created_at} run {label} state={entry.state}{source}: {entry.content}"
                        )
                    else:
                        lines.append(
                            f"- {entry.created_at} {entry.event}: {entry.content or ''}".rstrip()
                        )
                return "\n".join(lines)
            if action == "task":
                if len(_args) < 3:
                    return "peers: usage /peers task <create|claim|complete|list> <team-id> [...]"
                task_action = _args[1]
                team_id = _args[2]
                task_service = service.task_service(team_id)
                if task_action == "create":
                    if len(_args) < 4:
                        return "peers: usage /peers task create <team-id> <subject>"
                    task = task_service.create_task(" ".join(_args[3:]).strip())
                    return (
                        "peers: task created\n"
                        f"team_id: {team_id}\n"
                        f"task_id: {task.id}\n"
                        f"subject: {task.subject}"
                    )
                if task_action == "claim":
                    if len(_args) < 5:
                        return "peers: usage /peers task claim <team-id> <task-id> <agent-id>"
                    result = task_service.claim_task(
                        _args[3],
                        _args[4],
                        check_agent_busy=True,
                    )
                    return (
                        "peers: task claim\n"
                        f"team_id: {team_id}\n"
                        f"task_id: {_args[3]}\n"
                        f"owner: {_args[4]}\n"
                        f"success: {result.success}\n"
                        f"reason: {result.reason or 'claimed'}\n"
                        f"status: {result.task.status if result.task else 'unknown'}"
                    )
                if task_action == "complete":
                    if len(_args) < 4:
                        return "peers: usage /peers task complete <team-id> <task-id>"
                    task = task_service.update_status(_args[3], "completed")
                    return (
                        "peers: task completed\n"
                        f"team_id: {team_id}\n"
                        f"task_id: {task.id}\n"
                        f"status: {task.status}\n"
                        f"owner: {task.owner or ''}"
                    )
                if task_action == "list":
                    tasks = task_service.list_tasks()
                    if not tasks:
                        return f"peers: no tasks\nteam_id: {team_id}"
                    return "\n".join(
                        [f"peers: tasks\nteam_id: {team_id}"]
                        + [
                            f"- {task.id}: {task.subject} status={task.status} owner={task.owner or ''}"
                            for task in tasks
                        ]
                    )
                return "peers: usage /peers task <create|claim|complete|list> <team-id> [...]"
        team_id = service.create_team("peers")
        return (
            "peers: available\n"
            f"sample_team_id: {team_id}\n"
            f"teammate_mode: {mode}\n"
            f"effective_teammate_mode: {execution_mode}\n"
            "commands: create, show, spawn, send, inbox, history, task, memory, mode, plan, shutdown, cleanup\n"
            f"teams_root: {service.teams_root}\n"
            f"tasks_root: {service.tasks_root}"
        )

    async def _execute_feedback(self, _scheduler, _args: list[str]) -> str:
        raw_message = " ".join(_args).strip()
        cwd = str(Path.cwd())
        repo = remote_url(cwd=cwd)
        if repo.lower().startswith("error:"):
            repo = "No git remote configured."
        branch = current_branch(cwd=cwd)
        if not raw_message:
            return "\n".join(
                [
                    "feedback: capture and review feedback through the harness runtime.",
                    "saved: false",
                    "message: <empty>",
                    "usage: /feedback <message>",
                    f"repo: {repo}",
                    f"branch: {branch}",
                    f"cwd: {cwd}",
                ]
            )

        message = _redact_sensitive_debug_text(raw_message)
        feedback_path = harness_home_dir() / "feedback" / "feedback.jsonl"
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "cwd": cwd,
            "repo": repo,
            "branch": branch,
            "git_status": _redact_sensitive_debug_text(status_short(cwd=cwd)),
        }
        with feedback_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        lines = [
            "feedback: saved",
            f"message: {message}",
            f"repo: {repo}",
            f"branch: {branch}",
            f"cwd: {cwd}",
            f"path: {feedback_path}",
        ]
        return "\n".join(lines)

    async def _execute_statusline(self, scheduler, _args: list[str]) -> str:
        if _args and _args[0].strip().lower() in {"clear", "delete", "remove", "off", "disable"}:
            settings_path = update_user_statusline_config(None)
            return f"statusline: removed custom status line from {settings_path}"

        if not _args:
            imported = auto_configure_statusline_from_shell_prompt()
            if imported is None:
                return (
                    "statusline: couldn't auto-configure from your shell prompt. "
                    "No PS1 was found in ~/.zshrc, ~/.bashrc, ~/.bash_profile, or ~/.profile. "
                    "Run /statusline <description> to describe what you want."
                )
            return (
                f"statusline: configured from {imported.source_path}\n"
                f"settings_path: {imported.settings_path}\n"
                f"command: {imported.command}"
            )

        prompt = " ".join(_args).strip()
        definitions = getattr(scheduler, "agent_definitions", None) or get_agent_definitions(
            cwd=Path.cwd(),
            plugin_root=self.plugin_root,
            cli_agents_json=self.cli_agents_json,
        )
        selected = next(
            (
                agent
                for agent in definitions.active_agents
                if agent.agent_type == "statusline-setup"
            ),
            None,
        )
        if selected is None:
            return "statusline: setup agent is unavailable."

        seed_items = None
        if (
            scheduler is not None
            and hasattr(scheduler, "session")
            and hasattr(scheduler.session, "get_items")
        ):
            try:
                seed_items = await scheduler.session.get_items()
            except Exception:
                seed_items = None

        try:
            result = await self.agent_service.run_sync(
                agent_definition=selected,
                prompt=prompt,
                seed_items=seed_items,
                cwd=Path.cwd(),
            )
        except Exception as exc:
            return f"statusline: setup failed\n{exc}"
        return f"statusline: setup complete\n{result}"

    async def _execute_color(self, scheduler, _args: list[str]) -> str:
        color_list = ", ".join(AGENT_COLORS)
        if not _args or not _args[0].strip():
            return f"Please provide a color. Available colors: {color_list}, default"

        desired = _args[0].strip().lower()
        if desired in RESET_COLOR_ALIASES:
            self.current_color = "default"
            await self._set_session_color(scheduler, None)
            return "Session color reset to default"
        if desired not in AGENT_COLORS:
            return f'Invalid color "{desired}". Available colors: {color_list}, default'

        self.current_color = desired
        await self._set_session_color(scheduler, desired)
        return f"Session color set to: {desired}"

    async def _execute_btw(self, _scheduler, _args: list[str]) -> str:
        question = " ".join(_args).strip()
        if not question:
            return "Usage: /btw <question>"

        transcript_lines: list[str] = []
        if (
            _scheduler is not None
            and hasattr(_scheduler, "session")
            and hasattr(_scheduler.session, "get_items")
        ):
            try:
                items = await _scheduler.session.get_items()
            except Exception:
                items = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "")).strip().lower()
                if role not in {"user", "assistant"}:
                    continue
                text = self._flatten_session_text(item.get("content"))
                if text:
                    transcript_lines.append(f"{role}: {text}")

        prompt_parts = []
        if transcript_lines:
            prompt_parts.append("Current session context:")
            prompt_parts.append("\n".join(transcript_lines[-8:]))
            prompt_parts.append("")
        prompt_parts.append(f"Side question: {question}")
        prompt_parts.append(
            "Answer briefly and directly. Do not mention tools or internal process."
        )
        prompt_text = "\n".join(prompt_parts).strip()

        try:
            answer = await llm_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You answer quick side questions about the user's current coding session. "
                            "Use the provided session context when relevant. Keep the answer concise."
                        ),
                    },
                    {"role": "user", "content": prompt_text},
                ]
            )
            normalized = answer.strip()
            if normalized:
                return normalized
        except Exception as exc:
            return f"btw unavailable: {exc}"

        return "btw unavailable: no response"

    async def _execute_insights(self, scheduler, _args: list[str]) -> str:
        """Show session analytics."""
        items = await self._get_session_items(scheduler)
        role_counts = Counter(
            str(item.get("role", "")).strip().lower()
            for item in items
            if str(item.get("role", "")).strip()
        )
        tool_records = self._collect_tool_call_records(items)
        tool_call_count = sum(1 for record in tool_records if record["kind"] == "call")
        tool_result_count = sum(1 for record in tool_records if record["kind"] == "output")
        tool_name_counts = Counter(
            record["name"] for record in tool_records if record["kind"] == "call"
        )
        context_files = self._collect_context_file_paths(items)

        lines = [
            "Session Insights:",
            f"  Transcript items: {len(items)}",
            f"  Messages: {sum(role_counts.values())}",
            f"  User messages: {role_counts.get('user', 0)}",
            f"  Assistant messages: {role_counts.get('assistant', 0)}",
            f"  Tool results: {tool_result_count}",
            f"  Tool calls: {tool_call_count}",
            f"  Files in context: {len(context_files)}",
        ]
        if tool_name_counts:
            lines.append("  Tool usage:")
            for name, count in tool_name_counts.most_common(10):
                lines.append(f"    - {name}: {count}")
        if context_files:
            lines.append("  Context files:")
            lines.extend(f"    - {path}" for path in context_files)

        usage = getattr(getattr(scheduler, "usage_tracker", None), "session_usage", None)
        request_count = getattr(usage, "request_count", 0) if usage is not None else 0
        input_tokens = getattr(usage, "input_tokens", 0) if usage is not None else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage is not None else 0
        total_cost = getattr(usage, "total_cost", 0.0) if usage is not None else 0.0

        lines.extend(
            [
                f"  Requests: {request_count}",
                f"  Input tokens: {input_tokens:,}",
                f"  Output tokens: {output_tokens:,}",
                f"  Total cost: ${total_cost:.4f}",
            ]
        )
        if request_count > 0:
            avg_in = input_tokens // request_count
            avg_out = output_tokens // request_count
            lines.append(f"  Avg tokens/request: {avg_in:,} in / {avg_out:,} out")

        usage_tracker = getattr(scheduler, "usage_tracker", None)
        per_model = (
            usage_tracker.get_per_model_usage()
            if usage_tracker is not None and hasattr(usage_tracker, "get_per_model_usage")
            else {}
        )
        if per_model:
            lines.append("\n  Per-model breakdown:")
            for model, mu in sorted(per_model.items()):
                lines.append(
                    f"    {model}: {mu.request_count} requests, {mu.input_tokens:,}in/{mu.output_tokens:,}out"
                )
        return "\n".join(lines)

    async def _execute_sandbox(self, _scheduler, _args: list[str]) -> str:
        """Show and update sandbox settings."""
        from pathlib import Path

        from koder_agent.harness.sandbox.registry import BACKEND_IDS, normalize_backend_id

        cwd = Path.cwd()

        def _render_backend_choices(prefix: str | None = None) -> str:
            state = resolve_sandbox_settings(cwd)
            lines: list[str] = []
            if prefix:
                lines.append(prefix)
            lines.append("Available sandbox backends:")
            for status in state.backend_statuses:
                marker = "*" if status.backend_id == state.backend else "-"
                availability = "available" if status.available else "unavailable"
                lines.append(f"  {marker} {status.backend_id}: {availability} ({status.reason})")
            lines.append("Usage: /sandbox enable <backend>")
            lines.append("Example: /sandbox enable unix-local")
            return "\n".join(lines)

        def _render_status(prefix: str | None = None) -> str:
            state = resolve_sandbox_settings(cwd)
            lines: list[str] = []
            if prefix:
                lines.append(prefix)
            lines.extend(
                [
                    f"sandbox_enabled: {str(state.enabled).lower()}",
                    f"backend: {state.backend}",
                    f"backend_available: {str(state.backend_available).lower()}",
                    f"backend_reason: {state.backend_reason}",
                    f"mode: {state.policy_mode}",
                    f"auto_allow_bash_if_sandboxed: {str(state.auto_allow_bash_if_sandboxed).lower()}",
                    f"network_access: {str(state.network_access).lower()}",
                    "network_policy_enforcement: "
                    + next(
                        (
                            status.capabilities.supports_network_policy
                            for status in state.backend_statuses
                            if status.backend_id == state.backend
                        ),
                        "unknown",
                    ),
                    "allowed_domains: "
                    + (", ".join(state.allowed_domains) if state.allowed_domains else "none")
                    + (" (policy metadata, not enforced)" if state.allowed_domains else ""),
                    "denied_domains: "
                    + (", ".join(state.denied_domains) if state.denied_domains else "none")
                    + (" (policy metadata, not enforced)" if state.denied_domains else ""),
                    "protected_paths: "
                    + (", ".join(state.protected_paths) if state.protected_paths else "none"),
                    f"excluded_commands: {len(state.excluded_commands)}",
                    f"policy_locked: {str(state.policy_locked).lower()}",
                    f"settings_path: {state.settings_path}",
                    "backend_options: " + ", ".join(BACKEND_IDS),
                ]
            )
            if state.enabled and state.backend_available and state.platform_enabled:
                lines.append(
                    f"note: foreground shell commands use sandbox backend {state.backend}."
                )
            elif state.enabled:
                lines.append(
                    "note: sandbox is enabled, but the configured backend is unavailable; "
                    "non-excluded foreground shell commands are denied."
                )
            else:
                lines.append(
                    "note: sandbox disabled; shell commands use the normal local executor."
                )
            return "\n".join(lines)

        if not _args or _args[0].lower() == "status":
            return _render_status()

        subcommand = _args[0].lower()
        state = resolve_sandbox_settings(cwd)
        if subcommand in {"backends", "list"}:
            return _render_backend_choices()

        if state.policy_locked and subcommand in {"enable", "disable", "off", "exclude"}:
            return _render_status("sandbox: settings locked by managed policy")

        if subcommand == "enable":
            if len(_args) < 2:
                return _render_backend_choices("sandbox: choose a backend")
            backend = normalize_backend_id(_args[1])
            if backend not in BACKEND_IDS:
                return _render_backend_choices(f"sandbox: unknown backend {backend}")
            update_local_sandbox_settings(
                cwd,
                enabled=True,
                backend=backend,
            )
            return _render_status("sandbox: enabled")

        if subcommand in {"disable", "off"}:
            update_local_sandbox_settings(cwd, enabled=False)
            return _render_status("sandbox: disabled")

        if subcommand == "exclude":
            command_pattern = " ".join(_args[1:]).strip()
            if not command_pattern:
                return 'Usage: /sandbox exclude "command pattern"'
            if (
                len(command_pattern) >= 2
                and command_pattern[0] == command_pattern[-1]
                and command_pattern[0] in {'"', "'"}
            ):
                command_pattern = command_pattern[1:-1]
            target, normalized = add_excluded_command(cwd, command_pattern)
            state = resolve_sandbox_settings(cwd)
            return (
                "sandbox: excluded command added\n"
                f"pattern: {normalized}\n"
                f"excluded_commands: {len(state.excluded_commands)}\n"
                f"settings_path: {target}"
            )

        return (
            'Usage: /sandbox [status|backends|enable <backend>|disable|exclude "command pattern"]'
        )

    async def _execute_loop(self, _scheduler, args: list[str]) -> str:
        """Create, list, and delete scheduled loop jobs."""
        from koder_agent.harness.cron.loop import (
            LOOP_USAGE,
            LoopSpecError,
            format_loop_jobs,
            parse_loop_spec,
        )
        from koder_agent.tools.cron import _get_cron_storage, cron_create, cron_delete

        lowered = [arg.lower() for arg in args]
        if lowered and lowered[0] in {"help", "--help", "-h"}:
            return LOOP_USAGE

        default_storage_path = Path.home() / ".koder" / "scheduled_tasks.json"

        def _active_storage_path() -> Path:
            try:
                storage = _get_cron_storage()
            except Exception:
                return default_storage_path
            return Path(getattr(storage, "_path", default_storage_path))

        if not args or lowered[0] in {"list", "ls"}:
            if args and len(args) != 1:
                return LOOP_USAGE
            storage_path = default_storage_path
            try:
                storage = _get_cron_storage()
                storage_path = Path(getattr(storage, "_path", default_storage_path))
                jobs = storage.list_all()
            except Exception as exc:
                return (
                    "loop: failed to read scheduled task registry\n"
                    f"path: {storage_path}\n"
                    f"error: {exc}"
                )
            return format_loop_jobs(jobs)

        if lowered[0] in {"delete", "remove", "rm"}:
            if len(args) != 2:
                return "Usage: /loop delete <id>"
            try:
                result = json.loads(cron_delete(args[1]))
            except Exception as exc:
                return (
                    "loop: failed to update scheduled task registry\n"
                    f"path: {_active_storage_path()}\n"
                    f"error: {exc}"
                )
            if "error" in result:
                return f"loop: {result['error']}"
            return f"Loop job deleted: {result['id']}"

        try:
            spec = parse_loop_spec(args)
        except LoopSpecError as exc:
            return f"loop: {exc}\n{LOOP_USAGE}"

        try:
            result = json.loads(cron_create(spec.cron, spec.prompt, recurring=spec.recurring))
        except Exception as exc:
            return (
                "loop: failed to update scheduled task registry\n"
                f"path: {_active_storage_path()}\n"
                f"error: {exc}"
            )
        if "error" in result:
            return f"loop: {result['error']}"

        return "\n".join(
            [
                "Loop job created",
                f"id: {result['id']}",
                f"cron: {spec.cron}",
                f"human_schedule: {result.get('human_schedule', spec.cron)}",
                f"recurring: {str(spec.recurring).lower()}",
                f"prompt: {spec.prompt[:80]}",
            ]
        )

    async def _execute_schedule(self, _scheduler, _args: list[str]) -> str:
        """Show/manage cron tasks."""
        if _args:
            return "Usage: /schedule"

        storage_path = Path.home() / ".koder" / "scheduled_tasks.json"
        try:
            from koder_agent.tools.cron import _get_cron_storage, _human_schedule

            storage = _get_cron_storage()
            storage_path = getattr(storage, "_path", storage_path)
            jobs = storage.list_all()
        except Exception as exc:
            return (
                "schedule: failed to read scheduled task registry\n"
                f"path: {storage_path}\n"
                f"error: {exc}"
            )

        if not jobs:
            return "No scheduled tasks. Use cron_create or /loop to create recurring tasks."

        lines = [f"Scheduled tasks ({len(jobs)}):"]
        for index, job in enumerate(jobs, start=1):
            if not isinstance(job, dict):
                lines.extend(
                    [
                        f"  - index: {index}",
                        "    malformed: expected object",
                    ]
                )
                continue

            cron_expr = str(job.get("cron") or job.get("expression") or "?")
            prompt = str(job.get("prompt") or "")[:80]
            recurring = bool(job.get("recurring", True))
            lines.extend(
                [
                    f"  - id: {job.get('id', '?')}",
                    f"    cron: {cron_expr}",
                    f"    human_schedule: {_human_schedule(cron_expr)}",
                    f"    recurring: {str(recurring).lower()}",
                    f"    prompt: {prompt}",
                ]
            )
        return "\n".join(lines)

    async def _execute_torch(self, _scheduler, args: list[str]) -> str:
        """Deep code exploration — illuminate a topic."""
        topic = " ".join(args) if args else None
        if not topic:
            return "Usage: /torch <topic>\nDeeply explores a codebase topic using search, reading, and analysis."
        try:
            context_prompt = (
                f"The user wants to deeply explore: {topic}\n\n"
                "Provide a structured exploration plan:\n"
                "1. What files/modules to examine\n"
                "2. Key questions to answer\n"
                "3. What patterns to look for\n"
                "4. Suggested search queries"
            )
            result = await llm_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a code exploration assistant. Help the user explore a topic in their codebase.",
                    },
                    {"role": "user", "content": context_prompt},
                ]
            )
            return f"Torch: Exploring '{topic}'\n\n{result}"
        except Exception as e:
            return f"Exploration failed: {e}"

    async def _execute_ultraplan(self, _scheduler, args: list[str]) -> str:
        """Create a comprehensive implementation plan."""
        topic = " ".join(args) if args else None
        if not topic:
            return "Usage: /ultraplan <feature/task>\nCreates a detailed, comprehensive implementation plan."
        try:
            result = await llm_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior architect. Create detailed implementation plans with file paths, code snippets, test strategies, and rollout steps.",
                    },
                    {
                        "role": "user",
                        "content": f"Create a comprehensive implementation plan for: {topic}",
                    },
                ]
            )
            return f"Ultra Plan: {topic}\n\n{result}"
        except Exception as e:
            return f"Planning failed: {e}"

    async def _execute_fork(self, _scheduler, _args: list[str]) -> str:
        args = list(_args)

        def _render_fork_response(
            header: str,
            *,
            record,
            agent_type: str,
            permission_mode: str,
            context_mode: str | None = None,
        ) -> str:
            lines = [
                header,
                f"forked_agent_id: {record.id}",
                f"agent_type: {agent_type}",
                f"permission_mode: {permission_mode}",
            ]
            if context_mode:
                lines.append(f"context_mode: {context_mode}")
            lines.extend(
                [
                    "status: background",
                    f"session_id: {record.session_id}",
                    f"output_file: {record.output_path}",
                ]
            )
            if record.model_config:
                lines.append("model_config:")
                for key in (
                    "model_override",
                    "model_name",
                    "provider",
                    "base_url",
                    "native_openai",
                    "api_key_present",
                    "reasoning_effort",
                    "litellm_model",
                    "oauth_provider",
                    "oauth_headers_present",
                ):
                    if key in record.model_config:
                        value = record.model_config[key]
                        lines.append(f"  {key}: {value if value is not None else 'none'}")
            return "\n".join(lines)

        if not args:
            return "fork: provide a prompt to run in a background subagent."
        if args[0] == "--resume":
            if len(args) < 3:
                return "fork: usage /fork --resume <agent_id> <prompt>"
            agent_id = args[1]
            prompt = " ".join(args[2:]).strip()
            record = self.agent_service.get(agent_id)
            try:
                origin_cwd = resolve_agent_record_origin(record)
                execution_cwd = resolve_agent_record_execution_cwd(record)
            except ValueError as exc:
                return f"fork: cannot resume {agent_id}: {exc}"
            definitions = get_agent_definitions(
                cwd=origin_cwd,
                plugin_root=self.plugin_root,
                cli_agents_json=self.cli_agents_json,
            )
            selected = next(
                (
                    candidate
                    for candidate in definitions.active_agents
                    if candidate.agent_type == record.profile
                    and agent_definition_matches_record(record, candidate)
                ),
                None,
            )
            if selected is None:
                return (
                    "fork: cannot resume because the original agent definition "
                    f"is missing or changed: {record.profile}"
                )
            resumed = await self.agent_service.resume_background(
                agent_id=agent_id,
                agent_definition=selected,
                prompt=prompt,
                cwd=execution_cwd,
            )
            effective_permission_mode = (
                record.permission_mode or selected.permission_mode or "default"
            )
            return _render_fork_response(
                "fork: resumed background subagent",
                record=resumed,
                agent_type=selected.agent_type,
                permission_mode=effective_permission_mode,
            )
        definitions = getattr(_scheduler, "agent_definitions", None) or get_agent_definitions(
            cwd=Path.cwd(),
            plugin_root=self.plugin_root,
            cli_agents_json=self.cli_agents_json,
        )
        context_mode = "isolated"
        while args:
            if args[0] == "--context":
                if len(args) < 2 or args[1] not in {"isolated", "fork"}:
                    return "fork: usage /fork [--context isolated|fork] [agent_type] <prompt>"
                context_mode = args[1]
                args = args[2:]
                continue
            if args[0] in {"--with-context", "--fork-context"}:
                context_mode = "fork"
                args = args[1:]
                continue
            if args[0] == "--isolated":
                context_mode = "isolated"
                args = args[1:]
                continue
            break
        selected = None
        prompt = " ".join(args).strip()
        if args:
            candidate = next(
                (agent for agent in definitions.active_agents if agent.agent_type == args[0]),
                None,
            )
            if candidate is not None:
                selected = candidate
                prompt = " ".join(args[1:]).strip()
        if selected is None:
            selected = next(
                (
                    agent
                    for agent in definitions.active_agents
                    if agent.agent_type == "general-purpose"
                ),
                None,
            )
        if selected is None:
            return "fork: general-purpose agent is unavailable."
        current_main_agent = getattr(_scheduler, "agent_definition", None)
        if current_main_agent is not None and current_main_agent.tools is not None:
            allowed_types = current_main_agent.allowed_agent_types
            if allowed_types is None:
                return "fork: spawning subagents is not allowed for the active session agent."
            if allowed_types and selected.agent_type not in allowed_types:
                return (
                    "fork: requested agent is not allowed for the active session agent. "
                    f"Allowed: {', '.join(allowed_types)}"
                )
        if not prompt:
            return "fork: provide a prompt after the agent type."
        seed_items = None
        if (
            context_mode == "fork"
            and _scheduler is not None
            and hasattr(_scheduler.session, "get_items")
        ):
            seed_items = await _scheduler.session.get_items()
        record = await self.agent_service.launch_background(
            agent_definition=selected,
            prompt=prompt,
            description=prompt[:80],
            seed_items=seed_items,
            cwd=Path.cwd(),
            permission_mode=selected.permission_mode or "default",
        )
        effective_permission_mode = record.permission_mode or selected.permission_mode or "default"
        return _render_fork_response(
            "fork: launched background subagent",
            record=record,
            agent_type=selected.agent_type,
            permission_mode=effective_permission_mode,
            context_mode=context_mode,
        )

    async def _execute_issue(self, _scheduler, _args: list[str]) -> str:
        import subprocess

        if not _args:
            # List recent issues
            try:
                result = subprocess.run(
                    ["gh", "issue", "list", "--limit", "10"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return f"Recent issues:\n{result.stdout.strip()}"
                if result.returncode != 0:
                    detail = (
                        result.stderr.strip()
                        or result.stdout.strip()
                        or f"exit code {result.returncode}"
                    )
                    return f"Failed to fetch issues: {detail}"
                return "No open issues found. Create one with: /issue <title>"
            except FileNotFoundError:
                return "gh CLI not found. Install: https://cli.github.com"
            except subprocess.TimeoutExpired:
                return "Timeout while fetching issues. Try again."
            except Exception as e:
                return f"Error fetching issues: {e}"
        # Create issue with title
        title = " ".join(_args)
        return f"To create an issue, ask me: 'Create a GitHub issue titled \"{title}\"' and I'll use the gh CLI."

    async def _execute_pr_comments(self, _scheduler, _args: list[str]) -> str:
        return await run_pr_comments(cwd=Path.cwd())

    async def _execute_autofix_pr(self, _scheduler, _args: list[str]) -> str:
        import subprocess

        if not _args:
            return (
                "Usage: /autofix-pr #<PR-number>\n"
                "Inspects the PR diff size and tells you how to request an automated fix."
            )
        pr_num = _args[0].lstrip("#")
        try:
            diff = subprocess.run(
                ["gh", "pr", "diff", pr_num], capture_output=True, text=True, timeout=30
            )
            if diff.returncode != 0:
                return f"Failed to fetch PR #{pr_num}: {diff.stderr.strip()}"
            if not diff.stdout.strip():
                return f"PR #{pr_num} has no changes."
            lines = diff.stdout.strip().split("\n")
            return f"PR #{pr_num}: {len(lines)} diff lines. Ask me to 'review and fix PR #{pr_num}' for automated analysis."
        except FileNotFoundError:
            return "gh CLI not found. Install: https://cli.github.com"
        except subprocess.TimeoutExpired:
            return f"Timeout while fetching PR #{pr_num}."
        except Exception as e:
            return f"Error fetching PR: {e}"

    async def _execute_subscribe_pr(self, _scheduler, _args: list[str]) -> str:
        import subprocess

        try:
            result = subprocess.run(
                ["gh", "pr", "list", "--state", "open", "--limit", "5"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return (
                    f"Open PRs (subscribe via GitHub notifications):\n{result.stdout.strip()}\n\n"
                    "Use 'gh pr view <number>' for details."
                )
            if result.returncode != 0:
                detail = (
                    result.stderr.strip()
                    or result.stdout.strip()
                    or f"exit code {result.returncode}"
                )
                return f"Failed to fetch PRs: {detail}"
            return "No open PRs. PR subscription is managed via GitHub notifications."
        except FileNotFoundError:
            return "gh CLI not found. Install: https://cli.github.com"
        except subprocess.TimeoutExpired:
            return "Timeout while fetching PRs."
        except Exception as e:
            return f"Error: {e}"

    async def _execute_onboarding(self, _scheduler, _args: list[str]) -> str:
        from koder_agent.harness.onboarding import check_onboarding_state, get_onboarding_steps

        env = os.environ.copy()
        session_id = getattr(getattr(_scheduler, "session", None), "session_id", None)
        if isinstance(session_id, str) and session_id:
            env.update(load_session_env(session_id))
        state = check_onboarding_state(Path.cwd(), env=env)
        steps = get_onboarding_steps(state)
        if not steps:
            return "✓ Setup complete! All configuration is in place."
        lines = ["Setup checklist:"]
        for i, step in enumerate(steps, 1):
            lines.append(f"  {i}. {step}")
        lines.append(
            f"\nCompleted: API key={'✓' if state.api_key_configured else '✗'}, "
            f"Model={'✓' if state.model_selected else '✗'}, "
            f"Workspace={'✓' if state.workspace_trusted else '✗'}"
        )
        return "\n".join(lines)

    async def _execute_oauth_refresh(self, _scheduler, _args: list[str]) -> str:
        from koder_agent.auth.token_storage import TokenStorage

        if _args:
            return "Usage: /oauth-refresh"

        try:
            storage = TokenStorage()
            tokens = storage.get_all_tokens()
            if tokens:
                lines = ["oauth_refresh:", "providers:"]
                for provider in sorted(tokens):
                    token = tokens[provider]
                    state = "expired" if token.is_expired(buffer_ms=0) else "valid"
                    lines.append(f"- {provider}: {state}")
                    lines.append(f"  email: {token.email or 'unknown'}")
                    lines.append(f"  expires_at: {token.expires_at}")
                lines.append("refresh_command: koder auth login <provider>")
                return "\n".join(lines)
            return "oauth_refresh:\nproviders: none\nlogin_command: koder auth login <provider>"
        except Exception as e:
            return f"Token check failed: {e}"

    async def _execute_bughunter(self, _scheduler, args: list[str]) -> str:
        focus = " ".join(args).strip() or "workspace"
        cwd = os.getcwd()

        def run_git(git_args: list[str]) -> str:
            try:
                result = subprocess.run(
                    ["git", *git_args],
                    capture_output=True,
                    text=True,
                    cwd=cwd,
                    timeout=10,
                )
            except Exception as exc:
                return f"unavailable: {exc}"
            return result.stdout.strip() or result.stderr.strip() or "none"

        status = run_git(["status", "--short"])
        diff_stat = run_git(["diff", "--stat"])
        staged_stat = run_git(["diff", "--cached", "--stat"])
        diff_evidence = self._format_bughunter_diff_evidence(
            run_git(["diff", "--unified=0", "--no-ext-diff"]),
            run_git(["diff", "--cached", "--unified=0", "--no-ext-diff"]),
        )
        working_tree_state = "clean" if status == "none" and diff_stat == "none" else "dirty"
        lines = [
            "bughunter: local triage",
            f"focus: {focus}",
            f"cwd: {cwd}",
            f"working_tree: {working_tree_state}",
            "git_status:",
            status,
            "diff_stat:",
            diff_stat,
            "staged_diff_stat:",
            staged_stat,
            "diff_evidence:",
            diff_evidence,
            "recommended_checks:",
            "- /doctor for runtime health",
            "- /diff for exact pending changes",
            "- /security-review for security-focused review",
            "- uv run pytest -k <focused-test> for a targeted repro",
        ]
        return "\n".join(lines)

    def _format_bughunter_diff_evidence(self, unstaged: str, staged: str) -> str:
        parts = []
        if unstaged != "none":
            parts.append("unstaged:\n" + unstaged)
        if staged != "none":
            parts.append("staged:\n" + staged)
        if not parts:
            return "none"
        redacted = _redact_sensitive_debug_text("\n".join(parts))
        lines = redacted.splitlines()
        preview = "\n".join(lines[:80])
        if len(lines) > 80:
            preview += f"\n... truncated {len(lines) - 80} line(s)"
        if len(preview) > 6000:
            preview = preview[:6000].rstrip() + "\n... truncated"
        return preview

    def _collect_tool_call_records(self, items: list[dict]) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        for item in items:
            role = str(item.get("role", "")).strip().lower()
            if role == "assistant" and isinstance(item.get("tool_calls"), list):
                for call in item["tool_calls"]:
                    if not isinstance(call, dict):
                        continue
                    function = (
                        call.get("function") if isinstance(call.get("function"), dict) else {}
                    )
                    records.append(
                        {
                            "kind": "call",
                            "id": str(call.get("id") or "unknown"),
                            "name": str(function.get("name") or call.get("name") or "unknown"),
                            "preview": self._truncate_prompt_preview(
                                _redact_sensitive_debug_text(
                                    function.get("arguments") or call.get("arguments") or ""
                                ),
                                limit=140,
                            ),
                        }
                    )
            if role == "tool":
                records.append(
                    {
                        "kind": "output",
                        "id": str(item.get("tool_call_id") or "unknown"),
                        "name": str(item.get("name") or "tool"),
                        "preview": self._truncate_prompt_preview(
                            _redact_sensitive_debug_text(
                                self._flatten_session_text(item.get("content"))
                            ),
                            limit=140,
                        ),
                    }
                )
        return records

    async def _execute_debug_tool_call(self, scheduler, args: list[str]) -> str:
        records = self._collect_tool_call_records(await self._get_session_items(scheduler))
        if not records:
            return "debug-tool-call: no recorded tool calls in this session"

        if args and args[0] == "show":
            if len(args) != 2:
                return "Usage: /debug-tool-call [list|show <number>]"
            try:
                requested = int(args[1])
            except ValueError:
                return "Usage: /debug-tool-call [list|show <number>]"
            if requested < 1 or requested > len(records):
                return f"debug-tool-call: number must be between 1 and {len(records)}"
            record = records[requested - 1]
            return (
                "debug-tool-call: detail\n"
                f"number: {requested}\n"
                f"kind: {record['kind']}\n"
                f"id: {record['id']}\n"
                f"name: {record['name']}\n"
                f"preview: {record['preview']}"
            )
        if args and args[0] != "list":
            return "Usage: /debug-tool-call [list|show <number>]"

        lines = [f"debug-tool-call: {len(records)} recorded item(s)"]
        for index, record in enumerate(records[-10:], start=1):
            lines.append(
                f"{index}. {record['kind']} {record['name']} id={record['id']} preview={record['preview']}"
            )
        return "\n".join(lines)

    async def _execute_backfill_sessions(self, scheduler, _args: list[str]) -> str:
        try:
            session = getattr(scheduler, "session", None) if scheduler is not None else None
            db_path = getattr(session, "db_path", None) or EnhancedSQLiteSession._resolve_db_path()
            migrated = await migrate_legacy_sessions(str(db_path))
            sessions = await EnhancedSQLiteSession.list_sessions_with_titles()
            lines = ["backfill_sessions:", f"migrated: {migrated}"]
            if not sessions:
                lines.append("sessions: 0")
                return "\n".join(lines)
            lines.append(f"sessions: {len(sessions)}")
            for session_id, title in sessions[:20]:
                title_display = title or "untitled"
                sid = session_id if session_id else "unknown"
                lines.append(f"- session_id: {sid}")
                lines.append(f"  title: {title_display}")
            if len(sessions) > 20:
                lines.append(f"  ... and {len(sessions) - 20} more")
            return "\n".join(lines)
        except Exception as e:
            return f"Session listing failed: {e}"

    async def _execute_voice(self, _scheduler, _args: list[str]) -> str:
        if _args and _args[0] == "status":
            config = self.config_service.load()
            effective_provider = resolve_voice_provider(
                config, (get_config().model.provider or "").strip().lower()
            )
            return (
                f"voice_enabled: {config.voice.enabled}\n"
                f"voice_provider: {config.voice.provider}\n"
                f"voice_model: {config.voice.model}\n"
                f"voice_base_url: {config.voice.base_url}\n"
                f"voice_api_version: {config.voice.api_version}\n"
                f"effective_provider: {effective_provider}"
            )

        if _args and _args[0] == "provider":
            if len(_args) < 2:
                return "Usage: /voice provider <openai|chatgpt|google|gemini|azure|clear>"
            provider_arg = _args[1].strip().lower()
            config = self.config_service.load()
            if provider_arg == "clear":
                config.voice.provider = None
                self.config_service.save(config)
                return "Voice provider cleared."
            if provider_arg not in SUPPORTED_VOICE_PROVIDERS:
                return f"Unsupported voice provider: {provider_arg}."
            config.voice.provider = provider_arg
            self.config_service.save(config)
            return f"Voice provider set to: {provider_arg}"

        config = self.config_service.load()
        provider = resolve_voice_provider(
            config, (get_config().model.provider or "").strip().lower()
        )
        if provider not in SUPPORTED_VOICE_PROVIDERS:
            return f"Voice mode is not available for provider: {provider or 'unknown'}."

        try:
            resolve_voice_credentials(provider)
        except VoiceDictationError:
            oauth_provider = map_provider_to_oauth(provider)
            if oauth_provider:
                return (
                    f"Voice mode requires credentials for provider '{provider}'. "
                    f"Run `koder auth login {oauth_provider}` or configure the matching API key."
                )
            env_var_name = get_provider_api_env_var(provider)
            return (
                f"Voice mode requires credentials for provider '{provider}'. "
                f"Set `{env_var_name}` or configure `model.api_key`."
            )

        if (
            config.voice.enabled
            and resolve_voice_provider(config, (get_config().model.provider or "").strip().lower())
            == provider
        ):
            config.voice.enabled = False
            self.config_service.save(config)
            return "Voice mode disabled."

        config.voice.enabled = True
        if config.voice.provider is None:
            config.voice.provider = provider
        self.config_service.save(config)
        return (
            "Voice mode enabled.\n"
            f"provider: {provider}\n"
            "status: provider-backed voice routing configured\n"
            "shortcut: double-space to record, Space or Enter to stop, auto-send on completion"
        )

    @staticmethod
    def _ctx_viz_bar(percentage: float, width: int = 20) -> str:
        filled = max(0, min(width, round(width * percentage / 100)))
        return "[" + "#" * filled + "." * (width - filled) + "]"

    async def _execute_ctx_viz(self, _scheduler, _args: list[str]) -> str:
        from koder_agent.harness.session_flow import load_context

        base = await load_context()
        sections = [base]

        if (
            _scheduler is not None
            and hasattr(_scheduler, "session")
            and hasattr(_scheduler.session, "get_items")
        ):
            try:
                items = await _scheduler.session.get_items()
            except Exception:
                items = []
            transcript, message_count = session_transcript_from_items(items)
            sections.append(f"Session messages: {message_count}")

            # Token usage breakdown by category, mirroring the injected
            # context (project context + transcript roles + free space).
            window_size = get_context_window_size(get_model_name())
            base_tokens = estimate_text_tokens(base)
            dict_items = [item for item in items if isinstance(item, dict)]
            user_tokens = estimate_messages_tokens(
                [item for item in dict_items if item.get("role") == "user"]
            )
            assistant_tokens = estimate_messages_tokens(
                [item for item in dict_items if item.get("role") == "assistant"]
            )
            tool_tokens = estimate_messages_tokens(
                [item for item in dict_items if item.get("role") == "tool"]
            )
            other_tokens = estimate_messages_tokens(
                [
                    item
                    for item in dict_items
                    if item.get("role") not in {"user", "assistant", "tool"}
                ]
            )
            total_tokens = base_tokens + user_tokens + assistant_tokens + tool_tokens + other_tokens
            free_tokens = max(0, window_size - total_tokens)
            used_percentage = (total_tokens / window_size * 100) if window_size else 0.0

            breakdown = [
                ("project context", base_tokens),
                ("user messages", user_tokens),
                ("assistant messages", assistant_tokens),
                ("tool results", tool_tokens),
            ]
            if other_tokens:
                breakdown.append(("other items", other_tokens))
            breakdown.append(("free space", free_tokens))

            usage_lines = [
                "Context usage (estimated):",
                f"total: {total_tokens:,} / {window_size:,} tokens ({used_percentage:.1f}%)",
            ]
            for label, tokens in breakdown:
                pct = (tokens / window_size * 100) if window_size else 0.0
                usage_lines.append(f"{label}: {tokens:,} ({pct:.1f}%) {self._ctx_viz_bar(pct)}")
            if used_percentage >= 80:
                usage_lines.append(
                    "warning: context is over 80% full; consider /compact to free space"
                )
            sections.append("\n".join(usage_lines))

            context_files = self._collect_context_file_paths(items)
            if context_files:
                sections.append(
                    "Files in session context:\n" + "\n".join(f"- {p}" for p in context_files)
                )

            if transcript:
                recent_lines = transcript.splitlines()[-4:]
                sections.append("Recent transcript:\n" + "\n".join(recent_lines))

        return "\n\n".join(section for section in sections if section).strip()

    async def _execute_security_review(self, _scheduler, _args: list[str]) -> str:
        return await run_security_review(cwd=Path.cwd())

    async def _execute_summary(self, scheduler, _args: list[str]) -> str:
        lines = ["Session Summary:"]

        # Session info
        if scheduler and hasattr(scheduler, "session"):
            session = scheduler.session
            if hasattr(session, "get_title"):
                try:
                    title = await session.get_title()
                    if title:
                        lines.append(f"  Title: {title}")
                except Exception:
                    pass

            # Usage stats
            if hasattr(scheduler, "usage_tracker"):
                usage = scheduler.usage_tracker.session_usage
                lines.append(f"  Requests: {usage.request_count}")
                lines.append(f"  Tokens: {usage.input_tokens:,} in / {usage.output_tokens:,} out")
                if hasattr(usage, "total_cost") and usage.total_cost > 0:
                    lines.append(f"  Cost: ${usage.total_cost:.4f}")

        # Git changes in this session
        cwd = os.getcwd()
        try:
            diff_stat_output = subprocess.run(
                ["git", "diff", "--stat"], capture_output=True, text=True, timeout=5, cwd=cwd
            ).stdout.strip()
            if diff_stat_output:
                lines.append(f"\nUncommitted changes:\n{diff_stat_output}")

            recent = subprocess.run(
                ["git", "log", "--oneline", "-3"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=cwd,
            ).stdout.strip()
            if recent:
                lines.append(f"\nRecent commits:\n{recent}")
        except Exception:
            pass

        return "\n".join(lines)

    async def _execute_mcp_prompt(self, prompt, scheduler, args: list[str]) -> str:
        """Execute an MCP prompt command via the MCP server session."""
        from koder_agent.mcp.prompts import execute_prompt

        # Collect live MCP servers from the scheduler
        mcp_servers = getattr(scheduler, "_mcp_servers", []) if scheduler else []
        if not mcp_servers:
            # Fallback: try the agent's mcp_servers attribute
            agent = getattr(scheduler, "dev_agent", None) if scheduler else None
            if agent is not None:
                mcp_servers = list(getattr(agent, "mcp_servers", []) or [])

        if not mcp_servers:
            return (
                f"MCP prompt '{prompt.command_name}' cannot be executed: "
                f"no MCP servers are connected."
            )

        try:
            result = await execute_prompt(prompt, mcp_servers, args)
        except RuntimeError as exc:
            return f"MCP prompt error: {exc}"

        # Format the result for display
        lines: list[str] = []
        if result.description:
            lines.append(result.description)
            lines.append("")
        for msg in result.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"[{role}] {content}")

        rendered = "\n".join(lines)

        # Feed the prompt content into the scheduler so it becomes part
        # of the conversation context that the LLM acts on.
        if scheduler is not None and result.messages:
            user_content = "\n\n".join(
                msg["content"] for msg in result.messages if msg.get("content")
            )
            if user_content:
                return await scheduler.handle(user_content, render_output=self.emit_console)

        return rendered

    def _make_fallback_handler(self, name: str) -> InteractiveCommand:
        async def _handler(_scheduler, _args: list[str]) -> str:
            spec = self.registry.get(name)
            help_text = spec.help_text if spec else f"/{name}"
            return f"{name}: {help_text}\nStatus: registered in harness runtime."

        return _handler

    def _register_static_command_messages(self, mapping: dict[str, str]) -> None:
        for name, message in mapping.items():
            if name in self.commands:
                continue

            async def _handler(_scheduler, _args: list[str], *, _message=message) -> str:
                return _message

            _handler.__name__ = f"_execute_{name.replace('-', '_')}"
            self.commands[name] = _handler

    def _available_skills(self) -> dict[str, Skill]:
        config = get_config()
        if not config.skills.enabled:
            return {}
        return discover_merged_skills(
            cwd=Path.cwd(),
            user_dir=config.skills.user_skills_dir,
            project_dir=config.skills.project_skills_dir,
            plugin_root=self.plugin_root,
            additional_dirs=self.additional_skill_dirs,
        )

    def _user_visible_skills(self) -> dict[str, Skill]:
        return {
            name: skill for name, skill in self._available_skills().items() if skill.user_invocable
        }

    def _prompt_commands(self):
        """Discover standalone ``.koder/commands/*.md`` prompt commands."""
        from koder_agent.harness.commands.prompt_commands import discover_prompt_commands

        return discover_prompt_commands(cwd=Path.cwd())

    async def _execute_prompt_command(self, prompt_command, scheduler, args: list[str]) -> str:
        """Render a prompt command body as a prompt and dispatch it."""
        prompt = prompt_command.render_prompt(args)
        if not prompt:
            return f"prompt command '{prompt_command.name}' has an empty body"
        if scheduler is not None:
            return await scheduler.handle(prompt, render_output=self.emit_console)
        return prompt

    async def _execute_dynamic_skill(
        self,
        skill: Skill,
        scheduler,
        args: list[str],
    ) -> str:
        arguments_text = " ".join(args).strip()
        permission = await self.permission_service.evaluate_tool_call_async(
            "Skill",
            {
                "skill": skill.name,
                "arguments": arguments_text,
            },
        )
        if permission.requires_approval:
            return f"skills: permission required\nskill: {skill.name}\nreason: {permission.reason}"
        if not permission.allowed:
            return f"skills: blocked\nskill: {skill.name}\nreason: {permission.reason}"

        session_id = (
            scheduler.session.session_id
            if scheduler is not None and hasattr(scheduler, "session")
            else None
        )
        prompt = skill.render_prompt(args, session_id=session_id)

        with skill_invocation_scope(skill):
            if is_trusted_bundled_skill(skill, "remember"):
                return await self._execute_remember_skill(arguments_text)
            if skill.execution_context == "fork":
                definitions = get_agent_definitions(
                    cwd=Path.cwd(),
                    plugin_root=self.plugin_root,
                    cli_agents_json=self.cli_agents_json,
                )
                selected = None
                if skill.agent:
                    selected = next(
                        (
                            agent
                            for agent in definitions.active_agents
                            if agent.agent_type == skill.agent
                        ),
                        None,
                    )
                if selected is None:
                    selected = next(
                        (
                            agent
                            for agent in definitions.active_agents
                            if agent.agent_type == "general-purpose"
                        ),
                        None,
                    )
                if selected is None:
                    return f"skills: no agent available for {skill.name}"
                seed_items = None
                if scheduler is not None and hasattr(scheduler.session, "get_items"):
                    seed_items = await scheduler.session.get_items()
                return await self.agent_service.run_sync(
                    agent_definition=selected,
                    prompt=prompt,
                    seed_items=seed_items,
                    cwd=Path.cwd(),
                )

            if skill.disable_model_invocation:
                return prompt

            if scheduler is not None:
                return await scheduler.handle(prompt, render_output=self.emit_console)
            return prompt
