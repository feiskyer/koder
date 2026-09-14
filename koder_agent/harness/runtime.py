"""Minimal harness runtime shell."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from koder_agent.harness.config.service import RuntimeConfigService
from koder_agent.harness.paths import harness_home_dir
from koder_agent.harness.permissions.ai_classifier import AiShellClassifier
from koder_agent.harness.permissions.modes import PermissionMode
from koder_agent.harness.permissions.persistence import PermissionStore
from koder_agent.harness.permissions.rule_sources import RuleHierarchy
from koder_agent.harness.permissions.service import PermissionService
from koder_agent.harness.session_flow import run_harness_session_flow
from koder_agent.harness.version_info import render_cli_version_banner

_logger = logging.getLogger(__name__)


def _load_permission_hierarchy() -> RuleHierarchy:
    """Load permission rules from project, local, and user settings files."""
    hierarchy = RuleHierarchy()

    # Load project settings (.koder/settings.json)
    project_settings_path = Path.cwd() / ".koder" / "settings.json"
    if project_settings_path.exists():
        try:
            settings = json.loads(project_settings_path.read_text(encoding="utf-8"))
            hierarchy.load_from_settings(settings, source="project")
        except (json.JSONDecodeError, OSError):
            pass  # Ignore malformed or unreadable files

    # Load local project settings (.koder/settings.local.json, gitignored)
    local_settings_path = Path.cwd() / ".koder" / "settings.local.json"
    if local_settings_path.exists():
        try:
            settings = json.loads(local_settings_path.read_text(encoding="utf-8"))
            hierarchy.load_from_settings(settings, source="local")
        except (json.JSONDecodeError, OSError):
            pass  # Ignore malformed or unreadable files

    # Load user settings (~/.koder/settings.json)
    user_settings_path = harness_home_dir() / "settings.json"
    if user_settings_path.exists():
        try:
            settings = json.loads(user_settings_path.read_text(encoding="utf-8"))
            hierarchy.load_from_settings(settings, source="user")
        except (json.JSONDecodeError, OSError):
            pass  # Ignore malformed or unreadable files

    return hierarchy


@dataclass
class HarnessRuntime:
    request: object

    async def run(self) -> int:
        from koder_agent.harness.channels.state import channel_state_scope
        from koder_agent.mcp.notifications import notification_handler_scope

        with channel_state_scope(), notification_handler_scope():
            try:
                return await self._run()
            finally:
                from koder_agent.mcp import drain_orphaned_mcp_owners
                from koder_agent.mcp.reconnection import drain_orphaned_retirements

                for label, drain in (
                    ("owner", drain_orphaned_mcp_owners),
                    ("transport", drain_orphaned_retirements),
                ):
                    try:
                        await drain()
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        _logger.debug(
                            "Best-effort MCP %s orphan cleanup was incomplete; retaining it for retry",
                            label,
                            exc_info=True,
                        )

    async def _run(self) -> int:
        mode = getattr(self.request, "mode", "")
        argv = list(getattr(self.request, "argv", []))
        first_arg = getattr(self.request, "first_arg", None)
        # Maintenance must work even when the config it diagnoses is invalid.
        if mode == "help":
            from koder_agent.cli import _append_subcommand_help, _build_cli_parser

            help_text = getattr(self.request, "help_text", None)
            sys.stdout.write(
                help_text or _append_subcommand_help(_build_cli_parser(None).format_help())
            )
            return 0
        if mode == "version":
            sys.stdout.write(render_cli_version_banner() + "\n")
            return 0
        if first_arg == "config":
            return await run_harness_session_flow(first_arg=first_arg, argv=argv)

        # Create permission hierarchy and AI classifier
        rule_hierarchy = _load_permission_hierarchy()
        ai_classifier = AiShellClassifier()

        # Resolve effective permission mode: CLI > ENV > Config > Default
        config_service = RuntimeConfigService()
        config = config_service.load()
        cli_permission_mode = getattr(self.request, "permission_mode", None)
        effective_mode_str = config_service.get_effective_value(
            config.harness.permission_mode,
            "KODER_PERMISSION_MODE",
            cli_permission_mode,
        )
        try:
            effective_mode = PermissionMode(effective_mode_str)
        except ValueError:
            effective_mode = PermissionMode.DEFAULT

        # A disk-backed store makes "always allow" decisions survive across
        # sessions: PermissionService.add_rule/add_approval_rule only persist when
        # a store is present, so without this an approved-always rule was kept
        # in-memory for one process and forgotten on the next run.
        permission_store = PermissionStore(harness_home_dir() / "permissions.json")

        permission_service = PermissionService.default(
            mode=effective_mode,
            store=permission_store,
            rule_hierarchy=rule_hierarchy,
            ai_classifier=ai_classifier,
        )
        if mode == "interactive":
            return await run_harness_session_flow(
                first_arg=None,
                argv=argv,
                permission_service=permission_service,
            )

        if mode in {"prompt", "subcommand", "auth_passthrough"}:
            return await run_harness_session_flow(
                first_arg=first_arg,
                argv=argv,
                permission_service=permission_service,
            )
        return 0
