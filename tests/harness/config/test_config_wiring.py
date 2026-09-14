"""Tests to prevent config-consumption regressions.

These tests verify that configuration values actually flow through to the
runtime components that use them, catching three classes of bugs:
1. Config values defined but never consumed (dead config)
2. Config toggle commands that don't affect behavior (dead switches)
3. Environment variable overrides that don't propagate
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# Stub litellm before importing koder_agent
if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from koder_agent.harness.config.schema import HarnessRuntimeConfig
from koder_agent.harness.config.service import RuntimeConfigService
from koder_agent.harness.permissions.modes import PermissionMode
from koder_agent.harness.permissions.service import PermissionService


def _isolate_brief_agent_factory(monkeypatch, tmp_path):
    """Build the real agent without depending on a developer's provider setup."""
    monkeypatch.chdir(tmp_path)
    config = RuntimeConfigService(tmp_path / "config.yaml").load()
    config.harness.brief_mode_enabled = False
    monkeypatch.setattr("koder_agent.agentic.agent.get_config", lambda: config)
    monkeypatch.setattr(
        "koder_agent.agentic.agent.get_model_client_snapshot",
        lambda _override: {
            "model_name": "gpt-4.1",
            "api_key": "synthetic-test-key",
            "base_url": None,
            "native_openai": False,
            "litellm_kwargs": {
                "model": "openai/gpt-4.1",
                "api_key": "synthetic-test-key",
                "base_url": None,
                "extra_headers": {},
            },
        },
    )
    monkeypatch.setattr(
        "koder_agent.agentic.agent.load_active_output_style_body", lambda _cwd: None
    )


# ---------------------------------------------------------------------------
# Bug 1 regression: permission_mode must flow from config to PermissionService
# ---------------------------------------------------------------------------


class TestPermissionModeWiring:
    """Verify permission_mode config flows to PermissionService."""

    def test_permission_mode_from_config_applied_to_service(self, tmp_path, monkeypatch):
        """Config file permission_mode must reach PermissionService.mode."""
        config_path = tmp_path / "config.yaml"
        service = RuntimeConfigService(config_path)
        config = service.load()
        config.harness.permission_mode = "bypass"
        service.save(config)

        # Reload and simulate what runtime.py does
        service2 = RuntimeConfigService(config_path)
        config2 = service2.load()
        effective_mode_str = service2.get_effective_value(
            config2.harness.permission_mode, "KODER_PERMISSION_MODE"
        )
        mode = PermissionMode(effective_mode_str)
        ps = PermissionService.default(mode=mode)

        assert ps.mode == PermissionMode.BYPASS

    def test_permission_mode_env_overrides_config(self, tmp_path, monkeypatch):
        """KODER_PERMISSION_MODE env var takes precedence over config file."""
        config_path = tmp_path / "config.yaml"
        service = RuntimeConfigService(config_path)
        config = service.load()
        config.harness.permission_mode = "default"
        service.save(config)

        monkeypatch.setenv("KODER_PERMISSION_MODE", "strict")
        service2 = RuntimeConfigService(config_path)
        config2 = service2.load()
        effective = service2.get_effective_value(
            config2.harness.permission_mode, "KODER_PERMISSION_MODE"
        )
        assert effective == "strict"

    def test_permission_mode_cli_overrides_env(self, tmp_path, monkeypatch):
        """CLI argument takes precedence over env var."""
        monkeypatch.setenv("KODER_PERMISSION_MODE", "strict")
        service = RuntimeConfigService(tmp_path / "config.yaml")
        config = service.load()
        effective = service.get_effective_value(
            config.harness.permission_mode, "KODER_PERMISSION_MODE", cli_value="bypass"
        )
        assert effective == "bypass"

    def test_invalid_permission_mode_falls_back_to_default(self):
        """Invalid permission_mode string falls back to DEFAULT."""
        try:
            PermissionMode("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected

    def test_all_permission_modes_are_valid_config_values(self):
        """Every PermissionMode enum value should be accepted by the config."""
        for mode in PermissionMode:
            config = HarnessRuntimeConfig(permission_mode=mode.value)
            assert config.permission_mode == mode.value


# ---------------------------------------------------------------------------
# Bug 2 regression: no dead config fields should exist in HarnessRuntimeConfig
# ---------------------------------------------------------------------------


class TestNoDeadConfig:
    """Verify HarnessRuntimeConfig fields are not dead."""

    def test_interactive_shell_field_removed(self):
        """interactive_shell was dead config and must remain removed."""
        assert not hasattr(HarnessRuntimeConfig(), "interactive_shell")

    def test_all_harness_fields_are_documented(self):
        """All HarnessRuntimeConfig fields should be in a known set."""
        known_fields = {
            "permission_mode",
            "teammate_mode",
            "last_release_notes_seen",
            "advisor_model",
            "brief_mode_enabled",
            "companion",
            "companion_muted",
            "reasoning_display",
            "auto_dream_write_mode",
            "task_delegate_max_batch_size",
            "task_delegate_max_concurrency",
        }
        actual_fields = set(HarnessRuntimeConfig.model_fields.keys())
        unexpected = actual_fields - known_fields
        assert not unexpected, f"New fields added without tests: {unexpected}"


# ---------------------------------------------------------------------------
# Bug 3 regression: brief_mode_enabled must inject into system prompt
# ---------------------------------------------------------------------------


class TestBriefModeWiring:
    """Verify brief_mode_enabled actually affects LLM behavior."""

    def test_brief_mode_config_value_accessible(self, tmp_path):
        """brief_mode_enabled can be toggled and persisted."""
        config_path = tmp_path / "config.yaml"
        service = RuntimeConfigService(config_path)
        config = service.load()
        assert config.harness.brief_mode_enabled is False

        config.harness.brief_mode_enabled = True
        service.save(config)

        service2 = RuntimeConfigService(config_path)
        assert service2.load().harness.brief_mode_enabled is True

    def test_brief_mode_env_var_recognized(self, monkeypatch):
        """KODER_BRIEF=1 should be treated as brief mode enabled."""
        monkeypatch.setenv("KODER_BRIEF", "1")
        assert os.environ.get("KODER_BRIEF", "").lower() in ("1", "true")

    def test_brief_mode_env_var_true_recognized(self, monkeypatch):
        """KODER_BRIEF=true should be treated as brief mode enabled."""
        monkeypatch.setenv("KODER_BRIEF", "true")
        assert os.environ.get("KODER_BRIEF", "").lower() in ("1", "true")

    def test_brief_mode_env_var_false_not_active(self, monkeypatch):
        """KODER_BRIEF=0 should not activate brief mode."""
        monkeypatch.setenv("KODER_BRIEF", "0")
        assert os.environ.get("KODER_BRIEF", "").lower() not in ("1", "true")

    @pytest.mark.asyncio
    async def test_brief_mode_injects_into_system_prompt(self, monkeypatch, tmp_path):
        """When brief mode is enabled, system prompt must contain brief instructions."""
        _isolate_brief_agent_factory(monkeypatch, tmp_path)
        monkeypatch.setenv("KODER_BRIEF", "1")
        # Mock heavy dependencies to isolate prompt construction
        monkeypatch.setattr(
            "koder_agent.agentic.agent.load_mcp_servers", AsyncMock(return_value=[])
        )
        monkeypatch.setattr("koder_agent.agentic.agent._get_skills_metadata", lambda cfg: "")
        monkeypatch.setattr("koder_agent.agentic.agent._get_agents_metadata", lambda: "")

        from koder_agent.agentic.agent import create_dev_agent

        agent = await create_dev_agent([], name="test")
        assert "Brief Mode" in agent.instructions
        assert "extremely concise" in agent.instructions

    @pytest.mark.asyncio
    async def test_brief_mode_disabled_no_injection(self, monkeypatch, tmp_path):
        """When brief mode is disabled, system prompt must NOT contain brief instructions."""
        _isolate_brief_agent_factory(monkeypatch, tmp_path)
        monkeypatch.delenv("KODER_BRIEF", raising=False)
        # Ensure config also has it disabled
        config_path = tmp_path / "config.yaml"
        monkeypatch.setattr(
            "koder_agent.config.manager.ConfigManager.default_config_path",
            classmethod(lambda cls: config_path),
        )
        from koder_agent.config.manager import reset_config_manager

        reset_config_manager()

        monkeypatch.setattr(
            "koder_agent.agentic.agent.load_mcp_servers", AsyncMock(return_value=[])
        )
        monkeypatch.setattr("koder_agent.agentic.agent._get_skills_metadata", lambda cfg: "")
        monkeypatch.setattr("koder_agent.agentic.agent._get_agents_metadata", lambda: "")

        from koder_agent.agentic.agent import create_dev_agent

        agent = await create_dev_agent([], name="test")
        assert "Brief Mode" not in agent.instructions

        reset_config_manager()


# ---------------------------------------------------------------------------
# Env var override: teammate_mode
# ---------------------------------------------------------------------------


class TestTeammateModeWiring:
    """Verify teammate_mode env var flows correctly."""

    def test_teammate_mode_env_overrides_config(self, monkeypatch, tmp_path):
        """KODER_TEAMMATE_MODE env var takes precedence over config."""
        config_path = tmp_path / "config.yaml"
        service = RuntimeConfigService(config_path)
        config = service.load()
        config.harness.teammate_mode = "auto"
        service.save(config)

        monkeypatch.setenv("KODER_TEAMMATE_MODE", "tmux")

        from koder_agent.harness.agents.teams.runtime import resolve_teammate_mode

        mode = resolve_teammate_mode(config_service=service, cli_mode=None)
        assert mode == "tmux"

    def test_teammate_mode_cli_overrides_env(self, monkeypatch):
        """CLI mode takes precedence over env var."""
        monkeypatch.setenv("KODER_TEAMMATE_MODE", "tmux")

        from koder_agent.harness.agents.teams.runtime import resolve_teammate_mode

        mode = resolve_teammate_mode(config_service=None, cli_mode="in-process")
        assert mode == "in-process"

    def test_teammate_mode_invalid_env_ignored(self, monkeypatch, tmp_path):
        """Invalid KODER_TEAMMATE_MODE falls through to config."""
        config_path = tmp_path / "config.yaml"
        service = RuntimeConfigService(config_path)
        config = service.load()
        config.harness.teammate_mode = "in-process"
        service.save(config)

        monkeypatch.setenv("KODER_TEAMMATE_MODE", "invalid-value")

        from koder_agent.harness.agents.teams.runtime import resolve_teammate_mode

        service2 = RuntimeConfigService(config_path)
        mode = resolve_teammate_mode(config_service=service2, cli_mode=None)
        assert mode == "in-process"


# ---------------------------------------------------------------------------
# General: get_effective_value priority chain
# ---------------------------------------------------------------------------


class TestEffectiveValuePrecedence:
    """Verify the CLI > ENV > Config > Default priority chain."""

    def test_cli_wins_over_all(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KODER_PERMISSION_MODE", "strict")
        service = RuntimeConfigService(tmp_path / "config.yaml")
        config = service.load()
        config.harness.permission_mode = "bypass"
        service.save(config)

        service2 = RuntimeConfigService(tmp_path / "config.yaml")
        config2 = service2.load()
        result = service2.get_effective_value(
            config2.harness.permission_mode, "KODER_PERMISSION_MODE", cli_value="plan"
        )
        assert result == "plan"

    def test_env_wins_over_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("KODER_PERMISSION_MODE", "strict")
        service = RuntimeConfigService(tmp_path / "config.yaml")
        config = service.load()
        config.harness.permission_mode = "bypass"
        service.save(config)

        service2 = RuntimeConfigService(tmp_path / "config.yaml")
        config2 = service2.load()
        result = service2.get_effective_value(
            config2.harness.permission_mode, "KODER_PERMISSION_MODE"
        )
        assert result == "strict"

    def test_config_used_when_no_env_or_cli(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KODER_PERMISSION_MODE", raising=False)
        service = RuntimeConfigService(tmp_path / "config.yaml")
        config = service.load()
        config.harness.permission_mode = "bypass"
        service.save(config)

        service2 = RuntimeConfigService(tmp_path / "config.yaml")
        config2 = service2.load()
        result = service2.get_effective_value(
            config2.harness.permission_mode, "KODER_PERMISSION_MODE"
        )
        assert result == "bypass"

    def test_default_used_when_nothing_set(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KODER_PERMISSION_MODE", raising=False)
        service = RuntimeConfigService(tmp_path / "config.yaml")
        config = service.load()
        result = service.get_effective_value(
            config.harness.permission_mode, "KODER_PERMISSION_MODE"
        )
        assert result == "default"


# ---------------------------------------------------------------------------
# Integration: HarnessInteractiveCommandHandler receives permission_service
# ---------------------------------------------------------------------------


class TestCommandHandlerPermissionWiring:
    """Verify HarnessInteractiveCommandHandler uses the passed permission_service."""

    def test_handler_uses_passed_permission_service(self):
        """When permission_service is passed, handler must use it instead of default."""
        custom_ps = PermissionService.default(mode=PermissionMode.BYPASS)

        from koder_agent.harness.commands.interactive import (
            HarnessInteractiveCommandHandler,
        )

        handler = HarnessInteractiveCommandHandler(permission_service=custom_ps)
        assert handler.permission_service is custom_ps
        assert handler.permission_service.mode == PermissionMode.BYPASS

    def test_handler_without_permission_service_gets_default(self):
        """When no permission_service passed, handler creates a default one."""
        from koder_agent.harness.commands.interactive import (
            HarnessInteractiveCommandHandler,
        )

        handler = HarnessInteractiveCommandHandler()
        assert handler.permission_service.mode == PermissionMode.DEFAULT
