"""Sandbox honesty & fail-closed enforcement regression tests.

Covers the confirmed findings for the "Sandbox honesty & enforcement" group:
 1. unix-local reports unavailable on Linux (no kernel confinement).
 2. host env (API keys) is scrubbed before entering the sandbox.
 3. glob deny_write patterns (.env.*) are matched, not dropped.
 4. network policy non-enforcement is made explicit, not silently claimed.
 5. interpreter payloads / heredocs do not silently pass the preflight.
 6. invalid sandbox mode fails closed to workspace-write, not danger-full-access.
"""

from __future__ import annotations

import warnings

import pytest

from koder_agent.harness.sandbox.backend import SandboxExecutionContext
from koder_agent.harness.sandbox.policy import SandboxPolicy
from koder_agent.harness.sandbox.registry import _check_unix_local, get_backend_status
from koder_agent.harness.sandbox.sdk_backend import _scrub_env, execute_with_sdk_backend
from koder_agent.harness.sandbox.workspace import (
    interpreter_payload_violation,
    protected_write_violation,
)
from koder_agent.harness.session_env import (
    build_sandbox_env,
    build_subprocess_env,
    is_probably_secret_env_name,
)

# --- Finding #1: Linux unix-local reports no confinement -------------------


def test_unix_local_unavailable_on_linux(monkeypatch):
    monkeypatch.setattr("koder_agent.harness.sandbox.registry.sys.platform", "linux")
    ok, reason = _check_unix_local()
    assert ok is False
    assert reason is not None
    assert "no kernel confinement" in reason


def test_unix_local_status_not_available_on_linux(monkeypatch):
    monkeypatch.setattr("koder_agent.harness.sandbox.registry.sys.platform", "linux")
    status = get_backend_status("unix-local")
    assert status.available is False
    assert any("no kernel confinement" in item for item in status.unavailable_reasons)


def test_unix_local_available_on_macos_with_sandbox_exec(monkeypatch):
    monkeypatch.setattr("koder_agent.harness.sandbox.registry.sys.platform", "darwin")
    monkeypatch.setattr(
        "koder_agent.harness.sandbox.registry.shutil.which",
        lambda name: "/usr/bin/sandbox-exec",
    )
    ok, reason = _check_unix_local()
    assert ok is True
    assert reason is None


def test_unix_local_unavailable_on_macos_without_sandbox_exec(monkeypatch):
    monkeypatch.setattr("koder_agent.harness.sandbox.registry.sys.platform", "darwin")
    monkeypatch.setattr("koder_agent.harness.sandbox.registry.shutil.which", lambda name: None)
    ok, reason = _check_unix_local()
    assert ok is False
    assert reason == "missing /usr/bin/sandbox-exec"


# --- Finding #2: host env not leaked into the sandbox ----------------------


def test_secret_env_name_detection():
    assert is_probably_secret_env_name("OPENAI_API_KEY")
    assert is_probably_secret_env_name("ANTHROPIC_API_KEY")
    assert is_probably_secret_env_name("GITHUB_TOKEN")
    assert is_probably_secret_env_name("AWS_SECRET_ACCESS_KEY")
    assert is_probably_secret_env_name("MY_SERVICE_SECRET")
    assert is_probably_secret_env_name("DB_PASSWORD")
    # Benign names are not flagged.
    assert not is_probably_secret_env_name("PATH")
    assert not is_probably_secret_env_name("HOME")
    assert not is_probably_secret_env_name("LANG")


def test_scrub_env_strips_secrets():
    raw = {
        "PATH": "/usr/bin",
        "HOME": "/home/u",
        "OPENAI_API_KEY": "sk-secret",
        "ANTHROPIC_API_KEY": "sk-ant",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "GITHUB_TOKEN": "ghp_x",
    }
    scrubbed = _scrub_env(raw)
    assert scrubbed["PATH"] == "/usr/bin"
    assert scrubbed["HOME"] == "/home/u"
    assert "OPENAI_API_KEY" not in scrubbed
    assert "ANTHROPIC_API_KEY" not in scrubbed
    assert "AWS_SECRET_ACCESS_KEY" not in scrubbed
    assert "GITHUB_TOKEN" not in scrubbed


def test_build_sandbox_env_allowlist_only(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("HOME", "/home/u")
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("SOME_RANDOM_HOST_VAR", "leaked?")
    env = build_sandbox_env()
    assert env.get("PATH") == "/usr/bin"
    assert env.get("HOME") == "/home/u"
    assert env.get("LC_ALL") == "en_US.UTF-8"
    # Secrets and unknown host vars must NOT be forwarded.
    assert "OPENAI_API_KEY" not in env
    assert "SOME_RANDOM_HOST_VAR" not in env


def test_build_subprocess_env_still_carries_full_host_env(monkeypatch):
    # The non-sandboxed path must keep the full env (behavior preservation).
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    env = build_subprocess_env()
    assert env.get("OPENAI_API_KEY") == "sk-secret"


# --- Finding #3: glob deny_write patterns matched, not dropped --------------


def test_deny_write_globs_are_retained():
    policy = SandboxPolicy.from_config({"enabled": True})
    globs = policy.deny_write_globs()
    assert ".env.*" in globs


def test_glob_deny_write_flags_env_variants(tmp_path):
    policy = SandboxPolicy(mode="workspace-write")

    assert protected_write_violation("touch .env.local", policy=policy, repo_root=tmp_path)
    assert protected_write_violation("echo x > .env.production", policy=policy, repo_root=tmp_path)
    # An unrelated file must still be allowed.
    assert protected_write_violation("touch src/app.py", policy=policy, repo_root=tmp_path) is None
    assert (
        protected_write_violation("touch environment.md", policy=policy, repo_root=tmp_path) is None
    )


def test_glob_matcher_matches_nested_paths():
    policy = SandboxPolicy(mode="workspace-write")
    assert policy.matches_deny_write_glob("config/.env.production") == ".env.*"
    assert policy.matches_deny_write_glob(".env.local") == ".env.*"
    assert policy.matches_deny_write_glob("README.md") is None


# --- Finding #4: network non-enforcement is explicit ------------------------


def test_network_not_enforced_on_unix_local():
    policy = SandboxPolicy(mode="workspace-write", backend="unix-local", network_access=False)
    assert policy.network_enforced is False
    assert policy.network_restricted_but_unenforced is True


def test_network_enforced_flag_for_enforcing_backend():
    # Docker does NOT enforce network isolation by default (no --network=none),
    # so it is not in NETWORK_ENFORCING_BACKENDS.  Use a cloud backend instead.
    policy = SandboxPolicy(mode="workspace-write", backend="e2b", network_access=False)
    assert policy.network_enforced is True
    # Restriction requested AND backend can enforce -> not the dishonest case.
    assert policy.network_restricted_but_unenforced is False


def test_docker_network_not_enforced():
    """Docker backend does NOT enforce network isolation by default."""
    policy = SandboxPolicy(mode="workspace-write", backend="docker", network_access=False)
    assert policy.network_enforced is False
    assert policy.network_restricted_but_unenforced is True


def test_network_restriction_flag_with_domain_lists():
    policy = SandboxPolicy(
        mode="workspace-write",
        backend="unix-local",
        network_access=True,
        denied_domains=("metadata.google.internal",),
    )
    # network_access True but denied_domains set -> still a restriction the
    # unix-local backend cannot honor.
    assert policy.network_restricted_but_unenforced is True


@pytest.mark.asyncio
async def test_sandbox_execution_surfaces_network_non_enforcement(tmp_path, monkeypatch):
    status = get_backend_status("unix-local")
    if not status.available:
        pytest.skip(status.reason)
    policy = SandboxPolicy(mode="workspace-write", backend="unix-local", network_access=False)
    context = SandboxExecutionContext(
        cwd=tmp_path,
        repo_root=tmp_path,
        command="true",
        env={},
        timeout=10,
        background=False,
        session_id=None,
        policy=policy,
    )
    result = await execute_with_sdk_backend(context)
    assert result.sandboxed is False
    assert result.created is False
    assert result.executed is False
    assert "network access disabled" in (result.reason or "")
    assert "networkAccess=false is not enforced" in (result.reason or "")


# --- Finding #5: interpreter payloads don't silently pass -------------------


def test_interpreter_inline_payload_flagged():
    assert interpreter_payload_violation("python3 -c \"open('.git/config')\"") is not None
    assert interpreter_payload_violation("python -c 'print(1)'") is not None
    assert interpreter_payload_violation("bash -c 'rm .env'") is not None
    assert interpreter_payload_violation("perl -e 'unlink q(.git/config)'") is not None
    assert interpreter_payload_violation("node -e 'require(\"fs\")'") is not None


def test_heredoc_flagged():
    assert interpreter_payload_violation("cat <<EOF\nhi\nEOF") is not None
    assert interpreter_payload_violation("cat > f <<'END'\nx\nEND") is not None


def test_plain_commands_not_flagged_as_interpreter():
    assert interpreter_payload_violation("ls -la") is None
    assert interpreter_payload_violation("python3 script.py") is None
    assert interpreter_payload_violation("git status") is None
    assert interpreter_payload_violation("touch file.txt") is None


def test_interpreter_payload_blocks_preflight_in_workspace_write(tmp_path):
    policy = SandboxPolicy(mode="workspace-write")
    violation = protected_write_violation(
        "python3 -c \"open('.git/config','w').write('x')\"",
        policy=policy,
        repo_root=tmp_path,
    )
    assert violation is not None
    assert "payload" in violation


def test_interpreter_payload_ignored_in_danger_full_access(tmp_path):
    # When the sandbox is not confining (danger-full-access), the preflight is
    # not the enforcement layer; do not spuriously flag interpreter payloads.
    policy = SandboxPolicy(mode="danger-full-access")
    assert (
        protected_write_violation(
            "python3 -c \"open('.git/config')\"", policy=policy, repo_root=tmp_path
        )
        is None
    )


# --- Finding #6: invalid mode fails closed to workspace-write ---------------


def test_invalid_mode_when_enabled_falls_back_to_workspace_write():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        policy = SandboxPolicy.from_config({"enabled": True, "mode": "danger-fulll-acess"})
    assert policy.mode == "workspace-write"
    assert policy.enabled is True
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_invalid_mode_never_silently_becomes_danger_full_access():
    with pytest.warns(RuntimeWarning, match="invalid sandbox mode"):
        policy = SandboxPolicy.from_config({"enabled": True, "mode": "totally-bogus"})
    assert policy.mode != "danger-full-access"
    assert policy.mode == "workspace-write"


def test_valid_modes_unchanged():
    assert SandboxPolicy.from_config({"enabled": True, "mode": "read-only"}).mode == "read-only"
    assert (
        SandboxPolicy.from_config({"enabled": True, "mode": "workspace-write"}).mode
        == "workspace-write"
    )
    # Disabled with no mode -> the safe legacy default.
    assert SandboxPolicy.from_config({"enabled": False}).mode == "danger-full-access"


# --- Acceptance gap: the REAL sandboxed exec path must use the allowlist -----


def test_sandboxed_exec_path_uses_env_allowlist(monkeypatch, tmp_path):
    """execute_shell_command's sandbox branch must build env via the fail-closed
    allowlist (build_sandbox_env), so oddly-named secrets never reach the sandbox.

    Regression for the acceptance finding that the sandbox path fed the full host
    env through build_subprocess_env and only a pattern-based scrub applied.
    """
    import asyncio

    import koder_agent.harness.tools.shell_executor as se
    from koder_agent.harness.sandbox.backend import SandboxExecutionResult

    # Secrets whose names do NOT match the pattern denylist.
    monkeypatch.setenv("MYCUSTOMCREDS", "topsecret")
    monkeypatch.setenv("CI_JOB_JWT", "jwt-123")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-leak")

    captured: dict[str, dict] = {}

    async def _fake_backend(ctx):
        captured["env"] = dict(ctx.env)
        return SandboxExecutionResult(
            status="success", exit_code=0, stdout="", stderr="", sandboxed=True
        )

    # Force the sandboxed branch regardless of host platform: fabricate an
    # enabled state carrying a policy, and stub the excluded-command check.
    policy = SandboxPolicy.from_config({"enabled": True, "mode": "workspace-write"})
    real_state = se.resolve_sandbox_settings(str(tmp_path))
    forced = real_state.__class__(**{**real_state.__dict__, "enabled": True, "policy": policy})
    monkeypatch.setattr(se, "resolve_sandbox_settings", lambda *_a, **_k: forced)
    monkeypatch.setattr(se, "is_excluded_command", lambda *_a, **_k: False)
    monkeypatch.setattr(se, "execute_with_sdk_backend", _fake_backend)

    asyncio.run(
        se.execute_shell_command(
            "echo hi",
            timeout=5,
            session_id=None,
            sandbox_unavailable_approval=lambda _reason: True,
        )
    )

    env = captured.get("env")
    assert env is not None, "sandboxed backend was not reached"
    # Allowlist drops all secret-shaped and unknown host vars.
    assert "MYCUSTOMCREDS" not in env
    assert "CI_JOB_JWT" not in env
    assert "ANTHROPIC_API_KEY" not in env
    # But keeps benign essentials.
    assert "PATH" in env


# --- Finding H3: symlink TOCTOU protection -----------------------------------


class TestSymlinkToctouProtection:
    """Symlinks to protected paths must be detected regardless of link name."""

    def test_symlink_to_protected_git_dir_detected(self, tmp_path):
        """Symlink pointing to .git/ protected path must be caught."""
        protected = tmp_path / ".git" / "hooks" / "pre-commit"
        protected.parent.mkdir(parents=True, exist_ok=True)
        protected.touch()

        link = tmp_path / "innocent_link"
        link.symlink_to(protected)

        policy = SandboxPolicy(mode="workspace-write")
        cmd = f"echo malicious > {link}"
        result = protected_write_violation(cmd, policy=policy, repo_root=tmp_path)
        assert result is not None
        assert "protected path" in result

    def test_symlink_to_dot_koder_detected(self, tmp_path):
        """Symlink to .koder/ directory must be caught."""
        protected = tmp_path / ".koder" / "settings.json"
        protected.parent.mkdir(parents=True, exist_ok=True)
        protected.touch()

        link = tmp_path / "safe_name.txt"
        link.symlink_to(protected)

        policy = SandboxPolicy(mode="workspace-write")
        cmd = f"cp /tmp/evil {link}"
        result = protected_write_violation(cmd, policy=policy, repo_root=tmp_path)
        assert result is not None
        assert "protected path" in result

    def test_symlink_to_env_file_detected_via_glob(self, tmp_path):
        """Symlink whose real target matches a deny_write glob must be caught."""
        # Create a .env.production file and symlink to it with an innocent name
        env_file = tmp_path / ".env.production"
        env_file.touch()

        link = tmp_path / "config.txt"
        link.symlink_to(env_file)

        policy = SandboxPolicy(mode="workspace-write")
        cmd = f"echo SECRET=x > {link}"
        result = protected_write_violation(cmd, policy=policy, repo_root=tmp_path)
        assert result is not None
        assert "deny_write" in result or "protected path" in result

    def test_non_symlink_safe_path_still_allowed(self, tmp_path):
        """Regular files in non-protected locations must not be falsely flagged."""
        safe = tmp_path / "src" / "main.py"
        safe.parent.mkdir(parents=True, exist_ok=True)
        safe.touch()

        policy = SandboxPolicy(mode="workspace-write")
        cmd = f"echo code > {safe}"
        result = protected_write_violation(cmd, policy=policy, repo_root=tmp_path)
        assert result is None


# --- Finding H4: compound interpreter preflight bypass ------------------------


class TestCompoundInterpreterPreflight:
    """Interpreter payloads via various shells and wrappers must be caught."""

    def test_bash_c_with_rm_detected(self):
        result = interpreter_payload_violation("bash -c 'rm -rf /protected'")
        assert result is not None
        assert "bash" in result

    def test_sh_c_detected(self):
        result = interpreter_payload_violation("sh -c 'cat /etc/shadow'")
        assert result is not None
        assert "sh" in result

    def test_perl_e_with_unlink_detected(self):
        result = interpreter_payload_violation("perl -e 'unlink(\"/protected\")'")
        assert result is not None
        assert "perl" in result

    def test_python_c_with_os_remove_detected(self):
        result = interpreter_payload_violation("python3 -c 'import os; os.remove(\"/etc/passwd\")'")
        assert result is not None
        assert "python3" in result

    def test_ruby_e_detected(self):
        result = interpreter_payload_violation("ruby -e 'File.delete(\"/etc/passwd\")'")
        assert result is not None
        assert "ruby" in result

    def test_awk_detected(self):
        """awk with inline code must be flagged (H4 extension)."""
        result = interpreter_payload_violation("awk -e 'BEGIN{system(\"rm /etc/passwd\")}'")
        assert result is not None
        assert "awk" in result

    def test_gawk_detected(self):
        result = interpreter_payload_violation("gawk -e 'BEGIN{system(\"rm -rf /\")}'")
        assert result is not None
        assert "gawk" in result

    def test_mawk_detected(self):
        result = interpreter_payload_violation("mawk -e 'BEGIN{}'")
        assert result is not None
        assert "mawk" in result

    def test_env_bash_c_detected(self):
        """env prefix before interpreter must not hide the payload (H4 extension)."""
        result = interpreter_payload_violation("env bash -c 'rm -rf /'")
        assert result is not None
        assert "bash" in result

    def test_sudo_python_c_detected(self):
        """sudo prefix before interpreter must not hide the payload."""
        result = interpreter_payload_violation("sudo python3 -c 'import os; os.unlink(\"/x\")'")
        assert result is not None
        assert "python3" in result

    def test_node_eval_detected(self):
        result = interpreter_payload_violation('node --eval \'require("fs").unlinkSync("/x")\'')
        assert result is not None
        assert "node" in result

    def test_plain_script_not_flagged(self):
        """Running a script file (no inline flag) must not be flagged."""
        assert interpreter_payload_violation("python3 deploy.py") is None
        assert interpreter_payload_violation("bash ./run.sh") is None
        assert interpreter_payload_violation("perl script.pl") is None

    def test_non_interpreter_not_flagged(self):
        assert interpreter_payload_violation("git status") is None
        assert interpreter_payload_violation("ls -la /tmp") is None
        assert interpreter_payload_violation("cat README.md") is None
