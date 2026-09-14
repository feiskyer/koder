"""Git query names do not make their mutating modes safe to auto-run."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from koder_agent.harness.permissions import service as service_module
from koder_agent.harness.permissions.modes import PermissionMode
from koder_agent.harness.permissions.service import PermissionService
from koder_agent.harness.permissions.shell_classifier import classify_shell_command
from koder_agent.harness.sandbox.policy import SandboxPolicy
from koder_agent.harness.sandbox.workspace import read_only_violation

MUTATING_COMMANDS = (
    "git reflog expire --expire=all --all",
    "git reflog delete HEAD@{0}",
    "git reflog drop refs/heads/main",
    "git reflog write refs/heads/main old new fixture",
    "git fsck --lost-found",
    "git fsck --lost",
    "git fsck --lost-f",
    "git branch -Dtemporary",
    "git branch -vDtemporary",
    "git branch --set-upstream-to=origin/main",
    "git branch --unset-upstream",
    "git branch --edit-description",
    "git branch --edit-desc",
    "git tag -l -d temporary",
)

READ_COMMANDS = (
    "git reflog",
    "git reflog HEAD",
    "git reflog show --all",
    "git reflog list",
    "git reflog exists refs/heads/main",
    "git fsck",
    "git fsck --no-lost-found",
    "git fsck --connectivity-only",
    "git branch -avv",
    "git status --short",
    "git log -Sfoo",
    "git log -Gfoo",
    "git log -S -o",
    # Keep the attached Git pattern literal; unquoted shell globs are a
    # separate, approval-requiring argv-expansion boundary.
    pytest.param("git tag '-lrelease*'", id="git tag -lrelease*"),
)


@pytest.fixture
def permission_environment(tmp_path, monkeypatch):
    # This gate test needs no real config, model classifier or sandbox backend.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        service_module, "resolve_sandbox_settings", lambda _cwd: SimpleNamespace(enabled=False)
    )
    monkeypatch.setattr(service_module, "is_excluded_command", lambda *_args, **_kwargs: False)
    return tmp_path


@pytest.mark.parametrize("command", MUTATING_COMMANDS)
def test_mutating_git_modes_require_approval_and_fail_read_only_policy(command):
    decision = classify_shell_command(command)
    assert decision.allowed
    assert not decision.read_only
    assert decision.requires_approval
    policy = SandboxPolicy.from_config({"enabled": True, "mode": "read-only"})
    assert read_only_violation(command, policy=policy) is not None


@pytest.mark.parametrize("command", MUTATING_COMMANDS)
@pytest.mark.parametrize("tool_name", ["run_shell", "git_command"])
@pytest.mark.parametrize("mode", [PermissionMode.DEFAULT, PermissionMode.DONT_ASK])
def test_permission_service_cannot_auto_allow_mutating_git_modes(
    permission_environment, command, tool_name, mode
):
    service = PermissionService.default(mode=mode, workspace_root=permission_environment)
    value = command.removeprefix("git ") if tool_name == "git_command" else command
    result = service.evaluate_tool_call(tool_name, {"command": value})

    assert not result.allowed
    assert result.requires_approval == (mode == PermissionMode.DEFAULT)


@pytest.mark.parametrize("command", READ_COMMANDS)
def test_git_read_controls_remain_available(permission_environment, command):
    decision = classify_shell_command(command)
    assert decision.allowed and decision.read_only and not decision.requires_approval
    service = PermissionService.default(
        mode=PermissionMode.DONT_ASK, workspace_root=permission_environment
    )
    result = service.evaluate_tool_call("run_shell", {"command": command})
    assert result.allowed and not result.requires_approval
    policy = SandboxPolicy.from_config({"enabled": True, "mode": "read-only"})
    assert read_only_violation(command, policy=policy) is None


def _git(repo: Path, *args: str, input_text: str | None = None) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    ).stdout


@pytest.fixture
def temporary_git_repository(tmp_path):
    # Every actual mutation in this file stays in this disposable repository.
    repo = tmp_path / "git-fixture"
    repo.mkdir()
    _git(repo, "init", "-q", "--initial-branch=main")
    _git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "fixture",
    )
    return repo


def test_real_reflog_expiry_changes_metadata_and_is_not_read_only(temporary_git_repository):
    repo = temporary_git_repository
    before = _git(repo, "reflog", "show", "--format=%H")
    assert before.strip()

    _git(repo, "reflog", "expire", "--expire=all", "--all")
    assert _git(repo, "reflog", "show", "--format=%H").strip() == ""
    assert not classify_shell_command(MUTATING_COMMANDS[0]).read_only


@pytest.mark.parametrize("option", ["--lost-found", "--lost"])
def test_real_fsck_lost_found_writes_objects_and_is_not_read_only(temporary_git_repository, option):
    repo = temporary_git_repository
    contents = "synthetic dangling blob\n"
    blob = _git(repo, "hash-object", "-w", "--stdin", input_text=contents).strip()
    target = repo / ".git" / "lost-found" / "other" / blob
    assert not target.exists()

    _git(repo, "fsck", option)
    assert target.read_text(encoding="utf-8") == contents
    assert not classify_shell_command(f"git fsck {option}").read_only
