"""Tests for EnterWorktree and ExitWorktree tools."""

import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

import koder_agent.tools.worktree as worktree_module
from koder_agent.tools.todo import (
    TodoRuntimeIdentity,
    TodoStore,
    reset_todo_context,
    set_todo_context,
)
from koder_agent.tools.worktree import (
    WorktreeSession,
    _get_worktree_session,
    _set_worktree_session,
    enter_worktree,
    exit_worktree,
    exit_worktree_tool,
)


def _init_git_repo(repo: Path) -> Path:
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    (repo / "README.md").write_text("# Test")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, capture_output=True, check=True)
    return repo


@pytest.fixture()
def git_repo(tmp_path):
    """Create a minimal git repo for worktree tests."""
    return _init_git_repo(tmp_path / "repo")


@pytest.fixture(autouse=True, params=["standalone", "runtime"])
def _clear_session(request):
    token = None
    if request.param == "runtime":
        token = set_todo_context(
            TodoStore(TodoRuntimeIdentity("worktree-test", "main", request.node.nodeid))
        )
    _set_worktree_session(None)
    yield
    _set_worktree_session(None)
    if token is not None:
        reset_todo_context(token)


def _branch_exists(repo: Path, branch: str) -> bool:
    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _git_stdout(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit_worktree_file(worktree_path: Path, filename: str = "feature.txt") -> None:
    (worktree_path / filename).write_text("unmerged work")
    subprocess.run(
        ["git", "add", filename],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "unmerged work"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )


def _commit_all(repo: Path, message: str) -> None:
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _remove_worktree(discard_changes):
    if discard_changes is None:
        return json.loads(exit_worktree(action="remove"))
    return json.loads(exit_worktree(action="remove", discard_changes=discard_changes))


def _record_remove_commands(monkeypatch):
    real_run = subprocess.run
    remove_commands = []

    def record_remove(command, *args, **kwargs):
        if command[:3] == ["git", "worktree", "remove"]:
            remove_commands.append(command)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(worktree_module.subprocess, "run", record_remove)
    return remove_commands


def _assert_safe_removal_blocked(
    result: dict,
    worktree_path: Path,
    branch: str,
    session,
    remove_commands: list,
) -> None:
    assert result["error_code"] == "worktree_not_clean"
    assert result["worktree_removed"] is False
    assert result["branch_deleted"] is False
    assert result["worktree_path_exists"] is True
    assert result["worktree_registered"] is True
    assert result["branch_exists"] is True
    assert result["session_preserved"] is True
    assert remove_commands == []
    assert worktree_path.exists()
    assert _get_worktree_session() is session
    assert _branch_exists(worktree_path, branch)


def test_enter_worktree_creates_dir(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    result = json.loads(enter_worktree(name="test-feature"))
    assert "worktree_path" in result
    assert Path(result["worktree_path"]).exists()


def test_enter_worktree_rejects_double_entry(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_worktree(name="first")
    result = json.loads(enter_worktree(name="second"))
    assert "already in a worktree" in result["message"].lower()


def test_enter_worktree_rejects_existing_branch_without_resetting_it(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    main_branch = _git_stdout(git_repo, "branch", "--show-current")
    subprocess.run(
        ["git", "switch", "-c", "worktree-collision"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (git_repo / "valuable.txt").write_text("committed work")
    _commit_all(git_repo, "valuable branch work")
    valuable_commit = _git_stdout(git_repo, "rev-parse", "HEAD")
    subprocess.run(
        ["git", "switch", main_branch],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(enter_worktree(name="collision"))

    assert "already exists" in result["error"]
    assert _get_worktree_session() is None
    assert _git_stdout(git_repo, "rev-parse", "refs/heads/worktree-collision") == valuable_commit


def test_enter_worktree_rejects_existing_path(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    target = git_repo / ".koder" / "worktrees" / "occupied"
    target.mkdir(parents=True)
    marker = target / "keep.txt"
    marker.write_text("do not adopt")

    result = json.loads(enter_worktree(name="occupied"))

    assert "already exists" in result["error"]
    assert marker.read_text() == "do not adopt"
    assert _get_worktree_session() is None
    assert not _branch_exists(git_repo, "worktree-occupied")


def test_enter_worktree_rejects_symlink_ancestor_escape(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    outside = git_repo.parent / "outside-worktrees"
    outside.mkdir()
    (git_repo / ".koder").mkdir()
    (git_repo / ".koder" / "worktrees").symlink_to(outside, target_is_directory=True)

    result = json.loads(enter_worktree(name="escaped"))

    assert "symlink ancestor" in result["error"].lower()
    assert not (outside / "escaped").exists()
    assert not _branch_exists(git_repo, "worktree-escaped")
    assert _get_worktree_session() is None


def test_enter_worktree_retains_cleanup_session_after_post_checkout_failure(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    hook = git_repo / ".git" / "hooks" / "post-checkout"
    hook.write_text("#!/bin/sh\nexit 23\n")
    hook.chmod(0o755)

    result = json.loads(enter_worktree(name="hook-failure"))

    assert "error" in result
    assert result["worktree_created"] is True
    assert result["worktree_path_exists"] is True
    assert result["worktree_registered"] is True
    assert result["registered_branch"] == "worktree-hook-failure"
    assert result["branch_exists"] is True
    assert result["cleanup_required"] is True
    assert result["partial_state"] is True
    assert result["session_state"] == "worktree_cleanup_pending"
    assert result["session_preserved"] is True
    assert _get_worktree_session() is not None

    hook.unlink()
    cleanup = json.loads(exit_worktree(action="remove", discard_changes=True))

    assert cleanup["worktree_removed"] is True
    assert cleanup["branch_deleted"] is True
    assert _get_worktree_session() is None


def test_enter_worktree_reconciles_cancellation_after_git_created_state(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    real_run = subprocess.run

    def interrupt_after_add(command, *args, **kwargs):
        result = real_run(command, *args, **kwargs)
        if command[:3] == ["git", "worktree", "add"]:
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(worktree_module.subprocess, "run", interrupt_after_add)

    with pytest.raises(KeyboardInterrupt):
        enter_worktree(name="cancelled-add")

    session = _get_worktree_session()
    assert session is not None
    assert session.phase == "worktree_cleanup_pending"
    assert Path(session.worktree_path).exists()
    assert _branch_exists(git_repo, session.worktree_branch)

    monkeypatch.setattr(worktree_module.subprocess, "run", real_run)
    cleanup = json.loads(exit_worktree(action="remove", discard_changes=True))

    assert cleanup["branch_deleted"] is True
    assert _get_worktree_session() is None


def test_exit_worktree_keep(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_worktree(name="feat")
    result = json.loads(exit_worktree(action="keep"))
    assert result["action"] == "keep"
    assert _get_worktree_session() is None


def test_exit_worktree_without_enter():
    result = json.loads(exit_worktree(action="keep"))
    assert "no active" in result["message"].lower()


@pytest.mark.parametrize("discard_changes", [True, False, None])
def test_exit_worktree_remove(git_repo, monkeypatch, discard_changes):
    monkeypatch.chdir(git_repo)
    suffix = "omitted" if discard_changes is None else str(discard_changes).lower()
    enter_result = json.loads(enter_worktree(name=f"removable-{suffix}"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    result = _remove_worktree(discard_changes)
    assert result["action"] == "remove"
    assert result["worktree_removed"] is True
    assert result["branch_deleted"] is True
    assert not worktree_path.exists()
    assert _get_worktree_session() is None
    assert not _branch_exists(git_repo, branch)


@pytest.mark.parametrize("discard_changes", [False, None])
def test_exit_worktree_remove_without_discard_preserves_unmerged_commit(
    git_repo, monkeypatch, discard_changes
):
    monkeypatch.chdir(git_repo)
    suffix = "omitted" if discard_changes is None else "false"
    enter_result = json.loads(enter_worktree(name=f"unmerged-{suffix}"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    _commit_worktree_file(worktree_path)

    result = _remove_worktree(discard_changes)

    assert result["worktree_removed"] is True
    assert result["branch_deleted"] is False
    assert result["branch_exists"] is True
    assert result["session_preserved"] is True
    assert result["session_state"] == "branch_cleanup_pending"
    assert not worktree_path.exists()
    assert _branch_exists(git_repo, branch)
    assert _get_worktree_session() is not None

    retry = json.loads(exit_worktree(action="remove", discard_changes=True))

    assert retry["branch_deleted"] is True
    assert not _branch_exists(git_repo, branch)
    assert _get_worktree_session() is None


def test_exit_worktree_remove_with_discard_deletes_unmerged_commit(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="unmerged-discarded"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    _commit_worktree_file(worktree_path)

    result = _remove_worktree(True)

    assert result["worktree_removed"] is True
    assert result["branch_deleted"] is True
    assert not worktree_path.exists()
    assert not _branch_exists(git_repo, branch)
    assert _get_worktree_session() is None


@pytest.mark.parametrize("discard_changes", [False, None])
def test_exit_worktree_remove_without_discard_preserves_untracked_file(
    git_repo, monkeypatch, discard_changes
):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="dirty"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    (worktree_path / "dirty.txt").write_text("keep me")
    session = _get_worktree_session()
    remove_commands = _record_remove_commands(monkeypatch)

    result = _remove_worktree(discard_changes)

    _assert_safe_removal_blocked(result, worktree_path, branch, session, remove_commands)
    assert "discard_changes=true" in result["message"]
    assert result["local_state"] == [{"kind": "untracked", "status": "??", "path": "dirty.txt"}]
    assert (worktree_path / "dirty.txt").read_text() == "keep me"


def test_exit_worktree_remove_without_discard_preserves_ignored_file(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    (git_repo / ".gitignore").write_text("ignored.log\n")
    _commit_all(git_repo, "ignore local logs")
    enter_result = json.loads(enter_worktree(name="ignored"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    (worktree_path / "ignored.log").write_text("keep ignored data")
    session = _get_worktree_session()
    remove_commands = _record_remove_commands(monkeypatch)

    result = json.loads(exit_worktree(action="remove"))

    _assert_safe_removal_blocked(result, worktree_path, branch, session, remove_commands)
    assert result["local_state"] == [{"kind": "ignored", "status": "!!", "path": "ignored.log"}]
    assert (worktree_path / "ignored.log").read_text() == "keep ignored data"


def test_exit_worktree_remove_without_discard_preserves_tracked_change(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="tracked-dirty"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    (worktree_path / "README.md").write_text("changed locally")
    session = _get_worktree_session()
    remove_commands = _record_remove_commands(monkeypatch)

    result = json.loads(exit_worktree(action="remove", discard_changes=False))

    _assert_safe_removal_blocked(result, worktree_path, branch, session, remove_commands)
    assert result["local_state"] == [
        {"kind": "tracked_or_submodule", "status": " M", "path": "README.md"}
    ]
    assert (worktree_path / "README.md").read_text() == "changed locally"


@pytest.mark.parametrize(
    ("flag", "expected_kind"),
    [
        ("--assume-unchanged", "assume-unchanged"),
        ("--skip-worktree", "skip-worktree"),
    ],
)
def test_exit_worktree_default_preserves_tracked_change_hidden_by_index_flag(
    git_repo, monkeypatch, flag, expected_kind
):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name=expected_kind))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    subprocess.run(
        ["git", "update-index", flag, "README.md"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    valuable_content = f"valuable edit hidden by {expected_kind}"
    (worktree_path / "README.md").write_text(valuable_content)
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignored=matching",
            "--ignore-submodules=none",
        ],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""
    session = _get_worktree_session()
    remove_commands = _record_remove_commands(monkeypatch)

    result = json.loads(exit_worktree(action="remove"))

    _assert_safe_removal_blocked(result, worktree_path, branch, session, remove_commands)
    assert any(
        item["kind"] == expected_kind and item["path"] == "README.md"
        for item in result["local_state"]
    )
    assert (worktree_path / "README.md").read_text() == valuable_content


def test_exit_worktree_remove_without_discard_preserves_unmerged_index(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    main_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    enter_result = json.loads(enter_worktree(name="unmerged-index"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]

    (worktree_path / "README.md").write_text("worktree version")
    _commit_all(worktree_path, "change in worktree")
    (git_repo / "README.md").write_text("main version")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "change on main"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    merge_result = subprocess.run(
        ["git", "merge", main_branch],
        cwd=worktree_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert merge_result.returncode != 0

    session = _get_worktree_session()
    remove_commands = _record_remove_commands(monkeypatch)
    result = json.loads(exit_worktree(action="remove"))

    _assert_safe_removal_blocked(result, worktree_path, branch, session, remove_commands)
    assert {tuple(item.items()) for item in result["local_state"]} == {
        tuple({"kind": "unmerged", "status": "UU", "path": "README.md"}.items()),
        tuple({"kind": "operation", "status": "merge", "path": "MERGE_HEAD"}.items()),
    }


def test_exit_worktree_remove_with_discard_allows_ignored_file(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    (git_repo / ".gitignore").write_text("ignored.log\n")
    _commit_all(git_repo, "ignore local logs")
    enter_result = json.loads(enter_worktree(name="ignored-discarded"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    (worktree_path / "ignored.log").write_text("discard me")

    result = json.loads(exit_worktree(action="remove", discard_changes=True))

    assert result["worktree_removed"] is True
    assert result["branch_deleted"] is True
    assert not worktree_path.exists()
    assert not _branch_exists(git_repo, branch)
    assert _get_worktree_session() is None


def test_exit_worktree_remove_without_discard_preserves_dirty_submodule(git_repo, monkeypatch):
    submodule_repo = git_repo.parent / "submodule"
    submodule_repo.mkdir()
    subprocess.run(["git", "init"], cwd=submodule_repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=submodule_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=submodule_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (submodule_repo / "nested.txt").write_text("nested")
    _commit_all(submodule_repo, "init submodule")

    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(submodule_repo),
            "deps/submodule",
        ],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    _commit_all(git_repo, "add submodule")

    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="dirty-submodule"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "update",
            "--init",
        ],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    nested_file = worktree_path / "deps" / "submodule" / "nested.txt"
    nested_file.write_text("dirty nested data")
    session = _get_worktree_session()
    remove_commands = _record_remove_commands(monkeypatch)

    result = json.loads(exit_worktree(action="remove"))

    _assert_safe_removal_blocked(result, worktree_path, branch, session, remove_commands)
    assert result["local_state"] == [
        {
            "kind": "tracked_or_submodule",
            "status": " M",
            "path": "deps/submodule",
        }
    ]
    assert nested_file.read_text() == "dirty nested data"


def test_exit_worktree_remove_without_discard_preserves_in_progress_operation(
    git_repo, monkeypatch
):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="merge-operation"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    git_dir = Path(_git_stdout(worktree_path, "rev-parse", "--absolute-git-dir"))
    head = _git_stdout(worktree_path, "rev-parse", "HEAD")
    (git_dir / "MERGE_HEAD").write_text(f"{head}\n")
    session = _get_worktree_session()
    remove_commands = _record_remove_commands(monkeypatch)

    result = json.loads(exit_worktree(action="remove", discard_changes=False))

    _assert_safe_removal_blocked(result, worktree_path, branch, session, remove_commands)
    assert result["local_state"] == [{"kind": "operation", "status": "merge", "path": "MERGE_HEAD"}]


def test_exit_worktree_default_blocks_clean_detached_head_commit(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="detached"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    subprocess.run(
        ["git", "switch", "--detach"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (worktree_path / "detached.txt").write_text("committed but detached")
    _commit_all(worktree_path, "detached work")
    detached_commit = _git_stdout(worktree_path, "rev-parse", "HEAD")

    result = json.loads(exit_worktree(action="remove"))

    assert result["error_code"] == "worktree_checkout_mismatch"
    assert any(item["status"] == "detached" for item in result["local_state"])
    assert worktree_path.exists()
    assert _branch_exists(git_repo, branch)
    assert _git_stdout(worktree_path, "rev-parse", "HEAD") == detached_commit
    assert _get_worktree_session() is not None

    cleanup = json.loads(exit_worktree(action="remove", discard_changes=True))
    assert cleanup["branch_deleted"] is False
    assert cleanup["error_code"] == "branch_ownership_unproven"
    assert _branch_exists(git_repo, branch)
    assert _get_worktree_session() is not None


def test_exit_worktree_default_blocks_switched_branch(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="switched"))
    worktree_path = Path(enter_result["worktree_path"])
    expected_branch = enter_result["worktree_branch"]
    subprocess.run(
        ["git", "switch", "-c", "unexpected-branch"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(exit_worktree(action="remove"))

    assert result["error_code"] == "worktree_checkout_mismatch"
    assert any(item["status"] == "switched_branch" for item in result["local_state"])
    assert worktree_path.exists()
    assert _branch_exists(git_repo, expected_branch)
    assert _branch_exists(git_repo, "unexpected-branch")

    cleanup = json.loads(exit_worktree(action="remove", discard_changes=True))
    assert cleanup["branch_deleted"] is False
    assert cleanup["error_code"] == "branch_ownership_unproven"
    assert _branch_exists(git_repo, expected_branch)
    assert _branch_exists(git_repo, "unexpected-branch")
    assert _get_worktree_session() is not None


def test_exit_worktree_remove_without_discard_blocks_on_inspection_failure(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="inspection-failure"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    session = _get_worktree_session()
    real_run = subprocess.run
    remove_commands = []

    def fail_status(command, *args, **kwargs):
        if command[:2] == ["git", "status"]:
            raise subprocess.CalledProcessError(128, command, stderr="localized inspection failure")
        if command[:3] == ["git", "worktree", "remove"]:
            remove_commands.append(command)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(worktree_module.subprocess, "run", fail_status)

    result = json.loads(exit_worktree(action="remove", discard_changes=False))

    assert result["error_code"] == "worktree_inspection_failed"
    assert result["worktree_removed"] is False
    assert result["branch_deleted"] is False
    assert result["session_preserved"] is True
    assert result["inspection_errors"]
    assert remove_commands == []
    assert worktree_path.exists()
    assert _branch_exists(git_repo, branch)
    assert _get_worktree_session() is session


def test_exit_worktree_structures_git_root_oserror(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="root-oserror"))
    worktree_path = Path(enter_result["worktree_path"])
    session = _get_worktree_session()
    real_run = subprocess.run

    def fail_rev_parse(command, *args, **kwargs):
        if command[:3] == ["git", "rev-parse", "--show-toplevel"]:
            raise PermissionError("git execution denied")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(worktree_module.subprocess, "run", fail_rev_parse)

    result = json.loads(exit_worktree(action="remove"))

    assert "error" in result
    assert result["session_preserved"] is True
    assert result["worktree_path_exists"] is True
    assert result["worktree_registered"] is None
    assert result["branch_exists"] is None
    assert worktree_path.exists()
    assert _get_worktree_session() is session


def test_exit_worktree_preserves_branch_after_external_worktree_removal(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="externally-removed"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    subprocess.run(
        ["git", "worktree", "remove", str(worktree_path)],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(exit_worktree(action="remove"))

    assert result["worktree_removed"] is True
    assert result["branch_deleted"] is False
    assert result["error_code"] == "branch_ownership_unproven"
    assert result["worktree_path_exists"] is False
    assert result["worktree_registered"] is False
    assert result["branch_exists"] is True
    assert result["session_preserved"] is True
    assert _get_worktree_session() is not None
    assert _branch_exists(git_repo, branch)


def test_exit_worktree_uses_recorded_owner_after_cwd_changes(git_repo, tmp_path, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="owner-bound"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]

    other_repo = _init_git_repo(tmp_path / "other-repo")
    subprocess.run(
        ["git", "branch", branch],
        cwd=other_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    other_branch_head = _git_stdout(other_repo, "rev-parse", branch)
    monkeypatch.chdir(other_repo)

    result = json.loads(exit_worktree(action="remove"))

    assert result["owner_root"] == str(git_repo.resolve())
    assert result["worktree_removed"] is True
    assert result["branch_deleted"] is True
    assert not worktree_path.exists()
    assert not _branch_exists(git_repo, branch)
    assert _branch_exists(other_repo, branch)
    assert _git_stdout(other_repo, "rev-parse", branch) == other_branch_head


def test_exit_worktree_rejects_same_path_replacement_repository(git_repo, tmp_path, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="drift"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    session = _get_worktree_session()
    assert session is not None

    subprocess.run(
        ["git", "worktree", "remove", str(worktree_path)],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    monkeypatch.chdir(tmp_path)
    displaced_repo = tmp_path / "original-repo"
    git_repo.rename(displaced_repo)
    replacement_repo = _init_git_repo(git_repo)
    subprocess.run(
        ["git", "branch", branch],
        cwd=replacement_repo,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(exit_worktree(action="remove"))

    assert result["error_code"] == "repository_identity_mismatch"
    assert result["worktree_removed"] is False
    assert result["branch_deleted"] is False
    assert result["session_preserved"] is True
    assert _get_worktree_session() is session
    assert _branch_exists(replacement_repo, branch)
    assert _branch_exists(displaced_repo, branch)


def test_inplace_reinit_preserves_replacement_branch(git_repo, monkeypatch):
    """INPLACE_REINIT: retaining the marker/common inode cannot retain ownership."""
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="inplace-reinit"))
    branch = enter_result["worktree_branch"]
    session = _get_worktree_session()
    assert session is not None
    common_dir = Path(session.owner_common_dir)
    marker = common_dir / ".koder-repository-id"
    common_inode = common_dir.stat().st_ino
    marker_inode = marker.stat().st_ino

    for child in common_dir.iterdir():
        if child == marker:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    subprocess.run(["git", "init"], cwd=git_repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "replacement@test.com"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Replacement"],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (git_repo / "replacement.txt").write_text("replacement")
    _commit_all(git_repo, "replacement init")
    subprocess.run(
        ["git", "branch", branch], cwd=git_repo, check=True, capture_output=True, text=True
    )

    result = json.loads(exit_worktree(action="remove", discard_changes=True))

    assert common_dir.stat().st_ino == common_inode
    assert marker.stat().st_ino == marker_inode
    assert result["error_code"] == "repository_identity_mismatch"
    assert result["branch_deleted"] is False
    assert _branch_exists(git_repo, branch)
    assert _get_worktree_session() is session


def test_worktree_swap_preserves_replacement_worktree_and_branch(git_repo, monkeypatch):
    """WORKTREE_SWAP: an exact-path replacement is never adopted."""
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="worktree-swap"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    session = _get_worktree_session()
    assert session is not None

    subprocess.run(
        ["git", "worktree", "remove", str(worktree_path)],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "branch", "-D", branch],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "branch", branch], cwd=git_repo, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "worktree", "add", str(worktree_path), branch],
        cwd=git_repo,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(exit_worktree(action="remove", discard_changes=True))

    assert result["error_code"] in {
        "worktree_ownership_mismatch",
        "repository_identity_mismatch",
    }
    assert result["worktree_removed"] is False
    assert result["branch_deleted"] is False
    assert worktree_path.exists()
    assert _branch_exists(git_repo, branch)
    assert _get_worktree_session() is session


def test_check_command_race_mutates_only_descriptor_bound_repository(
    git_repo, tmp_path, monkeypatch
):
    """CHECK_COMMAND_RACE: a pathname swap cannot redirect update-ref."""
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="command-race"))
    branch = enter_result["worktree_branch"]
    displaced_repo = tmp_path / "displaced-repo"
    replacement_repo = git_repo
    real_run = subprocess.run
    swapped = False

    def swap_before_update_ref(command, *args, **kwargs):
        nonlocal swapped
        if command[:3] == ["git", "update-ref", "-d"] and not swapped:
            swapped = True
            monkeypatch.chdir(tmp_path)
            replacement_repo.rename(displaced_repo)
            _init_git_repo(replacement_repo)
            real_run(
                ["git", "branch", branch],
                cwd=replacement_repo,
                check=True,
                capture_output=True,
                text=True,
            )
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(worktree_module.subprocess, "run", swap_before_update_ref)

    result = json.loads(exit_worktree(action="remove"))

    assert swapped is True
    assert result["error_code"] == "repository_identity_mismatch"
    assert result["branch_deleted"] is True
    assert _branch_exists(replacement_repo, branch)
    assert not _branch_exists(displaced_repo, branch)


def test_repository_marker_creation_cleanup_uses_held_parent(git_repo, tmp_path, monkeypatch):
    """CREATE_CLEANUP_PATH_SWAP: cleanup cannot unlink a replacement marker."""
    common_dir = git_repo / ".git"
    displaced_common = tmp_path / "displaced-common"
    replacement_value = "a" * 64
    real_write = os.write
    triggered = False

    def swap_then_fail(descriptor, data):
        nonlocal triggered
        if not triggered:
            triggered = True
            common_dir.rename(displaced_common)
            common_dir.mkdir()
            replacement_marker = common_dir / ".koder-repository-id"
            replacement_marker.write_text(f"{replacement_value}\n")
            replacement_marker.chmod(0o600)
            raise OSError("simulated marker write failure")
        return real_write(descriptor, data)

    monkeypatch.setattr(worktree_module.os, "write", swap_then_fail)

    with pytest.raises(OSError, match="simulated marker write failure"):
        worktree_module._repository_fingerprint(common_dir, create=True)

    assert triggered is True
    assert (common_dir / ".koder-repository-id").read_text() == f"{replacement_value}\n"
    assert not (displaced_common / ".koder-repository-id").exists()


def test_hardlink_repository_marker_is_rejected(git_repo, tmp_path, monkeypatch):
    """HARDLINK_MARKER: multi-link marker inodes are never trusted."""
    source = tmp_path / "attacker-marker"
    source.write_text(f"{'b' * 64}\n")
    source.chmod(0o600)
    marker = git_repo / ".git" / ".koder-repository-id"
    os.link(source, marker)
    monkeypatch.chdir(git_repo)

    result = json.loads(enter_worktree(name="hardlink-marker"))

    assert marker.stat().st_nlink == 2
    assert "exactly one hard link" in result["error"]
    assert not _branch_exists(git_repo, "worktree-hardlink-marker")
    assert _get_worktree_session() is None


def test_worktree_session_identity_survives_json_restart(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="serialized"))
    branch = enter_result["worktree_branch"]
    session = _get_worktree_session()
    assert session is not None

    payload = json.loads(json.dumps(asdict(session)))
    _set_worktree_session(None)
    restored = WorktreeSession(**payload)
    _set_worktree_session(restored)

    result = json.loads(exit_worktree(action="remove"))

    assert restored.owner_fingerprint == session.owner_fingerprint
    assert restored.owner_structure == session.owner_structure
    assert restored.owner_device == session.owner_device
    assert restored.owner_inode == session.owner_inode
    assert restored.worktree_fingerprint == session.worktree_fingerprint
    assert restored.worktree_admin_inode == session.worktree_admin_inode
    assert result["worktree_removed"] is True
    assert result["branch_deleted"] is True
    assert not _branch_exists(git_repo, branch)
    assert _get_worktree_session() is None


def test_branch_cleanup_identity_survives_json_restart(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="serialized-branch-cleanup"))
    branch = enter_result["worktree_branch"]
    real_delete = worktree_module.WorktreeLifecycle.delete_owned_branch
    monkeypatch.setattr(
        worktree_module.WorktreeLifecycle,
        "delete_owned_branch",
        lambda lifecycle, force: PermissionError("pause branch cleanup"),
    )

    first = json.loads(exit_worktree(action="remove", discard_changes=True))

    assert first["worktree_removed"] is True
    assert first["branch_deleted"] is False
    session = _get_worktree_session()
    assert session is not None
    assert session.branch_cleanup_head is not None
    assert session.branch_cleanup_identity is not None
    payload = json.loads(json.dumps(asdict(session)))
    _set_worktree_session(WorktreeSession(**payload))
    monkeypatch.setattr(
        worktree_module.WorktreeLifecycle,
        "delete_owned_branch",
        real_delete,
    )

    retry = json.loads(exit_worktree(action="remove", discard_changes=True))

    assert retry["branch_deleted"] is True
    assert not _branch_exists(git_repo, branch)
    assert _get_worktree_session() is None


def test_legacy_serialized_session_without_identity_fails_closed(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="legacy-serialized"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    session = _get_worktree_session()
    assert session is not None

    payload = asdict(session)
    for field in (
        "owner_device",
        "owner_inode",
        "owner_fingerprint",
        "owner_structure",
        "owner_admin_dir",
        "approved_device",
        "approved_inode",
        "worktree_device",
        "worktree_inode",
        "worktree_admin_dir",
        "worktree_admin_relative",
        "worktree_admin_device",
        "worktree_admin_inode",
        "worktree_fingerprint",
    ):
        payload.pop(field)
    restored = WorktreeSession(**json.loads(json.dumps(payload)))
    _set_worktree_session(restored)

    result = json.loads(exit_worktree(action="remove", discard_changes=True))

    assert result["error_code"] == "repository_identity_mismatch"
    assert result["worktree_removed"] is False
    assert result["branch_deleted"] is False
    assert result["session_preserved"] is True
    assert worktree_path.exists()
    assert _branch_exists(git_repo, branch)
    assert _get_worktree_session() is restored


def test_exit_worktree_reconciles_cancellation_after_worktree_removal(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="remove-cancelled"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    real_remove = worktree_module.WorktreeLifecycle.remove_owned_worktree

    def interrupt_after_remove(lifecycle, state):
        assert real_remove(lifecycle, state) is None
        raise KeyboardInterrupt

    monkeypatch.setattr(
        worktree_module.WorktreeLifecycle,
        "remove_owned_worktree",
        interrupt_after_remove,
    )

    with pytest.raises(KeyboardInterrupt):
        exit_worktree(action="remove")

    session = _get_worktree_session()
    assert session is not None
    assert session.phase == "branch_cleanup_pending"
    assert not worktree_path.exists()
    assert _branch_exists(git_repo, branch)

    monkeypatch.setattr(
        worktree_module.WorktreeLifecycle,
        "remove_owned_worktree",
        real_remove,
    )
    retry = json.loads(exit_worktree(action="remove", discard_changes=True))

    assert retry["branch_deleted"] is True
    assert not _branch_exists(git_repo, branch)
    assert _get_worktree_session() is None


def test_exit_worktree_reconciles_cancellation_before_branch_cleanup(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="branch-cancelled"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    real_delete = worktree_module.WorktreeLifecycle.delete_owned_branch

    def interrupt_before_branch_delete(lifecycle, discard_changes):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        worktree_module.WorktreeLifecycle,
        "delete_owned_branch",
        interrupt_before_branch_delete,
    )

    with pytest.raises(KeyboardInterrupt):
        exit_worktree(action="remove")

    session = _get_worktree_session()
    assert session is not None
    assert session.phase == "branch_cleanup_pending"
    assert not worktree_path.exists()
    assert _branch_exists(git_repo, branch)

    monkeypatch.setattr(
        worktree_module.WorktreeLifecycle,
        "delete_owned_branch",
        real_delete,
    )
    retry = json.loads(exit_worktree(action="remove", discard_changes=True))

    assert retry["branch_deleted"] is True
    assert not _branch_exists(git_repo, branch)
    assert _get_worktree_session() is None


@pytest.mark.parametrize("failure_kind", ["called_process", "oserror"])
def test_exit_worktree_force_removal_failure_does_not_recommend_identical_retry(
    git_repo, monkeypatch, failure_kind
):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="force-failure"))
    worktree_path = Path(enter_result["worktree_path"])
    session = _get_worktree_session()

    def fail_remove(lifecycle, state):
        if failure_kind == "called_process":
            return subprocess.CalledProcessError(
                1, ["descriptor-bound-worktree-remove"], stderr="simulated removal failure"
            )
        return PermissionError("worktree removal denied")

    monkeypatch.setattr(
        worktree_module.WorktreeLifecycle,
        "remove_owned_worktree",
        fail_remove,
    )

    result = json.loads(exit_worktree(action="remove", discard_changes=True))

    assert "error" in result
    assert "already requested" in result["message"]
    assert "retry with discard_changes=true" not in result["message"]
    assert result["worktree_path_exists"] is True
    assert result["worktree_registered"] is True
    assert result["session_preserved"] is True
    assert worktree_path.exists()
    assert _get_worktree_session() is session


@pytest.mark.parametrize("discard_changes", [False, True])
@pytest.mark.parametrize("failure_kind", ["called_process", "oserror"])
def test_exit_worktree_branch_deletion_failure_preserves_branch(
    git_repo, monkeypatch, failure_kind, discard_changes
):
    monkeypatch.chdir(git_repo)
    mode = "force" if discard_changes else "safe"
    enter_result = json.loads(enter_worktree(name=f"branch-failure-{mode}-{failure_kind}"))
    worktree_path = Path(enter_result["worktree_path"])
    branch = enter_result["worktree_branch"]
    real_delete = worktree_module.WorktreeLifecycle.delete_owned_branch

    def fail_branch_delete(lifecycle, force):
        if failure_kind == "called_process":
            return subprocess.CalledProcessError(
                1, ["git", "update-ref"], stderr="simulated branch deletion failure"
            )
        return PermissionError("branch deletion denied")

    monkeypatch.setattr(
        worktree_module.WorktreeLifecycle,
        "delete_owned_branch",
        fail_branch_delete,
    )

    result = json.loads(exit_worktree(action="remove", discard_changes=discard_changes))

    assert "error" in result
    assert result["worktree_removed"] is True
    assert result["branch_deleted"] is False
    assert result["branch_exists"] is True
    assert result["session_preserved"] is True
    assert result["session_state"] == "branch_cleanup_pending"
    assert not worktree_path.exists()
    assert _branch_exists(git_repo, branch)
    assert _get_worktree_session() is not None

    monkeypatch.setattr(
        worktree_module.WorktreeLifecycle,
        "delete_owned_branch",
        real_delete,
    )
    retry = json.loads(exit_worktree(action="remove", discard_changes=True))

    assert retry["branch_deleted"] is True
    assert not _branch_exists(git_repo, branch)
    assert _get_worktree_session() is None


@pytest.mark.parametrize("replacement_kind", ["moved", "recreated"])
def test_branch_cleanup_refuses_moved_or_recreated_branch(git_repo, monkeypatch, replacement_kind):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name=f"branch-{replacement_kind}"))
    branch = enter_result["worktree_branch"]
    real_delete = worktree_module.WorktreeLifecycle.delete_owned_branch

    monkeypatch.setattr(
        worktree_module.WorktreeLifecycle,
        "delete_owned_branch",
        lambda lifecycle, force: PermissionError("pause before branch deletion"),
    )
    first = json.loads(exit_worktree(action="remove", discard_changes=True))
    assert first["worktree_removed"] is True
    assert first["branch_deleted"] is False
    original_head = _git_stdout(git_repo, "rev-parse", branch)

    monkeypatch.setattr(
        worktree_module.WorktreeLifecycle,
        "delete_owned_branch",
        real_delete,
    )
    if replacement_kind == "moved":
        (git_repo / "owner-change.txt").write_text("new owner commit")
        _commit_all(git_repo, "move owner head")
        replacement_head = _git_stdout(git_repo, "rev-parse", "HEAD")
        subprocess.run(
            ["git", "update-ref", f"refs/heads/{branch}", replacement_head],
            cwd=git_repo,
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        subprocess.run(
            ["git", "branch", "-D", branch],
            cwd=git_repo,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "branch", branch, original_head],
            cwd=git_repo,
            check=True,
            capture_output=True,
            text=True,
        )

    retry = json.loads(exit_worktree(action="remove", discard_changes=True))

    assert retry["branch_deleted"] is False
    assert "deleted, recreated, or moved" in retry["error"]
    assert _branch_exists(git_repo, branch)
    assert _get_worktree_session() is not None


def test_exit_worktree_rejects_invalid_action_without_changing_session(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="invalid-action"))
    worktree_path = Path(enter_result["worktree_path"])
    session = _get_worktree_session()

    result = json.loads(exit_worktree(action="delete"))

    assert "error" in result
    assert "keep" in result["error"]
    assert "remove" in result["error"]
    assert worktree_path.exists()
    assert _get_worktree_session() is session


def test_exit_worktree_action_schema_is_explicit():
    action_schema = exit_worktree_tool.params_json_schema["properties"]["action"]
    assert action_schema["enum"] == ["keep", "remove"]


@pytest.mark.asyncio
async def test_exit_worktree_tool_invokes_valid_schema(git_repo, monkeypatch):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="wrapper-valid"))

    raw_result = await exit_worktree_tool.on_invoke_tool(None, json.dumps({"action": "keep"}))
    result = json.loads(raw_result)

    assert result["action"] == "keep"
    assert Path(enter_result["worktree_path"]).exists()
    assert _get_worktree_session() is None


@pytest.mark.asyncio
async def test_exit_worktree_tool_rejects_invalid_schema_without_changing_session(
    git_repo, monkeypatch
):
    monkeypatch.chdir(git_repo)
    enter_result = json.loads(enter_worktree(name="wrapper-invalid"))
    session = _get_worktree_session()

    result = await exit_worktree_tool.on_invoke_tool(None, json.dumps({"action": "delete"}))

    assert "error" in result.lower()
    assert Path(enter_result["worktree_path"]).exists()
    assert _get_worktree_session() is session


def test_slug_validation():
    result = json.loads(enter_worktree(name="../escape"))
    assert (
        "invalid" in result.get("message", "").lower()
        or "error" in result.get("message", "").lower()
    )
