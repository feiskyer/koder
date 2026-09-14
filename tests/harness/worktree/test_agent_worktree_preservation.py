"""Exercise automatic agent worktree cleanup against real, temporary Git repos."""

import shutil
import subprocess
from pathlib import Path

import pytest

from koder_agent.harness.worktree.service import WorktreeService


def _git(cwd: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


@pytest.fixture
def worktree_service(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "README.md").write_text("original\n", encoding="utf-8")
    (repo / ".gitignore").write_text("local-notes.txt\n", encoding="utf-8")
    _git(repo, "add", "README.md", ".gitignore")
    _git(repo, "commit", "-m", "initial")
    return WorktreeService(repo / ".koder" / "worktrees", repo_root=repo)


def test_create_preserves_existing_branch_tip(worktree_service):
    repo = worktree_service.repo_root
    _git(repo, "switch", "-c", "reserved")
    (repo / "README.md").write_text("valuable committed work\n", encoding="utf-8")
    _git(repo, "commit", "-am", "valuable work")
    original_tip = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "main")

    with pytest.raises((FileExistsError, subprocess.CalledProcessError)):
        worktree_service.create("reserved")

    assert _git(repo, "rev-parse", "reserved") == original_tip


def test_create_does_not_adopt_existing_directory(worktree_service):
    occupied = worktree_service.root / "occupied"
    occupied.mkdir(parents=True)
    marker = occupied / "keep.txt"
    marker.write_text("unrelated user directory", encoding="utf-8")

    with pytest.raises(FileExistsError):
        worktree_service.create("occupied")

    assert marker.read_text(encoding="utf-8") == "unrelated user directory"


def test_automatic_cleanup_keeps_unmerged_commits(worktree_service):
    created = worktree_service.create("agent/review")
    (created.path / "result.txt").write_text("agent result\n", encoding="utf-8")
    _git(created.path, "add", "result.txt")
    _git(created.path, "commit", "-m", "agent result")
    committed_tip = _git(created.path, "rev-parse", "HEAD")
    assert _git(created.path, "status", "--porcelain") == ""

    assert worktree_service.remove_if_clean(created.path, branch=created.branch) is False

    assert (created.path / "result.txt").read_text(encoding="utf-8") == "agent result\n"
    assert _git(worktree_service.repo_root, "rev-parse", created.branch) == committed_tip


@pytest.mark.parametrize("hidden_kind", ["ignored", "assume-unchanged", "skip-worktree"])
def test_automatic_cleanup_keeps_changes_hidden_from_status(worktree_service, hidden_kind):
    created = worktree_service.create("agent/hidden")
    if hidden_kind == "ignored":
        filename = "local-notes.txt"
    else:
        filename = "README.md"
        _git(created.path, "update-index", f"--{hidden_kind}", filename)
    marker = created.path / filename
    marker.write_text("preserve local work\n", encoding="utf-8")
    assert _git(created.path, "status", "--porcelain") == ""

    assert worktree_service.remove_if_clean(created.path, branch=created.branch) is False

    assert marker.read_text(encoding="utf-8") == "preserve local work\n"


@pytest.mark.parametrize("checkout", ["detached", "other-branch"])
def test_automatic_cleanup_keeps_repurposed_checkout(worktree_service, checkout):
    created = worktree_service.create("agent/original")
    if checkout == "detached":
        _git(created.path, "switch", "--detach")
    else:
        _git(created.path, "switch", "-c", checkout)

    assert worktree_service.remove_if_clean(created.path, branch=created.branch) is False

    assert created.path.exists()
    assert _git(worktree_service.repo_root, "rev-parse", created.branch)


def test_automatic_cleanup_keeps_changes_written_after_preflight(worktree_service, monkeypatch):
    created = worktree_service.create("agent/race")
    original_run = subprocess.run
    marker = created.path / "late-result.txt"

    def write_before_remove(command, *args, **kwargs):
        if command[:3] == ["git", "worktree", "remove"]:
            marker.write_text("late user change\n", encoding="utf-8")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", write_before_remove)

    assert worktree_service.remove_if_clean(created.path, branch=created.branch) is False

    assert marker.read_text(encoding="utf-8") == "late user change\n"


def test_automatic_cleanup_removes_unchanged_owned_worktree(worktree_service):
    created = worktree_service.create("agent/clean")

    assert worktree_service.remove_if_clean(created.path, branch=created.branch) is True

    assert not created.path.exists()
    assert _git(worktree_service.repo_root, "branch", "--list", created.branch) == ""
