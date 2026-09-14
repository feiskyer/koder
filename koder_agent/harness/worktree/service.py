"""Worktree lifecycle service for isolated agent runs."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from koder_agent.harness.hooks.runtime import dispatch_command_hooks


@dataclass(frozen=True)
class WorktreeCreateResult:
    """Result of creating a worktree."""

    path: Path
    branch: str
    repo_root: Path | None = None


@dataclass(frozen=True)
class WorktreeTransitionResult:
    """Result of entering or exiting a worktree."""

    ok: bool
    path: Path


class WorktreeService:
    """Create agent worktrees and conservatively clean up unchanged checkouts."""

    def __init__(self, root: Path, *, repo_root: Path | None = None):
        self.root = root
        self.repo_root = repo_root
        self._active: set[Path] = set()

    @classmethod
    def for_test(cls, root: Path) -> "WorktreeService":
        root.mkdir(parents=True, exist_ok=True)
        repo_root = root if (root / ".git").exists() else None
        worktree_root = root / ".koder" / "worktrees" if repo_root else root
        worktree_root.mkdir(parents=True, exist_ok=True)
        return cls(worktree_root, repo_root=repo_root)

    def create(self, branch: str) -> WorktreeCreateResult:
        branch_slug = branch.replace("/", "-")
        path = self.root / branch_slug
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Worktree path already exists: {path}")
        if self.repo_root and (self.repo_root / ".git").exists():
            subprocess.run(
                ["git", "worktree", "add", "-b", branch, str(path), "HEAD"],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
            hook_result = dispatch_command_hooks(
                cwd=self.repo_root or path,
                event_name="WorktreeCreate",
                match_value=None,
                payload={
                    "event": "WorktreeCreate",
                    "branch": branch,
                    "worktree_path": str(path),
                },
            )
            if hook_result.worktree_path:
                path = Path(hook_result.worktree_path)
            return WorktreeCreateResult(path=path, branch=branch, repo_root=self.repo_root)

        path.mkdir(parents=True)
        hook_result = dispatch_command_hooks(
            cwd=self.repo_root or path,
            event_name="WorktreeCreate",
            match_value=None,
            payload={
                "event": "WorktreeCreate",
                "branch": branch,
                "worktree_path": str(path),
            },
        )
        if hook_result.worktree_path:
            path = Path(hook_result.worktree_path)
        return WorktreeCreateResult(path=path, branch=branch, repo_root=self.repo_root)

    def enter(self, path: Path) -> WorktreeTransitionResult:
        self._active.add(path)
        return WorktreeTransitionResult(ok=True, path=path)

    def exit(self, path: Path) -> WorktreeTransitionResult:
        self._active.discard(path)
        dispatch_command_hooks(
            cwd=self.repo_root or path,
            event_name="WorktreeRemove",
            match_value=None,
            payload={
                "event": "WorktreeRemove",
                "worktree_path": str(path),
            },
        )
        return WorktreeTransitionResult(ok=True, path=path)

    def is_clean(self, path: Path) -> bool:
        """Return True when the worktree has no local state worth preserving.

        Non-git worktrees (plain directories) count as clean when empty.
        Errors count as dirty so we never remove work we cannot assess.
        """
        if not path.exists() or path.is_symlink():
            return False
        if self.repo_root and (self.repo_root / ".git").exists():
            try:
                status = self._git(
                    path,
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--ignored=matching",
                    "--ignore-submodules=none",
                )
                if status:
                    return False
                # Git status deliberately hides assume-unchanged/skip-worktree
                # entries. Automatic cleanup must not discard their local data.
                flags = self._git(path, "ls-files", "-v", "-z")
                if any(
                    entry[:1].islower() or entry.startswith("S ") for entry in flags.split("\0")
                ):
                    return False
                git_dir = Path(self._git(path, "rev-parse", "--absolute-git-dir"))
                operation_markers = (
                    "MERGE_HEAD",
                    "CHERRY_PICK_HEAD",
                    "REVERT_HEAD",
                    "BISECT_LOG",
                    "rebase-merge",
                    "rebase-apply",
                    "sequencer",
                )
                return not any((git_dir / marker).exists() for marker in operation_markers)
            except Exception:
                return False
        try:
            return not any(path.iterdir())
        except OSError:
            return False

    @staticmethod
    def _git(cwd: Path, *arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()

    def remove_if_clean(self, path: Path, *, branch: str | None = None) -> bool:
        """Remove an unchanged, merged checkout; dispatch ``WorktreeRemove``.

        Unmerged commits, changed checkouts and unassessable state are kept.
        Git removal remains non-forcing to protect changes made after inspection.
        """
        if not self.is_clean(path):
            return False
        if self.repo_root and (self.repo_root / ".git").exists():
            try:
                if path.resolve().parent != self.root.resolve() or not (path / ".git").is_file():
                    return False
                current_branch = self._git(path, "symbolic-ref", "-q", "HEAD")
                if branch is not None and current_branch != f"refs/heads/{branch}":
                    return False
                head = self._git(path, "rev-parse", "HEAD")
                # A clean index says nothing about commits produced by an agent.
                # Keep those commits and their checkout until the owner merges.
                self._git(self.repo_root, "merge-base", "--is-ancestor", head, "HEAD")
                subprocess.run(
                    ["git", "worktree", "remove", str(path)],
                    cwd=self.repo_root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if branch:
                    subprocess.run(
                        ["git", "branch", "-d", branch],
                        cwd=self.repo_root,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
            except Exception:
                return False
        else:
            try:
                path.rmdir()
            except OSError:
                return False
        self.exit(path)
        return True
