"""Workspace path policy evaluation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DANGEROUS_DELETE_PATHS = {
    Path("/"),
    Path("/bin"),
    Path("/etc"),
    Path("/sbin"),
    Path("/usr"),
    Path("/var"),
    Path("/tmp"),
    Path.home(),
}

DANGEROUS_FILES = frozenset(
    {
        ".gitconfig",
        ".gitmodules",
        ".bashrc",
        ".bash_profile",
        ".zshrc",
        ".zprofile",
        ".profile",
        ".ripgreprc",
    }
)

DANGEROUS_DIRECTORIES = frozenset(
    {
        ".git",
        ".vscode",
        ".idea",
        ".koder",
    }
)


def has_shell_expansion_syntax(path: str) -> bool:
    """Check if path contains shell expansion syntax that could cause TOCTOU vulnerabilities."""
    if "$" in path or "%" in path or "`" in path:
        return True
    if path.startswith("="):
        return True
    if path.startswith("~") and not path.startswith("~/") and path != "~":
        return True
    return False


def resolve_with_symlinks(path: str) -> tuple[str, str]:
    """Resolve path and return both the normalized path and the real path (following symlinks)."""
    p = Path(path).expanduser()
    original = str(p.resolve())
    real = os.path.realpath(str(p))
    return original, real


def _is_dangerous_path(normalized: Path, roots: list[Path]) -> bool:
    """Check if the normalized path points to a dangerous file or directory."""
    name = normalized.name
    if name in DANGEROUS_FILES:
        return True
    containing = [root for root in roots if _path_within(normalized, root)]
    root = max(containing, key=lambda item: len(item.parts)) if containing else None
    # A worktree can live below .koder without making every source file a
    # configuration file. Protect sensitive paths inside its actual boundary.
    parts = normalized.relative_to(root).parts if root is not None else normalized.parts
    if root is not None and root.name in DANGEROUS_DIRECTORIES:
        return True
    for part in parts:
        if part in DANGEROUS_DIRECTORIES:
            return True
    return False


@dataclass(frozen=True)
class PathAccessDecision:
    """Decision for a single filesystem path access."""

    path: str
    normalized_path: str
    operation: str
    allowed: bool
    requires_approval: bool
    reason: str


def _normalize_root(root: Path | str | None) -> Path:
    if root is None:
        return Path.cwd().resolve()
    return Path(root).expanduser().resolve()


def _normalize_target(path: str, workspace_root: Path) -> Path:
    raw = Path(path.replace("\0", "")).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (workspace_root / raw).resolve()


def _path_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _all_roots(workspace_root: Path, additional_roots: Iterable[str | Path] | None) -> list[Path]:
    roots = [workspace_root]
    if additional_roots:
        roots.extend(_normalize_root(root) for root in additional_roots)
    return roots


def evaluate_path_access(
    path: str,
    *,
    operation: str,
    workspace_root: Path | str | None = None,
    additional_roots: Iterable[str | Path] | None = None,
    working_directory: Path | str | None = None,
) -> PathAccessDecision:
    """Evaluate whether a path operation is allowed within the current workspace."""
    # Check shell expansion FIRST before any normalization
    if has_shell_expansion_syntax(path):
        return PathAccessDecision(
            path=path,
            normalized_path=path,
            operation=operation,
            allowed=False,
            requires_approval=False,
            reason="shell expansion syntax detected",
        )

    root = _normalize_root(workspace_root)
    directory = _normalize_root(working_directory) if working_directory is not None else root

    if "\0" in path:
        normalized = _normalize_target(path, directory)
        return PathAccessDecision(
            path=path,
            normalized_path=str(normalized),
            operation=operation,
            allowed=False,
            requires_approval=False,
            reason="path contains null byte",
        )

    # Build the path without resolving symlinks first
    raw_path = Path(path.replace("\0", "")).expanduser()
    if not raw_path.is_absolute():
        raw_path = directory / raw_path

    # Now normalize (which resolves symlinks)
    normalized = _normalize_target(path, directory)
    roots = _all_roots(root, additional_roots)

    # Check if path is a symlink and where it points
    if raw_path.exists() and raw_path.is_symlink():
        # Get the real path following symlinks
        real_path = Path(os.path.realpath(str(raw_path)))

        # Check if the link itself is in workspace but target is not
        link_in_workspace = any(_path_within(raw_path, allowed_root) for allowed_root in roots)
        target_in_workspace = any(_path_within(real_path, allowed_root) for allowed_root in roots)

        if link_in_workspace and not target_in_workspace:
            return PathAccessDecision(
                path=path,
                normalized_path=str(normalized),
                operation=operation,
                allowed=False,
                requires_approval=False,
                reason="symlink resolves outside workspace",
            )

    if ".." in Path(path).parts and not _path_within(normalized, root):
        return PathAccessDecision(
            path=path,
            normalized_path=str(normalized),
            operation=operation,
            allowed=False,
            requires_approval=True,
            reason="path escapes workspace",
        )

    in_workspace = any(_path_within(normalized, allowed_root) for allowed_root in roots)

    if operation == "delete" and normalized in DANGEROUS_DELETE_PATHS:
        return PathAccessDecision(
            path=path,
            normalized_path=str(normalized),
            operation=operation,
            allowed=False,
            requires_approval=True,
            reason="dangerous delete path",
        )

    if not in_workspace:
        return PathAccessDecision(
            path=path,
            normalized_path=str(normalized),
            operation=operation,
            allowed=False,
            requires_approval=True,
            reason="path outside workspace",
        )

    if operation == "read":
        return PathAccessDecision(
            path=path,
            normalized_path=str(normalized),
            operation=operation,
            allowed=True,
            requires_approval=False,
            reason="workspace read allowed",
        )

    # Check for dangerous files/directories for write/delete operations
    if operation in ("write", "delete") and _is_dangerous_path(normalized, roots):
        return PathAccessDecision(
            path=path,
            normalized_path=str(normalized),
            operation=operation,
            allowed=True,
            requires_approval=True,
            reason="dangerous file or directory",
        )

    return PathAccessDecision(
        path=path,
        normalized_path=str(normalized),
        operation=operation,
        allowed=True,
        requires_approval=True,
        reason="workspace mutation requires approval",
    )
