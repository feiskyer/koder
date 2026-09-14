"""Task-local execution directories for in-process agents and their tools.

This routes relative paths and subprocess cwd; it is not an OS sandbox.
Permission and sandbox enforcement must use the same directory as execution.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class _ExecutionDirectory:
    cwd: Path
    workspace_root: Path | None


_EXECUTION_CWD: ContextVar[_ExecutionDirectory | None] = ContextVar(
    "koder_execution_cwd", default=None
)


def scoped_execution_cwd() -> Path | None:
    """Return the explicitly scoped directory, or None for the main process."""
    context = _EXECUTION_CWD.get()
    return context.cwd if context is not None else None


def execution_workspace_root() -> Path | None:
    """Return an agent's workspace boundary, leaving main policy roots intact."""
    context = _EXECUTION_CWD.get()
    return context.workspace_root if context is not None else None


def get_execution_cwd() -> Path:
    """Resolve the active directory without changing process-wide state."""
    scoped = scoped_execution_cwd()
    return scoped if scoped is not None else Path.cwd()


def execution_path(path: str | Path) -> Path:
    """Anchor a path without resolving symlinks or erasing traversal syntax."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else get_execution_cwd() / candidate


def set_execution_cwd(cwd: str | Path | None) -> Token:
    directory = (get_execution_cwd() if cwd is None else execution_path(cwd)).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("agent execution directory must be a directory")
    return _EXECUTION_CWD.set(_ExecutionDirectory(cwd=directory, workspace_root=directory))


def reset_execution_cwd(token: Token) -> None:
    _EXECUTION_CWD.reset(token)


@contextmanager
def execution_directory(cwd: str | Path | None) -> Iterator[Path]:
    """Scope an explicit directory across awaits, child tasks and to_thread."""
    token = set_execution_cwd(cwd)
    try:
        yield get_execution_cwd()
    finally:
        reset_execution_cwd(token)


@contextmanager
def capture_tool_directory() -> Iterator[Path]:
    """Keep permission evaluation and invocation on one directory snapshot.

    Main-session commands may run in a subdirectory of their allowed workspace.
    Capturing their cwd must not silently narrow or expand that allowed workspace.
    """
    if _EXECUTION_CWD.get() is not None:
        yield get_execution_cwd()
        return
    directory = Path.cwd().resolve()
    token = _EXECUTION_CWD.set(_ExecutionDirectory(cwd=directory, workspace_root=None))
    try:
        yield directory
    finally:
        _EXECUTION_CWD.reset(token)
