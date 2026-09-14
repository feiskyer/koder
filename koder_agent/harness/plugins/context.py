"""Task-local plugin roots shared by runtime discovery and tool execution."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Iterator

from koder_agent.harness.execution_context import execution_path
from koder_agent.harness.paths import harness_home_dir

_PLUGIN_ROOT: ContextVar[Path | None] = ContextVar("koder_plugin_root", default=None)


def get_plugin_root() -> Path:
    root = _PLUGIN_ROOT.get()
    return root if root is not None else harness_home_dir() / "plugins"


def normalize_plugin_root(root: str | Path) -> Path:
    # Preserve symlink components for the plugin filesystem guards to reject.
    return Path(os.path.abspath(os.fspath(execution_path(root))))


def set_plugin_root(root: str | Path | None) -> Token:
    return _PLUGIN_ROOT.set(normalize_plugin_root(get_plugin_root() if root is None else root))


def reset_plugin_root(token: Token) -> None:
    _PLUGIN_ROOT.reset(token)


@contextmanager
def plugin_root_scope(root: str | Path | None) -> Iterator[Path]:
    token = set_plugin_root(root)
    try:
        yield get_plugin_root()
    finally:
        reset_plugin_root(token)
