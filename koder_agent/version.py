"""Lightweight version resolution shared by package and CLI surfaces."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

_SOURCE_PROJECT_FILE = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _source_tree_version() -> str | None:
    try:
        try:
            import tomllib
        except ModuleNotFoundError:
            # Already present in the Python 3.10 development environment.
            # Installed packages use distribution metadata and need no parser.
            import tomli as tomllib
        document = tomllib.loads(_SOURCE_PROJECT_FILE.read_text(encoding="utf-8"))
    except (ImportError, OSError, UnicodeError, ValueError):
        return None
    project = document.get("project", {})
    if not isinstance(project, dict) or project.get("name") != "koder":
        return None
    value = project.get("version")
    return value.strip() if isinstance(value, str) and value.strip() else None


def resolve_package_version_info() -> tuple[str, str]:
    """Use installed metadata, then this checkout's static project metadata."""
    try:
        installed = package_version("koder")
    except PackageNotFoundError:
        installed = None
    if installed:
        return installed, "installed-package"
    source_version = _source_tree_version()
    if source_version:
        return source_version, "source-tree"
    return "unknown", "unavailable"
