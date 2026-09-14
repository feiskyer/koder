"""Runtime version helpers for harness-owned commands and CLI surfaces."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from koder_agent.version import resolve_package_version_info as resolve_runtime_version_info

PYPI_JSON_URL = "https://pypi.org/pypi/koder/json"
VERSION_CHECK_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours
VERSION_CHECK_ENV_OPT_OUT = "KODER_NO_UPDATE_CHECK"


def resolve_runtime_version() -> str:
    """Return the version of the runtime currently executing this session."""
    return resolve_runtime_version_info()[0]


def render_cli_version_banner() -> str:
    """Render the top-level `-v/--version` banner."""
    return f"{resolve_runtime_version()} (Koder)"


def render_command_version() -> str:
    """Render the interactive `/version` output."""
    resolved, source = resolve_runtime_version_info()
    build_time = os.environ.get("KODER_BUILD_TIME")
    return "\n".join(
        [
            f"version: {resolved}",
            "package: koder",
            f"source: {source}",
            f"build_time: {build_time or 'unavailable'}",
            f"cli_banner: {render_cli_version_banner()}",
        ]
    )


def _version_check_cache_path() -> Path:
    return Path.home() / ".koder" / "version_check.json"


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    """Parse a dotted version into a comparable integer tuple (best-effort)."""
    parts: list[int] = []
    for chunk in version.strip().split("."):
        digits = ""
        for char in chunk:
            if char.isdigit():
                digits += char
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts) or (0,)


def is_newer_version(latest: str, current: str) -> bool:
    """Return True when ``latest`` is strictly newer than ``current``."""
    try:
        return _parse_version_tuple(latest) > _parse_version_tuple(current)
    except Exception:
        return False


def is_update_check_allowed(*, interactive: bool) -> bool:
    """Return True when a startup update check should run.

    The check is opt-in and must never run in CI/headless environments. It is
    only allowed for interactive terminals with the environment opt-out unset.
    """
    if not interactive:
        return False
    if os.environ.get(VERSION_CHECK_ENV_OPT_OUT):
        return False
    if os.environ.get("CI"):
        return False
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    return True


def _read_version_cache(now: Optional[float] = None) -> Optional[dict]:
    now = time.time() if now is None else now
    path = _version_check_cache_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    checked_at = data.get("checked_at")
    if not isinstance(checked_at, (int, float)):
        return None
    if now - checked_at > VERSION_CHECK_CACHE_TTL_SECONDS:
        return None
    return data


def _write_version_cache(latest: str, now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    path = _version_check_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"latest": latest, "checked_at": now}),
            encoding="utf-8",
        )
    except OSError:
        pass


def _fetch_latest_pypi_version(timeout: float = 3.0) -> Optional[str]:
    """Fetch the latest koder version from PyPI (best-effort, network)."""
    import urllib.request

    try:
        with urllib.request.urlopen(PYPI_JSON_URL, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    version = payload.get("info", {}).get("version")
    return version if isinstance(version, str) and version else None


def get_latest_version(*, force: bool = False, now: Optional[float] = None) -> Optional[str]:
    """Return the latest known version, using the cache within its TTL.

    When the cache is fresh it is used directly. Otherwise PyPI is queried and
    the result cached. Returns None when the version cannot be determined.
    """
    if not force:
        cached = _read_version_cache(now=now)
        if cached is not None:
            latest = cached.get("latest")
            if isinstance(latest, str) and latest:
                return latest
    latest = _fetch_latest_pypi_version()
    if latest:
        _write_version_cache(latest, now=now)
    return latest


def check_for_update(*, interactive: bool) -> Optional[str]:
    """Return an upgrade-available message, or None when nothing to report.

    Respects the opt-in/CI/headless gating in :func:`is_update_check_allowed`.
    """
    if not is_update_check_allowed(interactive=interactive):
        return None
    latest = get_latest_version()
    if not latest:
        return None
    current, _source = resolve_runtime_version_info()
    if is_newer_version(latest, current):
        return (
            f"A new version of Koder is available: {latest} (installed: {current}).\n"
            "Run `koder upgrade` to update."
        )
    return None
