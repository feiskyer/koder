"""Local settings bundle import and export."""

from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml

from koder_agent.config.manager import _migrate_legacy_voice_fields
from koder_agent.harness.hooks.runtime import dispatch_command_hooks, snapshot_command_hooks
from koder_agent.utils.atomic_file import write_text_atomic

from .schema import parse_runtime_config_source
from .service import config_write_lock, read_config_text

SettingsBundleScope = Literal["all", "user", "project"]

BUNDLE_FORMAT = "koder-settings-bundle"
BUNDLE_VERSION = 1
MAX_BUNDLE_FILE_BYTES = 500 * 1024
# ConfigChange matches the settings source, not the bundle role or path ancestry.
# Memory documents are not configuration changes. Keybindings share user scope.
_CONFIG_CHANGE_SOURCES = {
    "user_config": "user_settings",
    "user_settings": "user_settings",
    "user_keybindings": "user_settings",
    "project_settings": "project_settings",
    "project_local_settings": "local_settings",
}


@dataclass(frozen=True)
class SettingsBundleExportResult:
    """Summary of a settings bundle export."""

    bundle_path: Path
    file_count: int
    skipped: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SettingsBundleImportResult:
    """Summary of a settings bundle import."""

    bundle_path: Path
    written: int
    unchanged: int
    backups: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    dry_run: bool = False


def export_settings_bundle(
    bundle_path: str | Path,
    *,
    scope: SettingsBundleScope = "all",
    cwd: str | Path | None = None,
    home: str | Path | None = None,
) -> SettingsBundleExportResult:
    """Export local Koder settings and memory files into a JSON bundle."""
    if scope not in {"all", "user", "project"}:
        raise ValueError("scope must be one of: all, user, project")

    home_dir = Path(home).expanduser() if home is not None else Path.home()
    cwd_dir = Path(cwd).resolve() if cwd is not None else Path.cwd()
    target = Path(bundle_path).expanduser()
    files: list[dict] = []
    skipped: list[str] = []

    for role, file_scope, path in _known_direct_files(home_dir, cwd_dir):
        if not _scope_included(file_scope, scope):
            continue
        _append_file_entry(
            files,
            skipped,
            role=role,
            file_scope=file_scope,
            path=path,
            relative_path=path.name,
            root=(home_dir if file_scope == "user" else cwd_dir) / ".koder",
        )

    for role, file_scope, base in _known_directory_files(home_dir, cwd_dir):
        if not _scope_included(file_scope, scope) or not base.exists():
            continue
        root = (home_dir if file_scope == "user" else cwd_dir) / ".koder"
        if _has_symlink(base, root):
            skipped.append(f"{base}: symlink skipped")
            continue
        for path in sorted(base.rglob("*")):
            if path.is_dir():
                continue
            try:
                relative_path = path.relative_to(base).as_posix()
            except ValueError:
                skipped.append(f"{path}: outside base directory")
                continue
            _append_file_entry(
                files,
                skipped,
                role=role,
                file_scope=file_scope,
                path=path,
                relative_path=relative_path,
                root=root,
            )

    payload = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "files": files,
    }
    if target.exists():
        target.chmod(0o600)
    write_text_atomic(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return SettingsBundleExportResult(bundle_path=target, file_count=len(files), skipped=skipped)


def import_settings_bundle(
    bundle_path: str | Path,
    *,
    scope: SettingsBundleScope = "all",
    cwd: str | Path | None = None,
    home: str | Path | None = None,
    dry_run: bool = False,
) -> SettingsBundleImportResult:
    """Import a JSON settings bundle into the current Koder home and project."""
    if scope not in {"all", "user", "project"}:
        raise ValueError("scope must be one of: all, user, project")

    source = Path(bundle_path).expanduser()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("format") != BUNDLE_FORMAT
        or payload.get("version") != BUNDLE_VERSION
        or not isinstance(payload.get("files"), list)
    ):
        raise ValueError("Unsupported Koder settings bundle format")

    home_dir = Path(home).expanduser() if home is not None else Path.home()
    cwd_dir = Path(cwd).resolve() if cwd is not None else Path.cwd()
    written = 0
    unchanged = 0
    backups: list[Path] = []
    skipped: list[str] = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    roles = {
        role: file_scope
        for role, file_scope, _ in (
            _known_direct_files(home_dir, cwd_dir) + _known_directory_files(home_dir, cwd_dir)
        )
    }
    changes: list[tuple[Path, str, str | None]] = []
    change_sources: dict[Path, str] = {}
    targets: set[Path] = set()

    # Validate the entire selected bundle before publishing any file or backup.
    for entry in payload["files"]:
        if not isinstance(entry, dict):
            raise ValueError("Invalid settings bundle file entry")
        role = entry.get("role")
        file_scope = entry.get("scope")
        if not isinstance(role, str) or role not in roles:
            raise ValueError("Unknown settings bundle role")
        if file_scope != roles[role]:
            raise ValueError(f"Invalid scope for settings bundle role: {role}")
        if not _scope_included(file_scope, scope):
            continue
        content = entry.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("Invalid settings bundle file entry")
        if len(content.encode("utf-8")) > MAX_BUNDLE_FILE_BYTES:
            raise ValueError(f"Settings bundle entry exceeds size limit: {role}")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest != entry.get("sha256"):
            raise ValueError(f"Checksum mismatch for bundle entry {role}")
        _validate_content(role, content)
        target = _target_path_for_entry(role, entry.get("relative_path"), home_dir, cwd_dir)
        if target is None:
            raise ValueError(f"Unknown settings bundle role: {role}")
        root = (home_dir if file_scope == "user" else cwd_dir) / ".koder"
        if _has_symlink(target, root):
            raise ValueError(f"Refusing to import over symlink target: {target}")
        identity = target.resolve()
        if identity in targets:
            raise ValueError(f"Duplicate settings bundle target: {target}")
        targets.add(identity)
        existing = read_config_text(target)
        if existing == content:
            unchanged += 1
            continue
        changes.append((target, content, existing))
        if role in _CONFIG_CHANGE_SOURCES:
            change_sources[target] = _CONFIG_CHANGE_SOURCES[role]
        written += 1

    if not dry_run:
        with ExitStack() as locks:
            # A stable lock order prevents opposite bundle orders deadlocking.
            # No lock/backup is created until whole-bundle validation succeeds.
            for target in sorted(target.resolve() for target, _, _ in changes):
                locks.enter_context(config_write_lock(target))
            for target, _, existing in changes:
                if read_config_text(target) != existing:
                    raise RuntimeError(f"Config changed during import; retry: {target}")
            # Retain pre-import definitions (including disable flags and project
            # trust payloads) before replacing any hook-bearing settings file.
            hook_snapshot = (
                locks.enter_context(
                    snapshot_command_hooks(cwd=cwd_dir, home=home_dir if home is not None else None)
                )
                if change_sources
                else None
            )
            applied: list[tuple[Path, str, str | None]] = []
            try:
                for target, content, existing in changes:
                    if existing is not None:
                        backup = _backup_path(target, stamp)
                        if backup.is_symlink():
                            raise ValueError(f"Refusing config backup symlink: {backup}")
                        write_text_atomic(backup, existing)
                        backups.append(backup)
                    write_text_atomic(target, content)
                    applied.append((target, content, existing))
                # Match ordinary config saves: hooks inspect already-published
                # candidates. Publish the whole bundle before any decision so
                # each hook sees the same complete proposed configuration.
                for target, change_source in change_sources.items():
                    result = dispatch_command_hooks(
                        cwd=cwd_dir,
                        event_name="ConfigChange",
                        match_value=change_source,
                        payload={
                            "event": "ConfigChange",
                            "source": change_source,
                            "file_path": str(target.resolve()),
                        },
                        snapshot=hook_snapshot,
                    )
                    if result.blocked:
                        raise RuntimeError(result.block_reason or "Config change blocked by hook")
                for target, content, _ in applied:
                    if read_config_text(target) != content:
                        raise RuntimeError(f"Config changed during ConfigChange hook: {target}")
            except BaseException as failure:
                rollback_errors = []
                for target, content, existing in reversed(applied):
                    try:
                        if read_config_text(target) != content:
                            raise RuntimeError("Newer external file preserved")
                        if existing is None:
                            target.unlink(missing_ok=True)
                        else:
                            write_text_atomic(target, existing)
                    except BaseException:
                        rollback_errors.append(str(target))
                if rollback_errors:
                    raise OSError(
                        "Settings import failed and rollback was incomplete for: "
                        + ", ".join(rollback_errors)
                    ) from failure
                raise

    return SettingsBundleImportResult(
        bundle_path=source,
        written=written,
        unchanged=unchanged,
        backups=backups,
        skipped=skipped,
        dry_run=dry_run,
    )


def _known_direct_files(home_dir: Path, cwd_dir: Path) -> list[tuple[str, str, Path]]:
    return [
        ("user_config", "user", home_dir / ".koder" / "config.yaml"),
        ("user_settings", "user", home_dir / ".koder" / "settings.json"),
        ("user_keybindings", "user", home_dir / ".koder" / "keybindings.json"),
        ("project_settings", "project", cwd_dir / ".koder" / "settings.json"),
        ("project_local_settings", "project", cwd_dir / ".koder" / "settings.local.json"),
    ]


def _known_directory_files(home_dir: Path, cwd_dir: Path) -> list[tuple[str, str, Path]]:
    return [
        ("user_memory", "user", home_dir / ".koder" / "memory"),
        ("project_memory", "project", cwd_dir / ".koder" / "memory"),
        ("project_session_memory", "project", cwd_dir / ".koder" / "session-memory"),
    ]


def _scope_included(file_scope: str, requested_scope: SettingsBundleScope) -> bool:
    return requested_scope == "all" or requested_scope == file_scope


def _append_file_entry(
    files: list[dict],
    skipped: list[str],
    *,
    role: str,
    file_scope: str,
    path: Path,
    relative_path: str,
    root: Path,
) -> None:
    if not path.exists():
        return
    if _has_symlink(path, root):
        skipped.append(f"{path}: symlink skipped")
        return
    if path.stat().st_size > MAX_BUNDLE_FILE_BYTES:
        skipped.append(f"{path}: exceeds {MAX_BUNDLE_FILE_BYTES} bytes")
        return
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        skipped.append(f"{path}: not UTF-8 text")
        return
    files.append(
        {
            "role": role,
            "scope": file_scope,
            "relative_path": relative_path,
            "size": len(content.encode("utf-8")),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "content": content,
        }
    )


def _target_path_for_entry(
    role: str,
    relative_path: object,
    home_dir: Path,
    cwd_dir: Path,
) -> Path | None:
    direct_targets = {
        "user_config": home_dir / ".koder" / "config.yaml",
        "user_settings": home_dir / ".koder" / "settings.json",
        "user_keybindings": home_dir / ".koder" / "keybindings.json",
        "project_settings": cwd_dir / ".koder" / "settings.json",
        "project_local_settings": cwd_dir / ".koder" / "settings.local.json",
    }
    if role in direct_targets:
        return direct_targets[role]

    directory_targets = {
        "user_memory": home_dir / ".koder" / "memory",
        "project_memory": cwd_dir / ".koder" / "memory",
        "project_session_memory": cwd_dir / ".koder" / "session-memory",
    }
    if role not in directory_targets or not isinstance(relative_path, str):
        return None
    safe_relative = _safe_relative_path(relative_path)
    return directory_targets[role] / safe_relative


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe bundle relative path: {value}")
    return path


def _validate_content(role: str, content: str) -> None:
    if role == "user_config":
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as error:
            raise ValueError("Invalid YAML in settings bundle user_config") from error
        parse_runtime_config_source(_migrate_legacy_voice_fields({} if data is None else data))
        return
    if role.endswith("settings") or role == "user_keybindings":
        if not isinstance(json.loads(content), dict):
            raise ValueError(f"Expected a JSON object for settings bundle role: {role}")


def _has_symlink(path: Path, root: Path) -> bool:
    """Check the file and all parents within the selected profile boundary."""
    while path != root.parent:
        if path.is_symlink():
            return True
        path = path.parent
    return False


def _backup_path(target: Path, stamp: str) -> Path:
    candidate = target.with_name(f"{target.name}.bak-{stamp}")
    index = 1
    while candidate.exists():
        candidate = target.with_name(f"{target.name}.bak-{stamp}-{index}")
        index += 1
    return candidate
