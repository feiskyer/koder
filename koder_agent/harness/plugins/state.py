"""Plugin state persistence (enabled/disabled/scope)."""

from __future__ import annotations

import errno
import json
import os
import re
import secrets
import stat
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .name_validation import canonical_marketplace_name, canonical_plugin_name
from .path_safety import _open_directory_no_symlinks


@dataclass(frozen=True)
class PluginInstallOrigin:
    """Installer-owned catalog identity; never taken from a plugin manifest."""

    marketplace: str
    source_digest: str

    def __post_init__(self) -> None:
        canonical, _reason = canonical_marketplace_name(self.marketplace)
        if canonical is None or canonical != self.marketplace:
            raise ValueError("Invalid installation marketplace")
        if not isinstance(self.source_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.source_digest
        ):
            raise ValueError("Invalid installation source digest")

    @classmethod
    def from_record(cls, data: object) -> "PluginInstallOrigin | None":
        if not isinstance(data, dict):
            return None
        try:
            return cls(marketplace=data.get("marketplace"), source_digest=data.get("source_digest"))
        except (TypeError, ValueError):
            # Malformed/legacy provenance confers no channel grant. Ordinary
            # installed plugin state can still be read and explicitly repaired.
            return None


@dataclass
class PluginState:
    """Tracks the runtime state of an installed plugin."""

    enabled: bool = True
    scope: str = "user"
    installed_at: str = ""
    origin: PluginInstallOrigin | None = None

    def __post_init__(self) -> None:
        if not self.installed_at:
            self.installed_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_record(cls, entry: dict) -> "PluginState":
        return cls(
            enabled=entry.get("enabled", True),
            scope=entry.get("scope", "user"),
            installed_at=entry.get("installed_at", ""),
            origin=PluginInstallOrigin.from_record(entry.get("origin")),
        )


class PluginStateStore:
    """Read and atomically write private state through a pinned parent fd."""

    def __init__(self, state_path: Path, *, root_fd: int | None = None):
        self._root = Path(os.path.abspath(os.fspath(state_path.parent.expanduser())))
        self._name = state_path.name
        if root_fd is None:
            self._root, self._root_fd = _open_directory_no_symlinks(
                state_path.parent,
                create=True,
            )
        else:
            self._root_fd = root_fd
        root_stat = os.fstat(self._root_fd)
        self._root_identity = (root_stat.st_dev, root_stat.st_ino)
        self._path = self._root / self._name

    @classmethod
    def for_test(cls, root: Path) -> "PluginStateStore":
        return cls(root / "state.json")

    def close(self) -> None:
        if self._root_fd >= 0:
            os.close(self._root_fd)
            self._root_fd = -1

    def __del__(self) -> None:  # pragma: no cover - best-effort descriptor cleanup
        try:
            if hasattr(self, "_root_fd"):
                self.close()
        except OSError:
            pass

    def _verify_root_identity(self) -> None:
        root_stat = os.fstat(self._root_fd)
        if (root_stat.st_dev, root_stat.st_ino) != self._root_identity:
            raise OSError(f"Plugin state root descriptor identity changed: {self._root}")

    def _entry_mode(self) -> int | None:
        try:
            return os.stat(self._name, dir_fd=self._root_fd, follow_symlinks=False).st_mode
        except FileNotFoundError:
            return None

    @staticmethod
    def _migrate_names(
        data: dict[str, dict],
    ) -> tuple[dict[str, dict], dict[str, object], bool]:
        migrated: dict[str, dict] = {}
        quarantined: dict[str, object] = {}
        changed = False

        # Preserve already-canonical entries first. A legacy spelling must
        # never overwrite a canonical record with different state.
        for name, entry in data.items():
            canonical_name, _ = canonical_plugin_name(name)
            if canonical_name is not None and isinstance(entry, dict):
                migrated[canonical_name] = entry
            elif canonical_name is not None:
                quarantined[name] = entry
                changed = True

        for name, entry in data.items():
            canonical_name, _ = canonical_plugin_name(name)
            if canonical_name is not None:
                continue
            lowered = name.lower() if isinstance(name, str) else ""
            migrated_name, _lower_error = canonical_plugin_name(lowered)
            if not isinstance(entry, dict) or migrated_name is None:
                quarantined[str(name)] = entry
                changed = True
                continue
            if migrated_name in migrated:
                quarantined[name] = entry
                changed = True
                continue
            migrated[migrated_name] = entry
            changed = True

        return migrated, quarantined, changed

    def _load(self) -> dict[str, dict]:
        self._verify_root_identity()
        mode = self._entry_mode()
        if mode is None:
            return {}
        if stat.S_ISLNK(mode):
            raise OSError(f"Refusing to read symlinked plugin state file: {self._path}")
        if not stat.S_ISREG(mode):
            raise OSError(f"Refusing to read non-regular plugin state file: {self._path}")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(self._name, flags, dir_fd=self._root_fd)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise OSError(
                    f"Refusing to read symlinked plugin state file: {self._path}"
                ) from exc
            raise
        try:
            with os.fdopen(descriptor, "r", encoding="utf-8") as state_file:
                descriptor = -1
                try:
                    data = json.load(state_file)
                except json.JSONDecodeError:
                    return {}
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if not isinstance(data, dict):
            return {}
        migrated, quarantined, changed = self._migrate_names(data)
        if changed:
            if quarantined:
                self._save_named("state.legacy.json", quarantined)
            self._save(migrated)
        return migrated

    def _save_named(self, name: str, data: dict[str, object]) -> None:
        self._verify_root_identity()
        try:
            mode = os.stat(name, dir_fd=self._root_fd, follow_symlinks=False).st_mode
        except FileNotFoundError:
            mode = None
        if mode is not None and stat.S_ISLNK(mode):
            raise OSError(f"Refusing to replace symlinked plugin state file: {self._root / name}")
        if mode is not None and not stat.S_ISREG(mode):
            raise OSError(f"Refusing to replace non-regular plugin state file: {self._root / name}")

        temporary_name = f".{name}-{secrets.token_hex(12)}.tmp"
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=self._root_fd,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as state_file:
                descriptor = -1
                json.dump(data, state_file, indent=2, sort_keys=True)
                state_file.write("\n")
                state_file.flush()
                os.fsync(state_file.fileno())

            # Replacing a raced-in symlink is safe because renameat replaces the
            # directory entry itself; it never opens or follows the link target.
            os.replace(
                temporary_name,
                name,
                src_dir_fd=self._root_fd,
                dst_dir_fd=self._root_fd,
            )
            os.fsync(self._root_fd)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=self._root_fd)
            except FileNotFoundError:
                pass

    def _save(self, data: dict[str, dict]) -> None:
        self._save_named(self._name, data)

    @staticmethod
    def _validated_name(name: object) -> str:
        canonical_name, error = canonical_plugin_name(name)
        if canonical_name is None:
            raise ValueError(f"Invalid plugin state name {name!r}: {error}")
        return canonical_name

    def get(self, name: str) -> PluginState | None:
        name = self._validated_name(name)
        entry = self._load().get(name)
        if entry is None:
            return None
        return PluginState.from_record(entry)

    def set(self, name: str, state: PluginState) -> None:
        name = self._validated_name(name)
        data = self._load()
        data[name] = asdict(state)
        self._save(data)

    def remove(self, name: str) -> bool:
        name = self._validated_name(name)
        data = self._load()
        if name not in data:
            return False
        del data[name]
        self._save(data)
        return True

    def is_enabled(self, name: str) -> bool:
        state = self.get(name)
        return True if state is None else state.enabled

    def list_all(self) -> dict[str, PluginState]:
        return {name: PluginState.from_record(entry) for name, entry in self._load().items()}
