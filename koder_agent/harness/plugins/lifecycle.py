"""Local plugin lifecycle service with atomic replacement and crash recovery."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .manifest import PluginManifest, find_manifest, parse_manifest
from .name_validation import canonical_marketplace_name
from .path_safety import (
    PinnedDirectory,
    PluginPathError,
    PluginRootGuard,
    copy_tree_without_links,
)
from .state import PluginInstallOrigin, PluginState, PluginStateStore

_JOURNAL_NAME = ".koder-lifecycle-transaction.json"
_INTERNAL_NAME = re.compile(r"^\.koder-(?:stage|backup)-[0-9a-f]{24}$")
_NON_PLUGIN_ENTRIES = {
    "state.json",
    "state.legacy.json",
    "marketplaces.json",
    "marketplace-cache",
}


@dataclass(frozen=True)
class PluginLifecycleResult:
    success: bool
    rollback_performed: bool
    plugin_name: str | None = None
    message: str = ""


class PluginLifecycleService:
    """Install plugins through a pinned root with journaled rollback."""

    def __init__(self, root: Path, *, state_store: PluginStateStore | None = None):
        self._paths = PluginRootGuard(root)
        self.root = self._paths.root
        self._state = state_store or PluginStateStore(
            self.root / "state.json",
            root_fd=self._paths.dup_fd(),
        )
        self._recover_incomplete_transaction()
        self._migrate_legacy_installations()

    @classmethod
    def for_test(cls, root: Path) -> "PluginLifecycleService":
        return cls(root / "installed_plugins")

    @property
    def state_store(self) -> PluginStateStore:
        return self._state

    def resolve_plugin_target(self, plugin_name: object) -> Path:
        """Resolve a canonical direct-child target without following symlinks."""
        return self._paths.target(plugin_name)

    @staticmethod
    def _result_name(name: object) -> str | None:
        return name if isinstance(name, str) else None

    def _target_result(
        self, plugin_name: object
    ) -> tuple[Path | None, PluginLifecycleResult | None]:
        try:
            return self.resolve_plugin_target(plugin_name), None
        except PluginPathError as exc:
            return None, PluginLifecycleResult(
                success=False,
                rollback_performed=False,
                plugin_name=self._result_name(plugin_name),
                message=str(exc),
            )

    def _new_internal_name(self, purpose: str) -> str:
        temporary = self._paths.staging_dir("temporary", purpose=purpose)
        name = temporary.name
        temporary.close()
        self._paths.remove_entry_name(name)
        return name

    @staticmethod
    def _state_data(state: PluginState | None) -> dict[str, Any] | None:
        return None if state is None else asdict(state)

    @staticmethod
    def _state_from_data(data: object) -> PluginState | None:
        if data is None:
            return None
        if not isinstance(data, dict):
            raise PluginPathError("Lifecycle journal contains invalid plugin state")
        return PluginState.from_record(data)

    def _write_journal(self, journal: dict[str, Any], phase: str) -> None:
        journal["phase"] = phase
        self._paths.write_json_atomic(_JOURNAL_NAME, journal)

    def _validate_journal(self, journal: dict[str, Any]) -> tuple[str, str, str | None, str, str]:
        operation = journal.get("operation")
        plugin_name = journal.get("plugin_name")
        stage_name = journal.get("stage_name")
        backup_name = journal.get("backup_name")
        legacy_name = journal.get("legacy_name")
        if operation not in {"install", "uninstall"}:
            raise PluginPathError("Lifecycle journal contains an invalid operation")
        target = self._paths.target(plugin_name, reject_symlink=False)
        if stage_name is not None and (
            not isinstance(stage_name, str) or not _INTERNAL_NAME.fullmatch(stage_name)
        ):
            raise PluginPathError("Lifecycle journal contains an invalid staging name")
        if backup_name is not None and (
            not isinstance(backup_name, str) or not _INTERNAL_NAME.fullmatch(backup_name)
        ):
            raise PluginPathError("Lifecycle journal contains an invalid backup name")
        restore_name = target.name
        if legacy_name is not None:
            migrated_name, error = canonical_marketplace_name(legacy_name)
            if migrated_name != target.name:
                raise PluginPathError(f"Lifecycle journal contains an invalid legacy name: {error}")
            restore_name = legacy_name
        return operation, target.name, backup_name, str(journal.get("phase", "")), restore_name

    def _restore_state(self, plugin_name: str, previous_state: PluginState | None) -> None:
        if previous_state is None:
            self._state.remove(plugin_name)
        else:
            self._state.set(plugin_name, previous_state)

    def _recover_incomplete_transaction(self) -> None:
        journal = self._paths.read_json(_JOURNAL_NAME)
        if journal is None:
            return
        operation, plugin_name, backup_name, phase, restore_name = self._validate_journal(journal)
        stage_name = journal.get("stage_name")
        previous_state = self._state_from_data(journal.get("previous_state"))
        new_state = self._state_from_data(journal.get("new_state"))
        target_existed = bool(journal.get("target_existed", previous_state is not None))
        target_exists = self._paths.entry_exists(plugin_name)
        backup_exists = bool(backup_name and self._paths.entry_exists(backup_name))

        if operation == "install":
            if phase == "state_written" and target_exists:
                if new_state is not None:
                    self._state.set(plugin_name, new_state)
            elif phase == "state_written" and not backup_exists:
                self._state.remove(plugin_name)
            elif phase == "target_published" and target_exists and not backup_exists:
                # The old payload backup was already removed. Keep the complete
                # published target and finish its registry state rather than
                # deleting the only surviving payload.
                if new_state is not None:
                    self._state.set(plugin_name, new_state)
            else:
                if target_exists and (
                    phase in {"target_published", "state_written", "backup_moved"}
                    or not target_existed
                ):
                    self._paths.remove_entry_name(plugin_name)
                if backup_exists and backup_name is not None:
                    if self._paths.entry_exists(plugin_name):
                        self._paths.remove_entry_name(plugin_name)
                    self._paths.replace(backup_name, restore_name)
                self._restore_state(plugin_name, previous_state)
        elif phase == "state_removed":
            # State removal is the uninstall commit point. Finish deleting any
            # orphan target/backup entries left by an interrupted cleanup.
            if target_exists:
                self._paths.remove_entry_name(plugin_name)
            if backup_exists and backup_name is not None:
                self._paths.remove_entry_name(backup_name)
            self._state.remove(plugin_name)
        else:
            if backup_exists and backup_name is not None:
                if target_exists:
                    self._paths.remove_entry_name(plugin_name)
                self._paths.replace(backup_name, restore_name)
                target_exists = True
            if target_exists:
                self._restore_state(plugin_name, previous_state)
            else:
                self._state.remove(plugin_name)

        if isinstance(stage_name, str) and self._paths.entry_exists(stage_name):
            self._paths.remove_entry_name(stage_name)
        if backup_name and self._paths.entry_exists(backup_name):
            self._paths.remove_entry_name(backup_name)
        self._paths.unlink(_JOURNAL_NAME)

    def _migrate_legacy_installations(self) -> None:
        """Canonicalize unambiguous mixed-case installations transactionally."""
        self._state.list_all()  # migrate/quarantine legacy state before payloads
        names = [name for name in self._paths.list_entries() if not name.startswith(".")]
        folded: dict[str, list[str]] = {}
        for name in names:
            folded.setdefault(name.casefold(), []).append(name)

        for legacy_name in names:
            canonical_name = legacy_name.lower()
            if legacy_name == canonical_name or len(folded[legacy_name.casefold()]) != 1:
                continue
            try:
                target = self._paths.target(canonical_name, reject_symlink=False)
            except PluginPathError:
                continue
            if self._paths.entry_is_symlink(legacy_name):
                continue

            staging: PinnedDirectory | None = None
            try:
                staging = self._paths.staging_dir(canonical_name)
                copy_tree_without_links(self.root / legacy_name, staging.fd)
                manifest_path = find_manifest(staging.path)
                if manifest_path is None:
                    continue
                raw = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict) or raw.get("name") != legacy_name:
                    continue
                raw["name"] = canonical_name
                manifest_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
                migrated_manifest, errors, _warnings = parse_manifest(staging.path)
                if migrated_manifest is None or errors or migrated_manifest.name != target.name:
                    continue

                previous_state = self._state.get(canonical_name)
                migrated_state = previous_state or PluginState()
                backup_name = self._new_internal_name("backup")
                journal = {
                    "operation": "install",
                    "plugin_name": canonical_name,
                    "stage_name": staging.name,
                    "backup_name": backup_name,
                    "previous_state": self._state_data(previous_state),
                    "new_state": self._state_data(migrated_state),
                    "legacy_name": legacy_name,
                    "target_existed": True,
                }
                self._write_journal(journal, "prepared")
                self._paths.replace(legacy_name, backup_name)
                self._write_journal(journal, "backup_moved")
                self._paths.replace(staging.name, canonical_name)
                self._write_journal(journal, "target_published")
                self._state.set(canonical_name, migrated_state)
                self._write_journal(journal, "state_written")
                self._paths.remove_entry_name(backup_name)
                self._paths.unlink(_JOURNAL_NAME)
            except (OSError, ValueError, PluginPathError, json.JSONDecodeError):
                self._recover_incomplete_transaction()
            finally:
                if staging is not None:
                    staging.close()
                    if self._paths.entry_exists(staging.name):
                        self._paths.remove_entry_name(staging.name)

    def install_from_dir(
        self,
        plugin_dir: Path,
        *,
        scope: str = "user",
        origin: PluginInstallOrigin | None = None,
        expected_name: str | None = None,
    ) -> PluginLifecycleResult:
        """Parse and install a plugin from a symlink-free local directory."""
        manifest, errors, _warnings = parse_manifest(plugin_dir)
        if manifest is None or errors:
            return PluginLifecycleResult(
                success=False,
                rollback_performed=False,
                message="; ".join(errors) if errors else "Invalid manifest",
            )
        if expected_name is not None and manifest.name != expected_name:
            return PluginLifecycleResult(
                success=False,
                rollback_performed=False,
                message=f"Selected plugin identity changed from '{expected_name}' to '{manifest.name}'",
            )
        return self.install_from_manifest(
            plugin_dir,
            manifest,
            state=PluginState(enabled=True, scope=scope, origin=origin),
        )

    def install_from_manifest(
        self,
        plugin_dir: Path,
        manifest: PluginManifest,
        *,
        state: PluginState,
    ) -> PluginLifecycleResult:
        """Install an untrusted parsed manifest using a private staged copy."""
        self._recover_incomplete_transaction()
        target_dir, invalid_result = self._target_result(manifest.name)
        if target_dir is None:
            assert invalid_result is not None
            return invalid_result
        plugin_name = target_dir.name

        try:
            previous_state = self._state.get(plugin_name)
        except Exception as exc:
            return PluginLifecycleResult(
                success=False,
                rollback_performed=False,
                plugin_name=plugin_name,
                message=str(exc),
            )

        staging: PinnedDirectory | None = None
        journal: dict[str, Any] | None = None
        phase = ""
        try:
            staging = self._paths.staging_dir(plugin_name)
            copy_tree_without_links(plugin_dir, staging.fd)

            staged_manifest, staged_errors, _warnings = parse_manifest(staging.path)
            if staged_manifest is None or staged_errors:
                reason = "; ".join(staged_errors) if staged_errors else "Invalid staged manifest"
                raise ValueError(f"Staged plugin validation failed: {reason}")
            staged_target = self.resolve_plugin_target(staged_manifest.name)
            if staged_target.name != plugin_name:
                raise ValueError(
                    f"Staged plugin identity changed from '{plugin_name}' to '{staged_target.name}'"
                )

            backup_name = self._new_internal_name("backup")
            target_existed = self._paths.entry_exists(plugin_name)
            journal = {
                "operation": "install",
                "plugin_name": plugin_name,
                "stage_name": staging.name,
                "backup_name": backup_name,
                "previous_state": self._state_data(previous_state),
                "new_state": self._state_data(state),
                "target_existed": target_existed,
            }
            self._write_journal(journal, "prepared")
            phase = "prepared"

            if target_existed:
                self._paths.replace(plugin_name, backup_name)
                self._write_journal(journal, "backup_moved")
                phase = "backup_moved"

            self._paths.replace(staging.name, plugin_name)
            self._write_journal(journal, "target_published")
            phase = "target_published"
            self._state.set(plugin_name, state)
            self._write_journal(journal, "state_written")
            phase = "state_written"

            if self._paths.entry_exists(backup_name):
                self._paths.remove_entry_name(backup_name)
            self._paths.unlink(_JOURNAL_NAME)
            journal = None
            return PluginLifecycleResult(
                success=True,
                rollback_performed=False,
                plugin_name=plugin_name,
                message=f"Installed {plugin_name}@{staged_manifest.version}",
            )
        except Exception as exc:
            rollback_errors: list[str] = []
            rollback_performed = False
            if journal is not None:
                backup_name = journal["backup_name"]
                backup_exists = self._paths.entry_exists(backup_name)
                if phase == "state_written" and not backup_exists:
                    try:
                        committed = self._paths.entry_exists(plugin_name) and (
                            self._state.get(plugin_name) == state
                        )
                    except Exception as verify_exc:
                        return PluginLifecycleResult(
                            success=False,
                            rollback_performed=False,
                            plugin_name=plugin_name,
                            message=(
                                f"{exc}; commit verification failed: {verify_exc}; "
                                "transaction journal retained"
                            ),
                        )
                    if committed:
                        try:
                            self._paths.unlink(_JOURNAL_NAME)
                        except Exception as journal_exc:
                            return PluginLifecycleResult(
                                success=False,
                                rollback_performed=False,
                                plugin_name=plugin_name,
                                message=(
                                    f"Installed {plugin_name}@{manifest.version}, but cleanup "
                                    f"reported: {exc}; journal cleanup failed: {journal_exc}"
                                ),
                            )
                        journal = None
                        return PluginLifecycleResult(
                            success=True,
                            rollback_performed=False,
                            plugin_name=plugin_name,
                            message=(
                                f"Installed {plugin_name}@{manifest.version}; "
                                f"backup cleanup reported after deletion: {exc}"
                            ),
                        )
                    return PluginLifecycleResult(
                        success=False,
                        rollback_performed=False,
                        plugin_name=plugin_name,
                        message=(
                            f"{exc}; committed upgrade postconditions could not be verified; "
                            "transaction journal retained"
                        ),
                    )
                try:
                    if self._paths.entry_exists(plugin_name) and (
                        phase in {"target_published", "state_written", "backup_moved"}
                        or not journal.get("target_existed", False)
                    ):
                        self._paths.remove_entry_name(plugin_name)
                    if self._paths.entry_exists(backup_name):
                        self._paths.replace(backup_name, plugin_name)
                    self._restore_state(plugin_name, previous_state)
                    rollback_performed = True
                except Exception as rollback_exc:
                    rollback_errors.append(f"rollback failed: {rollback_exc}")
                if not rollback_errors:
                    self._paths.unlink(_JOURNAL_NAME)
                    journal = None
            message = str(exc)
            if rollback_errors:
                message = f"{message}; {'; '.join(rollback_errors)}"
            return PluginLifecycleResult(
                success=False,
                rollback_performed=rollback_performed and not rollback_errors,
                plugin_name=plugin_name,
                message=message,
            )
        finally:
            if staging is not None:
                staging.close()
                if journal is None and self._paths.entry_exists(staging.name):
                    try:
                        self._paths.remove_entry_name(staging.name)
                    except Exception:
                        pass

    def uninstall(self, plugin_name: object) -> PluginLifecycleResult:
        """Uninstall a canonical plugin, rolling back filesystem and state together."""
        self._recover_incomplete_transaction()
        target_dir, invalid_result = self._target_result(plugin_name)
        if target_dir is None:
            assert invalid_result is not None
            return invalid_result
        canonical_name = target_dir.name
        if not self._paths.entry_exists(canonical_name):
            return PluginLifecycleResult(
                success=False,
                rollback_performed=False,
                plugin_name=canonical_name,
                message=f"Plugin '{canonical_name}' is not installed",
            )

        journal: dict[str, Any] | None = None
        previous_state: PluginState | None = None
        phase = ""
        try:
            previous_state = self._state.get(canonical_name)
            backup_name = self._new_internal_name("backup")
            journal = {
                "operation": "uninstall",
                "plugin_name": canonical_name,
                "stage_name": None,
                "backup_name": backup_name,
                "previous_state": self._state_data(previous_state),
                "new_state": None,
            }
            self._write_journal(journal, "prepared")
            phase = "prepared"
            self._paths.replace(canonical_name, backup_name)
            self._write_journal(journal, "backup_moved")
            phase = "backup_moved"
            self._state.remove(canonical_name)
            self._write_journal(journal, "state_removed")
            phase = "state_removed"
            self._paths.remove_entry_name(backup_name)
            self._paths.unlink(_JOURNAL_NAME)
            journal = None
            return PluginLifecycleResult(
                success=True,
                rollback_performed=False,
                plugin_name=canonical_name,
                message=f"Uninstalled {canonical_name}",
            )
        except Exception as exc:
            rollback_errors: list[str] = []
            if journal is not None:
                backup_name = journal["backup_name"]
                backup_exists = self._paths.entry_exists(backup_name)
                if phase == "state_removed" and not backup_exists:
                    try:
                        committed = (
                            not self._paths.entry_exists(canonical_name)
                            and self._state.get(canonical_name) is None
                        )
                    except Exception as verify_exc:
                        return PluginLifecycleResult(
                            success=False,
                            rollback_performed=False,
                            plugin_name=canonical_name,
                            message=(
                                f"{exc}; commit verification failed: {verify_exc}; "
                                "transaction journal retained"
                            ),
                        )
                    if committed:
                        try:
                            self._paths.unlink(_JOURNAL_NAME)
                        except Exception as journal_exc:
                            return PluginLifecycleResult(
                                success=False,
                                rollback_performed=False,
                                plugin_name=canonical_name,
                                message=(
                                    f"Uninstalled {canonical_name}, but cleanup reported: {exc}; "
                                    f"journal cleanup failed: {journal_exc}"
                                ),
                            )
                        journal = None
                        return PluginLifecycleResult(
                            success=True,
                            rollback_performed=False,
                            plugin_name=canonical_name,
                            message=(
                                f"Uninstalled {canonical_name}; "
                                f"backup cleanup reported after deletion: {exc}"
                            ),
                        )
                    return PluginLifecycleResult(
                        success=False,
                        rollback_performed=False,
                        plugin_name=canonical_name,
                        message=(
                            f"{exc}; committed uninstall postconditions could not be verified; "
                            "transaction journal retained"
                        ),
                    )
                try:
                    if backup_exists:
                        if self._paths.entry_exists(canonical_name):
                            self._paths.remove_entry_name(canonical_name)
                        self._paths.replace(backup_name, canonical_name)
                    self._restore_state(canonical_name, previous_state)
                except Exception as rollback_exc:
                    rollback_errors.append(f"rollback failed: {rollback_exc}")
                if not rollback_errors:
                    self._paths.unlink(_JOURNAL_NAME)
                    journal = None
            message = str(exc)
            if rollback_errors:
                message = f"{message}; {'; '.join(rollback_errors)}"
            return PluginLifecycleResult(
                success=False,
                rollback_performed=journal is None,
                plugin_name=canonical_name,
                message=message,
            )

    def _set_enabled(self, plugin_name: object, *, enabled: bool) -> PluginLifecycleResult:
        self._recover_incomplete_transaction()
        target_dir, invalid_result = self._target_result(plugin_name)
        if target_dir is None:
            assert invalid_result is not None
            return invalid_result
        canonical_name = target_dir.name
        if not self._paths.entry_exists(canonical_name):
            return PluginLifecycleResult(
                success=False,
                rollback_performed=False,
                plugin_name=canonical_name,
                message=f"Plugin '{canonical_name}' is not installed",
            )
        try:
            state = self._state.get(canonical_name) or PluginState()
            state.enabled = enabled
            self._state.set(canonical_name, state)
        except Exception as exc:
            return PluginLifecycleResult(
                success=False,
                rollback_performed=False,
                plugin_name=canonical_name,
                message=str(exc),
            )
        verb = "Enabled" if enabled else "Disabled"
        return PluginLifecycleResult(
            success=True,
            rollback_performed=False,
            plugin_name=canonical_name,
            message=f"{verb} {canonical_name}",
        )

    def enable(self, plugin_name: object) -> PluginLifecycleResult:
        return self._set_enabled(plugin_name, enabled=True)

    def disable(self, plugin_name: object) -> PluginLifecycleResult:
        return self._set_enabled(plugin_name, enabled=False)

    def is_enabled(self, plugin_name: str) -> bool:
        return self._state.is_enabled(plugin_name)

    def installed_manifests(self) -> list[dict]:
        manifests: list[dict] = []
        for manifest, _state in self.installed_plugins():
            plugin_dir = self.resolve_plugin_target(manifest.name)
            manifest_path = find_manifest(plugin_dir)
            if manifest_path is None:
                continue
            try:
                manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return manifests

    def manifest_errors(self) -> list[tuple[Path, str]]:
        """Return validation errors for direct-child plugin directories."""
        failures: list[tuple[Path, str]] = []
        try:
            names = self._paths.list_entries()
        except PluginPathError:
            return failures
        for name in names:
            if name.startswith(".") or name in _NON_PLUGIN_ENTRIES:
                continue
            try:
                plugin_dir = self.resolve_plugin_target(name)
                manifest, errors, _warnings = parse_manifest(plugin_dir)
            except (OSError, ValueError, PluginPathError) as exc:
                failures.append((self.root / name, str(exc)))
                continue
            if manifest is None or errors:
                failures.append(
                    (plugin_dir, "; ".join(errors) if errors else "Invalid plugin manifest")
                )
        return failures

    def installed_plugins(self) -> list[tuple[PluginManifest, PluginState]]:
        """List only canonical, symlink-free installed plugin directories."""
        result: list[tuple[PluginManifest, PluginState]] = []
        try:
            names = self._paths.list_entries()
        except PluginPathError:
            return result
        for name in names:
            if name.startswith(".") or name in _NON_PLUGIN_ENTRIES:
                continue
            try:
                plugin_dir = self.resolve_plugin_target(name)
                manifest, errors, _ = parse_manifest(plugin_dir)
                if manifest is None or errors or plugin_dir.name != manifest.name:
                    continue
                if self.resolve_plugin_target(manifest.name) != plugin_dir:
                    continue
                state = self._state.get(manifest.name) or PluginState()
            except (OSError, ValueError, PluginPathError):
                continue
            result.append((manifest, state))
        return result
