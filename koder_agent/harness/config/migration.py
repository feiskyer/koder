"""Backup-first runtime config migration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .service import RuntimeConfigService


@dataclass(frozen=True)
class ConfigMigrationResult:
    """Outcome of a config migration run."""

    config_path: Path
    backup_path: Path


def migrate_config_file(
    config_path: str | Path,
    *,
    legacy_db_path: str | Path | None = None,
) -> ConfigMigrationResult:
    """Rewrite the config file into the runtime schema after creating a backup."""
    path = Path(config_path)
    service = RuntimeConfigService(path)
    runtime_config = service.load()
    service.save(runtime_config, backup=True)
    resolved_path = path.resolve()
    backup_path = resolved_path.with_suffix(resolved_path.suffix + ".bak")

    # Explicitly do nothing to the legacy DB beyond accepting the path.
    if legacy_db_path is not None:
        _ = Path(legacy_db_path)

    return ConfigMigrationResult(config_path=path, backup_path=backup_path)
