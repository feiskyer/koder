"""Backup helpers for reversible config migrations."""

from __future__ import annotations

from pathlib import Path

from koder_agent.utils.atomic_file import write_text_atomic


def create_config_backup(config_path: Path) -> Path:
    """Create a sibling backup for a config file before rewriting it."""
    backup_path = config_path.with_suffix(config_path.suffix + ".bak")
    if backup_path.is_symlink():
        raise ValueError(f"Refusing config backup symlink: {backup_path}")
    if config_path.exists():
        if backup_path.exists():
            backup_path.chmod(0o600)
        write_text_atomic(backup_path, config_path.read_bytes().decode("utf-8"))
    return backup_path
