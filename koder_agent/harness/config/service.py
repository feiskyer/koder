"""Runtime config loading and saving."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from filelock import FileLock

from koder_agent.config.manager import _migrate_legacy_voice_fields
from koder_agent.harness.hooks.runtime import dispatch_command_hooks
from koder_agent.utils.atomic_file import write_text_atomic

from .backup import create_config_backup
from .schema import RuntimeConfig, parse_runtime_config_source

_UNLOADED = object()


def config_write_lock(path: Path) -> FileLock:
    """Serialize cooperating publishers through hook completion and rollback."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    if lock_path.is_symlink():
        raise ValueError(f"Refusing config lock symlink: {lock_path}")
    return FileLock(str(lock_path), timeout=5)


def read_config_text(path: Path) -> str | None:
    try:
        return path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        return None


def dispatch_config_change(path: Path) -> None:
    source = "user_settings" if path.is_relative_to(Path.home().resolve()) else "project_settings"
    result = dispatch_command_hooks(
        cwd=Path.cwd(),
        event_name="ConfigChange",
        match_value=source,
        payload={"event": "ConfigChange", "source": source, "file_path": str(path)},
    )
    if result.blocked:
        raise RuntimeError(result.block_reason or "Config change blocked by hook")


class RuntimeConfigService:
    """Loads and saves the runtime config schema at the existing path."""

    DEFAULT_CONFIG_PATH = Path.home() / ".koder" / "config.yaml"

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or (Path.home() / ".koder" / "config.yaml")
        self._config: RuntimeConfig | None = None
        self._source_text: str | None | object = _UNLOADED

    def load(self) -> RuntimeConfig:
        source_text = read_config_text(self.config_path)
        if self._config is not None and source_text == self._source_text:
            return self._config
        self._config = None
        if source_text is not None:
            data = yaml.safe_load(source_text)
            if data is None:
                data = {}
            data = _migrate_legacy_voice_fields(data)
            self._config = parse_runtime_config_source(data)
        else:
            self._config = RuntimeConfig()
        self._source_text = source_text
        return self._config

    def reload(self) -> RuntimeConfig:
        self._config = None
        return self.load()

    def save(self, config: RuntimeConfig | None = None, *, backup: bool = False) -> None:
        config = config or self._config or RuntimeConfig()
        self.save_text(
            yaml.safe_dump(
                config.model_dump(exclude_none=False),
                sort_keys=False,
                allow_unicode=True,
            ),
            expected_text=self._source_text,
            backup=backup,
        )

    def save_text(
        self,
        content: str,
        *,
        expected_text: str | None | object = _UNLOADED,
        backup: bool = False,
    ) -> None:
        """Validate and publish source YAML, preserving editor formatting.

        A supplied snapshot prevents stale edits. The lock covers the existing
        publish-then-hook contract, including rollback; readers remain lock-free.
        """
        try:
            data = yaml.safe_load(content)
            config = parse_runtime_config_source(
                _migrate_legacy_voice_fields({} if data is None else data)
            )
            path = self.config_path.resolve()
            with config_write_lock(path):
                previous_text = read_config_text(path)
                if expected_text is not _UNLOADED and previous_text != expected_text:
                    raise RuntimeError("Config changed since it was loaded; reload and retry")
                if backup:
                    create_config_backup(path)
                if previous_text != content:
                    write_text_atomic(path, content)
                    try:
                        dispatch_config_change(path)
                        if read_config_text(path) != content:
                            raise RuntimeError("Config changed during ConfigChange hook")
                    except BaseException as failure:
                        # Never erase a confirmed external winner that ignored
                        # the cooperative lock (for example, a hook's own write).
                        if read_config_text(path) != content:
                            raise RuntimeError(
                                "Config changed during hook; newer file preserved"
                            ) from failure
                        if previous_text is None:
                            path.unlink(missing_ok=True)
                        else:
                            write_text_atomic(path, previous_text)
                        raise
        except BaseException:
            self._config = None
            self._source_text = _UNLOADED
            raise
        self._config = config
        self._source_text = content

    def get_effective_value(
        self,
        config_value: Any,
        env_var_name: Optional[str],
        cli_value: Any = None,
    ) -> Any:
        if cli_value is not None:
            return cli_value
        if env_var_name:
            env_value = os.environ.get(env_var_name)
            if env_value is not None:
                return env_value
        return config_value
