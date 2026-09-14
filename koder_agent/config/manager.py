"""Configuration manager for Koder CLI."""

import os
from pathlib import Path
from typing import Any, Optional

from .models import KoderConfig


def _migrate_legacy_voice_fields(data: dict) -> dict:
    """Promote legacy harness.voice_* fields into top-level voice config."""
    if not isinstance(data, dict):
        return data

    harness = data.get("harness")
    voice = data.get("voice")
    if not isinstance(harness, dict):
        return data

    enabled = harness.pop("voice_enabled", None)
    provider = harness.pop("voice_provider", None)
    if enabled is None and provider is None:
        return data

    if not isinstance(voice, dict):
        voice = {}
    if enabled is not None and "enabled" not in voice:
        voice["enabled"] = enabled
    if provider is not None and "provider" not in voice:
        voice["provider"] = provider
    data["voice"] = voice
    return data


class ConfigManager:
    """Manages loading and saving of YAML configuration."""

    DEFAULT_CONFIG_PATH: Optional[Path] = None

    @classmethod
    def default_config_path(cls) -> Path:
        override = cls.DEFAULT_CONFIG_PATH
        if override is not None:
            return Path(override)
        return Path.home() / ".koder" / "config.yaml"

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the configuration manager.

        Args:
            config_path: Optional custom path to the config file.
                        Defaults to ~/.koder/config.yaml
        """
        self.config_path = config_path or self.default_config_path()
        from koder_agent.harness.config.service import RuntimeConfigService

        self._service = RuntimeConfigService(self.config_path)

    def load(self) -> KoderConfig:
        """Load configuration from file, creating default if not exists.

        Returns:
            KoderConfig instance with loaded or default configuration.
        """
        return self._service.load()

    def save(self, config: Optional[KoderConfig] = None) -> None:
        """Save configuration to file.

        Args:
            config: Optional config to save. Uses cached config if not provided.
        """
        self._service.save(config)

    def reload(self) -> KoderConfig:
        """Force reload configuration from file.

        Returns:
            Freshly loaded KoderConfig instance.
        """
        return self._service.reload()

    def get_effective_value(
        self, config_value: Any, env_var_name: Optional[str], cli_value: Any = None
    ) -> Any:
        """Get effective value with priority: CLI > ENV > Config > Default.

        Args:
            config_value: Value from config file.
            env_var_name: Name of environment variable to check (can be None).
            cli_value: Value from CLI argument (highest priority).

        Returns:
            The effective value based on priority order.
        """
        if cli_value is not None:
            return cli_value
        if env_var_name:
            env_value = os.environ.get(env_var_name)
            if env_value is not None:
                return env_value
        return config_value


# Global singleton instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global ConfigManager instance.

    Returns:
        The singleton ConfigManager instance.
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_config() -> KoderConfig:
    """Get the current configuration.

    Returns:
        The loaded KoderConfig instance.
    """
    return get_config_manager().load()


def reset_config_manager() -> None:
    """Reset the global config manager (useful for testing)."""
    global _config_manager
    _config_manager = None
