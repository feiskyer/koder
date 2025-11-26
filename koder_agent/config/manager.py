"""Configuration manager for Koder CLI."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from .models import KoderConfig


class ConfigManager:
    """Manages loading and saving of YAML configuration."""

    DEFAULT_CONFIG_PATH = Path.home() / ".koder" / "config.yaml"

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the configuration manager.

        Args:
            config_path: Optional custom path to the config file.
                        Defaults to ~/.koder/config.yaml
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self._config: Optional[KoderConfig] = None

    def load(self) -> KoderConfig:
        """Load configuration from file, creating default if not exists.

        Returns:
            KoderConfig instance with loaded or default configuration.
        """
        if self._config is not None:
            return self._config

        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                data = yaml.safe_load(f) or {}
            self._config = KoderConfig(**data)
        else:
            self._config = KoderConfig()

        return self._config

    def save(self, config: Optional[KoderConfig] = None) -> None:
        """Save configuration to file.

        Args:
            config: Optional config to save. Uses cached config if not provided.
        """
        config = config or self._config or KoderConfig()

        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and save as YAML
        data = config.model_dump(exclude_none=False)
        with open(self.config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        self._config = config

    def reload(self) -> KoderConfig:
        """Force reload configuration from file.

        Returns:
            Freshly loaded KoderConfig instance.
        """
        self._config = None
        return self.load()

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
