"""Malformed config and typoed updates must not silently become defaults."""

from types import SimpleNamespace

import pytest

from koder_agent.config.manager import ConfigManager
from koder_agent.harness.config import commands
from koder_agent.harness.config.service import RuntimeConfigService


@pytest.mark.parametrize("manager_type", [ConfigManager, RuntimeConfigService])
@pytest.mark.parametrize("source", ["[]", "false", "0", '""', "harness: []", "harness: typo"])
def test_malformed_source_is_not_silently_defaulted(tmp_path, manager_type, source):
    path = tmp_path / "config.yaml"
    path.write_text(source)
    with pytest.raises(ValueError):
        manager_type(path).load()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["model.nmae", "unknown.name", "model.name.child"])
async def test_config_set_rejects_unknown_or_non_mapping_paths(tmp_path, monkeypatch, key):
    path = tmp_path / "config.yaml"
    original = "model:\n  name: synthetic\n"
    path.write_text(original)
    monkeypatch.setattr(commands, "get_config_manager", lambda: ConfigManager(path))
    args = SimpleNamespace(config_action="set", key=key, value="replacement")
    assert await commands.handle_config_subcommand(args) == 1
    assert path.read_text() == original


@pytest.mark.asyncio
async def test_config_set_known_key_still_persists(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    monkeypatch.setattr(commands, "get_config_manager", lambda: ConfigManager(path))
    monkeypatch.setattr(
        "koder_agent.harness.config.service.dispatch_command_hooks",
        lambda **_kwargs: SimpleNamespace(blocked=False),
    )
    args = SimpleNamespace(config_action="set", key="model.name", value="synthetic-model")
    assert await commands.handle_config_subcommand(args) == 0
    assert ConfigManager(path).load().model.name == "synthetic-model"
