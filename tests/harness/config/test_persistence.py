"""Config writes must agree with hook decisions and cached runtime state."""

from types import SimpleNamespace

import pytest

from koder_agent.config.manager import ConfigManager
from koder_agent.harness.config import service as service_module
from koder_agent.harness.config.service import RuntimeConfigService


@pytest.mark.parametrize("manager_type", [ConfigManager, RuntimeConfigService])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("hook_raises", [False, True])
def test_rejected_config_write_restores_disk_and_invalidates_cache(
    tmp_path, monkeypatch, manager_type, existing, hook_raises
):
    path = tmp_path / "config.yaml"
    original = "harness:\n  permission_mode: default\n"
    if existing:
        path.write_text(original, encoding="utf-8")
    manager = manager_type(path)
    candidate = manager.load()
    candidate.harness.permission_mode = "bypass"

    def reject(**_kwargs):
        if hook_raises:
            raise RuntimeError("hook failed")
        return SimpleNamespace(blocked=True, block_reason="blocked change")

    monkeypatch.setattr(service_module, "dispatch_command_hooks", reject)

    with pytest.raises(RuntimeError, match="hook failed|blocked change"):
        manager.save(candidate)

    assert path.exists() is existing
    if existing:
        assert path.read_text(encoding="utf-8") == original
    assert manager.load().harness.permission_mode == "default"
