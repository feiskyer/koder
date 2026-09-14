"""Maintenance commands must remain usable when runtime configuration is broken."""

import pytest

from koder_agent.config.manager import ConfigManager
from koder_agent.harness import runtime
from koder_agent.harness.cli.entrypoint import build_runtime_request
from koder_agent.harness.config import commands
from koder_agent.harness.config.service import RuntimeConfigService


def _reject_runtime_bootstrap(monkeypatch):
    def reject(*_args, **_kwargs):
        raise AssertionError("maintenance command attempted agent bootstrap")

    monkeypatch.setattr(runtime, "_load_permission_hierarchy", reject)
    monkeypatch.setattr(runtime.RuntimeConfigService, "load", reject)


@pytest.mark.asyncio
@pytest.mark.parametrize("argv", [["--help"], ["--version"]])
async def test_help_and_version_do_not_bootstrap_agent_configuration(monkeypatch, capsys, argv):
    _reject_runtime_bootstrap(monkeypatch)

    assert await runtime.HarnessRuntime(build_runtime_request(argv)).run() == 0
    assert capsys.readouterr().out


@pytest.mark.asyncio
async def test_config_validate_reports_broken_yaml_without_agent_bootstrap(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "config.yaml"
    path.write_text("model: [broken", encoding="utf-8")
    _reject_runtime_bootstrap(monkeypatch)
    monkeypatch.setattr(commands, "get_config_manager", lambda: ConfigManager(path))
    monkeypatch.setattr(commands, "RuntimeConfigService", lambda: RuntimeConfigService(path))

    request = build_runtime_request(["--bare", "config", "validate"])
    assert await runtime.HarnessRuntime(request).run() == 1
    assert "Config invalid: YAML parse error" in capsys.readouterr().out
