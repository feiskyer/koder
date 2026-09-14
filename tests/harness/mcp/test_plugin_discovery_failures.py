"""One malformed plugin must not disable other plugins' MCP discovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import koder_agent.mcp as mcp_pkg
from koder_agent.harness.plugins.lifecycle import PluginLifecycleService


@pytest.mark.parametrize("bad_content", [b"{not-json", b"\xff\xfe"])
def test_invalid_plugin_mcp_file_does_not_hide_later_healthy_plugin(
    tmp_path, monkeypatch, bad_content
):
    monkeypatch.setenv("HOME", str(tmp_path))
    lifecycle = PluginLifecycleService(tmp_path / ".koder" / "plugins")
    healthy = json.dumps(
        {"mcpServers": {"healthy-server": {"type": "http", "url": "https://example.test/mcp"}}}
    ).encode()
    for name, content in (("broken", bad_content), ("healthy", healthy)):
        source = tmp_path / name
        source.mkdir()
        (source / "plugin.json").write_text(
            json.dumps({"name": name, "version": "1.0.0"}), encoding="utf-8"
        )
        (source / ".mcp.json").write_bytes(content)
        assert lifecycle.install_from_dir(source).success

    # Pin discovery order; filesystem enumeration order is not a contract.
    installed = sorted(lifecycle.installed_plugins(), key=lambda entry: entry[0].name)
    monkeypatch.setattr(PluginLifecycleService, "installed_plugins", lambda self: installed)

    with mcp_pkg._load_plugin_mcp_configs() as configs:
        assert [config.name for config in configs] == ["healthy-server"]
        snapshot = Path(configs[0].source_path).parent
        assert snapshot.is_dir()
    assert not snapshot.exists()
