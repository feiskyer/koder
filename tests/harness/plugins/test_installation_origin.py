"""Installer-owned provenance must survive lifecycle operations and gate MCP."""

import json
from pathlib import Path

import pytest

import koder_agent.mcp as mcp_pkg
from koder_agent.harness.channels.gate import find_channel_entry
from koder_agent.harness.channels.types import ChannelEntryPlugin
from koder_agent.harness.plugins.commands import _handle_install
from koder_agent.harness.plugins.lifecycle import PluginLifecycleService
from koder_agent.harness.plugins.marketplace import MarketplaceStore
from koder_agent.harness.plugins.session_root import build_session_plugin_root


def _plugin(directory, *, name="demo", version="1.0.0", claimed_marketplace=None):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"name": name, "version": version}
    if claimed_marketplace:
        manifest["marketplace"] = claimed_marketplace
    (directory / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
    (directory / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"raw-server": {"command": "synthetic-never-executed"}}}),
        encoding="utf-8",
    )
    (directory / "payload.txt").write_text(version, encoding="utf-8")
    return directory


@pytest.fixture
def installed_market(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    market = tmp_path / "Community"
    plugin = _plugin(market / "demo")
    store = MarketplaceStore.default()
    source, _message = store.add(str(market))
    assert source is not None
    lifecycle = PluginLifecycleService(home / ".koder" / "plugins")
    return home, plugin, store, lifecycle


def _discovered_origin():
    with mcp_pkg._load_plugin_mcp_configs() as configs:
        assert len(configs) == 1
        return configs[0].name, configs.channel_origins[id(configs[0])]


@pytest.mark.parametrize("reference", ["demo@community", "demo"])
def test_marketplace_cli_install_reaches_channel_gate(installed_market, reference):
    _home, _source, _store, lifecycle = installed_market
    assert _handle_install(lifecycle, reference, "user") == 0
    name, origin = _discovered_origin()
    assert name == "raw-server"
    assert origin.marketplace == "community"
    assert (
        find_channel_entry(
            name, [ChannelEntryPlugin(name="demo", marketplace="community")], plugin_origin=origin
        )
        is not None
    )


def test_origin_survives_reload_toggles_and_session_copy(installed_market):
    _home, _source, _store, lifecycle = installed_market
    assert _handle_install(lifecycle, "demo@community", "user") == 0
    original = getattr(lifecycle.state_store.get("demo"), "origin", None)
    assert original is not None
    reloaded = PluginLifecycleService(lifecycle.root)
    assert reloaded.state_store.list_all()["demo"].origin == original
    assert reloaded.disable("demo").success
    assert reloaded.enable("demo").success
    assert reloaded.state_store.get("demo").origin == original
    overlay = build_session_plugin_root("origin-copy", [], base_root=lifecycle.root)
    assert PluginLifecycleService(overlay).state_store.get("demo").origin == original
    assert _discovered_origin()[1].marketplace == "community"


def test_local_install_cannot_claim_or_inherit_marketplace(installed_market, tmp_path):
    _home, _source, _store, lifecycle = installed_market
    assert _handle_install(lifecycle, "demo@community", "user") == 0
    assert _discovered_origin()[1].marketplace == "community"
    local = _plugin(tmp_path / "local", version="2.0.0", claimed_marketplace="community")
    assert _handle_install(lifecycle, str(local), "user") == 0
    assert getattr(lifecycle.state_store.get("demo"), "origin", None) is None
    assert _discovered_origin()[1].marketplace is None


@pytest.mark.parametrize("change", ["remove", "rebind"])
def test_removed_or_rebound_marketplace_does_not_authorize_old_install(
    installed_market, tmp_path, change
):
    _home, _source, store, lifecycle = installed_market
    assert _handle_install(lifecycle, "demo@community", "user") == 0
    assert _discovered_origin()[1].marketplace == "community"
    assert store.remove("community")
    if change == "rebind":
        replacement = tmp_path / "replacement" / "Community"
        _plugin(replacement / "demo", version="2.0.0")
        assert store.add(str(replacement))[0] is not None
    assert _discovered_origin()[1].marketplace is None


def test_failed_upgrade_restores_payload_and_origin(installed_market, tmp_path, monkeypatch):
    _home, _source, store, lifecycle = installed_market
    assert _handle_install(lifecycle, "demo@community", "user") == 0
    old_origin = getattr(lifecycle.state_store.get("demo"), "origin", None)
    assert old_origin is not None
    other = tmp_path / "Other"
    _plugin(other / "demo", version="2.0.0")
    assert store.add(str(other))[0] is not None
    real_set = lifecycle.state_store.set
    failed = False

    def fail_once(name, state):
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("synthetic state failure")
        return real_set(name, state)

    monkeypatch.setattr(lifecycle.state_store, "set", fail_once)
    assert _handle_install(lifecycle, "demo@other", "user") == 1
    restored = PluginLifecycleService(lifecycle.root)
    assert restored.state_store.get("demo").origin == old_origin
    assert (lifecycle.root / "demo" / "payload.txt").read_text() == "1.0.0"
    assert _discovered_origin()[1].marketplace == "community"


@pytest.mark.parametrize(
    ("phase", "expected_market"), [("target_published", "community"), ("state_written", "other")]
)
def test_journal_recovery_preserves_selected_origin(
    installed_market, tmp_path, monkeypatch, phase, expected_market
):
    _home, _source, store, lifecycle = installed_market
    assert _handle_install(lifecycle, "demo@community", "user") == 0
    assert getattr(lifecycle.state_store.get("demo"), "origin", None) is not None
    other = tmp_path / "Other"
    _plugin(other / "demo", version="2.0.0")
    assert store.add(str(other))[0] is not None
    real_write = lifecycle._write_journal

    def interrupt(journal, next_phase):
        real_write(journal, next_phase)
        if next_phase == phase:
            raise SystemExit("synthetic process interruption")

    monkeypatch.setattr(lifecycle, "_write_journal", interrupt)
    with pytest.raises(SystemExit):
        _handle_install(lifecycle, "demo@other", "user")
    recovered = PluginLifecycleService(lifecycle.root)
    assert recovered.state_store.get("demo").origin.marketplace == expected_market
    assert _discovered_origin()[1].marketplace == expected_market


def test_legacy_state_is_not_backfilled_from_matching_marketplace(installed_market):
    _home, source, _store, lifecycle = installed_market
    assert lifecycle.install_from_dir(source).success
    assert getattr(lifecycle.state_store.get("demo"), "origin", None) is None
    assert _discovered_origin()[1].marketplace is None


def test_selected_plugin_name_is_revalidated_before_install(installed_market, monkeypatch):
    _home, source, store, lifecycle = installed_market
    real_find = MarketplaceStore.find_plugin

    def changed_manifest(self, reference):
        selected = real_find(self, reference)
        _plugin(source, name="different-plugin")
        return selected

    monkeypatch.setattr(MarketplaceStore, "find_plugin", changed_manifest)
    assert _handle_install(lifecycle, "demo@community", "user") == 1
    assert lifecycle.installed_plugins() == []


@pytest.mark.parametrize(
    "record",
    [
        {"marketplace": "community", "source_digest": "invalid"},
        {"marketplace": "../community", "source_digest": "0" * 64},
        {"marketplace": "community", "source_digest": "0" * 64},
        "community",
    ],
)
def test_malformed_or_unmatched_origin_never_grants_channel(installed_market, record):
    _home, source, _store, lifecycle = installed_market
    assert lifecycle.install_from_dir(source).success
    state_path = lifecycle.root / "state.json"
    data = json.loads(state_path.read_text())
    data["demo"]["origin"] = record
    state_path.write_text(json.dumps(data))
    assert _discovered_origin()[1].marketplace is None


def test_registry_source_change_is_detected_even_if_cache_path_is_same(installed_market):
    _home, _source, store, lifecycle = installed_market
    assert _handle_install(lifecycle, "demo@community", "user") == 0
    assert _discovered_origin()[1].marketplace == "community"
    data = store._load()
    data["community"]["raw_source"] = "a different registered source"
    store._save(data)
    assert _discovered_origin()[1].marketplace is None


def test_local_session_override_does_not_inherit_installed_origin(installed_market, tmp_path):
    _home, _source, _store, lifecycle = installed_market
    assert _handle_install(lifecycle, "demo@community", "user") == 0
    replacement = _plugin(tmp_path / "replacement", version="2.0.0")
    overlay = build_session_plugin_root("override", [replacement], base_root=lifecycle.root)
    assert PluginLifecycleService(overlay).state_store.get("demo").origin is None
    assert lifecycle.state_store.get("demo").origin is not None
