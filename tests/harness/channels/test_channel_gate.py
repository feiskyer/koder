"""Tests for channel gate logic."""

import pytest

from koder_agent.harness.channels.gate import (
    ChannelGateResult,
    PluginChannelOrigin,
    find_channel_entry,
    gate_channel_server,
)
from koder_agent.harness.channels.state import reset_channel_state, set_allowed_channels
from koder_agent.harness.channels.types import ChannelEntryPlugin, ChannelEntryServer


@pytest.fixture(autouse=True)
def _clean_state():
    reset_channel_state()
    yield
    reset_channel_state()


class TestFindChannelEntry:
    def test_server_exact_match(self):
        entries = [ChannelEntryServer(name="webhook")]
        result = find_channel_entry("webhook", entries)
        assert result is not None
        assert result.name == "webhook"

    def test_server_no_match(self):
        entries = [ChannelEntryServer(name="webhook")]
        assert find_channel_entry("other", entries) is None

    def test_plugin_segment_match(self):
        entries = [ChannelEntryPlugin(name="slack", marketplace="anthropic")]
        result = find_channel_entry(
            "plugin:slack:some-suffix",
            entries,
            plugin_origin=PluginChannelOrigin("slack", "anthropic"),
        )
        assert result is not None
        assert result.name == "slack"

    def test_plugin_exact_two_segment(self):
        entries = [ChannelEntryPlugin(name="telegram", marketplace="official")]
        result = find_channel_entry(
            "plugin:telegram",
            entries,
            plugin_origin=PluginChannelOrigin("telegram", "official"),
        )
        assert result is not None

    def test_plugin_no_match_wrong_name(self):
        entries = [ChannelEntryPlugin(name="slack", marketplace="anthropic")]
        assert find_channel_entry("plugin:discord", entries) is None

    def test_plugin_no_match_not_plugin_prefix(self):
        entries = [ChannelEntryPlugin(name="slack", marketplace="anthropic")]
        assert find_channel_entry("server:slack", entries) is None

    def test_empty_entries(self):
        assert find_channel_entry("anything", []) is None

    def test_mixed_entries(self):
        entries = [
            ChannelEntryServer(name="webhook"),
            ChannelEntryPlugin(name="slack", marketplace="anthropic"),
        ]
        assert find_channel_entry("webhook", entries) is not None
        assert (
            find_channel_entry(
                "plugin:slack",
                entries,
                plugin_origin=PluginChannelOrigin("slack", "anthropic"),
            )
            is not None
        )
        assert find_channel_entry("plugin:webhook", entries) is None

    def test_raw_plugin_server_key_matches_verified_discovery_origin(self):
        entries = [ChannelEntryPlugin(name="slack", marketplace="anthropic")]
        assert (
            find_channel_entry(
                "actual-transport-key",
                entries,
                plugin_origin=PluginChannelOrigin("slack", "anthropic"),
            )
            == entries[0]
        )

    @pytest.mark.parametrize(
        "origin",
        [None, PluginChannelOrigin("slack"), PluginChannelOrigin("slack", "other")],
    )
    def test_plugin_looking_server_name_does_not_verify_marketplace(self, origin):
        entries = [ChannelEntryPlugin(name="slack", marketplace="anthropic")]
        assert find_channel_entry("plugin:slack", entries, plugin_origin=origin) is None


class TestGateChannelServer:
    def test_no_capabilities(self):
        result = gate_channel_server("myserver", capabilities=None)
        assert result.action == "skip"
        assert result.kind == "capability"

    def test_no_experimental(self):
        """Capabilities exist but no experimental field."""

        class Caps:
            pass

        result = gate_channel_server("myserver", capabilities=Caps())
        assert result.action == "skip"
        assert result.kind == "capability"

    def test_no_channel_capability(self):
        """Experimental exists but doesn't include claude/channel."""

        class Caps:
            experimental = {"other/thing": {}}

        result = gate_channel_server("myserver", capabilities=Caps())
        assert result.action == "skip"
        assert result.kind == "capability"

    def test_capability_present_but_not_in_session(self):
        """Server declares channel capability but isn't in --channels list."""

        class Caps:
            experimental = {"claude/channel": {}}

        # No channels set → empty list
        result = gate_channel_server("myserver", capabilities=Caps())
        assert result.action == "skip"
        assert result.kind == "session"

    def test_register_success_server(self):
        set_allowed_channels([ChannelEntryServer(name="webhook")])

        class Caps:
            experimental = {"claude/channel": {}}

        result = gate_channel_server("webhook", capabilities=Caps())
        assert result.action == "register"
        assert result.kind is None

    def test_register_success_plugin(self):
        set_allowed_channels([ChannelEntryPlugin(name="slack", marketplace="anthropic")])

        class Caps:
            experimental = {"claude/channel": {}}

        result = gate_channel_server(
            "plugin:slack:runtime-id",
            capabilities=Caps(),
            plugin_origin=PluginChannelOrigin("slack", "anthropic"),
        )
        assert result.action == "register"

    def test_dict_capabilities(self):
        """Capabilities passed as a plain dict (not an object)."""
        set_allowed_channels([ChannelEntryServer(name="bot")])
        caps = {"experimental": {"claude/channel": {}}}
        result = gate_channel_server("bot", capabilities=caps)
        assert result.action == "register"

    def test_gate_result_is_frozen(self):
        result = ChannelGateResult(action="skip", kind="test")
        with pytest.raises(AttributeError):
            result.action = "register"  # type: ignore[misc]
