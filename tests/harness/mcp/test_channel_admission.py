"""Channel admission through load/factory/live-owner using in-memory transports."""

import json
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import anyio
import pytest
from mcp import ClientSession
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCNotification
from pydantic import TypeAdapter

import koder_agent.mcp as mcp_pkg
import koder_agent.mcp.notifications as notifications
from koder_agent.harness.channels.gate import PluginChannelOrigin
from koder_agent.harness.channels.interceptor import ChannelInterceptingStream
from koder_agent.harness.channels.notification import (
    CHANNEL_NOTIFICATION_METHOD,
    ChannelNotificationRouter,
)
from koder_agent.harness.channels.state import reset_channel_state, set_allowed_channels
from koder_agent.harness.channels.types import (
    ChannelEntryPlugin,
    ChannelEntryServer,
    parse_channel_entries,
)
from koder_agent.mcp.server_config import MCPServerConfig, MCPServerScope, MCPServerType

CHANNEL_CAPABILITIES = {"experimental": {"claude/channel": {}}}


class SyntheticServer:
    """Uses real MCP message/session types, but never starts an SDK receive loop."""

    def __init__(self, name, callback, capabilities):
        self.name = name
        self.callback = callback
        self.capabilities = capabilities
        self.session = None
        self.server_initialize_result = None
        self.send, read = anyio.create_memory_object_stream(4)
        self.read = (
            ChannelInterceptingStream(read, callback, name) if callback is not None else read
        )
        self.closed = False

    async def emit(self, content):
        if self.callback is None:
            return
        self.send.send_nowait(
            SessionMessage(
                TypeAdapter(JSONRPCMessage).validate_python(
                    JSONRPCNotification(
                        jsonrpc="2.0",
                        method=CHANNEL_NOTIFICATION_METHOD,
                        params={"content": content},
                    ).model_dump(by_alias=True)
                )
            )
        )
        # A non-channel sentinel ensures receive finishes after swallowing the
        # channel frame and proves ordinary MCP delivery is not blocked.
        sentinel = object()
        self.send.send_nowait(sentinel)
        assert await self.read.receive() is sentinel

    async def connect(self):
        await self.emit("before initialization")
        self.server_initialize_result = SimpleNamespace(capabilities=self.capabilities)
        self.session = ClientSession(self.read, self.send)
        if self.callback is not None:
            self.read.bind_session(self.session)
        await self.emit("initialized but not published")

    async def list_prompts(self):
        return SimpleNamespace(prompts=[])

    async def cleanup(self):
        if not self.closed:
            await self.emit("during cleanup")
            self.closed = True
            self.session = None
            await self.send.aclose()
            await self.read.aclose()


@pytest.fixture
def channel_runtime(monkeypatch):
    reset_channel_state()
    delivered = []
    created = []
    capability_sequence = []
    configs = []
    router = ChannelNotificationRouter()
    handler = notifications.MCPNotificationHandler()
    handler.set_channel_router(router)

    async def receive(server, content, _meta):
        delivered.append((server, content))

    router.on_channel_message(receive)
    monkeypatch.setattr(notifications, "_handler", handler)
    manager = SimpleNamespace(
        list_servers=AsyncMock(side_effect=lambda **_kwargs: list(configs)),
        revalidate_project_config=lambda _config: False,
    )
    monkeypatch.setattr(mcp_pkg, "MCPServerManager", lambda: manager)
    plugin_loader = mcp_pkg._load_plugin_mcp_configs
    monkeypatch.setattr(mcp_pkg, "_load_plugin_mcp_configs", lambda: [])

    async def create(config, channel_callback=None, *, trusted=True):
        assert trusted
        caps = capability_sequence.pop(0) if capability_sequence else CHANNEL_CAPABILITIES
        server = SyntheticServer(config.name, channel_callback, caps)
        created.append(server)
        return server

    monkeypatch.setattr(mcp_pkg.MCPServerFactory, "create_server", create)
    yield SimpleNamespace(
        configs=configs,
        capabilities=capability_sequence,
        delivered=delivered,
        created=created,
        manager=manager,
        plugin_loader=plugin_loader,
    )
    reset_channel_state()


@pytest.mark.asyncio
@pytest.mark.parametrize("registered", [True, False])
async def test_cli_installed_plugin_origin_controls_actual_channel_delivery(
    channel_runtime, monkeypatch, tmp_path, registered
):
    from pathlib import Path

    from koder_agent.harness.plugins.commands import _handle_install
    from koder_agent.harness.plugins.lifecycle import PluginLifecycleService
    from koder_agent.harness.plugins.marketplace import MarketplaceStore

    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    plugin = tmp_path / "Community" / "demo"
    plugin.mkdir(parents=True)
    (plugin / "plugin.json").write_text('{"name":"demo","version":"1.0.0"}')
    (plugin / ".mcp.json").write_text(
        '{"mcpServers":{"raw-plugin":{"command":"synthetic-never-executed"}}}'
    )
    store = MarketplaceStore.default()
    assert store.add(str(plugin.parent))[0] is not None
    lifecycle = PluginLifecycleService(home / ".koder" / "plugins")
    assert _handle_install(lifecycle, "demo@community", "user") == 0
    if not registered:
        assert store.remove("community")
    monkeypatch.setattr(mcp_pkg, "_load_plugin_mcp_configs", channel_runtime.plugin_loader)
    set_allowed_channels([ChannelEntryPlugin(name="demo", marketplace="community")])
    owner = await mcp_pkg.load_mcp_servers()
    try:
        assert channel_runtime.delivered == []
        await channel_runtime.created[0].emit("installed channel")
        expected = [("raw-plugin", "installed channel")] if registered else []
        assert channel_runtime.delivered == expected
    finally:
        await owner.aclose()


def config(name="channel", **kwargs):
    return MCPServerConfig(
        name=name, transport_type=MCPServerType.STDIO, command="synthetic-not-executed", **kwargs
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("capabilities", [None, {}, {"experimental": {"other/capability": {}}}])
async def test_unsupported_server_never_delivers_even_when_listed(channel_runtime, capabilities):
    runtime = channel_runtime
    runtime.configs.append(config())
    runtime.capabilities.append(capabilities)
    set_allowed_channels([ChannelEntryServer(name="channel")])
    owner = await mcp_pkg.load_mcp_servers()
    try:
        await runtime.created[0].emit("after capability rejection")
        assert runtime.delivered == []
        assert len(owner) == 1  # Only channels are rejected, not normal MCP functionality.
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_supported_listed_server_only_delivers_after_publication(channel_runtime):
    runtime = channel_runtime
    runtime.configs.append(config())
    set_allowed_channels([ChannelEntryServer(name="channel")])
    owner = await mcp_pkg.load_mcp_servers()
    try:
        assert runtime.delivered == []
        await runtime.created[0].emit("admitted")
        assert runtime.delivered == [("channel", "admitted")]
    finally:
        await owner.aclose()
    assert runtime.delivered == [("channel", "admitted")]
    assert runtime.created[0].closed


@pytest.mark.asyncio
async def test_unlisted_and_rejected_project_servers_cannot_deliver(channel_runtime):
    runtime = channel_runtime
    runtime.configs.extend([config("unlisted"), config("rejected", scope=MCPServerScope.PROJECT)])
    set_allowed_channels([ChannelEntryServer(name="rejected")])
    owner = await mcp_pkg.load_mcp_servers()
    try:
        assert [server.name for server in runtime.created] == ["unlisted"]
        assert runtime.created[0].callback is None
        assert runtime.delivered == []
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_allowlist_revocation_takes_effect_before_next_delivery(channel_runtime):
    runtime = channel_runtime
    runtime.configs.append(config())
    set_allowed_channels([ChannelEntryServer(name="channel")])
    owner = await mcp_pkg.load_mcp_servers()
    try:
        runtime.delivered.clear()
        set_allowed_channels([])
        await runtime.created[0].emit("revoked")
        assert runtime.delivered == []
    finally:
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement_supported", [False, True])
async def test_reconnect_rechecks_capability_and_rejects_retired_stream(
    channel_runtime, replacement_supported
):
    runtime = channel_runtime
    runtime.configs.append(config())
    runtime.capabilities.extend(
        [CHANNEL_CAPABILITIES, CHANNEL_CAPABILITIES if replacement_supported else {}]
    )
    set_allowed_channels([ChannelEntryServer(name="channel")])
    owner = await mcp_pkg.load_mcp_servers()
    try:
        runtime.delivered.clear()
        old = runtime.created[0]
        old.session = None
        assert await owner.reconnection_managers["channel"].reconnect_if_needed()
        new = runtime.created[1]
        assert runtime.delivered == []
        await new.emit("replacement")
        expected = [("channel", "replacement")] if replacement_supported else []
        assert runtime.delivered == expected
        # Simulate a late already-read event from the retired stream callback.
        await old.read._on_notification(
            "channel", CHANNEL_NOTIFICATION_METHOD, {"content": "retired connection"}
        )
        assert runtime.delivered == expected
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_closed_owner_callback_cannot_deliver_into_later_session(channel_runtime):
    runtime = channel_runtime
    runtime.configs.append(config())
    set_allowed_channels([ChannelEntryServer(name="channel")])
    owner = await mcp_pkg.load_mcp_servers()
    old_callback = runtime.created[0].read._on_notification
    await owner.aclose()
    runtime.delivered.clear()
    new_owner = await mcp_pkg.load_mcp_servers()
    try:
        runtime.delivered.clear()
        await old_callback("channel", CHANNEL_NOTIFICATION_METHOD, {"content": "old owner"})
        assert runtime.delivered == []
        await runtime.created[1].emit("new owner")
        assert runtime.delivered == [("channel", "new owner")]
    finally:
        await new_owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("marketplace", ["verified-market", None, "different-market"])
async def test_plugin_entry_matches_recorded_origin_not_raw_server_name(
    channel_runtime, monkeypatch, marketplace
):
    runtime = channel_runtime
    plugin_config = config("raw-plugin-transport")
    plugin_configs = mcp_pkg._PluginMCPConfigs()
    plugin_configs.append(plugin_config)
    plugin_configs.channel_origins[id(plugin_config)] = PluginChannelOrigin(
        "team-chat", marketplace
    )
    monkeypatch.setattr(mcp_pkg, "_load_plugin_mcp_configs", lambda: nullcontext(plugin_configs))
    set_allowed_channels(parse_channel_entries(["plugin:team-chat@verified-market"], "--channels"))
    owner = await mcp_pkg.load_mcp_servers()
    try:
        await runtime.created[0].emit("plugin message")
        assert owner[0].name == "raw-plugin-transport"
        expected = (
            [("raw-plugin-transport", "plugin message")] if marketplace == "verified-market" else []
        )
        assert runtime.delivered == expected
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_forged_plugin_prefix_without_origin_is_not_a_plugin(channel_runtime):
    runtime = channel_runtime
    runtime.configs.append(config("plugin:team-chat:transport"))
    set_allowed_channels([ChannelEntryPlugin(name="team-chat", marketplace="verified-market")])
    owner = await mcp_pkg.load_mcp_servers()
    try:
        assert runtime.created[0].callback is None
        await runtime.created[0].emit("unverified")
        assert runtime.delivered == []
    finally:
        await owner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("dev", [False, True])
async def test_explicit_raw_server_allowlist_still_works_for_plugins(
    channel_runtime, monkeypatch, dev
):
    runtime = channel_runtime
    plugin_config = config("raw-plugin-transport")
    plugin_configs = mcp_pkg._PluginMCPConfigs()
    plugin_configs.append(plugin_config)
    plugin_configs.channel_origins[id(plugin_config)] = PluginChannelOrigin("team-chat")
    monkeypatch.setattr(mcp_pkg, "_load_plugin_mcp_configs", lambda: nullcontext(plugin_configs))
    set_allowed_channels([ChannelEntryServer(name="raw-plugin-transport", dev=dev)])
    owner = await mcp_pkg.load_mcp_servers()
    try:
        await runtime.created[0].emit("explicitly enabled")
        assert runtime.delivered == [("raw-plugin-transport", "explicitly enabled")]
    finally:
        await owner.aclose()


@pytest.mark.asyncio
async def test_plugin_raw_server_identity_is_preserved_by_discovery(tmp_path, monkeypatch):
    """Document actual producer identity without using the user's plugin root."""
    import koder_agent.harness.plugins.path_safety as path_safety
    from koder_agent.harness.plugins.lifecycle import PluginLifecycleService
    from koder_agent.harness.plugins.manifest import PluginManifest
    from koder_agent.harness.plugins.state import PluginState

    source = tmp_path / "team-chat"
    source.mkdir()
    (source / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"chat-transport": {"command": "synthetic-not-executed"}}})
    )
    manifest = PluginManifest(name="team-chat", version="1.0.0")
    monkeypatch.setattr(PluginLifecycleService, "__init__", lambda self, _root: None)
    monkeypatch.setattr(
        PluginLifecycleService, "installed_plugins", lambda self: [(manifest, PluginState())]
    )
    monkeypatch.setattr(PluginLifecycleService, "resolve_plugin_target", lambda self, _name: source)

    @contextmanager
    def snapshot(root):
        assert root == source
        yield source

    monkeypatch.setattr(path_safety, "snapshot_plugin_tree", snapshot)
    with mcp_pkg._load_plugin_mcp_configs() as configs:
        assert [item.name for item in configs] == ["chat-transport"]
        assert configs[0].source_path == str(source / ".mcp.json")
        origin = configs.channel_origins[id(configs[0])]
        assert origin.name == "team-chat"
        assert origin.marketplace is None  # Never inferred as "local" or from a manifest.
