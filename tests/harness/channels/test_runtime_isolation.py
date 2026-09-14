"""Channel policy and delivery belong to the runtime that opened the owner."""

import asyncio
from types import SimpleNamespace

import pytest

from koder_agent.harness.channels import state
from koder_agent.harness.channels.admission import ChannelAdmission
from koder_agent.harness.channels.notification import (
    CHANNEL_NOTIFICATION_METHOD,
    ChannelNotificationRouter,
)
from koder_agent.harness.channels.types import ChannelEntryServer
from koder_agent.harness.cli.entrypoint import build_runtime_request, run_harness_runtime
from koder_agent.mcp import notifications


@pytest.fixture(autouse=True)
def isolated_channel_defaults(monkeypatch):
    state.reset_channel_state()
    monkeypatch.setattr(notifications, "_handler", notifications.MCPNotificationHandler())
    yield
    state.reset_channel_state()


def _request(name, channel=None):
    arguments = ["--session", name]
    if channel is not None:
        arguments.extend(["--channels", f"server:{channel}"])
    arguments.extend(["mcp", "list"])
    return build_runtime_request(arguments)


@pytest.mark.asyncio
async def test_mcp_runtime_preserves_callers_channel_policy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = [ChannelEntryServer(name="caller-channel", dev=True)]
    state.set_allowed_channels(original)
    state.set_has_dev_channels(True)
    handler = notifications.get_notification_handler()

    assert await run_harness_runtime(_request("maintenance")) == 0
    assert state.get_allowed_channels() == original
    assert state.get_has_dev_channels()
    assert notifications.get_notification_handler() is handler


@pytest.mark.asyncio
@pytest.mark.parametrize("check", ["policy", "delivery", "revocation"])
async def test_overlapping_runtimes_keep_policy_and_delivery_separate(tmp_path, monkeypatch, check):
    monkeypatch.chdir(tmp_path)
    entered = {name: asyncio.Event() for name in ("alpha", "beta")}
    inspect_alpha = asyncio.Event()
    inspected = asyncio.Event()
    revoke_alpha = asyncio.Event()
    revoked = asyncio.Event()
    finish = asyncio.Event()
    admissions = {}
    received = {"alpha": [], "beta": []}
    observed = {}

    async def command(args):
        name = args.session
        handler = notifications.get_notification_handler()
        router = handler.channel_router or ChannelNotificationRouter()
        handler.set_channel_router(router)
        active = True

        async def receive(server_name, content, _meta):
            received[name].append((server_name, content))

        unregister = router.on_channel_message(receive)
        admission = ChannelAdmission(name, router, lambda: active)
        admission.bind(
            SimpleNamespace(
                session=object(),
                server_initialize_result=SimpleNamespace(
                    capabilities={"experimental": {"claude/channel": {}}}
                ),
            )
        )
        admissions[name] = admission
        entered[name].set()
        try:
            if name == "alpha":
                await inspect_alpha.wait()
                observed["alpha"] = [entry.name for entry in state.get_allowed_channels()]
                inspected.set()
                await revoke_alpha.wait()
                state.set_allowed_channels([])
                revoked.set()
            await finish.wait()
            return 0
        finally:
            active = False
            unregister()

    # Keep CLI parsing, metadata, policy setup and runtime cleanup real; the
    # terminal action substitutes for external MCP configuration/network work.
    monkeypatch.setattr("koder_agent.harness.mcp.commands.handle_mcp_subcommand", command)
    tasks = []
    try:
        tasks.append(asyncio.create_task(run_harness_runtime(_request("alpha", "alpha"))))
        await asyncio.wait_for(entered["alpha"].wait(), 5)
        tasks.append(asyncio.create_task(run_harness_runtime(_request("beta", "beta"))))
        await asyncio.wait_for(entered["beta"].wait(), 5)
        inspect_alpha.set()
        await asyncio.wait_for(inspected.wait(), 5)

        # Delivery may run outside the caller's task. It must use the bound
        # owner's policy and router, not whichever runtime changed globals last.
        await admissions["alpha"]("alpha", CHANNEL_NOTIFICATION_METHOD, {"content": "first"})
        await admissions["beta"]("beta", CHANNEL_NOTIFICATION_METHOD, {"content": "second"})
        if check == "policy":
            assert observed["alpha"] == ["alpha"]
        if check == "delivery":
            assert received == {
                "alpha": [("alpha", "first")],
                "beta": [("beta", "second")],
            }

        before = {name: list(messages) for name, messages in received.items()}
        revoke_alpha.set()
        await asyncio.wait_for(revoked.wait(), 5)
        await admissions["alpha"]("alpha", CHANNEL_NOTIFICATION_METHOD, {"content": "blocked"})
        await admissions["beta"]("beta", CHANNEL_NOTIFICATION_METHOD, {"content": "still active"})
        if check == "revocation":
            assert received["alpha"] == before["alpha"]
            assert received["beta"] == before["beta"] + [("beta", "still active")]
    finally:
        inspect_alpha.set()
        revoke_alpha.set()
        finish.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_runtime_restores_parent_channel_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state.set_allowed_channels([ChannelEntryServer(name="parent")])
    parent_handler = notifications.get_notification_handler()

    async def fail(_args):
        state.set_has_dev_channels(True)
        raise RuntimeError("synthetic MCP command failure")

    monkeypatch.setattr("koder_agent.harness.mcp.commands.handle_mcp_subcommand", fail)
    with pytest.raises(RuntimeError, match="synthetic MCP command failure"):
        await run_harness_runtime(_request("failed", "child"))

    assert state.get_allowed_channels() == [ChannelEntryServer(name="parent")]
    assert not state.get_has_dev_channels()
    assert notifications.get_notification_handler() is parent_handler


@pytest.mark.asyncio
async def test_cancelled_runtime_restores_parent_channel_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state.set_allowed_channels([ChannelEntryServer(name="parent")])
    entered = asyncio.Event()
    release = asyncio.Event()

    async def command(_args):
        entered.set()
        await release.wait()
        return 0

    monkeypatch.setattr("koder_agent.harness.mcp.commands.handle_mcp_subcommand", command)
    task = asyncio.create_task(run_harness_runtime(_request("cancelled", "child")))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel("cancel-runtime")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state.get_allowed_channels() == [ChannelEntryServer(name="parent")]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_retired_runtime_cannot_reuse_a_later_same_name_allowlist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = {}
    delivered = []

    async def command(_args):
        router = ChannelNotificationRouter()

        async def receive(*args):
            delivered.append(args)

        captured["unregister"] = router.on_channel_message(receive)
        # Simulate an owner that has not completed cleanup. Runtime policy
        # retirement must still prevent a late notification from being admitted.
        admission = ChannelAdmission("same-name", router, lambda: True)
        admission.bind(
            SimpleNamespace(
                session=object(),
                server_initialize_result=SimpleNamespace(
                    capabilities={"experimental": {"claude/channel": {}}}
                ),
            )
        )
        captured["admission"] = admission
        return 0

    monkeypatch.setattr("koder_agent.harness.mcp.commands.handle_mcp_subcommand", command)
    try:
        assert await run_harness_runtime(_request("retired", "same-name")) == 0
        state.set_allowed_channels([ChannelEntryServer(name="same-name")])
        await captured["admission"]("same-name", CHANNEL_NOTIFICATION_METHOD, {"content": "late"})
        assert delivered == []
    finally:
        if "unregister" in captured:
            captured["unregister"]()
