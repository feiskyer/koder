from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from koder_agent.core.at_mentions import _read_mcp_resource
from koder_agent.mcp import _discover_prompts, discover_mcp_resources
from koder_agent.mcp.project_approvals import reset_project_choices, set_project_approval
from koder_agent.mcp.prompts import MCPPromptRegistry, execute_prompt
from koder_agent.mcp.reconnection import LiveMCPServer, ReconnectionConfig
from koder_agent.mcp.runtime_authorization import (
    MCPAuthorizationError,
    ProjectServerAuthorizationValidator,
    attach_project_authorization_validator,
    call_authorized_session,
)
from koder_agent.mcp.server_factory import MCPServerFactory
from koder_agent.mcp.server_manager import MCPServerManager


def _reset_project_choices_in_process(home: str, project: str) -> None:
    os.environ["HOME"] = home
    from koder_agent.mcp.project_approvals import reset_project_choices as reset

    if reset(project) != 1:
        raise RuntimeError("cross-process approval reset did not remove exactly one record")


class _ConnectedServer:
    name = "project-server"

    def __init__(self, session: SimpleNamespace):
        self.session = session
        self.cleaned = 0
        self.connect_impl = AsyncMock()
        self.list_tools_impl = AsyncMock(return_value=[])
        self.call_tool_impl = AsyncMock(return_value=SimpleNamespace(content=[]))

    async def connect(self):
        return await self.connect_impl()

    async def list_tools(self):
        return await self.list_tools_impl()

    async def call_tool(self, name, arguments=None):
        return await self.call_tool_impl(name, arguments)

    async def cleanup(self):
        self.cleaned += 1
        self.session = None


_PUBLIC_CALL_PATHS = (
    (("call_tool",), ("secret", {})),
    (("current", "call_tool"), ("secret", {})),
    (("session", "send_ping"), ()),
    (("current", "session", "send_ping"), ()),
    (("session", "experimental", "send_ping"), ()),
    (("send_ping",), ()),
    (("current", "send_ping"), ()),
    (("experimental", "send_ping"), ()),
)


def _resolve_public_call(handle, attribute_path):
    call = handle
    for attribute_name in attribute_path:
        call = getattr(call, attribute_name)
    return call


async def _approved_server(
    monkeypatch, tmp_path: Path
) -> tuple[Path, _ConnectedServer, SimpleNamespace]:
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    (project / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "project-server": {
                        "type": "http",
                        "url": "https://example.test/mcp",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config = await MCPServerManager().get_server("project-server", cwd=project, scope="project")
    assert config is not None
    set_project_approval(
        project_root=config.project_root,
        source_path=config.source_path,
        source_digest=config.source_digest,
        approved=True,
    )

    session = SimpleNamespace(
        list_prompts=AsyncMock(),
        get_prompt=AsyncMock(),
        list_resources=AsyncMock(),
        read_resource=AsyncMock(),
        list_resource_templates=AsyncMock(),
        subscribe_resource=AsyncMock(),
        unsubscribe_resource=AsyncMock(),
        send_ping=AsyncMock(),
    )
    server = _ConnectedServer(session)
    validator = attach_project_authorization_validator(server, config)
    assert validator is not None
    return project, server, session


async def _approved_connection_config(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    helper = tmp_path / "helper"
    helper.write_text('#!/bin/sh\nprintf \'{"Authorization": "safe"}\'\n', encoding="utf-8")
    helper.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    source = project / ".mcp.json"
    source.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "project-server": {
                        "type": "http",
                        "url": "https://example.test/mcp",
                        "headersHelper": "helper",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    config = await MCPServerManager().get_server("project-server", cwd=project, scope="project")
    assert config is not None
    set_project_approval(
        project_root=config.project_root,
        source_path=config.source_path,
        source_digest=config.source_digest,
        approved=True,
    )
    return project, source, helper, config


@pytest.mark.asyncio
async def test_registered_prompt_is_denied_after_project_approval_reset(monkeypatch, tmp_path):
    project, server, session = await _approved_server(monkeypatch, tmp_path)
    session.list_prompts.return_value = SimpleNamespace(
        prompts=[SimpleNamespace(name="review", description="Review", arguments=[])]
    )
    registry = MCPPromptRegistry()
    await _discover_prompts(server.name, server, registry)
    prompt = registry.get("mcp__project-server__review")
    assert prompt is not None
    session.list_prompts.assert_awaited_once_with()

    assert reset_project_choices(project) == 1

    with pytest.raises(RuntimeError, match="unavailable because approval was reset"):
        await execute_prompt(prompt, [server], [])

    session.get_prompt.assert_not_awaited()
    assert server.cleaned == 1
    assert server.session is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["list", "read"])
async def test_resource_discovery_and_read_are_denied_after_reset(monkeypatch, tmp_path, operation):
    project, server, session = await _approved_server(monkeypatch, tmp_path)
    assert reset_project_choices(project) == 1

    if operation == "list":
        assert await discover_mcp_resources([server]) == []
        session.list_resources.assert_not_awaited()
    else:
        assert await _read_mcp_resource(server, "config://secret") is None
        session.read_resource.assert_not_awaited()

    assert server.cleaned == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capability", "args"),
    [
        ("list_prompts", ()),
        ("get_prompt", ("review",)),
        ("list_resources", ()),
        ("read_resource", ("config://secret",)),
        ("list_resource_templates", ()),
        ("subscribe_resource", ("config://secret",)),
        ("unsubscribe_resource", ("config://secret",)),
        ("send_ping", ()),
    ],
)
async def test_every_direct_session_capability_uses_authorized_boundary(
    monkeypatch, tmp_path, capability, args
):
    project, server, session = await _approved_server(monkeypatch, tmp_path)
    assert reset_project_choices(project) == 1

    with pytest.raises(MCPAuthorizationError, match="unavailable because approval was reset"):
        await call_authorized_session(server, capability, *args)

    getattr(session, capability).assert_not_awaited()
    assert server.cleaned == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("capability", ["list_tools", "call_tool"])
async def test_tool_capabilities_share_the_same_authorization_boundary(
    monkeypatch, tmp_path, capability
):
    project, server, _session = await _approved_server(monkeypatch, tmp_path)
    assert reset_project_choices(project) == 1

    with pytest.raises(MCPAuthorizationError, match="unavailable because approval was reset"):
        if capability == "list_tools":
            await server.list_tools()
        else:
            await server.call_tool("secret", {})

    server.list_tools_impl.assert_not_awaited()
    server.call_tool_impl.assert_not_awaited()
    assert server.cleaned == 1


@pytest.mark.asyncio
async def test_concurrent_post_reset_operations_disable_once_without_server_contact(
    monkeypatch, tmp_path
):
    project, server, session = await _approved_server(monkeypatch, tmp_path)
    release = asyncio.Event()

    async def invoke_prompt():
        await release.wait()
        with pytest.raises(MCPAuthorizationError):
            await call_authorized_session(server, "get_prompt", "review")

    tasks = [asyncio.create_task(invoke_prompt()) for _ in range(12)]
    await asyncio.sleep(0)
    assert reset_project_choices(project) == 1
    release.set()
    await asyncio.gather(*tasks)

    session.get_prompt.assert_not_awaited()
    assert server.cleaned == 1


@pytest.mark.asyncio
async def test_reset_racing_runtime_validation_denies_the_operation(monkeypatch, tmp_path):
    project, server, session = await _approved_server(monkeypatch, tmp_path)
    entered_validation = threading.Event()
    reset_complete = threading.Event()
    original_revalidate = MCPServerManager.revalidate_project_config

    def gated_revalidate(manager, config, **kwargs):
        entered_validation.set()
        assert reset_complete.wait(timeout=5)
        return original_revalidate(manager, config, **kwargs)

    monkeypatch.setattr(MCPServerManager, "revalidate_project_config", gated_revalidate)

    def reset_during_validation():
        assert entered_validation.wait(timeout=5)
        assert reset_project_choices(project) == 1
        reset_complete.set()

    reset_thread = threading.Thread(target=reset_during_validation)
    reset_thread.start()
    try:
        with pytest.raises(MCPAuthorizationError, match="approval was reset"):
            await call_authorized_session(server, "get_prompt", "review")
    finally:
        reset_complete.set()
        reset_thread.join(timeout=5)

    session.get_prompt.assert_not_awaited()
    assert server.cleaned == 1


@pytest.mark.asyncio
async def test_source_drift_disables_resource_operation_without_server_contact(
    monkeypatch, tmp_path
):
    project, server, session = await _approved_server(monkeypatch, tmp_path)
    (project / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "project-server": {
                        "type": "http",
                        "url": "https://drifted.example.test/mcp",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(MCPAuthorizationError, match="source/executable drift"):
        await call_authorized_session(server, "list_resources")

    session.list_resources.assert_not_awaited()
    assert server.cleaned == 1


@pytest.mark.asyncio
async def test_reset_after_successful_preflight_before_session_admission_denies_contact(
    monkeypatch, tmp_path
):
    project, server, session = await _approved_server(monkeypatch, tmp_path)
    validator = server._koder_project_authorization_validator
    entered = asyncio.Event()
    release = asyncio.Event()

    async def pause_after_preflight():
        entered.set()
        await release.wait()

    monkeypatch.setattr(validator, "_after_preflight_validation", pause_after_preflight)
    operation = asyncio.create_task(call_authorized_session(server, "get_prompt", "review"))
    await entered.wait()
    assert reset_project_choices(project) == 1
    release.set()

    with pytest.raises(MCPAuthorizationError, match="approval was reset"):
        await operation

    session.get_prompt.assert_not_awaited()
    assert validator.in_flight == 0
    assert server.cleaned == 1


@pytest.mark.asyncio
async def test_cross_process_reset_between_preflight_and_final_admission_denies_contact(
    monkeypatch, tmp_path
):
    project, server, session = await _approved_server(monkeypatch, tmp_path)
    validator = server._koder_project_authorization_validator
    entered = asyncio.Event()
    release = asyncio.Event()

    async def pause_after_preflight():
        entered.set()
        await release.wait()

    monkeypatch.setattr(validator, "_after_preflight_validation", pause_after_preflight)
    operation = asyncio.create_task(call_authorized_session(server, "get_prompt", "review"))
    await entered.wait()

    context = multiprocessing.get_context("spawn")
    resetter = context.Process(
        target=_reset_project_choices_in_process,
        args=(str(tmp_path), str(project)),
    )
    resetter.start()
    await asyncio.to_thread(resetter.join, 60)
    assert resetter.exitcode == 0
    release.set()

    with pytest.raises(MCPAuthorizationError, match="approval was reset"):
        await operation

    session.get_prompt.assert_not_awaited()
    assert validator.in_flight == 0
    assert server.cleaned == 1


@pytest.mark.asyncio
async def test_source_drift_after_successful_preflight_before_tool_admission_denies_contact(
    monkeypatch, tmp_path
):
    project, server, _session = await _approved_server(monkeypatch, tmp_path)
    validator = server._koder_project_authorization_validator
    entered = asyncio.Event()
    release = asyncio.Event()

    async def pause_after_preflight():
        entered.set()
        await release.wait()

    monkeypatch.setattr(validator, "_after_preflight_validation", pause_after_preflight)
    operation = asyncio.create_task(server.call_tool("secret", {}))
    await entered.wait()
    (project / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "project-server": {
                        "type": "http",
                        "url": "https://drifted.example.test/mcp",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    release.set()

    with pytest.raises(MCPAuthorizationError, match="source/executable drift"):
        await operation

    server.call_tool_impl.assert_not_awaited()
    assert validator.in_flight == 0
    assert server.cleaned == 1


@pytest.mark.asyncio
async def test_admitted_operation_drains_before_cleanup_and_later_operation_is_denied(
    monkeypatch, tmp_path
):
    project, server, session = await _approved_server(monkeypatch, tmp_path)
    validator = server._koder_project_authorization_validator
    contacted = asyncio.Event()
    release = asyncio.Event()

    async def admitted_prompt(*_args, **_kwargs):
        contacted.set()
        await release.wait()
        return "admitted-result"

    session.get_prompt.side_effect = admitted_prompt
    admitted = asyncio.create_task(call_authorized_session(server, "get_prompt", "review"))
    await contacted.wait()
    assert validator.in_flight == 1
    assert reset_project_choices(project) == 1

    denied = asyncio.create_task(call_authorized_session(server, "list_resources"))
    await asyncio.sleep(0)
    session.list_resources.assert_not_awaited()
    assert server.cleaned == 0
    # Denial returns without waiting for admitted work. Otherwise a consumer
    # owning a paused stream could deadlock before it can close that stream.
    with pytest.raises(MCPAuthorizationError, match="approval was reset"):
        await denied

    release.set()
    assert await admitted == "admitted-result"

    assert validator.in_flight == 0
    session.list_resources.assert_not_awaited()
    assert server.cleaned == 1


@pytest.mark.asyncio
async def test_cleanup_requested_from_admitted_callback_does_not_deadlock(monkeypatch, tmp_path):
    _project, server, session = await _approved_server(monkeypatch, tmp_path)
    validator = server._koder_project_authorization_validator

    async def request_cleanup():
        await server.cleanup()
        return "done"

    session.send_ping.side_effect = request_cleanup

    assert await asyncio.wait_for(call_authorized_session(server, "send_ping"), timeout=2) == "done"
    assert validator.disabled is True
    assert validator.in_flight == 0
    assert server.cleaned == 1


@pytest.mark.asyncio
async def test_child_task_does_not_inherit_parent_admission_bypass(monkeypatch, tmp_path):
    project, server, session = await _approved_server(monkeypatch, tmp_path)
    child_created = asyncio.Event()
    release_child = asyncio.Event()

    async def spawn_child_operation(*_args, **_kwargs):
        async def child_operation():
            child_created.set()
            await release_child.wait()
            return await call_authorized_session(server, "list_resources")

        return await asyncio.create_task(child_operation())

    session.get_prompt.side_effect = spawn_child_operation
    parent = asyncio.create_task(call_authorized_session(server, "get_prompt", "review"))
    await child_created.wait()
    assert reset_project_choices(project) == 1
    release_child.set()

    with pytest.raises(MCPAuthorizationError, match="approval was reset"):
        await parent

    session.list_resources.assert_not_awaited()
    assert server.cleaned == 1


@pytest.mark.asyncio
async def test_connect_contact_uses_the_same_reset_before_admission_barrier(monkeypatch, tmp_path):
    project, server, _session = await _approved_server(monkeypatch, tmp_path)
    validator = server._koder_project_authorization_validator
    entered = asyncio.Event()
    release = asyncio.Event()

    async def pause_after_preflight():
        entered.set()
        await release.wait()

    monkeypatch.setattr(validator, "_after_preflight_validation", pause_after_preflight)
    connection = asyncio.create_task(server.connect())
    await entered.wait()
    assert reset_project_choices(project) == 1
    release.set()

    with pytest.raises(MCPAuthorizationError, match="approval was reset"):
        await connection

    server.connect_impl.assert_not_awaited()
    assert server.cleaned == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["reset", "source", "helper-executable"])
async def test_connection_preparation_revalidates_after_barrier_before_headers_helper(
    monkeypatch, tmp_path, drift
):
    project, source, helper, config = await _approved_connection_config(monkeypatch, tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()
    helper_call = AsyncMock(return_value={"Authorization": "unsafe"})
    transport_creation = AsyncMock(side_effect=AssertionError("transport must not be created"))

    async def pause_after_preflight(_validator):
        entered.set()
        await release.wait()

    monkeypatch.setattr(
        ProjectServerAuthorizationValidator,
        "_after_preflight_validation",
        pause_after_preflight,
    )
    monkeypatch.setattr("koder_agent.mcp.server_factory._resolve_headers_helper", helper_call)
    monkeypatch.setattr(MCPServerFactory, "_create_http_server", transport_creation)

    creation = asyncio.create_task(MCPServerFactory.create_server(config))
    await entered.wait()
    if drift == "reset":
        assert reset_project_choices(project) == 1
    elif drift == "source":
        source.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "project-server": {
                            "type": "http",
                            "url": "https://drifted.example.test/mcp",
                            "headersHelper": "helper",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
    else:
        helper.write_text('#!/bin/sh\nprintf \'{"Authorization": "drifted"}\'\n', encoding="utf-8")
        helper.chmod(0o755)
    release.set()

    with pytest.raises(MCPAuthorizationError):
        await creation

    helper_call.assert_not_awaited()
    transport_creation.assert_not_awaited()


@pytest.mark.asyncio
async def test_revocation_during_admitted_connection_preparation_cleans_once_after_boundary(
    monkeypatch, tmp_path
):
    project, _source, _helper, config = await _approved_connection_config(monkeypatch, tmp_path)
    preparation_started = asyncio.Event()
    release_preparation = asyncio.Event()
    captured_validator = None
    raw_session = SimpleNamespace(send_ping=AsyncMock())
    prepared_server = _ConnectedServer(raw_session)

    async def capture_validator(validator):
        nonlocal captured_validator
        captured_validator = validator

    async def prepare_transport(*_args, **_kwargs):
        preparation_started.set()
        await release_preparation.wait()
        return prepared_server

    monkeypatch.setattr(
        ProjectServerAuthorizationValidator,
        "_after_preflight_validation",
        capture_validator,
    )
    monkeypatch.setattr(MCPServerFactory, "_create_http_server", prepare_transport)

    creation = asyncio.create_task(MCPServerFactory.create_server(config))
    await preparation_started.wait()
    assert captured_validator is not None
    assert captured_validator.in_flight == 1
    assert reset_project_choices(project) == 1
    revocation = asyncio.create_task(captured_validator.validate())
    await asyncio.sleep(0)
    assert not revocation.done()
    assert prepared_server.cleaned == 0

    release_preparation.set()
    assert await creation is prepared_server
    assert await revocation is False
    assert prepared_server.cleaned == 1
    assert captured_validator.in_flight == 0

    with pytest.raises(MCPAuthorizationError):
        await prepared_server.connect()
    prepared_server.connect_impl.assert_not_awaited()
    assert prepared_server.cleaned == 1


@pytest.mark.asyncio
async def test_authorization_boundary_remains_on_stable_reconnect_handle(monkeypatch, tmp_path):
    project, _server, _session = await _approved_server(monkeypatch, tmp_path)
    first_session = SimpleNamespace(send_ping=AsyncMock(return_value="first"))
    second_session = SimpleNamespace(send_ping=AsyncMock(return_value="second"))

    class StableHandle:
        name = "project-server"

        def __init__(self):
            self.session = first_session
            self.next_session = second_session
            self.cleaned = 0

        async def connect(self):
            self.session = self.next_session

        async def list_tools(self):
            return []

        async def call_tool(self, _name, _arguments=None):
            return SimpleNamespace(content=[])

        async def cleanup(self):
            self.cleaned += 1
            self.session = None

    config = await MCPServerManager().get_server("project-server", cwd=project, scope="project")
    assert config is not None
    handle = StableHandle()
    stable_identity = id(handle)
    validator = attach_project_authorization_validator(handle, config)
    assert validator is not None

    assert await call_authorized_session(handle, "send_ping") == "first"
    await handle.connect()
    assert id(handle) == stable_identity
    assert await call_authorized_session(handle, "send_ping") == "second"
    first_session.send_ping.assert_awaited_once_with()
    second_session.send_ping.assert_awaited_once_with()

    # The combined approval+runtime branch supplies LiveMCPServer as this
    # stable reconnect identity; this test only fixes the authorization-layer
    # contract and deliberately does not duplicate that implementation here.
    assert reset_project_choices(project) == 1
    with pytest.raises(MCPAuthorizationError):
        await call_authorized_session(handle, "send_ping")
    assert handle.cleaned == 1


@pytest.mark.asyncio
async def test_factory_promotes_project_authorization_to_live_handle(monkeypatch, tmp_path):
    project, _source, _helper, config = await _approved_connection_config(monkeypatch, tmp_path)
    session = SimpleNamespace(send_ping=AsyncMock(return_value="connected"))
    concrete = _ConnectedServer(session)
    monkeypatch.setattr(
        MCPServerFactory,
        "_create_http_server",
        AsyncMock(return_value=concrete),
    )

    handle, manager = await MCPServerFactory.create_and_connect_with_retry(
        config,
        reconnection_config=ReconnectionConfig(
            initial_delay=0,
            max_delay=0,
            max_attempts=1,
        ),
    )

    assert isinstance(handle, LiveMCPServer)
    assert manager.server is handle
    assert getattr(handle, "_koder_project_authorization_validator", None) is not None
    assert getattr(concrete, "_koder_project_authorization_validator", None) is None
    assert await call_authorized_session(handle, "send_ping") == "connected"

    assert reset_project_choices(project) == 1
    with pytest.raises(MCPAuthorizationError, match="approval was reset"):
        await call_authorized_session(handle, "send_ping")

    assert concrete.cleaned == 1
    assert handle._koder_current_transport() is None
    assert await manager.reconnect_if_needed() is False
    assert manager.server is handle
    await manager.cleanup()
    assert manager.server is None
    assert concrete.cleaned == 1


@pytest.mark.asyncio
async def test_live_session_proxy_does_not_expose_concrete_session(monkeypatch, tmp_path):
    _project, _source, _helper, config = await _approved_connection_config(monkeypatch, tmp_path)
    session = SimpleNamespace(send_ping=AsyncMock(return_value="connected"))
    concrete = _ConnectedServer(session)
    monkeypatch.setattr(
        MCPServerFactory,
        "_create_http_server",
        AsyncMock(return_value=concrete),
    )

    handle, manager = await MCPServerFactory.create_and_connect_with_retry(
        config,
        reconnection_config=ReconnectionConfig(
            initial_delay=0,
            max_delay=0,
            max_attempts=1,
        ),
    )
    public_session = handle.session

    with pytest.raises(AttributeError, match="raw_session"):
        _ = public_session.raw_session
    with pytest.raises(AttributeError, match="_koder_raw_session"):
        _ = public_session._koder_raw_session

    assert handle._koder_raw_session() is session
    assert await public_session.send_ping() == "connected"

    await manager.cleanup()
    assert concrete.cleaned == 1


@pytest.mark.asyncio
async def test_project_reconnect_keeps_stable_authorization_and_cleanup(monkeypatch, tmp_path):
    project, _source, _helper, config = await _approved_connection_config(monkeypatch, tmp_path)
    first_session = SimpleNamespace(send_ping=AsyncMock(return_value="first"))
    second_session = SimpleNamespace(send_ping=AsyncMock(return_value="second"))
    first = _ConnectedServer(first_session)
    second = _ConnectedServer(second_session)
    create_transport = AsyncMock(side_effect=[first, second])
    monkeypatch.setattr(MCPServerFactory, "_create_http_server", create_transport)

    handle, manager = await MCPServerFactory.create_and_connect_with_retry(
        config,
        reconnection_config=ReconnectionConfig(
            initial_delay=0,
            max_delay=0,
            max_attempts=1,
        ),
    )
    stable_identity = id(handle)
    stable_validator = handle._koder_project_authorization_validator
    assert await call_authorized_session(handle, "send_ping") == "first"

    first.session = None
    assert await manager.reconnect_if_needed() is True
    assert id(handle) == stable_identity
    assert handle._koder_project_authorization_validator is stable_validator
    assert handle._koder_current_transport() is second
    assert getattr(second, "_koder_project_authorization_validator", None) is None
    assert first.cleaned == 1
    assert await call_authorized_session(handle, "send_ping") == "second"

    assert reset_project_choices(project) == 1
    with pytest.raises(MCPAuthorizationError, match="approval was reset"):
        await call_authorized_session(handle, "send_ping")

    assert second.cleaned == 1
    assert await manager.reconnect_if_needed() is False
    await manager.cleanup()
    assert first.cleaned == 1
    assert second.cleaned == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(("public_path", "call_args"), _PUBLIC_CALL_PATHS)
async def test_public_call_path_pins_old_transport_during_reconnect(
    monkeypatch, tmp_path, public_path, call_args
):
    _project, _source, _helper, config = await _approved_connection_config(monkeypatch, tmp_path)
    old_call_started = asyncio.Event()
    release_old_call = asyncio.Event()

    async def blocking_ping(*_args, **_kwargs):
        old_call_started.set()
        await release_old_call.wait()
        return "old"

    first_direct = AsyncMock(side_effect=blocking_ping)
    first_nested = AsyncMock(side_effect=blocking_ping)
    first_session = SimpleNamespace(
        send_ping=AsyncMock(side_effect=blocking_ping),
        experimental=SimpleNamespace(send_ping=AsyncMock(side_effect=blocking_ping)),
    )
    second_direct = AsyncMock(return_value="new")
    second_nested = AsyncMock(return_value="new")
    second_session = SimpleNamespace(
        send_ping=AsyncMock(return_value="new"),
        experimental=SimpleNamespace(send_ping=AsyncMock(return_value="new")),
    )
    first = _ConnectedServer(first_session)
    first.call_tool_impl = AsyncMock(side_effect=blocking_ping)
    first.send_ping = first_direct
    first.experimental = SimpleNamespace(send_ping=first_nested)
    second = _ConnectedServer(second_session)
    second.call_tool_impl = AsyncMock(return_value="new")
    second.send_ping = second_direct
    second.experimental = SimpleNamespace(send_ping=second_nested)
    monkeypatch.setattr(
        MCPServerFactory,
        "_create_http_server",
        AsyncMock(side_effect=[first, second]),
    )

    handle, manager = await MCPServerFactory.create_and_connect_with_retry(
        config,
        reconnection_config=ReconnectionConfig(
            initial_delay=0,
            max_delay=0,
            max_attempts=1,
        ),
    )
    retained_call = _resolve_public_call(handle, public_path)

    old_call = asyncio.create_task(retained_call(*call_args))
    await old_call_started.wait()
    first.session = None
    reconnect = asyncio.create_task(manager.reconnect_if_needed())
    for _ in range(100):
        if handle._koder_current_transport() is second:
            break
        await asyncio.sleep(0)

    assert handle._koder_current_transport() is second
    assert not reconnect.done()
    assert first.cleaned == 0
    assert await retained_call(*call_args) == "new"

    release_old_call.set()
    assert await old_call == "old"
    assert await reconnect is True
    assert first.cleaned == 1

    await manager.cleanup()
    assert second.cleaned == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "public_path",
    [
        ("session", "stream_events"),
        ("current", "session", "stream_events"),
        ("session", "experimental", "stream_events"),
        ("stream_events",),
        ("current", "stream_events"),
        ("experimental", "stream_events"),
    ],
)
async def test_public_async_iterator_path_pins_old_transport_during_reconnect(
    monkeypatch, tmp_path, public_path
):
    _project, _source, _helper, config = await _approved_connection_config(monkeypatch, tmp_path)
    old_iteration_started = asyncio.Event()
    release_old_iteration = asyncio.Event()

    async def old_events():
        old_iteration_started.set()
        yield "old-started"
        await release_old_iteration.wait()
        yield "old-finished"

    async def new_events():
        yield "new"

    first_session = SimpleNamespace(
        stream_events=old_events,
        experimental=SimpleNamespace(stream_events=old_events),
    )
    second_session = SimpleNamespace(
        stream_events=new_events,
        experimental=SimpleNamespace(stream_events=new_events),
    )
    first = _ConnectedServer(first_session)
    first.stream_events = old_events
    first.experimental = SimpleNamespace(stream_events=old_events)
    second = _ConnectedServer(second_session)
    second.stream_events = new_events
    second.experimental = SimpleNamespace(stream_events=new_events)
    monkeypatch.setattr(
        MCPServerFactory,
        "_create_http_server",
        AsyncMock(side_effect=[first, second]),
    )

    handle, manager = await MCPServerFactory.create_and_connect_with_retry(
        config,
        reconnection_config=ReconnectionConfig(
            initial_delay=0,
            max_delay=0,
            max_attempts=1,
        ),
    )
    retained_stream = _resolve_public_call(handle, public_path)

    async def consume_retained_stream():
        return [item async for item in retained_stream()]

    old_stream = asyncio.create_task(consume_retained_stream())
    await old_iteration_started.wait()
    first.session = None
    reconnect = asyncio.create_task(manager.reconnect_if_needed())
    for _ in range(100):
        if handle._koder_current_transport() is second:
            break
        await asyncio.sleep(0)

    assert handle._koder_current_transport() is second
    assert not reconnect.done()
    assert first.cleaned == 0
    assert await consume_retained_stream() == ["new"]

    release_old_iteration.set()
    assert await old_stream == ["old-started", "old-finished"]
    assert await reconnect is True
    assert first.cleaned == 1

    await manager.cleanup()
    assert second.cleaned == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["reset", "source"])
@pytest.mark.parametrize(("public_path", "call_args"), _PUBLIC_CALL_PATHS)
async def test_retained_public_call_path_denies_reset_and_source_drift_without_contact(
    monkeypatch, tmp_path, drift, public_path, call_args
):
    project, source, _helper, config = await _approved_connection_config(monkeypatch, tmp_path)
    session = SimpleNamespace(
        send_ping=AsyncMock(return_value="unsafe"),
        experimental=SimpleNamespace(send_ping=AsyncMock(return_value="unsafe")),
    )
    concrete = _ConnectedServer(session)
    concrete.send_ping = AsyncMock(return_value="unsafe")
    concrete.experimental = SimpleNamespace(send_ping=AsyncMock(return_value="unsafe"))
    monkeypatch.setattr(
        MCPServerFactory,
        "_create_http_server",
        AsyncMock(return_value=concrete),
    )

    handle, manager = await MCPServerFactory.create_and_connect_with_retry(
        config,
        reconnection_config=ReconnectionConfig(
            initial_delay=0,
            max_delay=0,
            max_attempts=1,
        ),
    )
    retained_call = _resolve_public_call(handle, public_path)

    if drift == "reset":
        assert reset_project_choices(project) == 1
    else:
        source.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "project-server": {
                            "type": "http",
                            "url": "https://drifted.example.test/mcp",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(MCPAuthorizationError, match="approval was reset|source/executable drift"):
        await retained_call(*call_args)

    concrete.call_tool_impl.assert_not_awaited()
    concrete.send_ping.assert_not_awaited()
    concrete.experimental.send_ping.assert_not_awaited()
    session.send_ping.assert_not_awaited()
    session.experimental.send_ping.assert_not_awaited()
    assert concrete.cleaned == 1
    assert handle._koder_current_transport() is None

    await manager.cleanup()


@pytest.mark.asyncio
async def test_reconnect_reset_before_final_admission_makes_no_candidate_contact(
    monkeypatch, tmp_path
):
    project, _source, _helper, config = await _approved_connection_config(monkeypatch, tmp_path)
    first = _ConnectedServer(SimpleNamespace(send_ping=AsyncMock(return_value="first")))
    candidate = _ConnectedServer(SimpleNamespace(send_ping=AsyncMock(return_value="candidate")))
    create_transport = AsyncMock(side_effect=[first, candidate])
    monkeypatch.setattr(MCPServerFactory, "_create_http_server", create_transport)

    handle, manager = await MCPServerFactory.create_and_connect_with_retry(
        config,
        reconnection_config=ReconnectionConfig(
            initial_delay=0,
            max_delay=0,
            max_attempts=1,
        ),
    )
    validator = handle._koder_project_authorization_validator
    entered = asyncio.Event()
    release = asyncio.Event()

    async def pause_after_preflight():
        entered.set()
        await release.wait()

    monkeypatch.setattr(validator, "_after_preflight_validation", pause_after_preflight)
    first.session = None
    reconnect = asyncio.create_task(manager.reconnect_if_needed())
    await entered.wait()
    assert reset_project_choices(project) == 1
    release.set()

    assert await reconnect is False
    assert create_transport.await_count == 1
    candidate.connect_impl.assert_not_awaited()
    assert manager.server is handle
    assert handle._koder_current_transport() is None
    assert first.cleaned == 1

    await manager.cleanup()


@pytest.mark.asyncio
async def test_authorization_cleanup_failure_remains_owned_for_shutdown_retry(
    monkeypatch, tmp_path
):
    project, _source, _helper, config = await _approved_connection_config(monkeypatch, tmp_path)

    class FailingCleanupServer(_ConnectedServer):
        def __init__(self, session):
            super().__init__(session)
            self.allow_cleanup = False

        async def cleanup(self):
            self.cleaned += 1
            if not self.allow_cleanup:
                raise RuntimeError("cleanup still failing")
            self.session = None

    concrete = FailingCleanupServer(SimpleNamespace(send_ping=AsyncMock(return_value="connected")))
    monkeypatch.setattr(
        MCPServerFactory,
        "_create_http_server",
        AsyncMock(return_value=concrete),
    )
    handle, manager = await MCPServerFactory.create_and_connect_with_retry(
        config,
        reconnection_config=ReconnectionConfig(
            initial_delay=0,
            max_delay=0,
            max_attempts=1,
        ),
    )
    assert reset_project_choices(project) == 1

    with pytest.raises(MCPAuthorizationError, match="approval was reset"):
        await call_authorized_session(handle, "send_ping")

    assert handle._koder_current_transport() is None
    assert manager.server is handle
    assert manager._retirement_owner.pending_count == 1
    assert concrete.cleaned == 3

    concrete.allow_cleanup = True
    await manager.cleanup()
    assert concrete.cleaned == 4
    assert manager._retirement_owner.pending_count == 0
    await manager.cleanup()
    assert concrete.cleaned == 4


@pytest.mark.asyncio
async def test_authorization_cleanup_cancellation_is_not_demoted_and_retries(monkeypatch, tmp_path):
    project, _source, _helper, config = await _approved_connection_config(monkeypatch, tmp_path)

    class CancelOnceCleanupServer(_ConnectedServer):
        async def cleanup(self):
            self.cleaned += 1
            if self.cleaned == 1:
                raise asyncio.CancelledError()
            self.session = None

    concrete = CancelOnceCleanupServer(
        SimpleNamespace(send_ping=AsyncMock(return_value="connected"))
    )
    monkeypatch.setattr(
        MCPServerFactory,
        "_create_http_server",
        AsyncMock(return_value=concrete),
    )
    handle, manager = await MCPServerFactory.create_and_connect_with_retry(
        config,
        reconnection_config=ReconnectionConfig(
            initial_delay=0,
            max_delay=0,
            max_attempts=1,
        ),
    )
    assert reset_project_choices(project) == 1

    with pytest.raises(asyncio.CancelledError):
        await call_authorized_session(handle, "send_ping")

    assert handle._koder_current_transport() is None
    assert manager.server is handle
    assert manager._retirement_owner.pending_count == 1
    assert concrete.cleaned == 1

    await manager.cleanup()
    assert concrete.cleaned == 2
    assert manager._retirement_owner.pending_count == 0
    await manager.cleanup()
    assert concrete.cleaned == 2
