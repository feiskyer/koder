"""Exercise the real installed SDK, OS pipes, handshake and lifecycle.

Each probe owns a subprocess group so a regression in AnyIO scope unwinding
fails within a deadline instead of hanging pytest or leaving its MCP child.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

PEER = Path(__file__).resolve().parents[1] / "fixtures/mcp_stdio_server.py"
PROBE = r"""
import asyncio
import json
import sys
from pathlib import Path
from koder_agent.mcp.server_config import MCPServerConfig, MCPServerType
from koder_agent.mcp.server_factory import MCPServerFactory
from koder_agent.mcp.reconnection import ReconnectionConfig
from koder_agent.harness.channels.admission import ChannelAdmission
from koder_agent.harness.channels.notification import ChannelNotificationRouter
from koder_agent.harness.channels.state import set_allowed_channels
from koder_agent.harness.channels.types import ChannelEntryServer

async def main():
    peer, trace, mode, channel, cancel = sys.argv[1:]
    received = []
    async def on_channel(*values):
        received.append(values)
    router = ChannelNotificationRouter()
    async def on_message(name, content, meta):
        await on_channel(name, "notifications/claude/channel", {"content": content})
    router.on_channel_message(on_message)
    set_allowed_channels([ChannelEntryServer(name="startup-fixture")])
    admission = ChannelAdmission("startup-fixture", router, lambda: True)
    config = MCPServerConfig(
        name="startup-fixture", transport_type=MCPServerType.STDIO,
        command=sys.executable, args=[peer, "--trace", trace, "--mode", mode],
    )
    factory = MCPServerFactory.create_and_connect_with_retry(
        config, channel_callback=admission if channel == "yes" else None,
        reconnection_config=ReconnectionConfig(max_attempts=1),
    )
    connecting = asyncio.create_task(factory)
    if cancel == "yes":
        for _ in range(500):
            if Path(trace).exists() and "initialize" in Path(trace).read_text():
                break
            if connecting.done():
                await connecting
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("SDK never sent initialize")
        connecting.cancel("cancel startup")
        try:
            await connecting
        except asyncio.CancelledError:
            print("CANCELLED_AND_JOINED")
        else:
            raise AssertionError("startup cancellation was swallowed")
        return
    try:
        server, manager = await connecting
        admission.bind(server)
    except Exception as exc:
        if mode != "normal":
            print("FAILED_WITHOUT_HANG", type(exc).__name__)
            return
        raise
    try:
        tools = await server.list_tools()
        assert [tool.name for tool in tools] == ["startup_echo"]
        result = await server.call_tool("startup_echo", {"channel": channel == "yes"})
        assert result.content[0].text == "MCP_HANDSHAKE_OK"
        result = await server.call_tool("startup_echo", {"elicit": True})
        assert result.content[0].text == "ELICITATION_OK"
        if channel == "yes":
            assert received[0][0] == "startup-fixture"
            assert received[0][2]["content"] == "channel handshake verified"
        print("HANDSHAKE_TOOLS_ELICITATION_OK")
    finally:
        # Real callers retire on another task; the transport must keep its own
        # same-task context-manager ownership despite concurrent shutdown.
        await asyncio.gather(server.cleanup(), server.cleanup())
    print("CLEANUP_JOINED")

asyncio.run(main())
"""


@pytest.mark.parametrize(
    ("mode", "channel", "cancel", "expected"),
    [
        ("normal", False, False, "HANDSHAKE_TOOLS_ELICITATION_OK"),
        ("normal", True, False, "HANDSHAKE_TOOLS_ELICITATION_OK"),
        ("fail", False, False, "FAILED_WITHOUT_HANG"),
        ("silent", False, False, "FAILED_WITHOUT_HANG"),
        ("silent", False, True, "CANCELLED_AND_JOINED"),
        ("silent", True, True, "CANCELLED_AND_JOINED"),
    ],
)
def test_real_sdk_startup_and_shutdown(
    tmp_path, python_child_environment, mode, channel, cancel, expected
):
    trace = tmp_path / "mcp-trace.jsonl"
    environment = {**python_child_environment, "MCP_TIMEOUT": "1500"}
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            PROBE,
            str(PEER),
            str(trace),
            mode,
            "yes" if channel else "no",
            "yes" if cancel else "no",
        ],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, stderr = process.communicate(timeout=5)
        pytest.fail(f"Real MCP startup/cleanup hung:\n{stdout}\n{stderr}")
    assert process.returncode == 0, stdout + stderr
    assert expected in stdout, stdout + stderr
    assert "cancel scope" not in stderr.lower(), stderr
    assert "TypeError" not in stderr, stderr
    if mode == "fail":
        assert "fixture failure" in stderr, stderr
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert any(event.get("method") == "initialize" for event in events)
    if mode == "normal":
        assert "CLEANUP_JOINED" in stdout
        assert any(event.get("method") == "tools/list" for event in events)
        capabilities = next(event["capabilities"] for event in events if "capabilities" in event)
        assert "elicitation" in capabilities
        assert any(event.get("result", {}).get("action") == "cancel" for event in events)
        assert "unsupported or invalid form data" not in stderr
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(events[0]["pid"], 0)
