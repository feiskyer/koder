import asyncio
import gc
import subprocess
import sys
from types import SimpleNamespace

import pytest

from koder_agent import cli
from koder_agent.harness.runtime import HarnessRuntime
from koder_agent.mcp import MCPServerSet, drain_orphaned_mcp_owners


def _abandon_cancel_once_owner(events: list[str]) -> None:
    class CancelOnceServer:
        name = "cancel-once"

        def __init__(self) -> None:
            self.cleanup_count = 0

        async def cleanup(self) -> None:
            self.cleanup_count += 1
            events.append(f"cleanup-{self.cleanup_count}")
            if self.cleanup_count == 1:
                raise asyncio.CancelledError

    owner = MCPServerSet([CancelOnceServer()])
    del owner
    gc.collect()


def test_cli_argv_flows_through_runtime_request_builder(monkeypatch):
    called = {}

    def fake_build(argv):
        called["argv"] = argv
        return type("Req", (), {"argv": argv, "mode": "help"})()

    monkeypatch.setattr("koder_agent.harness.cli.entrypoint.build_runtime_request", fake_build)
    cli._build_runtime_request_for_test(["--help"])
    assert called["argv"] == ["--help"]


def test_main_delegates_top_level_help_to_harness_runtime(monkeypatch):
    called = {}

    async def fake_run(request):
        called["request"] = request
        return 0

    monkeypatch.setattr("koder_agent.harness.cli.entrypoint.run_harness_runtime", fake_run)
    monkeypatch.setattr(cli.sys, "argv", ["koder", "--help"])

    assert asyncio.run(cli.main()) == 0
    assert called["request"].mode == "help"
    assert "usage: koder" in called["request"].help_text


def test_main_delegates_empty_boot_to_harness_runtime(monkeypatch):
    called = {}

    async def fake_run(request):
        called["request"] = request
        return 0

    monkeypatch.setattr("koder_agent.harness.cli.entrypoint.run_harness_runtime", fake_run)
    monkeypatch.setattr("koder_agent.utils.setup_openai_client", lambda: None)
    monkeypatch.setattr(cli.sys, "argv", ["koder"])

    assert asyncio.run(cli.main()) == 0
    assert called["request"].mode == "interactive"


def test_harness_runtime_interactive_uses_harness_session_flow(monkeypatch):
    called = {}

    async def fake_harness_flow(*, first_arg, argv, permission_service=None):
        called["first_arg"] = first_arg
        called["argv"] = argv
        return 0

    monkeypatch.setattr("koder_agent.harness.runtime.run_harness_session_flow", fake_harness_flow)

    runtime = HarnessRuntime(request=SimpleNamespace(mode="interactive", argv=[]))
    assert asyncio.run(runtime.run()) == 0
    assert called["first_arg"] is None
    assert called["argv"] == []


def test_harness_runtime_drains_orphaned_mcp_owners_on_shutdown(monkeypatch):
    drain_calls: list[str] = []

    async def fake_owner_drain():
        drain_calls.append("owner")

    async def fake_transport_drain():
        drain_calls.append("transport")

    monkeypatch.setattr(
        "koder_agent.mcp.drain_orphaned_mcp_owners",
        fake_owner_drain,
    )
    monkeypatch.setattr(
        "koder_agent.mcp.reconnection.drain_orphaned_retirements",
        fake_transport_drain,
    )

    runtime = HarnessRuntime(request=SimpleNamespace(mode="help", argv=[], help_text="help"))

    assert asyncio.run(runtime.run()) == 0
    assert drain_calls == ["owner", "transport"]


def test_harness_runtime_transport_drain_survives_owner_drain_failure(monkeypatch):
    drain_calls: list[str] = []

    async def fail_owner_drain():
        drain_calls.append("owner")
        raise RuntimeError("owner cleanup failed")

    async def fake_transport_drain():
        drain_calls.append("transport")

    monkeypatch.setattr(
        "koder_agent.mcp.drain_orphaned_mcp_owners",
        fail_owner_drain,
    )
    monkeypatch.setattr(
        "koder_agent.mcp.reconnection.drain_orphaned_retirements",
        fake_transport_drain,
    )

    runtime = HarnessRuntime(request=SimpleNamespace(mode="help", argv=[], help_text="help"))

    assert asyncio.run(runtime.run()) == 0
    assert drain_calls == ["owner", "transport"]


def test_harness_runtime_preserves_return_7_when_orphan_cleanup_is_cancelled(monkeypatch):
    events: list[str] = []
    _abandon_cancel_once_owner(events)

    async def return_seven(self):
        events.append("runtime-return-7")
        return 7

    monkeypatch.setattr(HarnessRuntime, "_run", return_seven)
    runtime = HarnessRuntime(request=SimpleNamespace(mode="help", argv=[], help_text="help"))

    assert asyncio.run(runtime.run()) == 7
    assert events == ["runtime-return-7", "cleanup-1"]

    asyncio.run(drain_orphaned_mcp_owners())
    assert events == ["runtime-return-7", "cleanup-1", "cleanup-2"]


def test_harness_runtime_preserves_original_exception_when_orphan_cleanup_is_cancelled(
    monkeypatch,
):
    events: list[str] = []
    _abandon_cancel_once_owner(events)

    async def fail_runtime(self):
        events.append("runtime-error")
        raise RuntimeError("runtime failed")

    monkeypatch.setattr(HarnessRuntime, "_run", fail_runtime)
    runtime = HarnessRuntime(request=SimpleNamespace(mode="help", argv=[], help_text="help"))

    with pytest.raises(RuntimeError, match="runtime failed"):
        asyncio.run(runtime.run())
    assert events == ["runtime-error", "cleanup-1"]

    asyncio.run(drain_orphaned_mcp_owners())
    assert events == ["runtime-error", "cleanup-1", "cleanup-2"]


def test_harness_runtime_propagates_caller_cancellation_during_orphan_drain(monkeypatch):
    async def scenario():
        cleanup_started = asyncio.Event()
        cleanup_release = asyncio.Event()

        class SlowServer:
            name = "slow"

            async def cleanup(self):
                cleanup_started.set()
                await cleanup_release.wait()

        owner = MCPServerSet([SlowServer()])
        del owner
        gc.collect()

        async def return_seven(self):
            return 7

        monkeypatch.setattr(HarnessRuntime, "_run", return_seven)
        runtime = HarnessRuntime(request=SimpleNamespace(mode="help", argv=[], help_text="help"))
        task = asyncio.create_task(runtime.run())
        await cleanup_started.wait()
        task.cancel()
        cleanup_release.set()

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_cli_process_exits_7_when_runtime_cleanup_is_cancelled_once(
    tmp_path, python_child_environment
):
    marker = tmp_path / "runtime-exit-7.txt"
    script = """
import asyncio
import gc
import sys
from pathlib import Path
from types import SimpleNamespace

from koder_agent import cli
from koder_agent.harness.runtime import HarnessRuntime
from koder_agent.mcp import MCPServerSet

marker = Path(sys.argv[1])


def record(event):
    with marker.open("a", encoding="utf-8") as handle:
        handle.write(f"{event}\\n")


class CancelOnceServer:
    name = "cancel-once"

    def __init__(self):
        self.cleanup_count = 0

    async def cleanup(self):
        self.cleanup_count += 1
        record(f"cleanup-{self.cleanup_count}")
        if self.cleanup_count == 1:
            raise asyncio.CancelledError


async def return_seven(self):
    record("runtime-return-7")
    return 7


async def main():
    runtime = HarnessRuntime(
        request=SimpleNamespace(mode="help", argv=[], help_text="help")
    )
    return await runtime.run()


HarnessRuntime._run = return_seven
cli.main = main
owner = MCPServerSet([CancelOnceServer()])
del owner
gc.collect()
cli.run()
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(marker)],
        cwd=tmp_path,
        env=python_child_environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 7, result.stderr
    assert marker.read_text(encoding="utf-8").splitlines() == [
        "runtime-return-7",
        "cleanup-1",
        "cleanup-2",
    ]
