"""Real CLI processes with synthetic sandbox compute and no model/network I/O."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

_PROBE = """
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import requests

output_format, cleanup_fails, log_path = sys.argv[1:]
events = Path(log_path)

def record(event):
    with events.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\\n")

def no_network(*args, **kwargs):
    record("unexpected-network")
    raise AssertionError("Network I/O is forbidden in the CLI fixture")

async def no_async_network(*args, **kwargs):
    return no_network(*args, **kwargs)

httpx.Client.send = no_network
httpx.AsyncClient.send = no_async_network
requests.Session.send = no_network

from koder_agent.harness.sandbox import registry, sdk_backend
from koder_agent.harness.sandbox.backend import (
    SandboxBackendCapabilities,
    SandboxBackendStatus,
)
from koder_agent.harness.sandbox.policy import SandboxPolicy
from koder_agent.harness.tools import shell_executor

capabilities = SandboxBackendCapabilities(
    supports_host_process_isolation="enforced",
    supports_workspace_isolation="enforced",
    supports_repository_sync="enforced",
    supports_read_only_filesystem="enforced",
    supports_network_policy="enforced",
    supports_domain_policy="enforced",
    supports_protected_paths="enforced",
)
status = SandboxBackendStatus(
    backend_id="docker", selected=True, available=True,
    reason="synthetic CLI compute", capabilities=capabilities,
)
policy = SandboxPolicy(backend="docker", network_access=True)
state = SimpleNamespace(
    enabled=True, backend="docker", policy=policy,
    backend_available=True, backend_statuses=(status,),
)
registry.get_backend_status = lambda *_a, **_kw: status
sdk_backend.get_backend_status = lambda *_a, **_kw: status
shell_executor.resolve_sandbox_settings = lambda *_a, **_kw: state
shell_executor.is_excluded_command = lambda *_a, **_kw: False

class Session:
    async def __aenter__(self):
        record("enter")
        return self

    async def __aexit__(self, *_error):
        record("exit")

    async def exec(self, command, *, timeout, shell):
        assert command == "printf sandbox-cli"
        assert timeout > 0 and shell is True
        record("exec")
        return SimpleNamespace(stdout=b"sandbox-cli-ran", stderr=b"", exit_code=0)

class Client:
    async def create(self, **_kwargs):
        record("create")
        return Session()

    async def delete(self, _session):
        record("delete")
        if cleanup_fails == "yes":
            raise RuntimeError("synthetic-cleanup-credential")

sdk_backend.create_backend_client_and_options = lambda *_a, **_kw: (Client(), None)

from koder_agent.cli import run
sys.argv = [
    "koder", "--bare", "--output-format", output_format,
    "--print", "!printf sandbox-cli",
]
run()
"""


@pytest.mark.parametrize("output_format", ["text", "json", "stream-json"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_real_cli_reports_sandbox_lifecycle_outcome(
    tmp_path, python_child_environment, output_format, cleanup_fails
):
    probe = tmp_path / "sandbox_cli_probe.py"
    probe.write_text(textwrap.dedent(_PROBE), encoding="utf-8")
    events = tmp_path / "sandbox-events.jsonl"
    environment = {
        **python_child_environment,
        "KODER_API_KEY": "synthetic-cli-key",
        "KODER_BASE_URL": "http://127.0.0.1:1/v1",
        "KODER_MODEL": "gpt-4.1",
        "KODER_SMALL_MODEL": "gpt-4.1",
        "KODER_NO_UPDATE_CHECK": "1",
    }

    completed = subprocess.run(
        [
            "uv",
            "run",
            "--quiet",
            "--no-project",
            "--no-env-file",
            "--python",
            sys.executable,
            "python",
            str(probe),
            output_format,
            "yes" if cleanup_fails else "no",
            str(events),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == int(cleanup_fails), completed.stderr
    assert [json.loads(line) for line in events.read_text().splitlines()] == [
        "create",
        "enter",
        "exec",
        "exit",
        "delete",
    ]
    assert "sandbox-cli-ran" in completed.stdout
    assert "synthetic-cleanup-credential" not in completed.stdout + completed.stderr
    if output_format != "text":
        payload = json.loads(completed.stdout)
        assert payload.get("is_error", False) is cleanup_fails
        if cleanup_fails:
            assert payload["exit_code"] == 1
        if output_format == "stream-json":
            assert payload["type"] == "result"
    if cleanup_fails:
        assert "sandbox cleanup failed" in completed.stdout
