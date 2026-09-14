# ruff: noqa: E402
"""Graceful-degradation tests for the sandbox init-failure path.

When the sandbox backend reports ``unavailable``/``unsupported``,
``execute_shell_command`` must:

- keep the fail-closed error when no approval callback is supplied (safe default),
- keep the error when an approval callback denies,
- fall through to a real UNSANDBOXED foreground run, with a visible warning,
  only when an approval callback approves,
- leave the sandboxed path untouched when the backend IS available.
"""

import asyncio
import sys
import types
from pathlib import Path

import pytest

# Stub litellm before importing koder_agent to avoid optional dependency issues.
if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import koder_agent.harness.tools.shell_executor as se
from koder_agent.harness.sandbox.backend import (
    SandboxBackendCapabilities,
    SandboxBackendStatus,
    SandboxExecutionResult,
)
from koder_agent.harness.sandbox.policy import SandboxPolicy
from koder_agent.tools.shell import build_sandbox_unavailable_approval


def _force_sandbox_state(
    monkeypatch,
    tmp_path,
    *,
    backend_result: SandboxExecutionResult,
    capabilities: SandboxBackendCapabilities | None = None,
    backend_id: str = "unix-local",
    policy_config: dict | None = None,
):
    """Force the sandboxed branch and stub the backend to a fixed result.

    Fabricates an enabled sandbox state carrying a policy, stubs the
    excluded-command check to False, and replaces the SDK backend with a fake
    that returns ``backend_result``. Returns the fake backend's call log.
    """
    policy = SandboxPolicy.from_config(
        {
            "enabled": True,
            "mode": "workspace-write",
            "backend": backend_id,
            **(policy_config or {}),
        }
    )
    capabilities = capabilities or SandboxBackendCapabilities(
        supports_host_process_isolation="enforced",
        supports_workspace_isolation="enforced",
        supports_repository_sync="enforced",
        supports_read_only_filesystem="enforced",
        supports_network_policy="enforced",
        supports_domain_policy="enforced",
        supports_protected_paths="enforced",
    )
    backend_status = SandboxBackendStatus(
        backend_id=backend_id,
        selected=True,
        available=True,
        reason="available",
        capabilities=capabilities,
    )
    real_state = se.resolve_sandbox_settings(str(tmp_path))
    forced = real_state.__class__(
        **{
            **real_state.__dict__,
            "enabled": True,
            "policy": policy,
            "backend": backend_id,
            "backend_available": True,
            "backend_statuses": (backend_status,),
        }
    )
    monkeypatch.setattr(se, "resolve_sandbox_settings", lambda *_a, **_k: forced)
    monkeypatch.setattr(se, "is_excluded_command", lambda *_a, **_k: False)

    calls: list[dict] = []

    async def _fake_backend(ctx):
        calls.append({"command": ctx.command})
        return backend_result

    monkeypatch.setattr(se, "execute_with_sdk_backend", _fake_backend)
    return calls


# --- unavailable + approval granted -> runs UNSANDBOXED with a warning --------


def test_unavailable_with_approval_runs_unsandboxed_with_warning(monkeypatch, tmp_path):
    _force_sandbox_state(
        monkeypatch,
        tmp_path,
        backend_result=SandboxExecutionResult(status="unavailable", reason="docker not found"),
    )
    monkeypatch.chdir(tmp_path)

    result = asyncio.run(
        se.execute_shell_command(
            "echo degraded-ok",
            timeout=10,
            sandbox_unavailable_approval=lambda reason: True,
        )
    )

    # The command actually ran unsandboxed.
    assert result.status == "success"
    assert result.exit_code == 0
    assert "degraded-ok" in result.output
    # A one-line warning made the degradation visible, including the reason.
    assert "warning: sandbox unavailable" in result.output
    assert "docker not found" in result.output
    assert "UNSANDBOXED" in result.output


def test_unavailable_fallback_reason_enumerates_exact_read_only_policy_losses(
    monkeypatch, tmp_path
):
    capabilities = SandboxBackendCapabilities(
        supports_host_process_isolation="enforced",
        supports_workspace_isolation="enforced",
        supports_repository_sync="enforced",
        supports_read_only_filesystem="enforced",
        supports_network_policy="enforced",
        supports_domain_policy="enforced",
        supports_protected_paths="enforced",
    )
    _force_sandbox_state(
        monkeypatch,
        tmp_path,
        policy_config={
            "mode": "read-only",
            "networkAccess": False,
            "writableRoots": ["build"],
            "allowRead": ["docs"],
            "denyRead": ["secrets"],
            "allowWrite": ["tmp"],
            "denyWrite": ["credentials.json"],
            "protectedPaths": [".git", "private"],
        },
        capabilities=capabilities,
        backend_result=SandboxExecutionResult(
            status="unavailable",
            reason="reviewer backend failure",
        ),
    )
    monkeypatch.chdir(tmp_path)
    approvals: list[str] = []

    result = asyncio.run(
        se.execute_shell_command(
            "python mutate.py",
            sandbox_unavailable_approval=lambda reason: (
                approvals.append(reason) or len(approvals) == 1
            ),
        )
    )

    assert result.status == "error"
    assert len(approvals) == 2
    reason = approvals[1]
    for expected in (
        "reviewer backend failure",
        "host process isolation",
        "workspace materialization and isolation",
        "repository synchronization",
        "read-only mode",
        "mode=read-only",
        "networkAccess=false",
        "writableRoots=['build']",
        "allowRead=['docs']",
        "denyRead=['secrets']",
        "allowWrite=['tmp']",
        "denyWrite=['credentials.json']",
        "protectedPaths=['.git', 'private']",
        "generic command or mutation approval does not accept these losses",
    ):
        assert expected in reason


def test_unsupported_with_async_approval_runs_unsandboxed(monkeypatch, tmp_path):
    """Async approval callbacks are awaited; 'unsupported' also degrades."""
    _force_sandbox_state(
        monkeypatch,
        tmp_path,
        backend_result=SandboxExecutionResult(status="unsupported", reason="non-darwin unix-local"),
    )
    monkeypatch.chdir(tmp_path)

    async def _approve(reason: str) -> bool:
        assert "unix-local" in reason
        return True

    result = asyncio.run(
        se.execute_shell_command(
            "echo async-ok",
            timeout=10,
            sandbox_unavailable_approval=_approve,
        )
    )

    assert result.status == "success"
    assert "async-ok" in result.output
    assert "warning: sandbox unavailable" in result.output


# --- unavailable + approval denied -> keeps the fail-closed error -------------


def test_unavailable_with_denied_approval_errors(monkeypatch, tmp_path):
    _force_sandbox_state(
        monkeypatch,
        tmp_path,
        backend_result=SandboxExecutionResult(status="unavailable", reason="docker not found"),
    )
    monkeypatch.chdir(tmp_path)

    result = asyncio.run(
        se.execute_shell_command(
            "echo should-not-run",
            timeout=10,
            sandbox_unavailable_approval=lambda reason: False,
        )
    )

    assert result.status == "error"
    assert "sandboxed: false" in result.output
    assert "docker not found" in result.output
    # The command did not run: its output is absent from the error.
    assert "should-not-run" not in result.output


def test_unavailable_without_callback_is_fail_closed(monkeypatch, tmp_path):
    """Default (no callback) must remain the safe fail-closed error."""
    _force_sandbox_state(
        monkeypatch,
        tmp_path,
        backend_result=SandboxExecutionResult(status="unavailable", reason="docker not found"),
    )
    monkeypatch.chdir(tmp_path)

    result = asyncio.run(se.execute_shell_command("echo nope", timeout=10))

    assert result.status == "error"
    assert "sandboxed: false" in result.output
    assert "nope" not in result.output


def test_raising_approval_callback_is_treated_as_denial(monkeypatch, tmp_path):
    """A callback that raises must not silently run unsandboxed."""
    _force_sandbox_state(
        monkeypatch,
        tmp_path,
        backend_result=SandboxExecutionResult(status="unavailable", reason="docker not found"),
    )
    monkeypatch.chdir(tmp_path)

    def _boom(reason: str) -> bool:
        raise RuntimeError("approval prompt blew up")

    result = asyncio.run(
        se.execute_shell_command(
            "echo boom",
            timeout=10,
            sandbox_unavailable_approval=_boom,
        )
    )

    assert result.status == "error"
    assert "sandboxed: false" in result.output
    assert "boom" not in result.output


@pytest.mark.parametrize(
    ("drift", "expected"),
    (
        ("cwd", "canonical cwd changed"),
        ("backend", "selected backend changed"),
        ("capability", "capability digest changed"),
        ("policy", "policy digest changed"),
        ("requirement", "requirement digest changed"),
    ),
)
def test_unavailable_fallback_reviewer_drift_reproduction_fails_closed(
    monkeypatch,
    tmp_path,
    drift,
    expected,
):
    capabilities = SandboxBackendCapabilities(
        supports_host_process_isolation="enforced",
        supports_workspace_isolation="enforced",
        supports_repository_sync="enforced",
        supports_read_only_filesystem="enforced",
        supports_network_policy="enforced",
        supports_domain_policy="enforced",
        supports_protected_paths="enforced",
    )
    backend_calls = _force_sandbox_state(
        monkeypatch,
        tmp_path,
        policy_config={"networkAccess": True},
        capabilities=capabilities,
        backend_result=SandboxExecutionResult(
            status="unavailable",
            reason="reviewer backend failure",
        ),
    )
    original_state = se.resolve_sandbox_settings(tmp_path)
    monkeypatch.chdir(tmp_path)
    host_calls: list[str] = []

    async def fake_host(command, **_kwargs):
        host_calls.append(command)
        return se.ShellExecutionResult(status="success", output="host-executed", exit_code=0)

    monkeypatch.setattr(se, "_run_foreground_unsandboxed", fake_host)

    def approve_then_drift(_reason: str) -> bool:
        if drift == "cwd":
            changed_cwd = tmp_path / "changed-cwd"
            changed_cwd.mkdir()
            monkeypatch.chdir(changed_cwd)
            return True

        if drift == "backend":
            changed_status = SandboxBackendStatus(
                backend_id="docker",
                selected=True,
                available=True,
                reason="available",
                capabilities=capabilities,
            )
            changed_state = original_state.__class__(
                **{
                    **original_state.__dict__,
                    "backend": "docker",
                    "backend_statuses": (changed_status,),
                }
            )
        elif drift == "capability":
            changed_capabilities = SandboxBackendCapabilities(
                **{
                    **capabilities.__dict__,
                    "supports_network_policy": "changed-after-approval",
                }
            )
            changed_status = SandboxBackendStatus(
                backend_id=original_state.backend,
                selected=True,
                available=True,
                reason="available",
                capabilities=changed_capabilities,
            )
            changed_state = original_state.__class__(
                **{**original_state.__dict__, "backend_statuses": (changed_status,)}
            )
        elif drift == "policy":
            changed_policy = SandboxPolicy.from_config(
                {
                    "enabled": True,
                    "mode": "read-only",
                    "backend": original_state.backend,
                    "networkAccess": True,
                }
            )
            changed_state = original_state.__class__(
                **{**original_state.__dict__, "policy": changed_policy}
            )
        else:
            changed_status = SandboxBackendStatus(
                backend_id=original_state.backend,
                selected=True,
                available=True,
                reason="availability detail changed after approval",
                capabilities=capabilities,
            )
            changed_state = original_state.__class__(
                **{**original_state.__dict__, "backend_statuses": (changed_status,)}
            )
        monkeypatch.setattr(se, "resolve_sandbox_settings", lambda _cwd: changed_state)
        return True

    result = asyncio.run(
        se.execute_shell_command(
            "python mutate.py",
            sandbox_unavailable_approval=approve_then_drift,
        )
    )

    assert len(backend_calls) == 1
    assert host_calls == []
    assert "executed: false" in result.output
    assert expected in result.output
    assert "new exact approval is required" in result.output


def test_unavailable_fallback_unchanged_state_executes_once(monkeypatch, tmp_path):
    _force_sandbox_state(
        monkeypatch,
        tmp_path,
        policy_config={"networkAccess": True},
        backend_result=SandboxExecutionResult(
            status="unavailable",
            reason="reviewer backend failure",
        ),
    )
    monkeypatch.chdir(tmp_path)
    approvals: list[str] = []
    host_calls: list[str] = []

    async def fake_host(command, **_kwargs):
        host_calls.append(command)
        return se.ShellExecutionResult(status="success", output="host-executed", exit_code=0)

    monkeypatch.setattr(se, "_run_foreground_unsandboxed", fake_host)

    result = asyncio.run(
        se.execute_shell_command(
            "python mutate.py",
            sandbox_unavailable_approval=lambda reason: approvals.append(reason) or True,
        )
    )

    assert host_calls == ["python mutate.py"]
    assert len(approvals) == 1
    assert result.output == "host-executed"


# --- non-regression: sandbox available -> path unchanged, callback ignored ----


def test_available_sandbox_path_unchanged_and_callback_not_consulted(monkeypatch, tmp_path):
    calls = _force_sandbox_state(
        monkeypatch,
        tmp_path,
        backend_result=SandboxExecutionResult(
            status="success",
            exit_code=0,
            stdout="hello",
            stderr="",
            sandboxed=True,
            backend_id="unix-local",
            created=True,
            executed=True,
        ),
    )
    monkeypatch.chdir(tmp_path)

    approval_consulted = {"called": False}

    def _approval(reason: str) -> bool:
        approval_consulted["called"] = True
        return True

    result = asyncio.run(
        se.execute_shell_command(
            "echo hello",
            timeout=10,
            sandbox_unavailable_approval=_approval,
        )
    )

    # Sandboxed path taken as before.
    assert result.status == "success"
    assert result.exit_code == 0
    assert "sandboxed: true" in result.output
    assert "backend: unix-local" in result.output
    # No degradation, so no warning and the approval callback was never asked.
    assert "warning: sandbox unavailable" not in result.output
    assert approval_consulted["called"] is False
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("backend_id", "policy_config", "capabilities", "expected_losses"),
    (
        (
            "unix-local",
            {"networkAccess": False},
            SandboxBackendCapabilities(
                supports_host_process_isolation="enforced",
                supports_workspace_isolation="enforced",
                supports_repository_sync="enforced",
                supports_network_policy="unsupported",
                supports_protected_paths="enforced",
            ),
            ("network access disabled", "networkAccess=false"),
        ),
        (
            "unix-local",
            {"networkAccess": True, "protectedPaths": ["private"]},
            SandboxBackendCapabilities(
                supports_host_process_isolation="enforced",
                supports_workspace_isolation="enforced",
                supports_repository_sync="enforced",
                supports_network_policy="enforced",
                supports_protected_paths="unsupported",
            ),
            ("protectedPaths", "protected subpaths"),
        ),
        (
            "e2b",
            {"networkAccess": True},
            SandboxBackendCapabilities(
                supports_host_process_isolation="remote-sandbox",
                supports_workspace_isolation="not-proven",
                supports_repository_sync="unsupported",
                supports_network_policy="enforced",
                supports_protected_paths="unsupported",
            ),
            (
                "host process isolation",
                "workspace materialization and isolation",
                "repository synchronization",
            ),
        ),
    ),
)
def test_incomplete_guarantees_require_exact_preexecution_approval(
    monkeypatch,
    tmp_path,
    backend_id,
    policy_config,
    capabilities,
    expected_losses,
):
    calls = _force_sandbox_state(
        monkeypatch,
        tmp_path,
        backend_id=backend_id,
        policy_config=policy_config,
        capabilities=capabilities,
        backend_result=SandboxExecutionResult(
            status="success",
            exit_code=0,
            stdout="must-not-run",
            sandboxed=True,
            backend_id=backend_id,
            created=True,
            executed=True,
        ),
    )
    monkeypatch.chdir(tmp_path)
    approvals: list[str] = []

    def deny_degradation(reason: str) -> bool:
        approvals.append(reason)
        return False

    result = asyncio.run(
        se.execute_shell_command(
            "python mutate.py",
            sandbox_unavailable_approval=deny_degradation,
        )
    )

    assert result.status == "error"
    assert "executed: false" in result.output
    assert "explicit sandbox degradation approval required" in result.output
    assert calls == []
    assert len(approvals) == 1
    for expected in expected_losses:
        assert expected in approvals[0]


def test_explicit_exact_degradation_approval_allows_backend_execution(monkeypatch, tmp_path):
    capabilities = SandboxBackendCapabilities(
        supports_host_process_isolation="remote-sandbox",
        supports_workspace_isolation="not-proven",
        supports_repository_sync="unsupported",
        supports_network_policy="enforced",
    )
    calls = _force_sandbox_state(
        monkeypatch,
        tmp_path,
        backend_id="e2b",
        policy_config={"networkAccess": True},
        capabilities=capabilities,
        backend_result=SandboxExecutionResult(
            status="success",
            exit_code=0,
            stdout="accepted",
            sandboxed=True,
            backend_id="e2b",
            created=True,
            executed=True,
        ),
    )
    monkeypatch.chdir(tmp_path)
    approvals: list[str] = []

    result = asyncio.run(
        se.execute_shell_command(
            "git status",
            sandbox_unavailable_approval=lambda reason: approvals.append(reason) or True,
        )
    )

    assert len(calls) == 1
    assert "accepted" in result.output
    assert "explicitly approved before execution" in result.output
    assert "workspace materialization and isolation" in approvals[0]
    assert "repository synchronization" in approvals[0]


def test_degradation_approval_rebinds_and_fails_closed_on_capability_change(monkeypatch, tmp_path):
    initial = SandboxBackendCapabilities(
        supports_host_process_isolation="enforced",
        supports_workspace_isolation="enforced",
        supports_repository_sync="enforced",
        supports_network_policy="unsupported",
    )
    changed = SandboxBackendCapabilities(
        supports_host_process_isolation="enforced",
        supports_workspace_isolation="enforced",
        supports_repository_sync="enforced",
        supports_network_policy="changed-after-approval",
    )
    calls = _force_sandbox_state(
        monkeypatch,
        tmp_path,
        policy_config={"networkAccess": False},
        capabilities=initial,
        backend_result=SandboxExecutionResult(
            status="success",
            exit_code=0,
            stdout="must-not-run",
            sandboxed=True,
            backend_id="unix-local",
            created=True,
            executed=True,
        ),
    )
    original_state = se.resolve_sandbox_settings(tmp_path)
    changed_status = SandboxBackendStatus(
        backend_id="unix-local",
        selected=True,
        available=True,
        reason="available",
        capabilities=changed,
    )
    changed_state = original_state.__class__(
        **{**original_state.__dict__, "backend_statuses": (changed_status,)}
    )
    monkeypatch.chdir(tmp_path)

    def approve_then_change(_reason: str) -> bool:
        monkeypatch.setattr(se, "resolve_sandbox_settings", lambda _cwd: changed_state)
        return True

    result = asyncio.run(
        se.execute_shell_command(
            "python mutate.py",
            sandbox_unavailable_approval=approve_then_change,
        )
    )

    assert calls == []
    assert "sandbox backend capabilities changed" in result.output
    assert "reapproval is required before execution" in result.output


@pytest.mark.asyncio
async def test_cancelling_degradation_approval_never_reaches_backend(monkeypatch, tmp_path):
    capabilities = SandboxBackendCapabilities(
        supports_host_process_isolation="enforced",
        supports_workspace_isolation="enforced",
        supports_repository_sync="enforced",
        supports_network_policy="unsupported",
    )
    calls = _force_sandbox_state(
        monkeypatch,
        tmp_path,
        policy_config={"networkAccess": False},
        capabilities=capabilities,
        backend_result=SandboxExecutionResult(
            status="success",
            exit_code=0,
            stdout="must-not-run",
            sandboxed=True,
            backend_id="unix-local",
            created=True,
            executed=True,
        ),
    )
    monkeypatch.chdir(tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def pending_approval(_reason: str) -> bool:
        entered.set()
        await release.wait()
        return True

    task = asyncio.create_task(
        se.execute_shell_command(
            "python mutate.py",
            sandbox_unavailable_approval=pending_approval,
        )
    )
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == []


def test_no_sandbox_foreground_path_output_is_unchanged(monkeypatch, tmp_path):
    """When sandbox is disabled the normal foreground output has no warning."""
    real_state = se.resolve_sandbox_settings(str(tmp_path))
    disabled = real_state.__class__(**{**real_state.__dict__, "enabled": False, "policy": None})
    monkeypatch.setattr(se, "resolve_sandbox_settings", lambda *_a, **_k: disabled)
    monkeypatch.chdir(tmp_path)

    result = asyncio.run(
        se.execute_shell_command(
            "echo plain",
            timeout=10,
            sandbox_unavailable_approval=lambda reason: True,
        )
    )

    assert result.status == "success"
    assert result.output == "plain"
    assert "warning" not in result.output


# --- the tool/permission-layer factory ----------------------------------------


def test_build_sandbox_unavailable_approval_defaults_to_deny():
    approval = build_sandbox_unavailable_approval()
    assert asyncio.run(approval("docker missing")) is False


def test_build_sandbox_unavailable_approval_delegates_to_approver():
    seen: list[str] = []

    def _approver(reason: str) -> bool:
        seen.append(reason)
        return True

    approval = build_sandbox_unavailable_approval(_approver)
    assert asyncio.run(approval("docker missing")) is True
    assert seen == ["docker missing"]


def test_build_sandbox_unavailable_approval_awaits_async_approver():
    async def _approver(reason: str) -> bool:
        return "unix-local" in reason

    approval = build_sandbox_unavailable_approval(_approver)
    assert asyncio.run(approval("non-darwin unix-local")) is True
    assert asyncio.run(approval("docker missing")) is False


def test_build_sandbox_unavailable_approval_threads_into_executor(monkeypatch, tmp_path):
    """End-to-end: the factory's callback drives real degradation."""
    _force_sandbox_state(
        monkeypatch,
        tmp_path,
        backend_result=SandboxExecutionResult(status="unavailable", reason="docker not found"),
    )
    monkeypatch.chdir(tmp_path)

    approval = build_sandbox_unavailable_approval(lambda reason: True)
    result = asyncio.run(
        se.execute_shell_command(
            "echo factory-run",
            timeout=10,
            sandbox_unavailable_approval=approval,
        )
    )

    assert result.status == "success"
    assert "factory-run" in result.output
    assert "warning: sandbox unavailable" in result.output
