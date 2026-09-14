"""A sandbox request owns creation, context exit and provider deletion."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from types import SimpleNamespace

import pytest

from koder_agent.harness.sandbox import sdk_backend
from koder_agent.harness.sandbox.backend import (
    SandboxBackendCapabilities,
    SandboxBackendStatus,
    SandboxExecutionContext,
    SandboxExecutionResult,
)
from koder_agent.harness.sandbox.policy import SandboxPolicy

_SESSION_CONTEXT = ContextVar("sandbox_lifecycle_test", default="outside")


class _Session:
    def __init__(self):
        self.calls = []
        self.exit_action = None
        self.exec_action = None
        self.exit_completed = False

    async def __aenter__(self):
        self.calls.append(("enter", asyncio.current_task()))
        self.token = _SESSION_CONTEXT.set("inside")
        return self

    async def __aexit__(self, *_error):
        self.calls.append(("exit", asyncio.current_task()))
        try:
            if self.exit_action is not None:
                await self.exit_action()
            self.exit_completed = True
        finally:
            _SESSION_CONTEXT.reset(self.token)

    async def exec(self, command, *, timeout, shell):
        assert _SESSION_CONTEXT.get() == "inside"
        self.calls.append(("exec", asyncio.current_task()))
        self.exec_arguments = (command, timeout, shell)
        if self.exec_action is not None:
            return await self.exec_action()
        return SimpleNamespace(stdout=b"command-completed", stderr=b"", exit_code=0)


class _Client:
    def __init__(self, session):
        self.session = session
        self.created = 0
        self.deleted = 0

    async def create(self, *, manifest, options):
        self.created += 1
        self.manifest = manifest
        self.options = options
        return self.session

    async def delete(self, session):
        assert session is self.session
        self.deleted += 1


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    session = _Session()
    client = _Client(session)
    status = SandboxBackendStatus(
        backend_id="docker",
        selected=True,
        available=True,
        reason="synthetic backend",
        capabilities=SandboxBackendCapabilities(
            supports_host_process_isolation="enforced",
            supports_workspace_isolation="enforced",
            supports_repository_sync="enforced",
            supports_read_only_filesystem="enforced",
            supports_network_policy="enforced",
            supports_domain_policy="enforced",
            supports_protected_paths="enforced",
        ),
    )
    options = object()
    monkeypatch.setattr(sdk_backend, "get_backend_status", lambda *_a, **_kw: status)
    monkeypatch.setattr(
        sdk_backend,
        "create_backend_client_and_options",
        lambda *_a, **_kw: (client, options),
    )
    context = SandboxExecutionContext(
        cwd=tmp_path,
        repo_root=tmp_path,
        command="printf command-completed",
        env={"LANG": "C", "SYNTHETIC_API_KEY": "not-for-the-sandbox"},
        timeout=7,
        background=False,
        session_id=None,
        policy=SandboxPolicy(backend="docker", network_access=True),
    )
    return SimpleNamespace(
        client=client, session=session, status=status, context=context, options=options
    )


async def _finish(baseline, *owned):
    """Drain test-created work after releasing every controlled barrier."""
    tasks = (asyncio.all_tasks() - baseline) | set(owned)
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2)


@pytest.mark.asyncio
async def test_success_keeps_sdk_context_and_result_contract(sandbox):
    result = await sdk_backend.execute_with_sdk_backend(sandbox.context)

    assert result.status == "success"
    assert result.exit_code == 0
    assert result.stdout == "command-completed"
    assert result.created is result.executed is result.sandboxed is True
    assert sandbox.client.created == sandbox.client.deleted == 1
    assert sandbox.client.options is sandbox.options
    assert sandbox.client.manifest.root == str(sandbox.context.cwd)
    assert sandbox.client.manifest.environment.value == {"LANG": "C"}
    assert sandbox.session.exec_arguments == ("printf command-completed", 7, True)
    assert [name for name, _task in sandbox.session.calls] == ["enter", "exec", "exit"]
    assert len({task for _name, task in sandbox.session.calls}) == 1
    assert _SESSION_CONTEXT.get() == "outside"


@pytest.mark.asyncio
async def test_repeated_cancel_during_delete_does_not_detach_cleanup(
    sandbox, monkeypatch, cancellation_observer
):
    observe, cancellations = cancellation_observer
    baseline = asyncio.all_tasks()
    deleting = asyncio.Event()
    release = asyncio.Event()
    deleted = asyncio.Event()

    async def delete(_session):
        deleting.set()
        await release.wait()
        deleted.set()

    monkeypatch.setattr(sandbox.client, "delete", delete)
    owner = asyncio.create_task(observe(sdk_backend.execute_with_sdk_backend(sandbox.context)))
    try:
        await asyncio.wait_for(deleting.wait(), timeout=2)
        owner.cancel("first sandbox cancellation")
        await asyncio.sleep(0)
        owner.cancel("second sandbox cancellation")
        await asyncio.sleep(0.02)
        assert not owner.done(), "request returned while its provider deletion was pending"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert deleted.is_set()
        assert [error.args for error in cancellations] == [("first sandbox cancellation",)]
        assert not (asyncio.all_tasks() - baseline)
    finally:
        release.set()
        await _finish(baseline, owner)


@pytest.mark.asyncio
async def test_failed_delete_is_not_reported_as_success(sandbox, monkeypatch):
    async def delete(_session):
        raise RuntimeError("private-cleanup-canary")

    monkeypatch.setattr(sandbox.client, "delete", delete)
    result = await sdk_backend.execute_with_sdk_backend(sandbox.context)

    assert result.status == "error"
    assert result.exit_code == 0  # The command succeeded; its resource lifecycle did not.
    assert result.stdout == "command-completed"
    assert result.created is result.executed is result.sandboxed is True
    assert "cleanup" in (result.reason or "").lower()
    assert "cleanup" in result.stderr.lower()
    assert "private-cleanup-canary" not in repr(result)


@pytest.mark.asyncio
async def test_delete_timeout_is_reported_after_cancellation_settles(sandbox, monkeypatch):
    baseline = asyncio.all_tasks()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()
    settled = asyncio.Event()

    async def delete(_session):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()
            raise
        finally:
            settled.set()

    monkeypatch.setattr(sandbox.client, "delete", delete)
    monkeypatch.setattr(sdk_backend, "DELETE_TIMEOUT_SECONDS", 0.01)
    owner = asyncio.create_task(sdk_backend.execute_with_sdk_backend(sandbox.context))
    try:
        await asyncio.wait_for(cancellation_seen.wait(), timeout=2)
        assert not owner.done()
        release.set()
        result = await owner
        assert settled.is_set()
        assert result.status == "error"
        assert result.exit_code == 0
        assert result.created is result.executed is True
        assert "cleanup" in (result.reason or "").lower()
        assert "timed out" in (result.reason or "").lower()
        assert not (asyncio.all_tasks() - baseline)
    finally:
        release.set()
        await _finish(baseline, owner)


@pytest.mark.asyncio
async def test_late_creation_after_cancel_is_deleted_without_running_command(
    sandbox, monkeypatch, cancellation_observer
):
    observe, cancellations = cancellation_observer
    baseline = asyncio.all_tasks()
    creating = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def create(**_kwargs):
        creating.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()
        return sandbox.session

    monkeypatch.setattr(sandbox.client, "create", create)
    owner = asyncio.create_task(observe(sdk_backend.execute_with_sdk_backend(sandbox.context)))
    try:
        await asyncio.wait_for(creating.wait(), timeout=2)
        owner.cancel("cancel creation")
        await asyncio.wait_for(cancellation_seen.wait(), timeout=2)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert [error.args for error in cancellations] == [("cancel creation",)]
        assert sandbox.session.calls == []
        assert sandbox.client.deleted == 1
    finally:
        release.set()
        await _finish(baseline, owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_exec_first", [False, True])
async def test_context_exit_finishes_before_cancel_returns(
    sandbox, cancellation_observer, cancel_exec_first
):
    observe, cancellations = cancellation_observer
    baseline = asyncio.all_tasks()
    executing = asyncio.Event()
    exiting = asyncio.Event()
    release = asyncio.Event()

    async def blocked_exec():
        executing.set()
        await asyncio.Event().wait()

    async def exit_action():
        exiting.set()
        await release.wait()

    sandbox.session.exit_action = exit_action
    if cancel_exec_first:
        sandbox.session.exec_action = blocked_exec
    owner = asyncio.create_task(observe(sdk_backend.execute_with_sdk_backend(sandbox.context)))
    try:
        if cancel_exec_first:
            await asyncio.wait_for(executing.wait(), timeout=2)
            owner.cancel("first cancellation")
        await asyncio.wait_for(exiting.wait(), timeout=2)
        owner.cancel("second cancellation" if cancel_exec_first else "first cancellation")
        await asyncio.sleep(0.02)
        assert not owner.done(), "request abandoned its SDK context exit"
        assert sandbox.client.deleted == 0
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert sandbox.session.exit_completed
        assert sandbox.client.deleted == 1
        assert [error.args for error in cancellations] == [("first cancellation",)]
        assert len({task for _name, task in sandbox.session.calls}) == 1
        assert _SESSION_CONTEXT.get() == "outside"
    finally:
        release.set()
        if not owner.done() and not exiting.is_set():
            owner.cancel()
        await _finish(baseline, owner)


@pytest.mark.asyncio
async def test_failed_context_entry_still_deletes_created_session(sandbox, monkeypatch):
    async def enter(_self):
        raise RuntimeError("entry failed")

    monkeypatch.setattr(_Session, "__aenter__", enter)
    result = await sdk_backend.execute_with_sdk_backend(sandbox.context)

    assert result.status == "error"
    assert result.created is result.sandboxed is True
    assert result.executed is False
    assert sandbox.client.deleted == 1
    assert sandbox.session.calls == []


@pytest.mark.asyncio
async def test_suppressed_execution_error_is_not_a_successful_result(sandbox, monkeypatch):
    original_exit = _Session.__aexit__

    async def suppress_error(self, *error):
        await original_exit(self, *error)
        return True

    async def fail_exec():
        raise RuntimeError("execution failed")

    monkeypatch.setattr(_Session, "__aexit__", suppress_error)
    sandbox.session.exec_action = fail_exec
    result = await sdk_backend.execute_with_sdk_backend(sandbox.context)

    assert result.status == "error"
    assert result.exit_code is None
    assert result.created is result.executed is True
    assert "no result" in (result.reason or "")
    assert sandbox.client.deleted == 1
    assert _SESSION_CONTEXT.get() == "outside"


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_code", [0, 7])
async def test_command_outcome_survives_context_exit_failure(sandbox, exit_code):
    async def execute():
        return SimpleNamespace(
            stdout=b"output before context failure",
            stderr=b"command diagnostic",
            exit_code=exit_code,
        )

    async def fail_exit():
        raise RuntimeError("private-context-cleanup-canary")

    sandbox.session.exec_action = execute
    sandbox.session.exit_action = fail_exit
    result = await sdk_backend.execute_with_sdk_backend(sandbox.context)

    assert result.status == "error"
    assert result.exit_code == exit_code
    assert result.stdout == "output before context failure"
    assert "command diagnostic" in result.stderr
    assert "cleanup" in result.stderr
    assert "private-context-cleanup-canary" not in repr(result)
    assert sandbox.client.deleted == 1
    assert _SESSION_CONTEXT.get() == "outside"


@pytest.mark.asyncio
async def test_exec_suppressing_cancel_cannot_return_success(sandbox, cancellation_observer):
    observe, cancellations = cancellation_observer
    baseline = asyncio.all_tasks()
    executing = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def execute():
        executing.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()
        return SimpleNamespace(stdout=b"late output", stderr=b"", exit_code=0)

    sandbox.session.exec_action = execute
    owner = asyncio.create_task(observe(sdk_backend.execute_with_sdk_backend(sandbox.context)))
    try:
        await asyncio.wait_for(executing.wait(), timeout=2)
        owner.cancel("cancel command")
        await asyncio.wait_for(cancellation_seen.wait(), timeout=2)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert [error.args for error in cancellations] == [("cancel command",)]
        assert sandbox.session.exit_completed
        assert sandbox.client.deleted == 1
    finally:
        release.set()
        await _finish(baseline, owner)


@pytest.mark.asyncio
async def test_concurrent_sandbox_requests_have_separate_cleanup(sandbox, monkeypatch):
    baseline = asyncio.all_tasks()
    executing = asyncio.Event()
    second_session = _Session()
    second_client = _Client(second_session)
    clients = iter((sandbox.client, second_client))
    monkeypatch.setattr(
        sdk_backend,
        "create_backend_client_and_options",
        lambda *_a, **_kw: (next(clients), sandbox.options),
    )

    async def blocked_exec():
        executing.set()
        await asyncio.Event().wait()

    sandbox.session.exec_action = blocked_exec
    owner = asyncio.create_task(sdk_backend.execute_with_sdk_backend(sandbox.context))
    try:
        await asyncio.wait_for(executing.wait(), timeout=2)
        second_result = await sdk_backend.execute_with_sdk_backend(sandbox.context)
        assert second_result.status == "success"
        assert second_session.exit_completed
        assert second_client.deleted == 1
        assert sandbox.client.deleted == 0
        assert not owner.done()
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert sandbox.client.deleted == second_client.deleted == 1
        assert _SESSION_CONTEXT.get() == "outside"
    finally:
        if not owner.done():
            owner.cancel()
        await _finish(baseline, owner)


@pytest.mark.asyncio
async def test_command_error_and_output_survive_delete_failure(sandbox, monkeypatch):
    async def failed_exec():
        return SimpleNamespace(stdout=b"partial output", stderr=b"command failed", exit_code=7)

    async def delete(_session):
        raise RuntimeError("private-cleanup-canary")

    sandbox.session.exec_action = failed_exec
    monkeypatch.setattr(sandbox.client, "delete", delete)
    result = await sdk_backend.execute_with_sdk_backend(sandbox.context)

    assert result.status == "error"
    assert result.exit_code == 7
    assert result.stdout == "partial output"
    assert "command failed" in result.stderr
    assert "cleanup" in result.stderr.lower()
    assert "private-cleanup-canary" not in repr(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_status", ["error", "ok"])
async def test_harness_does_not_turn_backend_failure_into_success(
    sandbox, monkeypatch, backend_status
):
    from koder_agent.harness.tools import shell_executor

    state = SimpleNamespace(
        enabled=True,
        backend="docker",
        policy=sandbox.context.policy,
        backend_available=True,
        backend_statuses=(sandbox.status,),
    )
    monkeypatch.setattr(shell_executor, "resolve_sandbox_settings", lambda _cwd: state)
    monkeypatch.setattr(shell_executor, "is_excluded_command", lambda *_a, **_kw: False)
    monkeypatch.setattr(shell_executor, "get_execution_cwd", lambda: sandbox.context.cwd)

    async def failed_backend(_context):
        return SandboxExecutionResult(
            status=backend_status,
            stdout="command-completed",
            exit_code=0,
            backend_id="docker",
            sandboxed=True,
            created=True,
            executed=True,
            reason="sandbox cleanup failed",
            stderr="sandbox cleanup failed",
        )

    async def no_host_execution(*_args, **_kwargs):
        raise AssertionError("a completed sandbox command must not be replayed on the host")

    monkeypatch.setattr(shell_executor, "execute_with_sdk_backend", failed_backend)
    monkeypatch.setattr(shell_executor, "_run_foreground_unsandboxed", no_host_execution)
    result = await shell_executor.execute_shell_command(sandbox.context.command)

    assert result.status == "error"
    assert result.exit_code == 0
    assert "command-completed" in result.output
    assert "cleanup" in result.output
