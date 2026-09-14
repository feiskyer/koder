"""Shell execution primitives for the harness runtime."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import platform
import re
import shlex
import shutil
import signal
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Union

from koder_agent.harness.execution_context import get_execution_cwd
from koder_agent.harness.sandbox.backend import (
    SandboxExecutionContext,
    SandboxExecutionRequirement,
    SandboxFallbackRequirement,
)
from koder_agent.harness.sandbox.enforcement import (
    autoapproval_blockers,
    backend_capability_digest,
    canonical_workspace_path,
    sandbox_degradation_reason,
    sandbox_fallback_requirement_digest,
    sandbox_policy_digest,
    unsandboxed_fallback_losses,
    unsandboxed_fallback_reason,
)
from koder_agent.harness.sandbox.sdk_backend import execute_with_sdk_backend
from koder_agent.harness.sandbox_settings import is_excluded_command, resolve_sandbox_settings
from koder_agent.harness.session_env import build_sandbox_env, build_subprocess_env

IS_WINDOWS = platform.system() == "Windows"

# Callback threaded in from the tool/permission layer. It is invoked with the
# exact unavailable guarantees before degraded sandbox execution, and again if
# an unavailable/unsupported backend would require UNSANDBOXED fallback.
# May be sync or async. When omitted, degradation stays fail-closed (error).
SandboxUnavailableApproval = Callable[[str], Union[bool, Awaitable[bool]]]


async def _resolve_sandbox_unavailable_approval(
    approval: SandboxUnavailableApproval | None, reason: str
) -> bool:
    """Ask the optional callback whether to accept the named sandbox degradation.

    Default is fail-closed: no callback -> not approved. Any error raised by the
    callback is treated as a denial so degradation never happens silently.
    """
    if approval is None:
        return False
    try:
        result = approval(reason)
        if inspect.isawaitable(result):
            result = await result
        return bool(result)
    except Exception:
        return False


def _selected_backend_status(sandbox_state):
    return next(
        (
            status
            for status in sandbox_state.backend_statuses
            if status.backend_id == sandbox_state.backend
        ),
        None,
    )


def _sandbox_snapshot(sandbox_state, backend_status) -> SandboxExecutionRequirement:
    return SandboxExecutionRequirement(
        backend_id=sandbox_state.backend,
        canonical_cwd=canonical_workspace_path(get_execution_cwd()),
        policy_digest=sandbox_policy_digest(sandbox_state.policy),
        capability_digest=backend_capability_digest(backend_status.capabilities),
    )


def _sandbox_snapshot_mismatch(
    required: SandboxExecutionRequirement,
    sandbox_state,
    *,
    use_sandbox: bool,
    require_complete_capabilities: bool,
) -> str | None:
    current_cwd = canonical_workspace_path(get_execution_cwd())
    if not sandbox_state.enabled:
        return "sandbox is no longer enabled"
    if not use_sandbox:
        return "command is now excluded from sandbox execution"
    if current_cwd != required.canonical_cwd:
        return f"canonical cwd changed from {required.canonical_cwd} to {current_cwd}"
    if sandbox_state.backend != required.backend_id:
        return f"selected backend changed to {sandbox_state.backend}"
    if sandbox_state.policy is None:
        return "sandbox policy is unavailable"

    backend_status = _selected_backend_status(sandbox_state)
    if backend_status is None or not backend_status.available:
        return "sandbox backend is no longer available"
    if sandbox_policy_digest(sandbox_state.policy) != required.policy_digest:
        return "sandbox policy changed after permission evaluation"
    if backend_capability_digest(backend_status.capabilities) != required.capability_digest:
        return "sandbox backend capabilities changed after permission evaluation"
    if require_complete_capabilities:
        blockers = autoapproval_blockers(
            sandbox_state.policy,
            backend_status.capabilities,
        )
        if blockers:
            return "sandbox no longer proves all auto-approval protections: " + "; ".join(
                item.restriction for item in blockers
            )
    return None


def capture_unsandboxed_fallback_requirement(
    sandbox_state,
    *,
    command: str,
    trigger: str,
) -> SandboxFallbackRequirement | None:
    """Capture the exact state and losses for one host-execution approval."""

    if sandbox_state.policy is None:
        return None
    backend_status = _selected_backend_status(sandbox_state)
    if backend_status is None:
        return None
    canonical_cwd = canonical_workspace_path(get_execution_cwd())
    losses = unsandboxed_fallback_losses(
        sandbox_state.policy,
        backend_status.capabilities,
        canonical_cwd=canonical_cwd,
    )
    reason = unsandboxed_fallback_reason(
        sandbox_state.backend,
        trigger=trigger,
        losses=losses,
    )
    return SandboxFallbackRequirement(
        backend_id=sandbox_state.backend,
        canonical_cwd=canonical_cwd,
        policy_digest=sandbox_policy_digest(sandbox_state.policy),
        capability_digest=backend_capability_digest(backend_status.capabilities),
        requirement_digest=sandbox_fallback_requirement_digest(
            command=command,
            trigger=trigger,
            losses=losses,
            sandbox_enabled=sandbox_state.enabled,
            platform_enabled=sandbox_state.platform_enabled,
            backend_available=sandbox_state.backend_available,
            backend_reason=sandbox_state.backend_reason,
            backend_status_available=backend_status.available,
            backend_status_reason=backend_status.reason,
        ),
        reason=reason,
    )


def unsandboxed_fallback_requirement_mismatch(
    required: SandboxFallbackRequirement,
    sandbox_state,
    *,
    command: str,
    trigger: str,
) -> str | None:
    """Rebuild and compare every field bound by a fallback approval."""

    if not sandbox_state.enabled:
        return "sandbox is no longer enabled"
    if is_excluded_command(command, cwd=get_execution_cwd()):
        return "command is now excluded from sandbox execution"
    current = capture_unsandboxed_fallback_requirement(
        sandbox_state,
        command=command,
        trigger=trigger,
    )
    if current is None:
        return "sandbox fallback state is unavailable"
    if current.canonical_cwd != required.canonical_cwd:
        return f"canonical cwd changed from {required.canonical_cwd} to {current.canonical_cwd}"
    if current.backend_id != required.backend_id:
        return f"selected backend changed from {required.backend_id} to {current.backend_id}"
    if current.policy_digest != required.policy_digest:
        return "sandbox policy digest changed after degradation approval"
    if current.capability_digest != required.capability_digest:
        return "sandbox capability digest changed after degradation approval"
    if current.requirement_digest != required.requirement_digest:
        return "sandbox fallback requirement digest changed after degradation approval"
    return None


def _blocked_sandbox_result(backend_id: str, reason: str) -> ShellExecutionResult:
    return ShellExecutionResult(
        status="error",
        output=(
            "sandboxed: false\n"
            "created: false\n"
            "executed: false\n"
            f"backend: {backend_id}\n"
            f"reason: {reason}"
        ),
    )


# Cap the per-shell output buffer so a verbose long-running background process
# does not leak memory for the whole session. Overridable via
# KODER_BG_SHELL_MAX_LINES; unset/empty/non-numeric/<=0 falls back to default.
DEFAULT_BG_SHELL_MAX_LINES = 10000


def _bg_shell_max_lines() -> int:
    """Resolve the max retained output lines per background shell."""
    raw = os.environ.get("KODER_BG_SHELL_MAX_LINES")
    if raw is None or raw.strip() == "":
        return DEFAULT_BG_SHELL_MAX_LINES
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_BG_SHELL_MAX_LINES
    if value <= 0:
        return DEFAULT_BG_SHELL_MAX_LINES
    return value


def _new_session_kwargs() -> dict:
    """Subprocess kwargs placing the child in its own process group (POSIX)."""
    if IS_WINDOWS:
        return {}
    return {"start_new_session": True}


def _signal_process_group(process: asyncio.subprocess.Process, sig: int) -> None:
    """Send ``sig`` to the child's whole process group (POSIX), guarded.

    Only signals the group when the child is a group leader distinct from our
    own process group — i.e. it was spawned with ``start_new_session=True``.
    This is a hard safety check: if the child shares our group, ``killpg`` would
    signal this very process, so we fall back to signalling just the child. Also
    falls back if the group can't be resolved/signalled or on Windows. Never
    raises.
    """
    pid = process.pid
    if not IS_WINDOWS and pid is not None:
        try:
            child_pgid = os.getpgid(pid)
            # Refuse to signal our own group — that would kill this process too.
            if child_pgid != os.getpgid(0):
                os.killpg(child_pgid, sig)
                return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    # Fall back to signalling the child process directly.
    with contextlib.suppress(ProcessLookupError, OSError):
        if sig == signal.SIGKILL:
            process.kill()
        else:
            process.terminate()


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    """SIGKILL the child and its whole process group (POSIX), guarded."""
    _signal_process_group(process, signal.SIGKILL)


async def _drain_after_kill(
    process: asyncio.subprocess.Process, timeout: float = 1.0
) -> tuple[str, str]:
    """Drain buffered output after a kill without blocking forever."""
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except Exception:
        return "", ""
    return (
        stdout.decode("utf-8", errors="replace").strip() if stdout else "",
        stderr.decode("utf-8", errors="replace").strip() if stderr else "",
    )


def resolve_powershell_executable() -> str | None:
    """Return a PowerShell executable available on this machine."""

    candidates = ["pwsh", "powershell"]
    if IS_WINDOWS:
        candidates.append("powershell.exe")
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


@dataclass
class BackgroundProcess:
    """Background shell process state.

    ``output_lines`` is a bounded ring buffer: old lines are evicted from the
    left once ``maxlen`` is hit, so a verbose long-running process cannot leak
    memory for the whole session. ``last_read_index`` is an index into the
    monotonic total-appended stream (see ``_total_appended``), not into the
    deque, so the reader stays correct even after left-eviction shifts positions.
    """

    shell_id: str
    command: str
    process: asyncio.subprocess.Process
    start_time: float
    output_lines: "deque[str]" = field(default_factory=lambda: deque(maxlen=_bg_shell_max_lines()))
    last_read_index: int = 0
    status: str = "running"
    exit_code: int | None = None
    # Monotonic count of all lines ever appended (including evicted ones).
    _total_appended: int = 0

    def __post_init__(self) -> None:
        # Accept a plain list (e.g. from callers/tests) and coerce to a bounded
        # deque; keep _total_appended consistent with any preseeded content.
        if not isinstance(self.output_lines, deque):
            seed = list(self.output_lines)
            self.output_lines = deque(seed, maxlen=_bg_shell_max_lines())
        elif self.output_lines.maxlen is None:
            self.output_lines = deque(self.output_lines, maxlen=_bg_shell_max_lines())
        if self._total_appended < len(self.output_lines):
            self._total_appended = len(self.output_lines)

    def add_output(self, line: str) -> None:
        self.output_lines.append(line)
        self._total_appended += 1

    def get_new_output(self, filter_pattern: str | None = None) -> list[str]:
        retained_start = self._total_appended - len(self.output_lines)
        # Clamp: never slice before the oldest retained line (evicted lines are
        # gone) and never re-read lines already consumed.
        effective_index = max(self.last_read_index, retained_start)
        offset = effective_index - retained_start
        new_lines = list(self.output_lines)[offset:]
        self.last_read_index = self._total_appended
        if filter_pattern:
            try:
                pattern = re.compile(filter_pattern)
                new_lines = [line for line in new_lines if pattern.search(line)]
            except re.error:
                pass
        return new_lines


class BackgroundProcessManager:
    """Singleton-style manager for runtime background shells."""

    _shells: dict[str, BackgroundProcess] = {}
    _monitor_tasks: dict[str, asyncio.Task] = {}

    @classmethod
    def add(cls, shell: BackgroundProcess) -> None:
        cls._shells[shell.shell_id] = shell

    @classmethod
    def get(cls, shell_id: str) -> BackgroundProcess | None:
        return cls._shells.get(shell_id)

    @classmethod
    def list_ids(cls) -> list[str]:
        return list(cls._shells.keys())

    @classmethod
    async def start_monitor(cls, shell_id: str) -> None:
        shell = cls.get(shell_id)
        if not shell:
            return

        async def monitor() -> None:
            try:
                process = shell.process
                while True:
                    if not process.stdout:
                        break
                    try:
                        line = await asyncio.wait_for(process.stdout.readline(), timeout=0.1)
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0.05)
                        continue
                    if line:
                        shell.add_output(line.decode("utf-8", errors="replace").rstrip("\n"))
                        continue
                    if process.returncode is not None:
                        break
                    await asyncio.sleep(0.05)

                shell.exit_code = await process.wait()
                shell.status = "completed" if shell.exit_code == 0 else "failed"
            except Exception as exc:
                shell.status = "error"
                shell.add_output(f"Monitor error: {exc}")
            finally:
                cls._monitor_tasks.pop(shell_id, None)

        cls._monitor_tasks[shell_id] = asyncio.create_task(monitor())

    @classmethod
    async def terminate(cls, shell_id: str) -> BackgroundProcess:
        shell = cls.get(shell_id)
        if not shell:
            raise ValueError(f"Shell not found: {shell_id}")
        if shell.process.returncode is None:
            # SIGTERM the whole process group (not just the wrapper) so
            # grandchildren spawned via '&' are asked to exit too.
            _signal_process_group(shell.process, signal.SIGTERM)
            try:
                await asyncio.wait_for(shell.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                # Escalate to SIGKILL of the whole group so nothing is orphaned;
                # then wait (guarded) for the child to reap.
                _kill_process_group(shell.process)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(shell.process.wait(), timeout=5)
        shell.status = "terminated"
        shell.exit_code = shell.process.returncode
        monitor = cls._monitor_tasks.pop(shell_id, None)
        if monitor and not monitor.done():
            monitor.cancel()
        cls._shells.pop(shell_id, None)
        return shell


@dataclass(frozen=True)
class ShellExecutionResult:
    """Result of a shell execution request."""

    status: str
    output: str
    exit_code: int | None = None
    shell_id: str | None = None


def _sandbox_output(result, *, output_style: str = "shell") -> str:
    lines: list[str] = [
        f"sandboxed: {'true' if result.sandboxed else 'false'}",
        f"created: {'true' if result.created else 'false'}",
        f"executed: {'true' if result.executed else 'false'}",
    ]
    if result.backend_id:
        lines.append(f"backend: {result.backend_id}")
    if output_style == "git":
        body = result.stdout.strip()
        stderr = result.stderr.strip()
        if stderr:
            body = f"{body}\n{stderr}" if body else stderr
        if result.exit_code not in (None, 0) and not body:
            body = f"Git command failed with exit code {result.exit_code}"
        if result.violation:
            body = f"sandbox violation: {result.violation}" + (f"\n{body}" if body else "")
    else:
        body = result.combined_output().strip()
    if result.reason and (not body or body == "(no output)"):
        body = f"sandbox error: {result.reason}"
    if body:
        lines.append(body)
    return "\n".join(lines) if lines else "(no output)"


async def _run_foreground_unsandboxed(
    command: str,
    *,
    timeout: int,
    session_id: str | None,
    warning: str | None = None,
    output_style: str = "shell",
) -> ShellExecutionResult:
    """Run ``command`` in the foreground without a sandbox.

    Reuses the Wave-1 process-group hardening: the child is placed in its own
    session/group so a timeout kills the whole tree (via ``_kill_process_group``)
    rather than orphaning grandchildren, and buffered output is drained under a
    bounded timeout. ``warning`` (if given) is prepended as a one-line notice to
    the output, used to make sandbox degradation visible to the caller.
    """
    child_env = build_subprocess_env(session_id)
    # Own process group so a timeout kills the whole tree, not just the wrapper.
    if IS_WINDOWS:
        process = await asyncio.create_subprocess_exec(
            "powershell.exe",
            "-NoProfile",
            "-Command",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env,
            cwd=get_execution_cwd(),
            **_new_session_kwargs(),
        )
    else:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env,
            cwd=get_execution_cwd(),
            **_new_session_kwargs(),
        )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        _kill_process_group(process)
        drained = await _drain_after_kill(process, timeout=1)
        partial_stdout, partial_stderr = drained or ("", "")
        label = "Git command" if output_style == "git" else "Command"
        timeout_output = f"{label} timed out after {timeout} seconds"
        if output_style != "git" and (partial_stdout or partial_stderr):
            timeout_output += f". Partial output:\n{partial_stdout}"
            if partial_stderr:
                timeout_output += f"\n[stderr]: {partial_stderr}"
        if warning:
            timeout_output = f"{warning}\n{timeout_output}"
        return ShellExecutionResult(status="error", output=timeout_output)
    except asyncio.CancelledError:
        # ESC/cancel: kill the detached process tree before propagating
        _kill_process_group(process)
        # Brief drain so killpg's SIGKILL has time to reap
        try:
            await asyncio.wait_for(process.communicate(), timeout=2)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        raise  # Re-raise so the SDK knows the turn was cancelled

    output = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()
    if stderr_text:
        if output_style == "git":
            output += f"\n{stderr_text}" if output else stderr_text
        else:
            output += f"\n[stderr]: {stderr_text}" if output else f"[stderr]: {stderr_text}"
    if process.returncode != 0 and output_style != "git":
        output += (
            f"\n[exit code]: {process.returncode}"
            if output
            else f"[exit code]: {process.returncode}"
        )
    if process.returncode != 0 and not output and output_style == "git":
        output = f"Git command failed with exit code {process.returncode}"
    output = output or "(no output)"
    if warning:
        output = f"{warning}\n{output}"
    return ShellExecutionResult(
        status="success" if process.returncode == 0 else "error",
        output=output,
        exit_code=process.returncode,
    )


async def execute_shell_command(
    command: str,
    *,
    timeout: int = 120,
    run_in_background: bool = False,
    session_id: str | None = None,
    sandbox_unavailable_approval: SandboxUnavailableApproval | None = None,
    required_sandbox: SandboxExecutionRequirement | None = None,
    output_style: str = "shell",
) -> ShellExecutionResult:
    """Execute a shell command without performing permission checks.

    ``sandbox_unavailable_approval`` lets the caller (which owns the permission
    context) explicitly accept sandbox degradation. It is consulted before a
    command enters a backend with incomplete configured guarantees and again if
    an unavailable/unsupported backend would require UNSANDBOXED fallback. The
    safe default without a callback is fail-closed.
    """
    parts = shlex.split(command)
    if not parts:
        return ShellExecutionResult(status="error", output="Empty command")

    timeout = max(1, min(timeout, 600))

    sandbox_state = resolve_sandbox_settings(get_execution_cwd())
    use_sandbox = sandbox_state.enabled and not is_excluded_command(
        command, cwd=get_execution_cwd()
    )
    if required_sandbox is not None:
        mismatch_reason = _sandbox_snapshot_mismatch(
            required_sandbox,
            sandbox_state,
            use_sandbox=use_sandbox,
            require_complete_capabilities=True,
        )
        if mismatch_reason is not None:
            return _blocked_sandbox_result(
                required_sandbox.backend_id,
                "permission was auto-approved only for sandbox execution, but "
                f"{mismatch_reason}; host execution was blocked",
            )
    if use_sandbox:
        if run_in_background:
            return ShellExecutionResult(
                status="error",
                output=(
                    "sandboxed: false\n"
                    f"backend: {sandbox_state.backend}\n"
                    "reason: background sandbox execution is not implemented; run the "
                    "command in the foreground, add a sandbox exclusion with "
                    "/sandbox exclude, or run /sandbox disable"
                ),
            )
        elif sandbox_state.policy is not None:
            degradation_warning = None
            backend_status = _selected_backend_status(sandbox_state)
            if backend_status is not None and backend_status.available:
                blockers = autoapproval_blockers(
                    sandbox_state.policy,
                    backend_status.capabilities,
                )
                if blockers:
                    degradation_reason = sandbox_degradation_reason(
                        sandbox_state.backend,
                        blockers,
                    )
                    degradation_snapshot = _sandbox_snapshot(sandbox_state, backend_status)
                    approved = await _resolve_sandbox_unavailable_approval(
                        None if required_sandbox is not None else sandbox_unavailable_approval,
                        degradation_reason,
                    )
                    if not approved:
                        return _blocked_sandbox_result(
                            sandbox_state.backend,
                            "explicit sandbox degradation approval required before execution; "
                            f"{degradation_reason}",
                        )

                    refreshed_state = resolve_sandbox_settings(get_execution_cwd())
                    refreshed_use_sandbox = refreshed_state.enabled and not is_excluded_command(
                        command,
                        cwd=get_execution_cwd(),
                    )
                    mismatch_reason = _sandbox_snapshot_mismatch(
                        degradation_snapshot,
                        refreshed_state,
                        use_sandbox=refreshed_use_sandbox,
                        require_complete_capabilities=False,
                    )
                    if mismatch_reason is not None:
                        return _blocked_sandbox_result(
                            degradation_snapshot.backend_id,
                            "sandbox state changed after degradation approval; "
                            f"{mismatch_reason}; reapproval is required before execution",
                        )
                    sandbox_state = refreshed_state
                    degradation_warning = (
                        "warning: sandbox degradation explicitly approved before execution; "
                        f"{degradation_reason}"
                    )

            # Fail-closed allowlist for the sandboxed path: forward only benign
            # vars + explicit session vars, never the full host env. A
            # pattern-based scrub alone misses oddly-named secrets
            # (MYCUSTOMCREDS, CI_JOB_JWT); the allowlist drops everything unknown.
            child_env = build_sandbox_env(session_id)
            sandbox_result = await execute_with_sdk_backend(
                SandboxExecutionContext(
                    cwd=get_execution_cwd().resolve(),
                    repo_root=get_execution_cwd().resolve(),
                    command=command,
                    env=child_env,
                    timeout=timeout,
                    background=False,
                    session_id=session_id,
                    policy=sandbox_state.policy,
                    degradation_approved=degradation_warning is not None,
                )
            )
            if sandbox_result.status not in {"unavailable", "unsupported"}:
                output = _sandbox_output(sandbox_result, output_style=output_style)
                if degradation_warning:
                    output = f"{degradation_warning}\n{output}"
                return ShellExecutionResult(
                    status=(
                        "success"
                        if sandbox_result.status == "success" and sandbox_result.exit_code == 0
                        else "error"
                    ),
                    output=output,
                    exit_code=sandbox_result.exit_code,
                )
            # Sandbox backend cannot run this command. Offer graceful
            # degradation to the caller: run UNSANDBOXED only behind an explicit
            # approval, with a visible warning. Absent approval, keep the
            # existing fail-closed error so nothing runs unsandboxed silently.
            backend_reason = sandbox_result.reason or "sandbox backend unavailable"
            fallback_trigger = f"sandbox backend became unavailable ({backend_reason})"
            fallback_requirement = capture_unsandboxed_fallback_requirement(
                sandbox_state,
                command=command,
                trigger=fallback_trigger,
            )
            if fallback_requirement is None:
                return _blocked_sandbox_result(
                    sandbox_state.backend,
                    "unable to capture exact sandbox fallback state; host execution was blocked",
                )
            approved = await _resolve_sandbox_unavailable_approval(
                sandbox_unavailable_approval,
                fallback_requirement.reason,
            )
            if approved:
                refreshed_state = resolve_sandbox_settings(get_execution_cwd())
                mismatch_reason = unsandboxed_fallback_requirement_mismatch(
                    fallback_requirement,
                    refreshed_state,
                    command=command,
                    trigger=fallback_trigger,
                )
                if mismatch_reason is not None:
                    return _blocked_sandbox_result(
                        fallback_requirement.backend_id,
                        "sandbox state changed after exact unsandboxed fallback approval; "
                        f"{mismatch_reason}; a new exact approval is required before execution",
                    )
                warning = (
                    f"warning: sandbox unavailable ({backend_reason}); "
                    "running command UNSANDBOXED with exact state-bound approval"
                )
                return await _run_foreground_unsandboxed(
                    command,
                    timeout=timeout,
                    session_id=session_id,
                    warning=warning,
                    output_style=output_style,
                )
            return _blocked_sandbox_result(
                sandbox_state.backend,
                "exact unsandboxed fallback approval was not granted; "
                f"{fallback_requirement.reason}",
            )

    if run_in_background:
        shell_id = str(uuid.uuid4())[:8]
        child_env = build_subprocess_env(session_id)
        # Own process group so terminate() can signal the whole tree.
        if IS_WINDOWS:
            process = await asyncio.create_subprocess_exec(
                "powershell.exe",
                "-NoProfile",
                "-Command",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=child_env,
                cwd=get_execution_cwd(),
                **_new_session_kwargs(),
            )
        else:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=child_env,
                cwd=get_execution_cwd(),
                **_new_session_kwargs(),
            )
        shell = BackgroundProcess(
            shell_id=shell_id,
            command=command,
            process=process,
            start_time=time.time(),
        )
        BackgroundProcessManager.add(shell)
        await BackgroundProcessManager.start_monitor(shell_id)
        return ShellExecutionResult(
            status="background",
            output=(
                f"Command started in background.\n"
                f"shell_id: {shell_id}\n"
                f"Use shell_output(shell_id='{shell_id}') to monitor output.\n"
                f"Use shell_kill(shell_id='{shell_id}') to terminate."
            ),
            shell_id=shell_id,
        )

    return await _run_foreground_unsandboxed(
        command,
        timeout=timeout,
        session_id=session_id,
        output_style=output_style,
    )


async def execute_powershell_command(
    command: str,
    *,
    timeout: int = 120,
    run_in_background: bool = False,
    session_id: str | None = None,
) -> ShellExecutionResult:
    """Execute a PowerShell command without performing permission checks."""

    if not command.strip():
        return ShellExecutionResult(status="error", output="Empty command")

    executable = resolve_powershell_executable()
    if executable is None:
        return ShellExecutionResult(
            status="error",
            output="PowerShell executable not found. Install PowerShell (pwsh) or run on Windows.",
        )

    timeout = max(1, min(timeout, 600))
    argv = [executable, "-NoProfile", "-NonInteractive", "-Command", command]
    child_env = build_subprocess_env(session_id)

    if run_in_background:
        shell_id = str(uuid.uuid4())[:8]
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=child_env,
            cwd=get_execution_cwd(),
            **_new_session_kwargs(),
        )
        shell = BackgroundProcess(
            shell_id=shell_id,
            command=f"powershell: {command}",
            process=process,
            start_time=time.time(),
        )
        BackgroundProcessManager.add(shell)
        await BackgroundProcessManager.start_monitor(shell_id)
        return ShellExecutionResult(
            status="background",
            output=(
                f"PowerShell command started in background.\n"
                f"shell_id: {shell_id}\n"
                f"Use shell_output(shell_id='{shell_id}') to monitor output.\n"
                f"Use shell_kill(shell_id='{shell_id}') to terminate."
            ),
            shell_id=shell_id,
        )

    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=child_env,
        cwd=get_execution_cwd(),
        **_new_session_kwargs(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        _kill_process_group(process)
        await _drain_after_kill(process, timeout=1)
        return ShellExecutionResult(
            status="error",
            output=f"PowerShell command timed out after {timeout} seconds",
        )

    output = stdout.decode("utf-8", errors="replace").strip()
    stderr_text = stderr.decode("utf-8", errors="replace").strip()
    if stderr_text:
        output += f"\n[stderr]: {stderr_text}" if output else f"[stderr]: {stderr_text}"
    if process.returncode != 0:
        output += (
            f"\n[exit code]: {process.returncode}"
            if output
            else f"[exit code]: {process.returncode}"
        )

    return ShellExecutionResult(
        status="success" if process.returncode == 0 else "error",
        output=output or "(no output)",
        exit_code=process.returncode,
    )


async def get_background_output(
    shell_id: str, filter_str: str | None = None
) -> ShellExecutionResult:
    """Get incremental output for a background shell."""
    shell = BackgroundProcessManager.get(shell_id)
    if not shell:
        available = BackgroundProcessManager.list_ids()
        return ShellExecutionResult(
            status="error",
            output=f"Shell not found: {shell_id}\nAvailable: {available or 'none'}",
        )

    new_lines = shell.get_new_output(filter_pattern=filter_str)
    output = "\n".join(new_lines) if new_lines else "(no new output)"
    status_info = f"\n[status]: {shell.status}"
    if shell.exit_code is not None:
        status_info += f"\n[exit_code]: {shell.exit_code}"
    return ShellExecutionResult(status="success", output=output + status_info, shell_id=shell_id)


async def terminate_background_command(shell_id: str) -> ShellExecutionResult:
    """Terminate a background shell and return its final output summary."""
    shell = BackgroundProcessManager.get(shell_id)
    remaining_lines = shell.get_new_output() if shell else []
    try:
        shell = await BackgroundProcessManager.terminate(shell_id)
    except ValueError as exc:
        available = BackgroundProcessManager.list_ids()
        return ShellExecutionResult(
            status="error",
            output=f"{exc}\nAvailable: {available or 'none'}",
        )
    output = "\n".join(remaining_lines) if remaining_lines else "(no remaining output)"
    return ShellExecutionResult(
        status="success",
        output=f"Shell {shell_id} terminated.\n{output}\n[exit_code]: {shell.exit_code}",
        exit_code=shell.exit_code,
        shell_id=shell_id,
    )
