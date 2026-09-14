"""Shell command execution tools with background process management.

Supports both bash (Unix/Linux/macOS) and PowerShell (Windows).
"""

import asyncio
import contextlib
import inspect
import os
import platform
import re
import shlex
import signal
import time
import uuid
from collections import deque
from typing import Awaitable, Callable, List, Optional, Union

from pydantic import BaseModel

from ..core.security import SecurityGuard
from ..harness.execution_context import get_execution_cwd
from ..harness.tools import shell_executor as harness_shell_executor
from ..utils.async_tasks import await_owned_task
from .compat import function_tool
from .permission_context import approve_sandbox_degradation, required_sandbox_execution
from .todo import TodoRuntimeIdentity, get_todo_store_or_none

# Signature of the callback threaded into the harness shell executor to decide,
# at runtime, whether a command may fall back to UNSANDBOXED execution when the
# sandbox backend is unavailable/unsupported. Given a one-line reason, it must
# return True to approve the (warned) unsandboxed run or False to keep the
# fail-closed error. May be sync or async.
SandboxUnavailableApproval = Callable[[str], Union[bool, Awaitable[bool]]]

# Detect OS once at module load
IS_WINDOWS = platform.system() == "Windows"
SHELL_NAME = "PowerShell" if IS_WINDOWS else "bash"

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
    """Subprocess kwargs that place the child in its own process group/session.

    On POSIX this makes the spawned shell wrapper a session/group leader, so we
    can signal the whole group (wrapper + grandchildren) on timeout instead of
    orphaning grandchildren. No-op on Windows.
    """
    if IS_WINDOWS:
        return {}
    return {"start_new_session": True}


def _signal_process_group(process: "asyncio.subprocess.Process", sig: int) -> None:
    """Send ``sig`` to the child's whole process group (POSIX), guarded.

    Only signals the group when the child is a group leader distinct from our
    own process group — i.e. it was spawned with ``start_new_session=True``.
    This is a hard safety check: if the child shares our group (e.g. it was not
    started in a new session), ``killpg`` would signal this very process, so we
    fall back to signalling just the child. Also falls back if the group can't
    be resolved/signalled or on Windows. Never raises.
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
            # Group already gone or not signalable; fall through to plain signal.
            pass
    with contextlib.suppress(ProcessLookupError, OSError):
        if sig == signal.SIGKILL:
            process.kill()
        else:
            process.terminate()


def _kill_process_group(process: "asyncio.subprocess.Process") -> None:
    """SIGKILL the child and its whole process group (POSIX), guarded."""
    _signal_process_group(process, signal.SIGKILL)


async def _drain_after_kill(
    process: "asyncio.subprocess.Process", timeout: float = 1.0
) -> tuple[str, str]:
    """Drain buffered output after a kill without blocking forever.

    ``communicate()`` on a killed process can still hang (e.g. an inherited pipe
    held open by a surviving grandchild), so wrap it in a timeout. Returns
    decoded ``(stdout, stderr)`` or empty strings if nothing could be read.
    """
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, Exception):
        return "", ""
    stdout = out.decode("utf-8", errors="replace").strip() if out else ""
    stderr = err.decode("utf-8", errors="replace").strip() if err else ""
    return stdout, stderr


class ShellModel(BaseModel):
    command: str
    timeout: int = 120
    run_in_background: bool = False


class ShellOutputModel(BaseModel):
    shell_id: str
    filter_str: Optional[str] = None


class ShellKillModel(BaseModel):
    shell_id: str


class GitModel(BaseModel):
    command: str
    timeout: int = 60


class BackgroundShell:
    """Background shell data container.

    Pure data class that stores process state and output.
    IO operations are managed externally by BackgroundShellManager.
    """

    def __init__(
        self,
        shell_id: str,
        command: str,
        process: "asyncio.subprocess.Process",
        start_time: float,
        owner: TodoRuntimeIdentity | None = None,
    ):
        self.shell_id = shell_id
        self.command = command
        self.process = process
        self.start_time = start_time
        self.owner = owner
        # Bounded ring buffer: old lines are evicted from the left once the cap
        # is hit, keeping session memory bounded for verbose processes.
        self.output_lines: "deque[str]" = deque(maxlen=_bg_shell_max_lines())
        # Monotonic count of all lines ever appended (including evicted ones).
        # last_read_index is an index into this logical stream, not into the
        # deque, so it stays correct even after left-eviction shifts positions.
        self._total_appended = 0
        self.last_read_index = 0
        self.status = "running"  # running, completed, failed, terminated, error
        self.exit_code: Optional[int] = None
        self._termination_task: Optional[asyncio.Task["BackgroundShell"]] = None

    def add_output(self, line: str):
        """Add new output line."""
        self.output_lines.append(line)
        self._total_appended += 1

    def get_new_output(self, filter_pattern: Optional[str] = None) -> List[str]:
        """Get new output since last check, optionally filtered by regex.

        ``last_read_index`` counts against the monotonic total-appended stream.
        The deque only retains the most recent ``maxlen`` lines, so the oldest
        retained line has logical index ``_total_appended - len(output_lines)``.
        If lines were evicted since the last read we clamp forward to that start
        (silently skipping the dropped lines) and never re-emit read lines.
        """
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
                # Invalid regex, return all lines
                pass

        return new_lines

    def update_status(self, is_alive: bool, exit_code: Optional[int] = None):
        """Update process status."""
        if not is_alive:
            self.status = "completed" if exit_code == 0 else "failed"
            self.exit_code = exit_code
        else:
            self.status = "running"

    async def terminate(self):
        """Terminate the background process (and its process group on POSIX)."""
        if self.process.returncode is None:
            # SIGTERM the whole group (not just the wrapper) so grandchildren
            # spawned via '&' are asked to exit too.
            _signal_process_group(self.process, signal.SIGTERM)
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                # Escalate to SIGKILL of the whole group so nothing is orphaned.
                _kill_process_group(self.process)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self.process.wait(), timeout=5)
        self.status = "terminated"
        self.exit_code = self.process.returncode


class BackgroundShellManager:
    """Manager for all background shell processes (singleton via class variables)."""

    _shells: dict[str, BackgroundShell] = {}
    _monitor_tasks: dict[str, asyncio.Task] = {}

    @classmethod
    def add(cls, shell: BackgroundShell) -> None:
        """Add a background shell to management."""
        cls._shells[shell.shell_id] = shell

    @classmethod
    def get(cls, shell_id: str) -> Optional[BackgroundShell]:
        """Get a background shell by ID."""
        return cls._shells.get(shell_id)

    @classmethod
    def get_available_ids(cls) -> List[str]:
        """Get all available shell IDs."""
        return list(cls._shells.keys())

    @classmethod
    def get_owned_ids(cls, owner: TodoRuntimeIdentity) -> List[str]:
        """Get shell IDs created within one explicit runtime owner scope."""
        return [shell_id for shell_id, shell in cls._shells.items() if shell.owner == owner]

    @classmethod
    def _remove(cls, shell_id: str) -> None:
        """Remove a background shell from management (internal use only)."""
        if shell_id in cls._shells:
            del cls._shells[shell_id]

    @classmethod
    async def start_monitor(cls, shell_id: str) -> None:
        """Start monitoring a background shell's output."""
        shell = cls.get(shell_id)
        if not shell:
            return

        async def monitor():
            try:
                process = shell.process
                # Continuously read output until process stdout reaches EOF
                while True:
                    if not process.stdout:
                        break
                    try:
                        line = await asyncio.wait_for(process.stdout.readline(), timeout=0.1)
                        if line:
                            decoded_line = line.decode("utf-8", errors="replace").rstrip("\n")
                            shell.add_output(decoded_line)
                            continue
                        # No line returned: check if process ended before breaking
                        if process.returncode is not None:
                            break
                        await asyncio.sleep(0.05)
                        continue
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0.1)
                        continue
                    except Exception:
                        await asyncio.sleep(0.1)
                        continue

                # Process ended, wait for exit code
                try:
                    returncode = await process.wait()
                except Exception:
                    returncode = -1

                shell.update_status(is_alive=False, exit_code=returncode)

            except Exception as e:
                if shell_id in cls._shells:
                    cls._shells[shell_id].status = "error"
                    cls._shells[shell_id].add_output(f"Monitor error: {str(e)}")
            finally:
                if cls._monitor_tasks.get(shell_id) is asyncio.current_task():
                    del cls._monitor_tasks[shell_id]

        task = asyncio.create_task(monitor())
        cls._monitor_tasks[shell_id] = task

    @classmethod
    async def _cancel_monitor(cls, shell_id: str, task: Optional[asyncio.Task]) -> None:
        """Join the captured monitor without removing a later generation."""
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if cls._monitor_tasks.get(shell_id) is task:
            del cls._monitor_tasks[shell_id]

    @classmethod
    async def terminate(cls, shell_id: str) -> BackgroundShell:
        """Terminate a background shell and clean up all resources.

        Args:
            shell_id: The unique identifier of the background shell

        Returns:
            The terminated BackgroundShell object

        Raises:
            ValueError: If shell not found
        """
        shell = cls.get(shell_id)
        if not shell:
            raise ValueError(f"Shell not found: {shell_id}")

        if shell._termination_task is None:
            monitor = cls._monitor_tasks.get(shell_id)

            async def terminate_and_join():
                await shell.terminate()
                await cls._cancel_monitor(shell_id, monitor)
                if cls.get(shell_id) is shell:
                    cls._remove(shell_id)
                return shell

            shell._termination_task = asyncio.create_task(terminate_and_join())

        # Multiple stop requests share ownership. Cancelling a waiter must not
        # interrupt the process/reader cleanup or leave a detached monitor.
        cleanup = shell._termination_task
        try:
            return await asyncio.shield(cleanup)
        except asyncio.CancelledError as original:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await await_owned_task(cleanup)
            raise original
        finally:
            if cleanup.done() and (cleanup.cancelled() or cleanup.exception() is not None):
                if shell._termination_task is cleanup:
                    shell._termination_task = None


def build_sandbox_unavailable_approval(
    approver: Optional[Callable[[str], Union[bool, Awaitable[bool]]]] = None,
    *,
    tool_name: str | None = None,
    arguments: dict | None = None,
) -> SandboxUnavailableApproval:
    """Build a sandbox-degradation approval callback for the shell executor.

    This lives in the tool/permission layer (which owns the interactive approval
    context) and is threaded into ``execute_shell_command`` so that, when the
    sandbox backend is unavailable/unsupported, a command may still run
    UNSANDBOXED — but only behind an explicit yes from ``approver`` and always
    with a visible warning surfaced by the executor.

    ``approver`` receives the one-line degradation reason and returns True to
    approve or False to keep the fail-closed error. It may be sync or async. If
    ``approver`` is omitted the default is fail-closed (deny), matching the safe
    behaviour of ``execute_shell_command`` when no callback is supplied.
    """

    async def _approval(reason: str) -> bool:
        if approver is not None:
            result = approver(reason)
            if inspect.isawaitable(result):
                result = await result
            return bool(result)
        if tool_name is None:
            return False
        return await approve_sandbox_degradation(tool_name, arguments or {}, reason)

    return _approval


@function_tool
async def run_shell(command: str, timeout: int = 120, run_in_background: bool = False) -> str:
    """Execute a shell command with security checks.

    IMPORTANT - use dedicated tools instead of shell equivalents. Dedicated tools
    are easier to review and to grant permission for:
    - Read files: use read_file, NOT cat/head/tail
    - Edit files: use edit_file, NOT sed/awk
    - Create files: use write_file, NOT echo redirection or heredocs
    - Find files by name: use glob_search, NOT find/ls
    - Search file contents: use grep_search, NOT grep/rg
    - Communicate with the user: output text directly in your response,
      NEVER via echo or shell comments

    Usage notes:
    - Set run_in_background=true for long-running commands (dev servers, watchers,
      long builds); monitor with shell_output and stop with shell_kill. Do not use
      a trailing '&' instead.
    - No interactive input is supported: avoid interactive flags (e.g. -i) and
      commands that prompt; use non-interactive alternatives (e.g. --yes).
    - Oversized output is truncated (~30,000 characters, head and tail kept), so
      pipe verbose commands through filters (e.g. tail, --quiet) to keep the
      relevant part.

    Args:
        command: The shell command to execute
        timeout: Timeout in seconds (default: 120, max: 600). Only for foreground.
        run_in_background: Set true for long-running commands. Use shell_output to monitor.

    Returns:
        Command output, or shell_id if run_in_background=True
    """
    try:
        # Security validation
        error = SecurityGuard.validate_command(command)
        if error:
            return error

        # Validate command not empty
        parts = shlex.split(command)
        if not parts:
            return "Empty command"

        # Clamp timeout to valid range
        timeout = max(1, min(timeout, 600))

        if run_in_background:
            # Background execution
            shell_id = str(uuid.uuid4())[:8]

            # Start background process with combined stdout/stderr. Place it in
            # its own process group so shell_kill can signal the whole tree.
            if IS_WINDOWS:
                process = await asyncio.create_subprocess_exec(
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=get_execution_cwd(),
                    **_new_session_kwargs(),
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=get_execution_cwd(),
                    **_new_session_kwargs(),
                )

            # Create background shell and add to manager
            todo_store = get_todo_store_or_none()
            bg_shell = BackgroundShell(
                shell_id=shell_id,
                command=command,
                process=process,
                start_time=time.time(),
                owner=todo_store.identity if todo_store is not None else None,
            )
            BackgroundShellManager.add(bg_shell)

            # Start monitoring task
            await BackgroundShellManager.start_monitor(shell_id)

            return (
                f"Command started in background.\n"
                f"shell_id: {shell_id}\n"
                f"Use shell_output(shell_id='{shell_id}') to monitor output.\n"
                f"Use shell_kill(shell_id='{shell_id}') to terminate."
            )

        else:
            result = await harness_shell_executor.execute_shell_command(
                command,
                timeout=timeout,
                sandbox_unavailable_approval=build_sandbox_unavailable_approval(
                    tool_name="run_shell",
                    arguments={
                        "command": command,
                        "timeout": timeout,
                        "run_in_background": run_in_background,
                    },
                ),
                required_sandbox=required_sandbox_execution("run_shell", command),
            )
            return result.output

    except Exception as e:
        return f"Error executing command: {str(e)}"


@function_tool
async def shell_output(shell_id: str, filter_str: Optional[str] = None) -> str:
    """Retrieve output from a background shell started via run_shell(run_in_background=true).

    Returns only output produced since the last check, plus the process status
    (e.g. running, completed, failed, terminated). Poll periodically for
    long-running commands.

    Args:
        shell_id: The ID returned when starting a background command
        filter_str: Optional regex to filter output lines

    Returns:
        New output since last check, process status, and available shell IDs
    """
    try:
        bg_shell = BackgroundShellManager.get(shell_id)
        if not bg_shell:
            available = BackgroundShellManager.get_available_ids()
            return f"Shell not found: {shell_id}\nAvailable: {available or 'none'}"

        new_lines = bg_shell.get_new_output(filter_pattern=filter_str)
        output = "\n".join(new_lines) if new_lines else "(no new output)"

        # Add status info
        status_info = f"\n[status]: {bg_shell.status}"
        if bg_shell.exit_code is not None:
            status_info += f"\n[exit_code]: {bg_shell.exit_code}"

        return output + status_info

    except Exception as e:
        return f"Error retrieving output: {str(e)}"


@function_tool
async def shell_kill(shell_id: str) -> str:
    """Terminate a background shell started via run_shell(run_in_background=true).

    Use this to stop dev servers, watchers, or other long-running background
    commands when they are no longer needed.

    Args:
        shell_id: The ID of the background shell to terminate

    Returns:
        Termination status and any remaining output
    """
    try:
        # Get remaining output before termination
        bg_shell = BackgroundShellManager.get(shell_id)
        if bg_shell:
            remaining_lines = bg_shell.get_new_output()
        else:
            remaining_lines = []

        # Terminate
        bg_shell = await BackgroundShellManager.terminate(shell_id)

        output = "\n".join(remaining_lines) if remaining_lines else "(no remaining output)"
        return f"Shell {shell_id} terminated.\n{output}\n[exit_code]: {bg_shell.exit_code}"

    except ValueError as e:
        available = BackgroundShellManager.get_available_ids()
        return f"{str(e)}\nAvailable: {available or 'none'}"
    except Exception as e:
        return f"Error terminating shell: {str(e)}"


@function_tool
async def git_command(command: str, timeout: int = 60) -> str:
    """Execute a git command.

    Git Safety Protocol:
    - NEVER modify git config.
    - NEVER run destructive commands (push --force, reset --hard, clean -f,
      checkout/restore that discards changes) unless the user explicitly requests it.
    - NEVER use --no-verify or otherwise skip hooks.
    - If a pre-commit hook fails, the commit did NOT happen: fix the issue and
      create a NEW commit; do not use --amend (it would rewrite the previous commit).

    Args:
        command: The git command to execute (with or without leading 'git')
        timeout: Timeout in seconds (default: 60, max: 300). Increase for clone/fetch/gc.

    Returns:
        Command output or error message
    """
    try:
        # Clamp timeout to valid range
        timeout = max(1, min(timeout, 300))
        requested_command = command

        # Ensure command starts with 'git'
        if not command.strip().startswith("git"):
            command = f"git {command}"

        # Security validation
        error = SecurityGuard.validate_command(command)
        if error:
            return error

        result = await harness_shell_executor.execute_shell_command(
            command,
            timeout=timeout,
            sandbox_unavailable_approval=build_sandbox_unavailable_approval(
                tool_name="git_command",
                arguments={"command": requested_command, "timeout": timeout},
            ),
            required_sandbox=required_sandbox_execution("git_command", requested_command),
            output_style="git",
        )
        return result.output

    except Exception as e:
        return f"Error executing git command: {str(e)}"
