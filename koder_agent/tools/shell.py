"""Shell command execution tools with background process management.

Supports both bash (Unix/Linux/macOS) and PowerShell (Windows).
"""

import asyncio
import platform
import re
import shlex
import time
import uuid
from typing import List, Optional

from agents import function_tool
from pydantic import BaseModel

from ..core.security import SecurityGuard

# Detect OS once at module load
IS_WINDOWS = platform.system() == "Windows"
SHELL_NAME = "PowerShell" if IS_WINDOWS else "bash"


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
    ):
        self.shell_id = shell_id
        self.command = command
        self.process = process
        self.start_time = start_time
        self.output_lines: List[str] = []
        self.last_read_index = 0
        self.status = "running"  # running, completed, failed, terminated, error
        self.exit_code: Optional[int] = None

    def add_output(self, line: str):
        """Add new output line."""
        self.output_lines.append(line)

    def get_new_output(self, filter_pattern: Optional[str] = None) -> List[str]:
        """Get new output since last check, optionally filtered by regex."""
        new_lines = self.output_lines[self.last_read_index :]
        self.last_read_index = len(self.output_lines)

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
        """Terminate the background process."""
        if self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
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
                if shell_id in cls._monitor_tasks:
                    del cls._monitor_tasks[shell_id]

        task = asyncio.create_task(monitor())
        cls._monitor_tasks[shell_id] = task

    @classmethod
    def _cancel_monitor(cls, shell_id: str) -> None:
        """Cancel and remove a monitoring task (internal use only)."""
        if shell_id in cls._monitor_tasks:
            task = cls._monitor_tasks[shell_id]
            if not task.done():
                task.cancel()
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

        # Terminate the process
        await shell.terminate()

        # Clean up monitoring and remove from manager
        cls._cancel_monitor(shell_id)
        cls._remove(shell_id)

        return shell


@function_tool
async def run_shell(command: str, timeout: int = 120, run_in_background: bool = False) -> str:
    """Execute a shell command with security checks.

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

            # Start background process with combined stdout/stderr
            if IS_WINDOWS:
                process = await asyncio.create_subprocess_exec(
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )

            # Create background shell and add to manager
            bg_shell = BackgroundShell(
                shell_id=shell_id,
                command=command,
                process=process,
                start_time=time.time(),
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
            # Foreground execution
            if IS_WINDOWS:
                process = await asyncio.create_subprocess_exec(
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                process.kill()
                return f"Command timed out after {timeout} seconds"

            # Decode output
            output = stdout.decode("utf-8", errors="replace").strip()
            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace").strip()
                if stderr_text:
                    output += f"\n[stderr]: {stderr_text}"

            if process.returncode != 0:
                output += f"\n[exit code]: {process.returncode}"

            return output or "(no output)"

    except Exception as e:
        return f"Error executing command: {str(e)}"


@function_tool
async def shell_output(shell_id: str, filter_str: Optional[str] = None) -> str:
    """Retrieve output from a background shell process.

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
    """Terminate a background shell process.

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
async def git_command(command: str) -> str:
    """Execute a git command."""
    try:
        # Ensure command starts with 'git'
        if not command.strip().startswith("git"):
            command = f"git {command}"

        # Security validation
        error = SecurityGuard.validate_command(command)
        if error:
            return error

        # Execute git command (always foreground, 30s timeout)
        if IS_WINDOWS:
            process = await asyncio.create_subprocess_exec(
                "powershell.exe",
                "-NoProfile",
                "-Command",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        except asyncio.TimeoutError:
            process.kill()
            return "Git command timed out after 30 seconds"

        # Decode output
        output = stdout.decode("utf-8", errors="replace").strip()
        if stderr:
            # Git often uses stderr for informational messages
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            if stderr_text:
                output += f"\n{stderr_text}"

        if process.returncode != 0 and not output:
            output = f"Git command failed with exit code {process.returncode}"

        return output or "(no output)"

    except Exception as e:
        return f"Error executing git command: {str(e)}"
