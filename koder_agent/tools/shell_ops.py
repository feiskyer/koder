"""Shell command execution tools."""

import shlex
import subprocess

from agents import function_tool
from pydantic import BaseModel

from ..core.security import SecurityGuard


class ShellModel(BaseModel):
    command: str


class GitModel(BaseModel):
    command: str


@function_tool
def run_shell(command: str) -> str:
    """Execute a shell command with security checks."""
    try:
        # Security validation
        error = SecurityGuard.validate_command(command)
        if error:
            return error

        # Additional check for allowed commands
        parts = shlex.split(command)
        if not parts:
            return "Empty command"

        # Execute command with timeout
        result = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=30,  # 30 second timeout
            cwd=None,  # Use current working directory
        )

        output = result.stdout.strip()
        if result.stderr:
            output += f"\n[stderr]: {result.stderr.strip()}"

        if result.returncode != 0:
            output += f"\n[exit code]: {result.returncode}"

        return output or "(no output)"

    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds"
    except subprocess.CalledProcessError as e:
        return f"Command failed: {str(e)}"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@function_tool
def git_command(command: str) -> str:
    """Execute a git command."""
    try:
        # Ensure command starts with 'git'
        if not command.strip().startswith("git"):
            command = f"git {command}"

        # Security validation
        error = SecurityGuard.validate_command(command)
        if error:
            return error

        # Execute git command
        result = subprocess.run(
            command, shell=True, text=True, capture_output=True, timeout=30, cwd=None
        )

        output = result.stdout.strip()
        if result.stderr:
            # Git often uses stderr for informational messages
            output += f"\n{result.stderr.strip()}"

        if result.returncode != 0 and not output:
            output = f"Git command failed with exit code {result.returncode}"

        return output or "(no output)"

    except subprocess.TimeoutExpired:
        return "Git command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing git command: {str(e)}"
