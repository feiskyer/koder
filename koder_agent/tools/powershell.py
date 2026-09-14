"""PowerShell command execution tool."""

from __future__ import annotations

from pydantic import BaseModel

from koder_agent.harness.execution_context import get_execution_cwd
from koder_agent.harness.permissions.powershell_classifier import classify_powershell_command
from koder_agent.harness.sandbox_settings import is_excluded_command, resolve_sandbox_settings
from koder_agent.harness.tools.shell_executor import (
    capture_unsandboxed_fallback_requirement,
    execute_powershell_command,
    unsandboxed_fallback_requirement_mismatch,
)

from .compat import function_tool
from .permission_context import approve_sandbox_degradation


class PowerShellModel(BaseModel):
    command: str
    timeout: int = 120
    run_in_background: bool = False


@function_tool
async def run_powershell(command: str, timeout: int = 120, run_in_background: bool = False) -> str:
    """Execute a PowerShell command with Koder permission and background support.

    Args:
        command: The PowerShell command to execute.
        timeout: Timeout in seconds for foreground commands. The value is clamped to 1-600.
        run_in_background: Set true for long-running commands. Use shell_output to monitor.

    Returns:
        Command output, or a shell_id if run_in_background=True.
    """

    decision = classify_powershell_command(command)
    if not decision.allowed:
        return decision.reason

    cwd = get_execution_cwd()
    sandbox_state = resolve_sandbox_settings(cwd)
    degraded = sandbox_state.enabled and not is_excluded_command(command, cwd=cwd)
    if degraded:
        fallback_trigger = "PowerShell sandbox execution is unsupported"
        fallback_requirement = capture_unsandboxed_fallback_requirement(
            sandbox_state,
            command=command,
            trigger=fallback_trigger,
        )
        if fallback_requirement is None:
            return (
                "sandboxed: false\ncreated: false\nexecuted: false\n"
                "reason: unable to capture exact PowerShell sandbox fallback state; "
                "host execution was blocked"
            )
        approved = await approve_sandbox_degradation(
            "run_powershell",
            {
                "command": command,
                "timeout": timeout,
                "run_in_background": run_in_background,
            },
            fallback_requirement.reason,
        )
        if not approved:
            return (
                "sandboxed: false\ncreated: false\nexecuted: false\n"
                "reason: exact PowerShell unsandboxed fallback approval was not granted; "
                f"{fallback_requirement.reason}"
            )
        refreshed_state = resolve_sandbox_settings(get_execution_cwd())
        mismatch_reason = unsandboxed_fallback_requirement_mismatch(
            fallback_requirement,
            refreshed_state,
            command=command,
            trigger=fallback_trigger,
        )
        if mismatch_reason is not None:
            return (
                "sandboxed: false\ncreated: false\nexecuted: false\n"
                "reason: sandbox state changed after exact PowerShell fallback approval; "
                f"{mismatch_reason}; a new exact approval is required before execution"
            )

    result = await execute_powershell_command(
        command,
        timeout=timeout,
        run_in_background=run_in_background,
    )
    if degraded:
        return (
            "warning: PowerShell sandbox execution is unsupported; running UNSANDBOXED "
            f"with exact state-bound approval\n{result.output}"
        )
    return result.output
