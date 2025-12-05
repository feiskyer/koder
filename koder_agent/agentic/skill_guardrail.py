"""Guardrail for enforcing skill-based tool restrictions.

This module provides a ToolInputGuardrail that intercepts tool calls and
rejects those not permitted by the currently active skill restrictions.

When a skill with `allowed_tools` is loaded via get_skill(), subsequent
tool calls are validated against the allowed list. Tools not in the list
are rejected with an informative message.
"""

from agents import (
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolInputGuardrailData,
)

from ..tools.skill_context import get_active_restrictions


def skill_tool_restriction_guardrail(
    data: ToolInputGuardrailData,
) -> ToolGuardrailFunctionOutput:
    """Guardrail that enforces skill-declared tool restrictions.

    When a skill is loaded with `allowed_tools`, only those tools
    (plus always-allowed tools like get_skill, todo_read, todo_write)
    can be used.

    Supports pattern matching for fine-grained control:
    - "read_file"           - Exact tool name match
    - "run_shell:git *"     - Shell commands matching glob pattern
    - "run_shell:*"         - All shell commands allowed
    - "*"                   - Wildcard, all tools allowed

    Args:
        data: The guardrail input data containing tool context

    Returns:
        ToolGuardrailFunctionOutput indicating whether to allow or reject
    """
    restrictions = get_active_restrictions()

    # No active restrictions - allow all tools
    if restrictions is None:
        return ToolGuardrailFunctionOutput.allow()

    # Guard against missing or empty tool_name (defensive coding)
    tool_name = getattr(data.context, "tool_name", "") or ""
    if not tool_name:
        # Unknown tool - reject with clear message
        return ToolGuardrailFunctionOutput.reject_content(
            message="Unable to verify tool: tool name is missing or empty",
            output_info={"error": "missing_tool_name"},
        )

    # Get tool arguments for pattern matching (e.g., shell command matching)
    tool_args = getattr(data.context, "tool_arguments", None)

    # Check if tool is allowed (with pattern matching support)
    if restrictions.is_tool_allowed(tool_name, tool_args):
        return ToolGuardrailFunctionOutput.allow()

    # Tool is not allowed - reject with informative message
    allowed_list = ", ".join(sorted(restrictions.allowed_tools))
    skills_list = ", ".join(restrictions.loaded_skills)

    message = (
        f"Tool '{tool_name}' is not permitted while skill(s) [{skills_list}] "
        f"are active with tool restrictions. "
        f"Allowed tools: {allowed_list}. "
        f"You can load a skill without restrictions to clear these limits."
    )

    return ToolGuardrailFunctionOutput.reject_content(
        message=message,
        output_info={
            "blocked_tool": tool_name,
            "active_skills": restrictions.loaded_skills,
            "allowed_tools": list(restrictions.allowed_tools),
        },
    )


# Create the guardrail instance for use in agent configuration
skill_restriction_guardrail = ToolInputGuardrail(
    guardrail_function=skill_tool_restriction_guardrail,
    name="skill_tool_restrictions",
)
