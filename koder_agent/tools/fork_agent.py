"""Fork subagent: context inheritance for child agents.

When a subagent is forked, it inherits the parent's conversation
context (system prompt + message history). This enables:
- Prompt cache sharing (child reuses parent's cached prefix)
- Context continuity (child knows what parent has done)
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from ..harness.memory.budget import estimate_text_tokens


@dataclass
class ForkContext:
    """Captured parent context for forking to a child agent."""

    system_prompt: str | None = None
    conversation_messages: list[dict] = field(default_factory=list)

    def to_messages(self) -> list[dict]:
        """Reconstruct full message list for child agent."""
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.extend(self.conversation_messages)
        return msgs

    def estimate_tokens(self) -> int:
        """Estimate total token count of the fork context."""
        total = 0
        if self.system_prompt:
            total += estimate_text_tokens(self.system_prompt)
        for msg in self.conversation_messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += estimate_text_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        total += estimate_text_tokens(block["text"])
        return total


def build_fork_context(messages: list[dict]) -> ForkContext:
    """Build a fork context from parent's message history.

    Extracts system prompt and conversation messages,
    filtering out incomplete tool call/result pairs.
    """
    if not messages:
        return ForkContext()

    system_prompt = None
    conversation = []

    for msg in messages:
        if msg.get("role") == "system" and system_prompt is None:
            system_prompt = msg.get("content", "")
        else:
            conversation.append(deepcopy(msg))

    # Filter incomplete tool calls
    conversation = filter_incomplete_tool_calls(conversation)

    return ForkContext(
        system_prompt=system_prompt,
        conversation_messages=conversation,
    )


def filter_incomplete_tool_calls(messages: list[dict]) -> list[dict]:
    """Remove tool_calls that don't have matching tool results.

    This prevents the child agent from seeing orphaned tool calls
    that were in-flight when the fork happened.
    """
    # Collect all tool_call_ids that have results
    result_ids = {
        msg["tool_call_id"]
        for msg in messages
        if msg.get("role") == "tool" and "tool_call_id" in msg
    }

    filtered = []
    for msg in messages:
        if msg.get("tool_calls"):
            # Filter to only completed tool calls
            completed = [tc for tc in msg["tool_calls"] if tc.get("id") in result_ids]
            if completed:
                new_msg = {**msg, "tool_calls": completed}
                filtered.append(new_msg)
            elif msg.get("content"):
                # Keep message without tool_calls if it has text content
                new_msg = {k: v for k, v in msg.items() if k != "tool_calls"}
                filtered.append(new_msg)
            # Skip entirely if no completed calls and no content
        else:
            filtered.append(msg)

    # Remove orphaned tool results (tool_call_id not in any remaining assistant's tool_calls)
    remaining_call_ids = set()
    for msg in filtered:
        for tc in msg.get("tool_calls", []):
            remaining_call_ids.add(tc.get("id"))

    filtered = [
        msg
        for msg in filtered
        if msg.get("role") != "tool" or msg.get("tool_call_id") in remaining_call_ids
    ]

    # The Agents SDK persists Responses items, not only Chat Completions
    # tool_calls. Keep their pairing namespace separate from tool_call_id.
    native_calls = {
        msg["call_id"]
        for msg in filtered
        if msg.get("type") == "function_call"
        and isinstance(msg.get("call_id"), str)
        and msg["call_id"]
    }
    native_outputs = {
        msg["call_id"]
        for msg in filtered
        if msg.get("type") == "function_call_output"
        and isinstance(msg.get("call_id"), str)
        and msg["call_id"]
    }
    paired_native_ids = native_calls & native_outputs
    return [
        msg
        for msg in filtered
        if msg.get("type") not in {"function_call", "function_call_output"}
        or (isinstance(msg.get("call_id"), str) and msg["call_id"] in paired_native_ids)
    ]
