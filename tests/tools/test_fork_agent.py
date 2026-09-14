"""Tests for fork subagent with context inheritance."""

from koder_agent.tools.fork_agent import (
    ForkContext,
    build_fork_context,
    filter_incomplete_tool_calls,
)


def test_build_fork_context_from_messages():
    """Should extract conversation messages for forking."""
    messages = [
        {"role": "system", "content": "You are a helper."},
        {"role": "user", "content": "Fix the bug in auth.py"},
        {"role": "assistant", "content": "I'll look at auth.py"},
        {"role": "user", "content": "Also check the tests"},
    ]
    ctx = build_fork_context(messages)
    assert ctx.system_prompt == "You are a helper."
    assert len(ctx.conversation_messages) == 3  # Excludes system
    assert ctx.conversation_messages[0]["role"] == "user"


def test_build_fork_context_empty():
    ctx = build_fork_context([])
    assert ctx.system_prompt is None
    assert ctx.conversation_messages == []


def test_build_fork_context_no_system():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    ctx = build_fork_context(messages)
    assert ctx.system_prompt is None
    assert len(ctx.conversation_messages) == 2


def test_filter_incomplete_tool_calls():
    """Should remove tool_calls without matching tool results."""
    messages = [
        {"role": "user", "content": "Read the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "tc2", "function": {"name": "grep_search", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "content": "file content", "tool_call_id": "tc1"},
        # tc2 has no tool result — incomplete
    ]
    filtered = filter_incomplete_tool_calls(messages)

    # The assistant message should only have tc1 (the completed one)
    assistant_msg = next(m for m in filtered if m.get("role") == "assistant")
    assert len(assistant_msg.get("tool_calls", [])) == 1
    assert assistant_msg["tool_calls"][0]["id"] == "tc1"


def test_filter_keeps_complete_pairs():
    """Complete tool call/result pairs should be preserved."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "tc1", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "content": "result", "tool_call_id": "tc1"},
    ]
    filtered = filter_incomplete_tool_calls(messages)
    assert len(filtered) == 2


def test_fork_context_dataclass():
    ctx = ForkContext(
        system_prompt="system",
        conversation_messages=[{"role": "user", "content": "hi"}],
    )
    assert ctx.system_prompt == "system"
    assert len(ctx.conversation_messages) == 1


def test_fork_context_to_messages():
    """Should reconstruct full message list for child agent."""
    ctx = ForkContext(
        system_prompt="You are helpful",
        conversation_messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    )
    msgs = ctx.to_messages()
    assert msgs[0] == {"role": "system", "content": "You are helpful"}
    assert len(msgs) == 3


def test_fork_context_to_messages_no_system():
    ctx = ForkContext(
        system_prompt=None,
        conversation_messages=[{"role": "user", "content": "hi"}],
    )
    msgs = ctx.to_messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


def test_fork_context_token_estimate():
    ctx = ForkContext(
        system_prompt="short",
        conversation_messages=[{"role": "user", "content": "x" * 1000}],
    )
    tokens = ctx.estimate_tokens()
    assert tokens > 0


def test_fork_filters_incomplete_responses_pairs():
    messages = [
        {"role": "user", "content": "Inspect the repository"},
        {"type": "function_call", "call_id": "done", "name": "read_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "done", "output": "file contents"},
        {"type": "function_call", "call_id": "pending", "name": "read_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "orphan", "output": "unmatched result"},
    ]

    assert build_fork_context(messages).to_messages() == messages[:3]
    assert len(messages) == 5


def test_fork_keeps_chat_and_responses_pairs_separate():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "shared", "function": {"name": "read_file", "arguments": "{}"}}],
        },
        {"type": "function_call_output", "call_id": "shared", "output": "wrong protocol"},
        {"type": "function_call", "call_id": "other", "name": "read_file", "arguments": "{}"},
        {"role": "tool", "tool_call_id": "other", "content": "wrong protocol"},
    ]

    assert filter_incomplete_tool_calls(messages) == []
