"""Preserved tool tails must satisfy the session's two-way replay contract."""

from copy import deepcopy

import pytest

from koder_agent.harness.memory.compact import (
    _trailing_replayable_tool_items,
    replayable_session_items,
)


def _call(call_id):
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": "read_file",
        "arguments": '{"path":"synthetic.py"}',
    }


def _output(call_id):
    return {"type": "function_call_output", "call_id": call_id, "output": "synthetic"}


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        ([_call("unfinished")], []),
        ([_output("orphan")], []),
        ([_output("reversed"), _call("reversed")], []),
        ([_call("complete"), _output("complete")], [_call("complete"), _output("complete")]),
        (
            [_call("complete"), _output("complete"), _call("unfinished")],
            [_call("complete"), _output("complete")],
        ),
        (
            [_call("complete"), _call("unfinished"), _output("complete")],
            [_call("complete"), _output("complete")],
        ),
        (
            [_call("unfinished"), _call("complete"), _output("complete")],
            [_call("complete"), _output("complete")],
        ),
        (
            [_call("complete"), _output("complete"), _output("orphan")],
            [_call("complete"), _output("complete")],
        ),
        (
            [_call("a"), _call("b"), _output("b"), _output("a")],
            [_call("a"), _call("b"), _output("b"), _output("a")],
        ),
        ([_call(None), _output(None)], []),
        ([_call(""), _output("")], []),
    ],
    ids=[
        "unfinished-call",
        "orphan-output",
        "output-before-call",
        "complete-pair",
        "unfinished-after",
        "unfinished-between",
        "unfinished-before",
        "orphan-after",
        "parallel-pairs",
        "null-id",
        "empty-id",
    ],
)
def test_preserved_tail_contains_only_complete_ordered_pairs(tail, expected):
    history = [{"role": "user", "content": "inspect"}, *tail]
    original = deepcopy(history)

    kept = _trailing_replayable_tool_items(history)

    assert kept == expected
    assert replayable_session_items(kept) == kept
    assert history == original


def test_tail_does_not_repair_a_pair_across_a_conversation_boundary():
    history = [
        _call("outside"),
        {"role": "assistant", "content": "text breaks the tool run"},
        _output("outside"),
        _call("complete"),
        _output("complete"),
    ]

    assert _trailing_replayable_tool_items(history) == history[-2:]
