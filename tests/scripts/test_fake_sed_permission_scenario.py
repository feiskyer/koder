"""Sed terminal checks require correlated model tool results for both paths."""

import json

import pytest

from scripts.fake_openai_chat_server import _Handler


@pytest.mark.parametrize("read", [False, True])
def test_sed_fixture_uses_the_current_user_marker(read):
    handler = object.__new__(_Handler)
    handler.scenario = "sed_query_mutation"
    marker = "KODER_SED_READ" if read else "KODER_SED_WRITE"
    previous = "KODER_SED_WRITE" if read else "KODER_SED_READ"
    call = handler._tool_call_payload(
        {
            "messages": [
                {"role": "user", "content": previous},
                {"role": "tool", "content": "previous result"},
                {"role": "user", "content": marker},
            ]
        }
    )
    assert call["function"]["name"] == "run_shell"
    assert call["id"] == f"call_koder_sed_{'read' if read else 'write'}"
    assert json.loads(call["function"]["arguments"]) == {
        "command": (
            "sed -n '1p' sample.txt" if read else "sed -n 'w sed-write-proof.txt' sample.txt"
        )
    }


@pytest.mark.parametrize(
    ("call_id", "content", "flag"),
    [
        (
            "call_koder_sed_write",
            "Permission denied for run_shell: dontAsk mode: approval auto-denied",
            "sed_write_denied",
        ),
        ("call_koder_sed_read", "initial\n", "sed_read_succeeded"),
    ],
)
def test_sed_receipts_require_matching_tool_results(call_id, content, flag):
    message = {"role": "tool", "tool_call_id": call_id, "content": content}
    flags = _Handler._sed_permission_outcomes({"messages": [message]})
    assert flags[flag]
    assert sum(flags.values()) == 1
    for changed in (
        {**message, "role": "assistant"},
        {**message, "tool_call_id": "unrelated"},
        {**message, "content": "different output"},
    ):
        assert not any(_Handler._sed_permission_outcomes({"messages": [changed]}).values())
