"""The terminal fixture must dispatch the current model tool, not a past turn."""

import json

import pytest

from scripts.fake_openai_chat_server import _Handler


@pytest.mark.parametrize(
    ("marker", "command", "call_id"),
    [
        (
            "KODER_GIT_REFLOG",
            "git reflog expire --expire=all --all",
            "call_koder_git_reflog",
        ),
        ("KODER_GIT_FSCK", "git fsck --lost", "call_koder_git_fsck"),
    ],
)
def test_git_fixture_uses_latest_user_marker(marker, command, call_id):
    handler = object.__new__(_Handler)
    handler.scenario = "git_query_mutation"
    previous = "KODER_GIT_FSCK" if marker == "KODER_GIT_REFLOG" else "KODER_GIT_REFLOG"
    body = {
        "messages": [
            {"role": "user", "content": previous},
            {"role": "tool", "content": "previous denial"},
            {"role": "user", "content": marker},
        ]
    }
    call = handler._tool_call_payload(body)
    assert call["id"] == call_id
    assert call["function"]["name"] == "run_shell"
    assert json.loads(call["function"]["arguments"]) == {"command": command}


@pytest.mark.parametrize("name", ["reflog", "fsck"])
def test_git_denial_receipt_requires_the_matching_tool_result(name):
    message = {
        "role": "tool",
        "tool_call_id": f"call_koder_git_{name}",
        "content": "Permission denied for run_shell: dontAsk mode: approval auto-denied",
    }
    flags = _Handler._git_permission_denials({"messages": [message]})
    assert flags[f"git_{name}_denied"]
    assert sum(flags.values()) == 1

    for changed in (
        {**message, "role": "assistant"},
        {**message, "tool_call_id": "unrelated"},
        {**message, "content": "command succeeded"},
    ):
        assert not any(_Handler._git_permission_denials({"messages": [changed]}).values())
