"""Compaction retains actual input shapes and never rewrites a valid no-op."""

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from koder_agent.harness.memory import compact
from koder_agent.utils.image_input import build_multimodal_input


@pytest.fixture(params=["responses", "chat"])
def image_turn(tmp_path, request):
    path = tmp_path / "synthetic.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-image")
    message = build_multimodal_input("Exact image-related user intent", [str(path)])[0]
    if request.param == "chat":
        message = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": message["content"][0]["image_url"]}},
                {"type": "text", "text": message["content"][1]["text"]},
            ],
        }
    return message


def _tool_pair():
    return [
        {"type": "function_call", "call_id": "complete", "name": "read_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "complete", "output": "synthetic"},
    ]


@pytest.mark.asyncio
async def test_recent_multimodal_turn_stays_verbatim(monkeypatch, image_turn):
    history = [
        *[{"role": "user", "content": f"old text {index}"} for index in range(8)],
        image_turn,
    ]
    original = deepcopy(history)
    completion = AsyncMock(return_value="<summary>Older messages.</summary>")
    monkeypatch.setattr(compact, "llm_completion", completion)

    result = await compact.llm_compact_messages(history, keep_recent=1)

    assert result.kept_messages == [image_turn]
    assert "Exact image-related user intent" not in completion.await_args.args[0][0]["content"]
    assert history == original


@pytest.mark.asyncio
async def test_old_multimodal_text_reaches_summary_without_image_payload(monkeypatch, image_turn):
    history = [image_turn, {"role": "user", "content": "latest text"}]
    original = deepcopy(history)
    completion = AsyncMock(return_value="<summary>Older image discussion.</summary>")
    monkeypatch.setattr(compact, "llm_completion", completion)

    result = await compact.llm_compact_messages(history, keep_recent=1)

    source = completion.await_args.args[0][0]["content"]
    assert "Exact image-related user intent" in source
    assert "[image]" in source
    assert "data:image" not in source
    assert result.kept_messages == history[-1:]
    assert history == original


@pytest.mark.parametrize("include_text", [False, True])
@pytest.mark.asyncio
async def test_minimal_multimodal_turn_is_a_verbatim_noop(monkeypatch, image_turn, include_text):
    message = deepcopy(image_turn)
    if not include_text:
        message["content"] = message["content"][:1]
    history = [message]
    completion = AsyncMock(side_effect=AssertionError("No summary is needed"))
    monkeypatch.setattr(compact, "llm_completion", completion)

    result = await compact.llm_compact_messages(history, keep_recent=1)

    assert result.summary is None
    assert compact.build_compacted_session_items(result) == history
    completion.assert_not_awaited()


@pytest.mark.parametrize(
    "instruction_roles",
    [[], ["system"], ["developer"], ["system", "developer"]],
    ids=["no_instructions", "system", "developer", "both"],
)
@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.asyncio
async def test_compacted_multimodal_context_is_a_verbatim_noop(
    monkeypatch, image_turn, instruction_roles, with_tools
):
    history = [
        *[{"role": role, "content": f"{role} invariant"} for role in instruction_roles],
        {"role": "user", "content": "[Conversation compacted]\n\nOlder messages."},
        image_turn,
        *(_tool_pair() if with_tools else []),
    ]
    original = deepcopy(history)
    completion = AsyncMock(side_effect=AssertionError("No summary is needed"))
    monkeypatch.setattr(compact, "llm_completion", completion)

    result = await compact.llm_compact_messages(history, keep_recent=1)

    assert result.summary is None
    assert compact.build_compacted_session_items(result) == original
    assert history == original
    completion.assert_not_awaited()


@pytest.mark.parametrize("instruction_role", ["system", "developer"])
@pytest.mark.asyncio
async def test_compacted_typed_message_keeps_its_original_fields(monkeypatch, instruction_role):
    history = [
        {"role": instruction_role, "content": []},
        {"role": "user", "content": "[Conversation compacted]\n\nOlder messages."},
        {
            "type": "message",
            "role": "assistant",
            "id": "synthetic-message",
            "status": "completed",
            "content": [{"type": "output_text", "text": "exact answer", "annotations": []}],
        },
    ]
    original = deepcopy(history)
    completion = AsyncMock(side_effect=AssertionError("No summary is needed"))
    monkeypatch.setattr(compact, "llm_completion", completion)

    result = await compact.llm_compact_messages(history, keep_recent=1)

    assert result.summary is None
    assert compact.build_compacted_session_items(result) == original
    assert history == original
    completion.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_conversation_beyond_retention_budget_is_not_a_false_noop(
    monkeypatch, image_turn
):
    history = [
        {"role": "system", "content": "invariant"},
        {"role": "user", "content": "[Conversation compacted]\n\nOlder messages."},
        image_turn,
        {"role": "user", "content": "genuinely new work"},
    ]
    completion = AsyncMock(return_value="<summary>Older image discussion and summary.</summary>")
    monkeypatch.setattr(compact, "llm_completion", completion)

    result = await compact.llm_compact_messages(history, keep_recent=1)

    assert result.summary == "Older image discussion and summary."
    assert result.kept_messages == [history[0], history[-1]]
    assert "Exact image-related user intent" in completion.await_args.args[0][0]["content"]
    completion.assert_awaited_once()
