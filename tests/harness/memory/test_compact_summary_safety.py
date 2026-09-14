"""A missing summary must never authorize dropping the summarized source."""

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from koder_agent.harness.memory import compact


@pytest.mark.parametrize("response", ["", " \n\t ", "<summary> \n </summary>"])
@pytest.mark.asyncio
async def test_empty_summary_is_rejected_without_mutating_source(monkeypatch, response):
    history = [{"role": "user", "content": f"original-{index}"} for index in range(8)]
    before = deepcopy(history)
    monkeypatch.setattr(compact, "llm_completion", AsyncMock(return_value=response))
    with pytest.raises(ValueError, match="summary"):
        await compact.llm_compact_messages(history, keep_recent=2)
    assert history == before


@pytest.mark.asyncio
async def test_minimal_history_is_a_noop_without_calling_summarizer(monkeypatch):
    history = [
        {"role": "user", "content": "latest intent"},
        {"role": "assistant", "content": "latest answer"},
    ]
    completion = AsyncMock(side_effect=AssertionError("A no-op must not call a model"))
    monkeypatch.setattr(compact, "llm_completion", completion)
    result = await compact.llm_compact_messages(history, keep_recent=2)
    assert result.summary is None
    assert compact.build_compacted_session_items(result) == history
    completion.assert_not_awaited()


@pytest.mark.parametrize(
    "response",
    [
        "<analysis>Thinking only; no final summary.</analysis>",
        "<analysis>Unfinished analysis",
        "<summary>Unfinished summary",
        "Summary with unmatched closing tag</summary>",
        "<summary />",
        "<summary>outer <summary>inner</summary></summary>",
        "<summary>first</summary><summary>second</summary>",
        "<analysis>An example <summary>not a final summary</summary>.</analysis>",
        "<summary><analysis>Only analysis.</analysis></summary>",
    ],
    ids=[
        "analysis_only",
        "incomplete_analysis",
        "incomplete_summary",
        "unmatched_close",
        "self_closing",
        "nested_summary",
        "multiple_summaries",
        "summary_inside_analysis",
        "empty_after_analysis_removal",
    ],
)
@pytest.mark.asyncio
async def test_malformed_summary_is_rejected_without_mutating_source(monkeypatch, response):
    history = [{"role": "user", "content": f"original-{index}"} for index in range(8)]
    before = deepcopy(history)
    monkeypatch.setattr(compact, "llm_completion", AsyncMock(return_value=response))

    with pytest.raises(ValueError, match="summary"):
        await compact.llm_compact_messages(history, keep_recent=2)

    assert history == before


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Plain final summary.", "Plain final summary."),
        (" \n<summary>Final summary.</summary>\n ", "Final summary."),
        (
            "<analysis>Internal preamble.</analysis>\n<summary>Final summary.</summary>",
            "Final summary.",
        ),
        (
            "<ANALYSIS>Internal preamble.</ANALYSIS>\n<SUMMARY>Final summary.</SUMMARY>",
            "Final summary.",
        ),
        (
            "<analysis>An example <summary>not the final answer</summary>.</analysis>"
            "<summary>Actual final summary.</summary>",
            "Actual final summary.",
        ),
        ("<analysis>Internal preamble.</analysis>\nPlain final summary.", "Plain final summary."),
        ("<summary>Keep a literal <path> token.</summary>", "Keep a literal <path> token."),
    ],
    ids=[
        "plain",
        "tagged",
        "analysis_and_summary",
        "uppercase_tags",
        "ignore_analysis_example",
        "analysis_and_plain",
        "non_protocol_markup",
    ],
)
@pytest.mark.asyncio
async def test_valid_summary_uses_only_the_final_content(monkeypatch, response, expected):
    history = [{"role": "user", "content": f"original-{index}"} for index in range(8)]
    monkeypatch.setattr(compact, "llm_completion", AsyncMock(return_value=response))

    result = await compact.llm_compact_messages(history, keep_recent=2)

    assert result.summary == expected
    assert result.kept_messages == history[-2:]
