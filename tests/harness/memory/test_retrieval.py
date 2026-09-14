import sys
import types
from pathlib import Path

# Stub litellm before importing koder_agent to avoid optional dependency issues
if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from koder_agent.harness.memory.retrieval import retrieve_relevant_memories


def test_retrieve_relevant_memories_prefers_matching_fixture_notes():
    fixtures_dir = project_root / "tests/fixtures/memory/retrieval"
    result = retrieve_relevant_memories("dashboard latency", [fixtures_dir], max_tokens=200)

    assert result.memories
    assert result.memories[0].path.name == "project-note.md"


def test_retrieve_relevant_memories_respects_token_budget():
    fixtures_dir = project_root / "tests/fixtures/memory/retrieval"
    result = retrieve_relevant_memories("user", [fixtures_dir], max_tokens=20)
    assert result.token_count <= 20


def test_word_boundary_scoring_does_not_match_substrings(tmp_path):
    """'cat' must not match 'category' (word-boundary tokenization)."""
    note = tmp_path / "note.md"
    note.write_text(
        "---\ntype: note\ndescription: product taxonomy\n---\n"
        "This document lists every product category and its subcategory.\n",
        encoding="utf-8",
    )

    result = retrieve_relevant_memories("cat", [tmp_path], max_tokens=200)
    assert result.memories == []


def test_word_boundary_scoring_matches_whole_word(tmp_path):
    """'cat' matches a standalone 'cat' token even next to 'category' text."""
    note = tmp_path / "note.md"
    note.write_text(
        "---\ntype: note\ndescription: pets\n---\nThe cat sat in the category aisle.\n",
        encoding="utf-8",
    )

    result = retrieve_relevant_memories("cat", [tmp_path], max_tokens=200)
    assert len(result.memories) == 1
    assert result.memories[0].parsed.description == "pets"


def test_stopwords_and_short_tokens_are_ignored(tmp_path):
    """Stopwords and <3-char tokens do not create spurious matches."""
    note = tmp_path / "note.md"
    note.write_text(
        "---\ntype: note\ndescription: misc\n---\nThe project is done.\n",
        encoding="utf-8",
    )

    # "the", "is", "a" are stopwords/short -> no term should match -> no memories.
    result = retrieve_relevant_memories("the a is", [tmp_path], max_tokens=200)
    assert result.memories == []
