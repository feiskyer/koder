import json
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

from koder_agent.harness.memory.compact import compact_messages


def test_compact_matches_golden_fixture():
    fixtures_dir = project_root / "tests/fixtures/memory"
    before = json.loads((fixtures_dir / "compact_before.json").read_text(encoding="utf-8"))
    expected = json.loads((fixtures_dir / "compact_after.json").read_text(encoding="utf-8"))

    result = compact_messages(before["messages"], max_messages=expected["config"]["max_messages"])
    actual = {
        "summary": result.summary,
        "kept_messages": result.kept_messages,
        "original_count": result.original_count,
    }
    assert actual == expected["result"]
