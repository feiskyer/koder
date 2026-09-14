import sqlite3
import sys
import types
from contextlib import closing
from pathlib import Path

# Stub litellm before importing koder_agent to avoid optional dependency issues
if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from koder_agent.harness.memory.legacy_db import LegacyDB
from koder_agent.harness.memory.transcript_store import TranscriptStore


def test_legacy_db_access_is_read_only(tmp_path):
    with closing(TranscriptStore.for_test(tmp_path)) as store:
        assert store.legacy_db().is_read_only is True


def test_legacy_db_reports_preserved_tables(tmp_path):
    legacy_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(legacy_path)
    conn.execute("CREATE TABLE ctx (sid TEXT, msgs TEXT, title TEXT)")
    conn.commit()
    conn.close()

    legacy = LegacyDB(legacy_path)
    assert legacy.exists is True
    assert "ctx" in legacy.list_tables()
