import sys
import types
from contextlib import closing
from pathlib import Path

import pytest

# Stub litellm before importing koder_agent to avoid optional dependency issues
if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from koder_agent.harness.memory.transcript_store import TranscriptStore


def test_failed_schema_initialization_closes_its_connection(tmp_path, monkeypatch):
    import sqlite3

    created = []
    connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        created.append(connection)
        return connection

    def fail_schema(_self):
        raise RuntimeError("synthetic schema initialization failure")

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    monkeypatch.setattr(TranscriptStore, "_init_schema", fail_schema)
    try:
        with pytest.raises(RuntimeError, match="schema initialization failure"):
            TranscriptStore.for_test(tmp_path)
        assert len(created) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            created[0].execute("SELECT 1")
    finally:
        for connection in created:
            connection.close()


def test_transcript_store_writes_runtime_session_without_touching_legacy_db(tmp_path):
    with closing(TranscriptStore.for_test(tmp_path)) as store:
        session_id = store.create_session("demo")
        store.append_user_message(session_id, "hello")

        messages = store.read_messages(session_id)
        assert messages[0].content == "hello"
        assert [message.role for message in messages] == ["user"]
        assert store.runtime_db_path.exists()


def test_transcript_store_persists_across_restart(tmp_path):
    with closing(TranscriptStore.for_test(tmp_path)) as store:
        session_id = store.create_session("demo")
        store.append_user_message(session_id, "hello")
        store.append_assistant_message(session_id, "world")

    with closing(TranscriptStore.for_test(tmp_path)) as reopened:
        messages = reopened.read_messages(session_id)

        assert [message.content for message in messages] == ["hello", "world"]


def test_transcript_store_rolls_back_failed_write(tmp_path):
    class Unserializable:
        pass

    with closing(TranscriptStore.for_test(tmp_path)) as store:
        session_id = store.create_session("demo")
        store.append_user_message(session_id, "first")
        with pytest.raises(TypeError):
            store.append_message(
                session_id, "assistant", "second", metadata={"bad": Unserializable()}
            )

        messages = store.read_messages(session_id)
        assert [message.content for message in messages] == ["first"]
