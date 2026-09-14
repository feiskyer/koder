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

from koder_agent.harness.memory.recovery import (
    create_backup,
    default_backup_path,
    recover_partial_write,
)
from koder_agent.harness.memory.transcript_store import TranscriptStore


def test_recover_partial_write_restores_last_known_good_state(tmp_path):
    runtime_db = tmp_path / "runtime.db"
    backup_db = tmp_path / "runtime.db.bak"

    with closing(
        TranscriptStore(runtime_db_path=runtime_db, legacy_db_path=tmp_path / "legacy.db")
    ) as store:
        session_id = store.create_session("demo")
        store.append_user_message(session_id, "hello")

    backup_db.write_bytes(runtime_db.read_bytes())
    runtime_db.write_text("not a sqlite db", encoding="utf-8")

    result = recover_partial_write(runtime_db, backup_db)

    assert result.recovered is True
    with closing(
        TranscriptStore(runtime_db_path=runtime_db, legacy_db_path=tmp_path / "legacy.db")
    ) as reopened:
        assert reopened.read_messages(session_id)[0].content == "hello"


def test_recover_partial_write_is_noop_when_primary_db_is_healthy(tmp_path):
    runtime_db = tmp_path / "runtime.db"
    backup_db = tmp_path / "runtime.db.bak"

    with closing(
        TranscriptStore(runtime_db_path=runtime_db, legacy_db_path=tmp_path / "legacy.db")
    ) as store:
        store.create_session("demo")

    result = recover_partial_write(runtime_db, backup_db)

    assert result.recovered is False


def test_create_backup_then_recover_round_trip(tmp_path):
    """create_backup must produce a .bak that recover_partial_write can restore from."""
    runtime_db = tmp_path / "runtime.db"

    with closing(
        TranscriptStore(runtime_db_path=runtime_db, legacy_db_path=tmp_path / "legacy.db")
    ) as store:
        session_id = store.create_session("demo")
        store.append_user_message(session_id, "hello")

    # No explicit path -> uses the same <db>.bak convention recovery expects.
    backup_result = create_backup(runtime_db)
    assert backup_result.created is True
    assert default_backup_path(runtime_db).exists()

    # Corrupt the primary, then recover using the backup create_backup made.
    runtime_db.write_text("not a sqlite db", encoding="utf-8")
    result = recover_partial_write(runtime_db)

    assert result.recovered is True
    with closing(
        TranscriptStore(runtime_db_path=runtime_db, legacy_db_path=tmp_path / "legacy.db")
    ) as reopened:
        assert reopened.read_messages(session_id)[0].content == "hello"


def test_create_backup_refuses_corrupt_source(tmp_path):
    """A corrupt runtime DB must never clobber an existing good backup."""
    runtime_db = tmp_path / "runtime.db"

    with closing(
        TranscriptStore(runtime_db_path=runtime_db, legacy_db_path=tmp_path / "legacy.db")
    ) as store:
        session_id = store.create_session("demo")
        store.append_user_message(session_id, "good state")

    # Establish a known-good backup.
    assert create_backup(runtime_db).created is True

    # Now corrupt the primary and attempt another backup -- it must refuse so the
    # existing good backup is preserved.
    runtime_db.write_text("garbage", encoding="utf-8")
    result = create_backup(runtime_db)
    assert result.created is False

    # The good backup still restores the original content.
    recovery = recover_partial_write(runtime_db)
    assert recovery.recovered is True
    with closing(
        TranscriptStore(runtime_db_path=runtime_db, legacy_db_path=tmp_path / "legacy.db")
    ) as reopened:
        assert reopened.read_messages(session_id)[0].content == "good state"
