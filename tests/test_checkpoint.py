"""Tests for file checkpointing + /rewind code restoration.

Covers:
- CheckpointStore snapshot/restore semantics (happy path + tombstones + bound).
- File tools (write_file/edit_file/append_file) recording pre-edit snapshots.
- End-to-end: edit a file across two turns, then a "code" rewind to turn 1
  restores the original content; "conversation" mode leaves files untouched;
  "both" mode does both.
"""

import json
import sys
import types
from pathlib import Path

import pytest

# Stub litellm before importing koder_agent to avoid optional dependency issues.
if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from koder_agent.harness import checkpoint as cp  # noqa: E402
from koder_agent.harness.checkpoint import CheckpointStore  # noqa: E402
from koder_agent.tools.file import (  # noqa: E402
    _file_state,
    append_file,
    edit_file,
    read_file,
    write_file,
)


@pytest.fixture
def checkpoint_root(tmp_path, monkeypatch):
    """Point the checkpoint store at an isolated temp dir and reset state."""
    root = tmp_path / "checkpoints"
    monkeypatch.setenv("KODER_CHECKPOINT_DIR", str(root))
    monkeypatch.delenv("KODER_FILE_CHECKPOINTS", raising=False)
    cp.reset_state_for_tests()
    _file_state.clear()
    yield root
    cp.reset_state_for_tests()
    _file_state.clear()


# ---------------------------------------------------------------------------
# CheckpointStore unit behaviour
# ---------------------------------------------------------------------------
class TestCheckpointStore:
    def test_record_and_restore_content(self, checkpoint_root, tmp_path):
        target = tmp_path / "f.txt"
        target.write_text("original\n", encoding="utf-8")

        store = CheckpointStore("sess1", root=checkpoint_root)
        # Turn 1 == checkpoint 1: record pre-edit (original) then change file.
        store.record(str(target), checkpoint=1)
        target.write_text("changed\n", encoding="utf-8")

        # Restore to before turn 1 (checkpoint 0) -> original content.
        restored = store.restore_to(0)
        assert restored == [str(target.resolve())]
        assert target.read_text(encoding="utf-8") == "original\n"

    def test_restore_keeps_earliest_snapshot(self, checkpoint_root, tmp_path):
        target = tmp_path / "f.txt"
        target.write_text("v0\n", encoding="utf-8")
        store = CheckpointStore("sess", root=checkpoint_root)

        # Two edits within the same turn: only the first (true pre-edit) counts.
        store.record(str(target), checkpoint=1)
        target.write_text("v1\n", encoding="utf-8")
        store.record(str(target), checkpoint=1)
        target.write_text("v2\n", encoding="utf-8")

        store.restore_to(0)
        assert target.read_text(encoding="utf-8") == "v0\n"

    def test_restore_only_after_target_checkpoint(self, checkpoint_root, tmp_path):
        target = tmp_path / "f.txt"
        target.write_text("t1\n", encoding="utf-8")
        store = CheckpointStore("sess", root=checkpoint_root)

        # Turn 1 edit.
        store.record(str(target), checkpoint=1)
        target.write_text("t2\n", encoding="utf-8")
        # Turn 2 edit.
        store.record(str(target), checkpoint=2)
        target.write_text("t3\n", encoding="utf-8")

        # Restore to before turn 2 (checkpoint 1): revert only turn-2 edit ->
        # the content as it was at the start of turn 2, which is "t2".
        restored = store.restore_to(1)
        assert restored == [str(target.resolve())]
        assert target.read_text(encoding="utf-8") == "t2\n"

    def test_tombstone_deletes_created_file(self, checkpoint_root, tmp_path):
        target = tmp_path / "new.txt"
        assert not target.exists()
        store = CheckpointStore("sess", root=checkpoint_root)

        # File absent at snapshot -> tombstone; then it gets created.
        store.record(str(target), checkpoint=1)
        target.write_text("created\n", encoding="utf-8")

        store.restore_to(0)
        assert not target.exists()

    def test_restore_returns_empty_when_nothing_after(self, checkpoint_root, tmp_path):
        target = tmp_path / "f.txt"
        target.write_text("x\n", encoding="utf-8")
        store = CheckpointStore("sess", root=checkpoint_root)
        store.record(str(target), checkpoint=1)
        # Nothing recorded after checkpoint 5.
        assert store.restore_to(5) == []

    def test_snapshot_store_is_bounded(self, checkpoint_root, tmp_path, monkeypatch):
        monkeypatch.setattr(cp, "MAX_SNAPSHOTS", 3)
        store = CheckpointStore("sess", root=checkpoint_root)
        for i in range(6):
            f = tmp_path / f"f{i}.txt"
            f.write_text(f"c{i}\n", encoding="utf-8")
            store.record(str(f), checkpoint=i + 1)

        records = store._load()
        assert len(records) == 3
        # Oldest three were pruned; only checkpoints 4,5,6 remain.
        assert sorted(r.checkpoint for r in records) == [4, 5, 6]

    def test_tracked_paths_after(self, checkpoint_root, tmp_path):
        store = CheckpointStore("sess", root=checkpoint_root)
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("a\n", encoding="utf-8")
        b.write_text("b\n", encoding="utf-8")
        store.record(str(a), checkpoint=1)
        store.record(str(b), checkpoint=2)

        after0 = store.tracked_paths_after(0)
        assert str(a.resolve()) in after0 and str(b.resolve()) in after0
        after1 = store.tracked_paths_after(1)
        assert after1 == [str(b.resolve())]

    def test_clear_removes_store(self, checkpoint_root, tmp_path):
        target = tmp_path / "f.txt"
        target.write_text("x\n", encoding="utf-8")
        store = CheckpointStore("sess", root=checkpoint_root)
        store.record(str(target), checkpoint=1)
        assert store.dir.exists()
        store.clear()
        assert not store.dir.exists()


# ---------------------------------------------------------------------------
# Enable/disable gating
# ---------------------------------------------------------------------------
class TestGating:
    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("KODER_FILE_CHECKPOINTS", raising=False)
        assert cp.checkpoints_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "OFF", "Disabled"])
    def test_disabled_by_flag(self, monkeypatch, val):
        monkeypatch.setenv("KODER_FILE_CHECKPOINTS", val)
        assert cp.checkpoints_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
    def test_enabled_by_flag(self, monkeypatch, val):
        monkeypatch.setenv("KODER_FILE_CHECKPOINTS", val)
        assert cp.checkpoints_enabled() is True

    def test_record_pre_edit_noop_when_disabled(self, checkpoint_root, tmp_path, monkeypatch):
        monkeypatch.setenv("KODER_FILE_CHECKPOINTS", "0")
        cp.set_active_session("s")
        target = tmp_path / "f.txt"
        target.write_text("x\n", encoding="utf-8")
        cp.record_pre_edit(str(target))
        # No store dir created.
        assert not (checkpoint_root / "s").exists()

    def test_record_pre_edit_noop_without_active_session(self, checkpoint_root, tmp_path):
        cp.set_active_session(None)
        target = tmp_path / "f.txt"
        target.write_text("x\n", encoding="utf-8")
        cp.record_pre_edit(str(target))  # should not raise, no store
        assert not any(checkpoint_root.glob("*")) if checkpoint_root.exists() else True


# ---------------------------------------------------------------------------
# Module-level counter / active session
# ---------------------------------------------------------------------------
class TestModuleState:
    def test_begin_turn_advances_counter(self, checkpoint_root):
        cp.set_active_session("s")
        assert cp.current_checkpoint() == 0
        assert cp.begin_turn() == 1
        assert cp.begin_turn() == 2
        assert cp.current_checkpoint() == 2

    def test_switching_session_resets_counter(self, checkpoint_root):
        cp.set_active_session("a")
        cp.begin_turn()
        cp.begin_turn()
        assert cp.current_checkpoint() == 2
        cp.set_active_session("b")
        assert cp.current_checkpoint() == 0


# ---------------------------------------------------------------------------
# File tools record pre-edit snapshots
# ---------------------------------------------------------------------------
async def _write(path, content):
    return await write_file.on_invoke_tool(
        None, json.dumps({"path": str(path), "content": content})
    )


async def _read(path):
    return await read_file.on_invoke_tool(None, json.dumps({"path": str(path)}))


async def _edit_replace(path, old, new):
    return await edit_file.on_invoke_tool(
        None, json.dumps({"path": str(path), "old_string": old, "new_string": new})
    )


async def _append(path, content):
    return await append_file.on_invoke_tool(
        None, json.dumps({"path": str(path), "content": content})
    )


class TestFileToolHooks:
    @pytest.mark.asyncio
    async def test_write_file_records_pre_edit(self, checkpoint_root, tmp_path):
        cp.set_active_session("sess")
        cp.begin_turn()  # checkpoint 1
        target = tmp_path / "f.txt"
        target.write_text("original\n", encoding="utf-8")
        await _read(target)

        await _write(target, "overwritten\n")
        assert target.read_text(encoding="utf-8") == "overwritten\n"

        restored = cp.restore_to("sess", 0)
        assert restored == [str(target.resolve())]
        assert target.read_text(encoding="utf-8") == "original\n"

    @pytest.mark.asyncio
    async def test_write_new_file_tombstone_restore_deletes(self, checkpoint_root, tmp_path):
        cp.set_active_session("sess")
        cp.begin_turn()
        target = tmp_path / "brand_new.txt"
        await _write(target, "hello\n")
        assert target.exists()

        cp.restore_to("sess", 0)
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_edit_file_records_pre_edit(self, checkpoint_root, tmp_path):
        cp.set_active_session("sess")
        cp.begin_turn()
        target = tmp_path / "f.txt"
        target.write_text("foo bar baz\n", encoding="utf-8")
        await _read(target)

        await _edit_replace(target, "bar", "QUX")
        assert "QUX" in target.read_text(encoding="utf-8")

        cp.restore_to("sess", 0)
        assert target.read_text(encoding="utf-8") == "foo bar baz\n"

    @pytest.mark.asyncio
    async def test_append_file_records_pre_edit(self, checkpoint_root, tmp_path):
        cp.set_active_session("sess")
        cp.begin_turn()
        target = tmp_path / "log.txt"
        target.write_text("line1\n", encoding="utf-8")

        # Must read before appending to an existing file (read-before-write guard).
        await _read(target)

        await _append(target, "line2\n")
        assert target.read_text(encoding="utf-8") == "line1\nline2\n"

        cp.restore_to("sess", 0)
        assert target.read_text(encoding="utf-8") == "line1\n"


# ---------------------------------------------------------------------------
# End-to-end: two turns then rewind to turn 1
# ---------------------------------------------------------------------------
class TestTwoTurnRewind:
    @pytest.mark.asyncio
    async def test_code_rewind_to_turn_one_restores_original(self, checkpoint_root, tmp_path):
        cp.set_active_session("sess")
        target = tmp_path / "code.py"

        # Turn 1: create the file with initial content.
        cp.begin_turn()  # checkpoint 1
        await _write(target, "print('v1')\n")
        assert target.read_text(encoding="utf-8") == "print('v1')\n"

        # Turn 2: edit the file.
        cp.begin_turn()  # checkpoint 2
        await _read(target)
        await _edit_replace(target, "v1", "v2")
        assert target.read_text(encoding="utf-8") == "print('v2')\n"

        # Rewind code to turn 1 (undo everything from turn 1 onward): file gone.
        restored = cp.restore_to("sess", 0)
        assert str(target.resolve()) in restored
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_code_rewind_to_turn_two_restores_turn_one_content(
        self, checkpoint_root, tmp_path
    ):
        cp.set_active_session("sess")
        target = tmp_path / "code.py"
        target.write_text("print('v0')\n", encoding="utf-8")

        # Turn 1 edit.
        cp.begin_turn()  # checkpoint 1
        await _read(target)
        await _edit_replace(target, "v0", "v1")

        # Turn 2 edit.
        cp.begin_turn()  # checkpoint 2
        await _read(target)
        await _edit_replace(target, "v1", "v2")
        assert target.read_text(encoding="utf-8") == "print('v2')\n"

        # Undo only turn 2 -> content at start of turn 2 == "v1".
        cp.restore_to("sess", 1)
        assert target.read_text(encoding="utf-8") == "print('v1')\n"


# ---------------------------------------------------------------------------
# /rewind command integration (conversation | code | both)
# ---------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from koder_agent.core.session import EnhancedSQLiteSession  # noqa: E402
from koder_agent.harness.commands.interactive import (  # noqa: E402
    HarnessInteractiveCommandHandler,
)


def _user(text):
    return {"role": "user", "content": text}


def _assistant(text):
    return {"role": "assistant", "content": text}


async def _seed_two_turn_session(session_id, request):
    """Create a session with two user turns and align file checkpoints.

    Returns the session with conversation:
      turn 1 -> user "prompt one" + assistant reply
      turn 2 -> user "prompt two" + assistant reply
    """
    session = EnhancedSQLiteSession(session_id, db_path=":memory:")
    request.addfinalizer(session.close)
    cp.set_active_session(session_id)

    cp.begin_turn()  # checkpoint 1 (turn 1)
    await session.add_items([_user("prompt one"), _assistant("reply one")])

    cp.begin_turn()  # checkpoint 2 (turn 2)
    await session.add_items([_user("prompt two"), _assistant("reply two")])
    return session


class TestRewindCommand:
    @pytest.mark.asyncio
    async def test_conversation_mode_trims_history_without_touching_files(
        self, checkpoint_root, tmp_path, request
    ):
        handler = HarnessInteractiveCommandHandler(emit_console=False)
        session = await _seed_two_turn_session("rw-conv", request)
        scheduler = SimpleNamespace(session=session)

        # A tracked file edited in turn 2.
        target = tmp_path / "f.txt"
        target.write_text("turn2\n", encoding="utf-8")
        cp._get_store("rw-conv").record(str(target), checkpoint=2)

        # Rewind to prompt 2 (newest) in conversation mode (default).
        out = await handler._execute_rewind(scheduler, ["2"])
        assert "Rewound conversation to prompt 2" in out
        assert handler.consume_pending_input_text() == "prompt one"

        # File is untouched by conversation-only rewind.
        assert target.read_text(encoding="utf-8") == "turn2\n"
        # Conversation trimmed to before the oldest user prompt.
        items = await session.get_items()
        assert items == []

    @pytest.mark.asyncio
    async def test_code_mode_restores_files_without_trimming_history(
        self, checkpoint_root, tmp_path, request
    ):
        handler = HarnessInteractiveCommandHandler(emit_console=False)
        session = await _seed_two_turn_session("rw-code", request)
        scheduler = SimpleNamespace(session=session)

        target = tmp_path / "code.py"
        target.write_text("v1\n", encoding="utf-8")
        # Turn 2 pre-edit snapshot recorded (content "v1"), then edited.
        cp._get_store("rw-code").record(str(target), checkpoint=2)
        target.write_text("v2\n", encoding="utf-8")

        items_before = await session.get_items()

        # Rewind to prompt 1 (newest = "prompt two") in code mode.
        # total=2, selection=1 -> target_checkpoint = 2 - 1 = 1, so files
        # snapshotted after checkpoint 1 (i.e. turn 2's edit) are reverted.
        out = await handler._execute_rewind(scheduler, ["code", "1"])
        assert "Restored" in out and str(target.resolve()) in out
        # Undo turn-2 edit -> back to "v1".
        assert target.read_text(encoding="utf-8") == "v1\n"

        # Conversation history unchanged; no pending input.
        assert await session.get_items() == items_before
        assert handler.consume_pending_input_text() is None

    @pytest.mark.asyncio
    async def test_both_mode_trims_history_and_restores_files(
        self, checkpoint_root, tmp_path, request
    ):
        handler = HarnessInteractiveCommandHandler(emit_console=False)
        session = await _seed_two_turn_session("rw-both", request)
        scheduler = SimpleNamespace(session=session)

        target = tmp_path / "code.py"
        target.write_text("v1\n", encoding="utf-8")
        cp._get_store("rw-both").record(str(target), checkpoint=2)
        target.write_text("v2\n", encoding="utf-8")

        # Rewind to prompt 1 (newest) in both mode.
        out = await handler._execute_rewind(scheduler, ["1", "both"])
        assert "Rewound conversation to prompt 1" in out
        assert "Restored" in out
        # Newest prompt is "prompt two"; conversation trimmed to before it.
        items = await session.get_items()
        assert items == [_user("prompt one"), _assistant("reply one")]
        assert handler.consume_pending_input_text() == "prompt two"
        # Code restored to before turn 2.
        assert target.read_text(encoding="utf-8") == "v1\n"

    @pytest.mark.asyncio
    async def test_listing_targets_shows_modes(self, checkpoint_root, request):
        handler = HarnessInteractiveCommandHandler(emit_console=False)
        session = await _seed_two_turn_session("rw-list", request)
        scheduler = SimpleNamespace(session=session)

        out = await handler._execute_rewind(scheduler, [])
        assert "Rewind targets" in out
        assert "conversation" in out and "code" in out and "both" in out
        assert "1. prompt two" in out
        assert "2. prompt one" in out

    @pytest.mark.asyncio
    async def test_help_lists_modes(self, checkpoint_root):
        handler = HarnessInteractiveCommandHandler(emit_console=False)
        out = await handler._execute_rewind(SimpleNamespace(session=None), ["help"])
        assert "Usage: /rewind" in out
        assert "code" in out and "both" in out

    @pytest.mark.asyncio
    async def test_invalid_number_rejected(self, checkpoint_root, request):
        handler = HarnessInteractiveCommandHandler(emit_console=False)
        session = await _seed_two_turn_session("rw-bad", request)
        scheduler = SimpleNamespace(session=session)
        out = await handler._execute_rewind(scheduler, ["notanumber"])
        assert out == handler._REWIND_USAGE

    @pytest.mark.asyncio
    async def test_out_of_range_number_rejected(self, checkpoint_root, request):
        handler = HarnessInteractiveCommandHandler(emit_console=False)
        session = await _seed_two_turn_session("rw-range", request)
        scheduler = SimpleNamespace(session=session)
        out = await handler._execute_rewind(scheduler, ["99", "code"])
        assert "between 1 and 2" in out

    @pytest.mark.asyncio
    async def test_code_mode_no_changes_message(self, checkpoint_root, request):
        handler = HarnessInteractiveCommandHandler(emit_console=False)
        session = await _seed_two_turn_session("rw-nochange", request)
        scheduler = SimpleNamespace(session=session)
        # No file snapshots recorded -> code restore has nothing to do.
        out = await handler._execute_rewind(scheduler, ["1", "code"])
        assert "No tracked file changes" in out

    @pytest.mark.asyncio
    async def test_code_mode_disabled_message(self, checkpoint_root, monkeypatch, request):
        monkeypatch.setenv("KODER_FILE_CHECKPOINTS", "0")
        handler = HarnessInteractiveCommandHandler(emit_console=False)
        session = await _seed_two_turn_session("rw-disabled", request)
        scheduler = SimpleNamespace(session=session)
        out = await handler._execute_rewind(scheduler, ["1", "code"])
        assert "disabled" in out.lower()
