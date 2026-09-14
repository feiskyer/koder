"""Tests for scheduler runtime wiring: tool call counting, auto-compact, session memory."""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from koder_agent.harness.memory.auto_compact import AutoCompactManager
from koder_agent.harness.memory.budget import estimate_messages_tokens
from koder_agent.harness.memory.compact import CompactionResult
from koder_agent.harness.memory.extraction import ExtractionResult
from koder_agent.harness.memory.session_memory import SessionMemoryManager


def test_streaming_output_does_not_clear_terminal_scrollback():
    """Streaming turns must leave prior terminal history scrollable."""
    from koder_agent.core.scheduler import AgentScheduler

    source = inspect.getsource(AgentScheduler._handle_streaming)

    assert "ClearScrollback" not in source
    assert "\\033[3J" not in source
    assert "ESC[3J" not in source


def test_reasoning_stream_payload_respects_display_mode():
    from koder_agent.core.scheduler import _reasoning_stream_payload

    class Event:
        type = "response.reasoning_summary_text.delta"
        delta = "checking"
        item_id = "rs_1"
        output_index = 0
        summary_index = 0

    assert _reasoning_stream_payload(Event(), "off") is None

    payload = _reasoning_stream_payload(Event(), "summary")

    assert payload == {
        "kind": "summary",
        "text": "checking",
        "done": False,
        "item_id": "rs_1",
        "output_index": 0,
        "part_index": 0,
    }


def test_format_execution_error_hides_copilot_raw_details():
    from koder_agent.core.scheduler import _format_execution_error

    message = _format_execution_error(
        Exception(
            "litellm.AuthenticationError: Failed to refresh API key: "
            "Failed to refresh API key after maximum retries\n\n"
            "original model: github_copilot/claude-sonnet-4.6"
        )
    )

    assert "koder auth login github_copilot" in message
    assert "Details:" not in message
    assert "original model" not in message


@pytest.mark.asyncio
async def test_scheduler_serializes_concurrent_handle_calls():
    from koder_agent.core.scheduler import AgentScheduler

    active = 0
    max_active = 0

    async def fake_run(*_args, **_kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1

        class Result:
            final_output = "ok"
            context_wrapper = None

        return Result()

    with (
        patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
        patch("koder_agent.core.scheduler.get_display_hooks"),
        patch("koder_agent.core.scheduler.ApprovalHooks"),
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
        patch("koder_agent.core.scheduler.Runner.run", side_effect=fake_run),
        patch("koder_agent.core.scheduler.get_companion", return_value=None),
    ):
        mock_session = AsyncMock()
        mock_session.session_id = "test-session"
        mock_session.get_items = AsyncMock(return_value=[{"role": "user", "content": "old"}])
        mock_session.db_path = ":memory:"
        mock_session_cls.return_value = mock_session

        scheduler = AgentScheduler(session_id="test-session")
        scheduler.dev_agent = object()
        scheduler._agent_initialized = True
        scheduler._migration_done = True
        scheduler._capture_usage = AsyncMock()
        scheduler._refresh_magic_docs_after_turn = AsyncMock()

        try:
            await asyncio.gather(
                scheduler.handle("first", render_output=False),
                scheduler.handle("second", render_output=False),
            )
        finally:
            await scheduler.cleanup()

    assert max_active == 1


@pytest.mark.asyncio
async def test_stream_json_emits_reasoning_deltas_when_enabled(tmp_path, monkeypatch):
    from agents import RawResponsesStreamEvent
    from openai.types.responses.response_reasoning_summary_text_delta_event import (
        ResponseReasoningSummaryTextDeltaEvent,
    )
    from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KODER_REASONING_DISPLAY", "summary")

    class FakeResult:
        final_output = "Visible answer."

        async def stream_events(self):
            yield RawResponsesStreamEvent(
                data=ResponseReasoningSummaryTextDeltaEvent(
                    delta="checking",
                    item_id="rs_1",
                    output_index=0,
                    sequence_number=1,
                    summary_index=0,
                    type="response.reasoning_summary_text.delta",
                )
            )
            yield RawResponsesStreamEvent(
                data=ResponseTextDeltaEvent(
                    content_index=0,
                    delta="Visible answer.",
                    item_id="msg_1",
                    logprobs=[],
                    output_index=1,
                    sequence_number=2,
                    type="response.output_text.delta",
                )
            )

    with (
        patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
        patch("koder_agent.core.scheduler.get_display_hooks"),
        patch("koder_agent.core.scheduler.ApprovalHooks"),
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
        patch("koder_agent.core.scheduler.Runner.run_streamed", return_value=FakeResult()),
    ):
        mock_session = AsyncMock()
        mock_session.session_id = "test-session"
        mock_session.get_items = AsyncMock(return_value=[])
        mock_session.db_path = ":memory:"
        mock_session_cls.return_value = mock_session

        from koder_agent.core.scheduler import AgentScheduler

        scheduler = AgentScheduler(session_id="test-session")
        scheduler.dev_agent = object()
        scheduler._agent_initialized = True
        scheduler._migration_done = True
        scheduler._capture_usage = AsyncMock()
        scheduler._refresh_magic_docs_after_turn = AsyncMock()
        events = []

        try:
            response = await scheduler.handle_stream_json(
                "hello",
                on_event=events.append,
                include_partial_messages=True,
            )
        finally:
            await scheduler.cleanup()

    assert response == "Visible answer."
    assert events[0]["event"]["delta"] == {
        "type": "reasoning_summary_delta",
        "text": "checking",
    }
    assert events[1]["event"]["delta"] == {
        "type": "text_delta",
        "text": "Visible answer.",
    }


@pytest.mark.asyncio
async def test_stream_json_continues_active_goal_until_complete(tmp_path, monkeypatch):
    from koder_agent.core.goals import GoalStatus, GoalUpdate

    monkeypatch.setenv("HOME", str(tmp_path))

    calls = []
    scheduler = None

    class FakeResult:
        def __init__(self, final_output: str, *, complete_goal: bool = False):
            self.final_output = final_output
            self.complete_goal = complete_goal

        async def stream_events(self):
            if self.complete_goal:
                await scheduler.goal_store.update_goal(
                    "stream-goal",
                    GoalUpdate(status=GoalStatus.COMPLETE),
                )
            for event in ():
                yield event

    def fake_run_streamed(_agent, user_input, **_kwargs):
        calls.append(user_input)
        return FakeResult(f"turn {len(calls)}", complete_goal=len(calls) >= 2)

    with (
        patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
        patch("koder_agent.core.scheduler.get_display_hooks"),
        patch("koder_agent.core.scheduler.ApprovalHooks"),
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
        patch("koder_agent.core.scheduler.Runner.run_streamed", side_effect=fake_run_streamed),
    ):
        mock_session = AsyncMock()
        mock_session.session_id = "stream-goal"
        mock_session.get_items = AsyncMock(return_value=[{"role": "user", "content": "old"}])
        mock_session.db_path = str(tmp_path / "koder.db")
        mock_session_cls.return_value = mock_session

        from koder_agent.core.scheduler import AgentScheduler

        scheduler = AgentScheduler(session_id="stream-goal")
        scheduler.dev_agent = object()
        scheduler._agent_initialized = True
        scheduler._migration_done = True
        scheduler._capture_usage = AsyncMock()
        scheduler._refresh_magic_docs_after_turn = AsyncMock()

        try:
            goal = await scheduler.goal_store.replace_goal(
                "stream-goal", "finish the streamed goal", GoalStatus.ACTIVE, token_budget=None
            )

            events = []
            response = await scheduler.handle_stream_json("start", on_event=events.append)
            final = await scheduler.goal_store.get_goal("stream-goal")
        finally:
            await scheduler.cleanup()

    assert response == "turn 2"
    assert len(calls) == 2
    assert calls[0] == "start"
    assert calls[1].startswith("[Goal continuation]")
    assert "finish the streamed goal" in calls[1]
    assert final.status is GoalStatus.COMPLETE
    assert final.goal_id == goal.goal_id


@pytest.mark.asyncio
async def test_streaming_ui_receives_stream_updates_without_rich_live(monkeypatch):
    """Interactive streaming should let the fixed-bottom TUI own terminal rendering."""
    from agents import RawResponsesStreamEvent
    from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

    class FakeResult:
        final_output = "Visible answer."

        async def stream_events(self):
            yield RawResponsesStreamEvent(
                data=ResponseTextDeltaEvent(
                    content_index=0,
                    delta="Visible answer.",
                    item_id="msg_1",
                    logprobs=[],
                    output_index=0,
                    sequence_number=1,
                    type="response.output_text.delta",
                )
            )

    class FakeStreamingUI:
        def __init__(self):
            self.updates = []
            self.final_content = None

        def update_output(self, renderable):
            self.updates.append(renderable)

        def set_final_content(self, renderable):
            self.final_content = renderable

    with (
        patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
        patch("koder_agent.core.scheduler.get_display_hooks"),
        patch("koder_agent.core.scheduler.ApprovalHooks"),
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
        patch("koder_agent.core.scheduler.Runner.run_streamed", return_value=FakeResult()),
        patch("koder_agent.core.scheduler.Live") as mock_live,
    ):
        mock_session = AsyncMock()
        mock_session.session_id = "test-session"
        mock_session.get_items = AsyncMock(return_value=[])
        mock_session.db_path = ":memory:"
        mock_session_cls.return_value = mock_session

        from koder_agent.core.scheduler import AgentScheduler

        scheduler = AgentScheduler(session_id="test-session", streaming=True)
        scheduler.dev_agent = object()
        scheduler._agent_initialized = True
        scheduler._migration_done = True
        scheduler._capture_usage = AsyncMock()

        streaming_ui = FakeStreamingUI()
        response = await scheduler._handle_streaming("hello", streaming_ui=streaming_ui)

    assert response == "Visible answer."
    assert streaming_ui.updates
    assert streaming_ui.final_content is not None
    mock_live.assert_not_called()


class TestToolCallCounter:
    """Verify _tool_call_count increments on tool output events."""

    @pytest.fixture
    def scheduler_with_mocks(self):
        """Create a minimally patched AgentScheduler for testing."""
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
        ):
            mock_session = AsyncMock()
            mock_session.session_id = "test-session"
            mock_session.get_items = AsyncMock(return_value=[])
            mock_session.db_path = ":memory:"
            mock_session_cls.return_value = mock_session

            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test-session")
            yield scheduler

    def test_initial_tool_call_count_is_zero(self, scheduler_with_mocks):
        assert scheduler_with_mocks._tool_call_count == 0

    def test_tool_call_count_increments_directly(self, scheduler_with_mocks):
        """Verify the counter attribute can be incremented (basic sanity)."""
        scheduler_with_mocks._tool_call_count += 1
        assert scheduler_with_mocks._tool_call_count == 1
        scheduler_with_mocks._tool_call_count += 1
        assert scheduler_with_mocks._tool_call_count == 2


class TestMagicDocsRefresh:
    """Verify Magic Docs refresh is wired after completed turns."""

    @pytest.mark.asyncio
    async def test_refresh_magic_docs_after_turn_dispatches_best_effort_refresh(self, monkeypatch):
        calls = []

        def fake_refresh(user_input, response, *, cwd=None, **_kwargs):
            calls.append((user_input, response, cwd))
            return []

        monkeypatch.setattr(
            "koder_agent.harness.magic_docs.refresh_tracked_magic_docs",
            fake_refresh,
        )

        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
        ):
            mock_session = AsyncMock()
            mock_session.session_id = "test-session"
            mock_session.get_items = AsyncMock(return_value=[])
            mock_session.db_path = ":memory:"
            mock_session_cls.return_value = mock_session

            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test-session")
            await scheduler._refresh_magic_docs_after_turn("capture docs", "done")

        assert calls
        assert calls[0][0] == "capture docs"
        assert calls[0][1] == "done"


class TestAutoCompact:
    """Verify auto-compaction is triggered and wired correctly."""

    @pytest.fixture
    def compact_manager(self):
        """Create an AutoCompactManager with a low threshold for testing."""
        return AutoCompactManager(context_window=50_000, max_output_tokens=10_000)

    @pytest.mark.asyncio
    async def test_run_auto_compact_calls_llm_compact(self):
        """Verify _run_auto_compact calls llm_compact_messages and rewrites history."""
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
            patch("koder_agent.core.scheduler.llm_compact_messages") as mock_compact,
        ):
            # Set up session mock
            mock_session = AsyncMock()
            mock_session.session_id = "test"
            mock_session.get_items = AsyncMock(
                return_value=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                    {"role": "user", "content": "fix bug"},
                    {"role": "assistant", "content": "done"},
                ]
            )
            mock_session.replace_items = AsyncMock()
            mock_session.db_path = ":memory:"
            mock_session_cls.return_value = mock_session

            # Set up compact mock
            mock_compact.return_value = CompactionResult(
                summary="User greeted and asked to fix a bug.",
                kept_messages=[
                    {"role": "user", "content": "fix bug"},
                    {"role": "assistant", "content": "done"},
                ],
                token_count=100,
                original_count=4,
            )

            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test")
            scheduler._auto_compact = AutoCompactManager(
                context_window=50_000, max_output_tokens=10_000
            )

            # File restoration still appends through add_items.
            mock_session.add_items = AsyncMock()
            mock_session.summarization_threshold = None

            await scheduler._run_auto_compact()

            # Verify compaction was called
            mock_compact.assert_called_once()
            # Verify session history was replaced in one operation.
            mock_session.replace_items.assert_called_once()
            # Verify the compacted items include summary + kept messages
            added_items = mock_session.replace_items.call_args[0][0]
            assert added_items[0]["content"].startswith("[Conversation compacted]")
            assert len(added_items) == 3  # summary + 2 kept messages
            # Verify circuit breaker success was recorded
            assert scheduler._auto_compact._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_run_auto_compact_refreshes_context_tokens_after_persist(self):
        """Verify auto-compact updates status-line context tokens after rewriting history."""
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
            patch("koder_agent.core.scheduler.llm_compact_messages") as mock_compact,
        ):
            mock_session = AsyncMock()
            mock_session.session_id = "test"
            mock_session.get_items = AsyncMock(
                return_value=[
                    {"role": "user", "content": "old"},
                    {"role": "assistant", "content": "old answer"},
                    {"role": "user", "content": "latest"},
                    {"role": "assistant", "content": "latest answer"},
                ]
            )
            mock_session.replace_items = AsyncMock()
            mock_session.add_items = AsyncMock()
            mock_session.summarization_threshold = None
            mock_session.db_path = ":memory:"
            mock_session_cls.return_value = mock_session
            mock_compact.return_value = CompactionResult(
                summary="Old turns summarized.",
                kept_messages=[
                    {"role": "user", "content": "latest"},
                    {"role": "assistant", "content": "latest answer"},
                ],
                token_count=100,
                original_count=4,
            )
            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test")
            scheduler._auto_compact = AutoCompactManager(
                context_window=50_000, max_output_tokens=10_000
            )
            scheduler._estimate_static_context_tokens = MagicMock(return_value=17)

            await scheduler._run_auto_compact()

            compacted_items = mock_session.replace_items.call_args[0][0]
            expected = 17 + estimate_messages_tokens(compacted_items)
            assert scheduler.usage_tracker.session_usage.current_context_tokens == expected

    @pytest.mark.asyncio
    async def test_repair_unreplayable_session_items_drops_unknown_role(self):
        """Verify old poisoned compact output is removed before the next SDK run."""
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
        ):
            valid_item = {"role": "user", "content": "hello"}
            mock_session = AsyncMock()
            mock_session.session_id = "test"
            mock_session.get_items = AsyncMock(
                return_value=[valid_item, {"role": "unknown", "content": ""}]
            )
            mock_session.replace_items = AsyncMock()
            mock_session.summarization_threshold = None
            mock_session.db_path = ":memory:"
            mock_session_cls.return_value = mock_session

            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test")
            scheduler._estimate_static_context_tokens = MagicMock(return_value=0)

            await scheduler._repair_unreplayable_session_items()

            mock_session.replace_items.assert_called_once_with([valid_item])
            assert scheduler.usage_tracker.session_usage.current_context_tokens == (
                estimate_messages_tokens([valid_item])
            )

    @pytest.mark.asyncio
    async def test_run_auto_compact_records_failure_on_error(self):
        """Verify failure is recorded when llm_compact_messages raises."""
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
            patch(
                "koder_agent.core.scheduler.llm_compact_messages",
                side_effect=RuntimeError("LLM unavailable"),
            ),
        ):
            mock_session = AsyncMock()
            mock_session.session_id = "test"
            mock_session.get_items = AsyncMock(return_value=[{"role": "user", "content": "hello"}])
            mock_session.db_path = ":memory:"
            mock_session_cls.return_value = mock_session

            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test")
            scheduler._auto_compact = AutoCompactManager(
                context_window=50_000, max_output_tokens=10_000
            )

            await scheduler._run_auto_compact()

            # Verify failure was recorded
            assert scheduler._auto_compact._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_run_auto_compact_noop_does_not_record_failure(self):
        """A legitimate no-op (no summary, kept == original minimal history) must
        NOT advance the circuit breaker.

        Previously this recorded a failure, so three no-op compactions in a row
        would wedge auto-compaction forever. Only genuine failures (add_items
        rollback or the outer except) may trip the breaker now.
        """
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
            patch("koder_agent.core.scheduler.llm_compact_messages") as mock_compact,
        ):
            mock_session = AsyncMock()
            mock_session.session_id = "test"
            mock_session.get_items = AsyncMock(return_value=[{"role": "user", "content": "hello"}])
            mock_session.replace_items = AsyncMock()
            mock_session.db_path = ":memory:"
            mock_session_cls.return_value = mock_session

            mock_compact.return_value = CompactionResult(
                summary=None,
                kept_messages=[{"role": "user", "content": "hello"}],
                token_count=10,
                original_count=1,
            )

            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test")
            scheduler._auto_compact = AutoCompactManager(
                context_window=50_000, max_output_tokens=10_000
            )

            await scheduler._run_auto_compact()

            assert scheduler._auto_compact._consecutive_failures == 0
            # A no-op never rewrites the session.
            mock_session.replace_items.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_auto_compact_records_atomic_replace_failure(self):
        """A failed atomic replacement records failure without a restore write."""
        original_items = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "fix bug"},
            {"role": "assistant", "content": "done"},
        ]
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
            patch("koder_agent.core.scheduler.llm_compact_messages") as mock_compact,
        ):
            mock_session = AsyncMock()
            mock_session.session_id = "test"
            mock_session.get_items = AsyncMock(return_value=list(original_items))
            mock_session.replace_items = AsyncMock(side_effect=RuntimeError("disk full"))
            mock_session.summarization_threshold = None
            mock_session.db_path = ":memory:"
            mock_session.add_items = AsyncMock()
            mock_session_cls.return_value = mock_session

            mock_compact.return_value = CompactionResult(
                summary="User greeted and asked to fix a bug.",
                kept_messages=[
                    {"role": "user", "content": "fix bug"},
                    {"role": "assistant", "content": "done"},
                ],
                token_count=100,
                original_count=4,
            )

            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test")
            scheduler._auto_compact = AutoCompactManager(
                context_window=50_000, max_output_tokens=10_000
            )

            await scheduler._run_auto_compact()

            mock_session.replace_items.assert_called_once()
            mock_session.add_items.assert_not_called()
            assert scheduler._auto_compact._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_run_auto_compact_rollback_removes_partial_compacted_items(self):
        """A partial failed write is cleared before originals are restored."""
        original_items = [
            {"role": "user", "content": "ORIGINAL_A"},
            {"role": "assistant", "content": "ORIGINAL_B"},
        ]

        class PartialWriteSession:
            session_id = "test"
            db_path = ":memory:"
            summarization_threshold = None

            def __init__(self):
                self.items = list(original_items)
                self.add_calls = 0

            async def get_items(self):
                return list(self.items)

            async def clear_session(self):
                self.items.clear()

            async def add_items(self, batch):
                self.add_calls += 1
                if self.add_calls == 1:
                    self.items.append({"role": "assistant", "content": "PARTIAL_COMPACTED"})
                    raise RuntimeError("partial disk write")
                self.items.extend(batch)

        session = PartialWriteSession()
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession", return_value=session),
            patch("koder_agent.core.scheduler.llm_compact_messages") as mock_compact,
        ):
            mock_compact.return_value = CompactionResult(
                summary="PARTIAL_COMPACTED",
                kept_messages=[],
                token_count=10,
                original_count=2,
            )
            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test")
            scheduler._auto_compact = AutoCompactManager(
                context_window=50_000, max_output_tokens=10_000
            )

            await scheduler._run_auto_compact()

        assert session.items == original_items
        assert all("PARTIAL_COMPACTED" not in str(item) for item in session.items)
        assert scheduler._auto_compact._consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_restore_prefers_replace_items_capability(self):
        original_items = [{"role": "user", "content": "ORIGINAL"}]

        class ReplaceCapableSession:
            session_id = "test"
            db_path = ":memory:"

            def __init__(self):
                self.items = [{"role": "assistant", "content": "PARTIAL_COMPACTED"}]
                self.replace_calls = 0

            async def replace_items(self, items):
                self.replace_calls += 1
                self.items = list(items)

            async def get_items(self):
                return list(self.items)

            async def clear_session(self):
                raise AssertionError("replace_items should be preferred")

            async def add_items(self, _items):
                raise AssertionError("replace_items should be preferred")

        session = ReplaceCapableSession()
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession", return_value=session),
        ):
            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test")
            await scheduler._restore_session_items(original_items)

        assert session.replace_calls == 1
        assert session.items == original_items

    @pytest.mark.asyncio
    async def test_run_auto_compact_adds_file_restoration_attachments(self):
        """After a successful compaction, recently-read files are re-attached.

        The read_file target is collected from the ORIGINAL items (a koder
        function_call item) and appended via a second add_items call.
        """
        import json as _json
        import tempfile
        from pathlib import Path as _Path

        with tempfile.TemporaryDirectory() as tmpdir:
            read_path = _Path(tmpdir) / "restored.py"
            read_path.write_text("def restored(): return 42\n", encoding="utf-8")

            original_items = [
                {"role": "user", "content": "please read a file"},
                {
                    "type": "function_call",
                    "name": "read_file",
                    "arguments": _json.dumps({"path": str(read_path)}),
                    "call_id": "call_1",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "..."},
                {"role": "assistant", "content": "done reading"},
            ]
            with (
                patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
                patch("koder_agent.core.scheduler.get_display_hooks"),
                patch("koder_agent.core.scheduler.ApprovalHooks"),
                patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
                patch("koder_agent.core.scheduler.llm_compact_messages") as mock_compact,
            ):
                mock_session = AsyncMock()
                mock_session.session_id = "test"
                mock_session.get_items = AsyncMock(return_value=list(original_items))
                replaced_batches = []

                async def replace_items_side_effect(batch):
                    replaced_batches.append(batch)

                mock_session.replace_items = AsyncMock(side_effect=replace_items_side_effect)
                mock_session.summarization_threshold = None
                mock_session.db_path = ":memory:"

                added_batches = []

                async def add_items_side_effect(batch):
                    added_batches.append(batch)

                mock_session.add_items = AsyncMock(side_effect=add_items_side_effect)
                mock_session_cls.return_value = mock_session

                mock_compact.return_value = CompactionResult(
                    summary="Read a file and reported success.",
                    kept_messages=[{"role": "assistant", "content": "done reading"}],
                    token_count=50,
                    original_count=4,
                )

                from koder_agent.core.scheduler import AgentScheduler

                scheduler = AgentScheduler(session_id="test")
                scheduler._auto_compact = AutoCompactManager(
                    context_window=50_000, max_output_tokens=10_000
                )

                await scheduler._run_auto_compact()

                assert len(replaced_batches) == 1
                # Replacement is atomic; restoration attachments append afterward.
                assert len(added_batches) == 1
                attachments = added_batches[0]
                assert len(attachments) == 1
                assert "def restored" in attachments[0]["content"]
                assert str(read_path) in attachments[0]["content"]
                assert scheduler._auto_compact._consecutive_failures == 0


class TestRepairUnreplayableAtomicity:
    """Verify repair delegates rollback guarantees to atomic replacement."""

    @pytest.mark.asyncio
    async def test_repair_does_not_issue_fragile_restore_when_replace_fails(self):
        original_items = [
            {"role": "user", "content": "hello"},
            {"role": "unknown", "content": ""},  # unreplayable -> triggers rewrite
        ]
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
        ):
            mock_session = AsyncMock()
            mock_session.session_id = "test"
            mock_session.get_items = AsyncMock(return_value=list(original_items))
            mock_session.replace_items = AsyncMock(side_effect=RuntimeError("disk full"))
            mock_session.summarization_threshold = None
            mock_session.db_path = ":memory:"
            mock_session.add_items = AsyncMock()
            mock_session_cls.return_value = mock_session

            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test")
            scheduler._estimate_static_context_tokens = MagicMock(return_value=0)

            await scheduler._repair_unreplayable_session_items()

            mock_session.replace_items.assert_called_once_with([original_items[0]])
            mock_session.add_items.assert_not_called()

    @pytest.mark.asyncio
    async def test_repair_rollback_removes_partial_and_restores_once(self):
        original_items = [
            {"role": "user", "content": "ORIGINAL_A"},
            {"role": "unknown", "content": "ORIGINAL_B"},
        ]

        class PartialRepairSession:
            session_id = "test"
            db_path = ":memory:"
            summarization_threshold = None

            def __init__(self):
                self.items = list(original_items)
                self.add_calls = 0

            async def get_items(self):
                return list(self.items)

            async def clear_session(self):
                self.items.clear()

            async def add_items(self, batch):
                self.add_calls += 1
                if self.add_calls == 1:
                    self.items.append({"role": "assistant", "content": "PARTIAL_COMPACTED"})
                    raise RuntimeError("partial disk write")
                self.items.extend(batch)

        session = PartialRepairSession()
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession", return_value=session),
        ):
            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test")
            scheduler._estimate_static_context_tokens = MagicMock(return_value=0)
            await scheduler._repair_unreplayable_session_items()

        assert session.items == original_items
        assert all("PARTIAL_COMPACTED" not in str(item) for item in session.items)


class TestSessionMemoryExtraction:
    """Verify session memory extraction is triggered and wired correctly."""

    @pytest.mark.asyncio
    async def test_run_session_memory_extraction_calls_llm(self):
        """Verify extraction calls llm_extract_memories and persists results."""
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
            patch("koder_agent.core.scheduler.llm_extract_memories") as mock_extract,
        ):
            mock_session = AsyncMock()
            mock_session.session_id = "test"
            mock_session.get_items = AsyncMock(
                return_value=[
                    {"role": "user", "content": "I prefer dark mode"},
                    {"role": "assistant", "content": "Noted."},
                ]
            )
            mock_session.db_path = ":memory:"
            mock_session_cls.return_value = mock_session

            mock_extract.return_value = ExtractionResult(
                memories=[
                    {"type": "user", "content": "User prefers dark mode"},
                ],
                errors=[],
            )

            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test")

            # Mock the session memory manager to avoid filesystem operations
            scheduler._session_memory = MagicMock(spec=SessionMemoryManager)
            scheduler._session_memory.ensure_notes_file.return_value = "/tmp/test_notes.md"
            scheduler._session_memory.record_extraction = MagicMock()

            mock_open = MagicMock()
            with patch("builtins.open", mock_open):
                await scheduler._run_session_memory_extraction(
                    context_tokens=15_000, tool_call_count=5
                )

            mock_extract.assert_called_once()
            scheduler._session_memory.record_extraction.assert_called_once_with(15_000, 5)

    @pytest.mark.asyncio
    async def test_run_session_memory_extraction_handles_errors_gracefully(self):
        """Verify extraction failure is caught and extraction is still recorded."""
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
            patch(
                "koder_agent.core.scheduler.llm_extract_memories",
                side_effect=RuntimeError("LLM down"),
            ),
        ):
            mock_session = AsyncMock()
            mock_session.session_id = "test"
            mock_session.get_items = AsyncMock(return_value=[{"role": "user", "content": "hello"}])
            mock_session.db_path = ":memory:"
            mock_session_cls.return_value = mock_session

            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test")
            scheduler._session_memory = MagicMock(spec=SessionMemoryManager)
            scheduler._session_memory.record_extraction = MagicMock()

            # Should not raise
            await scheduler._run_session_memory_extraction(
                context_tokens=20_000, tool_call_count=10
            )

            # Extraction should still be recorded to avoid immediate retry
            scheduler._session_memory.record_extraction.assert_called_once_with(20_000, 10)

    @pytest.mark.asyncio
    async def test_run_session_memory_extraction_skips_empty_history(self):
        """Verify extraction is skipped when session has no messages."""
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
            patch("koder_agent.core.scheduler.llm_extract_memories") as mock_extract,
        ):
            mock_session = AsyncMock()
            mock_session.session_id = "test"
            mock_session.get_items = AsyncMock(return_value=[])
            mock_session.db_path = ":memory:"
            mock_session_cls.return_value = mock_session

            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test")

            await scheduler._run_session_memory_extraction(context_tokens=15_000, tool_call_count=5)

            # LLM should not be called for empty history
            mock_extract.assert_not_called()


class TestCaptureUsageIntegration:
    """Verify _capture_usage triggers compaction and extraction at the right thresholds."""

    @pytest.mark.asyncio
    async def test_capture_usage_triggers_auto_compact(self):
        """Verify that _capture_usage calls _run_auto_compact when threshold is met."""
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
            patch("koder_agent.core.scheduler.get_model_name", return_value="gpt-4o"),
        ):
            mock_session = AsyncMock()
            mock_session.session_id = "test"
            mock_session.get_items = AsyncMock(return_value=[])
            mock_session.db_path = ":memory:"
            mock_session_cls.return_value = mock_session

            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test")

            # Set up auto-compact with a very low threshold
            scheduler._auto_compact = AutoCompactManager(context_window=1000, max_output_tokens=100)
            # Threshold will be (1000 - 100) - 13000 = negative, so ANY tokens should trigger
            # Actually with a small context window the threshold is negative.
            # Let's use a realistic scenario instead:
            scheduler._auto_compact = AutoCompactManager(
                context_window=100_000, max_output_tokens=10_000
            )
            # Threshold = (100000 - 10000) - 13000 = 77000

            # Mock the actual compact method
            scheduler._run_auto_compact = AsyncMock()
            scheduler._run_session_memory_extraction = AsyncMock()

            # Create a mock result with usage exceeding the threshold
            mock_result = MagicMock()
            mock_result.context_wrapper.usage.input_tokens = 70_000
            mock_result.context_wrapper.usage.output_tokens = 5_000
            usage_entry = MagicMock()
            usage_entry.total_tokens = 80_000  # Above 77000 threshold
            mock_result.context_wrapper.usage.request_usage_entries = [usage_entry]

            await scheduler._capture_usage(mock_result)

            scheduler._run_auto_compact.assert_called_once()

    @pytest.mark.asyncio
    async def test_capture_usage_triggers_session_memory(self):
        """Verify that _capture_usage triggers extraction when conditions are met."""
        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
            patch("koder_agent.core.scheduler.get_model_name", return_value="gpt-4o"),
        ):
            mock_session = AsyncMock()
            mock_session.session_id = "test"
            mock_session.get_items = AsyncMock(return_value=[])
            mock_session.db_path = ":memory:"
            mock_session_cls.return_value = mock_session

            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="test")

            # No auto-compact manager to avoid that path
            scheduler._auto_compact = None

            # Set tool call count high enough
            scheduler._tool_call_count = 5

            # Mock the extraction method
            scheduler._run_session_memory_extraction = AsyncMock()

            # Create a mock result with enough tokens to trigger extraction (>= 10000)
            mock_result = MagicMock()
            mock_result.context_wrapper.usage.input_tokens = 8_000
            mock_result.context_wrapper.usage.output_tokens = 3_000
            usage_entry = MagicMock()
            usage_entry.total_tokens = 11_000
            mock_result.context_wrapper.usage.request_usage_entries = [usage_entry]

            await scheduler._capture_usage(mock_result)

            scheduler._run_session_memory_extraction.assert_called_once_with(11_000, 5)


@pytest.mark.asyncio
async def test_streaming_ui_esc_callback_registered_and_cleared(monkeypatch):
    """Fixed-bottom streaming must wire ESC cancellation through the TUI."""
    from agents import RawResponsesStreamEvent
    from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

    class FakeResult:
        final_output = "Answer."

        def cancel(self, mode="immediate"):
            self.cancelled_mode = mode

        async def stream_events(self):
            yield RawResponsesStreamEvent(
                data=ResponseTextDeltaEvent(
                    content_index=0,
                    delta="Answer.",
                    item_id="msg_1",
                    logprobs=[],
                    output_index=0,
                    sequence_number=1,
                    type="response.output_text.delta",
                )
            )

    class FakeStreamingUI:
        def __init__(self):
            self.cancel_callbacks = []
            self.final_content = None

        def update_output(self, renderable):
            pass

        def set_final_content(self, renderable):
            self.final_content = renderable

        def set_final_text(self, text):
            pass

        def set_cancel_callback(self, callback):
            self.cancel_callbacks.append(callback)

    with (
        patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
        patch("koder_agent.core.scheduler.get_display_hooks"),
        patch("koder_agent.core.scheduler.ApprovalHooks"),
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
        patch("koder_agent.core.scheduler.Runner.run_streamed", return_value=FakeResult()),
    ):
        mock_session = AsyncMock()
        mock_session.session_id = "test-session"
        mock_session.get_items = AsyncMock(return_value=[])
        mock_session.db_path = ":memory:"
        mock_session_cls.return_value = mock_session

        from koder_agent.core.scheduler import AgentScheduler

        scheduler = AgentScheduler(session_id="test-session", streaming=True)
        scheduler.dev_agent = object()
        scheduler._agent_initialized = True
        scheduler._migration_done = True
        scheduler._capture_usage = AsyncMock()

        streaming_ui = FakeStreamingUI()
        await scheduler._handle_streaming("hello", streaming_ui=streaming_ui)

    # Registered a real callback during streaming, cleared it afterwards.
    assert len(streaming_ui.cancel_callbacks) == 2
    assert callable(streaming_ui.cancel_callbacks[0])
    assert streaming_ui.cancel_callbacks[1] is None


@pytest.mark.asyncio
async def test_streaming_ui_esc_callback_cancels_turn(monkeypatch):
    """Invoking the registered ESC callback must cancel the in-flight stream."""
    import asyncio

    from agents import RawResponsesStreamEvent
    from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent

    first_event_seen = asyncio.Event()

    class FakeResult:
        final_output = "Never finished."
        cancelled_mode = None

        def cancel(self, mode="immediate"):
            self.cancelled_mode = mode

        async def stream_events(self):
            yield RawResponsesStreamEvent(
                data=ResponseTextDeltaEvent(
                    content_index=0,
                    delta="partial ",
                    item_id="msg_1",
                    logprobs=[],
                    output_index=0,
                    sequence_number=1,
                    type="response.output_text.delta",
                )
            )
            first_event_seen.set()
            await asyncio.sleep(30)  # Simulate a stalled stream

    class FakeStreamingUI:
        def __init__(self):
            self.cancel_callback = None
            self.final_texts = []

        def update_output(self, renderable):
            pass

        def set_final_content(self, renderable):
            pass

        def set_final_text(self, text):
            self.final_texts.append(text)

        def set_cancel_callback(self, callback):
            if callback is not None:
                self.cancel_callback = callback

    fake_result = FakeResult()
    with (
        patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
        patch("koder_agent.core.scheduler.get_display_hooks"),
        patch("koder_agent.core.scheduler.ApprovalHooks"),
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as mock_session_cls,
        patch("koder_agent.core.scheduler.Runner.run_streamed", return_value=fake_result),
    ):
        mock_session = AsyncMock()
        mock_session.session_id = "test-session"
        mock_session.get_items = AsyncMock(return_value=[])
        mock_session.db_path = ":memory:"
        mock_session_cls.return_value = mock_session

        from koder_agent.core.scheduler import AgentScheduler

        scheduler = AgentScheduler(session_id="test-session", streaming=True)
        scheduler.dev_agent = object()
        scheduler._agent_initialized = True
        scheduler._migration_done = True
        scheduler._capture_usage = AsyncMock()

        streaming_ui = FakeStreamingUI()

        async def press_escape():
            await asyncio.wait_for(first_event_seen.wait(), timeout=5)
            assert streaming_ui.cancel_callback is not None
            streaming_ui.cancel_callback()

        stream_task = asyncio.create_task(
            scheduler._handle_streaming("hello", streaming_ui=streaming_ui)
        )
        esc_task = asyncio.create_task(press_escape())
        response = await asyncio.wait_for(stream_task, timeout=10)
        await esc_task

    assert fake_result.cancelled_mode == "immediate"
    assert scheduler._last_turn_cancelled is True
    assert "partial" in response or "cancelled" in response.lower()
