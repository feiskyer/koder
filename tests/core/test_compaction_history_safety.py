"""Compaction must preserve actual persisted history when no summary is produced."""

import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from koder_agent.core import scheduler as scheduler_module
from koder_agent.core.session import EnhancedSQLiteSession
from koder_agent.harness.commands import interactive
from koder_agent.harness.memory import compact
from koder_agent.harness.memory.auto_compact import AutoCompactManager
from koder_agent.harness.memory.budget import estimate_messages_tokens


def raw_rows(database):
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        return connection.execute(
            "SELECT id, session_id, message_data, created_at FROM agent_messages ORDER BY id"
        ).fetchall()


@pytest_asyncio.fixture
async def runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    database = tmp_path / "history.db"
    session = EnhancedSQLiteSession("summary-safety", db_path=str(database))
    monkeypatch.setattr(scheduler_module, "EnhancedSQLiteSession", lambda **_: session)
    monkeypatch.setattr(scheduler_module, "get_all_tools", lambda: [])
    monkeypatch.setattr(scheduler_module, "get_display_hooks", lambda **_: SimpleNamespace())
    monkeypatch.setattr(
        scheduler_module, "ApprovalHooks", lambda *_args, **_kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(scheduler_module, "get_companion", lambda: None)
    scheduler = scheduler_module.AgentScheduler(session_id="summary-safety", project_root=tmp_path)
    scheduler._auto_compact = AutoCompactManager(50_000, 10_000)
    monkeypatch.setattr(scheduler, "_compact_keep_recent", lambda: 2)
    scheduler._append_post_compact_file_restoration = AsyncMock(return_value=[])
    scheduler._replace_session_items = AsyncMock(wraps=scheduler._replace_session_items)
    hooks = []
    monkeypatch.setattr(
        scheduler, "_dispatch_compact_hooks", lambda event, payload: hooks.append((event, payload))
    )
    monkeypatch.setattr(
        interactive,
        "dispatch_command_hooks",
        lambda **kwargs: hooks.append((kwargs["event_name"], kwargs["payload"])),
    )

    async def refresh(items=None):
        return estimate_messages_tokens(items if items is not None else await session.get_items())

    monkeypatch.setattr(scheduler, "refresh_context_usage_from_session", refresh)
    handler = interactive.HarnessInteractiveCommandHandler(
        agent_service=scheduler.get_agent_service(), emit_console=False
    )
    try:
        yield SimpleNamespace(
            session=session, scheduler=scheduler, handler=handler, database=database, hooks=hooks
        )
    finally:
        await scheduler.cleanup()


@pytest.mark.parametrize("route", ["auto", "manual"])
@pytest.mark.parametrize("response", ["", " \n\t ", "<summary> \n </summary>"])
@pytest.mark.asyncio
async def test_empty_summary_preserves_database_rows(runtime, monkeypatch, route, response):
    history = [
        {"role": "system", "content": "preserve system instructions"},
        {"role": "developer", "content": "preserve developer instructions"},
        *[{"role": "user", "content": f"original-intent-{index}"} for index in range(8)],
    ]
    await runtime.session.add_items(history)
    before = raw_rows(runtime.database)
    monkeypatch.setattr(compact, "llm_completion", AsyncMock(return_value=response))
    if route == "auto":
        await runtime.scheduler._run_auto_compact()
        assert runtime.scheduler._auto_compact._consecutive_failures == 1
    else:
        result = await runtime.handler.handle_slash_input("/compact", runtime.scheduler)
        assert "Error executing command" in result and "summary" in result
    runtime.scheduler._replace_session_items.assert_not_awaited()
    runtime.scheduler._append_post_compact_file_restoration.assert_not_awaited()
    assert raw_rows(runtime.database) == before
    assert await runtime.session.get_items() == history
    assert not any(event == "PostCompact" for event, _payload in runtime.hooks)


@pytest.mark.asyncio
async def test_valid_summary_after_failure_can_replace_history_and_reset_failure_count(
    runtime, monkeypatch
):
    history = [{"role": "user", "content": f"intent-{index}"} for index in range(8)]
    await runtime.session.add_items(history)
    completion = AsyncMock(side_effect=["", "<summary>Earlier intents retained here.</summary>"])
    monkeypatch.setattr(compact, "llm_completion", completion)
    await runtime.scheduler._run_auto_compact()
    assert await runtime.session.get_items() == history
    assert runtime.scheduler._auto_compact._consecutive_failures == 1
    await runtime.scheduler._run_auto_compact()
    result = await runtime.session.get_items()
    assert len(result) == 3
    assert result[0]["content"] == "[Conversation compacted]\n\nEarlier intents retained here."
    assert result[1:] == history[-2:]
    assert runtime.scheduler._auto_compact._consecutive_failures == 0


@pytest.mark.asyncio
async def test_existing_minimal_history_does_not_trip_failure_counter(runtime, monkeypatch):
    history = [{"role": "user", "content": "short current intent"}]
    await runtime.session.add_items(history)
    before = raw_rows(runtime.database)
    runtime.scheduler._auto_compact._consecutive_failures = 2
    completion = AsyncMock(side_effect=AssertionError("No summary is needed"))
    monkeypatch.setattr(compact, "llm_completion", completion)
    await runtime.scheduler._run_auto_compact()
    assert raw_rows(runtime.database) == before
    assert runtime.scheduler._auto_compact._consecutive_failures == 2
    runtime.scheduler._replace_session_items.assert_not_awaited()
    completion.assert_not_awaited()


@pytest.mark.parametrize("route", ["auto", "manual"])
@pytest.mark.asyncio
async def test_compaction_summarizes_unfinished_tools_and_persists_complete_pairs(
    runtime, monkeypatch, route
):
    history = [
        *[{"role": "user", "content": f"older intent {index}"} for index in range(8)],
        {
            "type": "function_call",
            "call_id": "complete",
            "name": "read_file",
            "arguments": '{"path":"synthetic.py"}',
        },
        {"type": "function_call_output", "call_id": "complete", "output": "synthetic"},
        {
            "type": "function_call",
            "call_id": "unfinished",
            "name": "read_file",
            "arguments": '{"path":"unfinished.py"}',
        },
    ]
    await runtime.session.add_items(history)
    completion = AsyncMock(return_value="<summary>Earlier intent and unfinished work.</summary>")
    monkeypatch.setattr(compact, "llm_completion", completion)

    if route == "auto":
        await runtime.scheduler._run_auto_compact()
    else:
        result = await runtime.handler.handle_slash_input("/compact", runtime.scheduler)
        assert result.startswith("compacted,")

    persisted = await runtime.session.get_items()
    assert compact.replayable_session_items(persisted) == persisted
    assert persisted[-2:] == history[-3:-1]
    runtime.scheduler._replace_session_items.assert_awaited_once()
    source = completion.await_args.args[0][0]["content"]
    assert "id=unfinished" in source
    assert "id=complete" not in source


@pytest.mark.parametrize("route", ["auto", "manual"])
@pytest.mark.asyncio
async def test_multimodal_compaction_preserves_content_and_repeated_call_preserves_rows(
    runtime, tmp_path, monkeypatch, route
):
    from koder_agent.utils.image_input import build_multimodal_input

    image = tmp_path / "synthetic.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nsynthetic-image")
    image_turn = build_multimodal_input("Exact image-related user intent", [str(image)])[0]
    history = [
        {"role": "system", "content": "system invariant"},
        {"role": "developer", "content": "developer invariant"},
        *[{"role": "user", "content": f"older intent {index}"} for index in range(8)],
        image_turn,
    ]
    await runtime.session.add_items(history)
    completion = AsyncMock(return_value="<summary>Earlier intent.</summary>")
    monkeypatch.setattr(compact, "llm_completion", completion)

    async def run_compaction():
        if route == "auto":
            await runtime.scheduler._run_auto_compact()
        else:
            result = await runtime.handler.handle_slash_input("/compact", runtime.scheduler)
            assert result.startswith("compacted,")

    await run_compaction()
    persisted = await runtime.session.get_items()
    assert persisted[:2] == history[:2]
    assert persisted[-1] == image_turn
    rows = raw_rows(runtime.database)
    runtime.scheduler._replace_session_items.reset_mock()

    await run_compaction()

    assert raw_rows(runtime.database) == rows
    assert await runtime.session.get_items() == persisted
    runtime.scheduler._replace_session_items.assert_not_awaited()
    completion.assert_awaited_once()


@pytest.mark.parametrize("route", ["auto", "manual"])
@pytest.mark.parametrize(
    "response",
    [
        "<analysis>No final summary.</analysis>",
        "<summary>Unfinished summary",
        "<summary>first</summary><summary>second</summary>",
        "<analysis>An example <summary>not final</summary>.</analysis>",
    ],
    ids=["analysis_only", "incomplete_summary", "multiple_summaries", "analysis_example"],
)
@pytest.mark.asyncio
async def test_malformed_summary_preserves_database_rows(runtime, monkeypatch, route, response):
    history = [{"role": "user", "content": f"original-intent-{index}"} for index in range(8)]
    await runtime.session.add_items(history)
    before = raw_rows(runtime.database)
    monkeypatch.setattr(compact, "llm_completion", AsyncMock(return_value=response))

    if route == "auto":
        await runtime.scheduler._run_auto_compact()
        assert runtime.scheduler._auto_compact._consecutive_failures == 1
    else:
        result = await runtime.handler.handle_slash_input("/compact", runtime.scheduler)
        assert "Error executing command" in result and "summary" in result

    runtime.scheduler._replace_session_items.assert_not_awaited()
    runtime.scheduler._append_post_compact_file_restoration.assert_not_awaited()
    assert raw_rows(runtime.database) == before
    assert await runtime.session.get_items() == history
    assert not any(event == "PostCompact" for event, _payload in runtime.hooks)


@pytest_asyncio.fixture
async def cached_file(runtime, tmp_path):
    from koder_agent.harness.agents.runtime_context import agent_session_scope
    from koder_agent.tools import file as file_tools

    state = runtime.session.file_read_state
    path = tmp_path / "cached.txt"
    body = "synthetic file body that will be removed from context"
    path.write_text(body, encoding="utf-8")
    arguments = json.dumps({"path": str(path)})

    async def read():
        with agent_session_scope(runtime.session):
            return await file_tools.read_file.on_invoke_tool(None, arguments)

    def validate():
        with agent_session_scope(runtime.session):
            return file_tools.validate_read_file_for_edit(str(path))

    original_output = await read()
    assert body in original_output
    assert "prior content still in context" in await read()
    return SimpleNamespace(
        state=state,
        path=path,
        body=body,
        read=read,
        validate=validate,
        history=[
            {"role": "user", "content": "Read the synthetic file."},
            {
                "type": "function_call",
                "call_id": "cached-read",
                "name": "read_file",
                "arguments": arguments,
            },
            {"type": "function_call_output", "call_id": "cached-read", "output": original_output},
        ],
    )


@pytest.mark.parametrize("route", ["auto", "manual"])
@pytest.mark.parametrize("post_failure", [False, True])
@pytest.mark.asyncio
async def test_compaction_invalidates_removed_file_content(
    runtime, cached_file, monkeypatch, route, post_failure
):
    history = [
        *cached_file.history,
        *[{"role": "user", "content": f"later intent {index}"} for index in range(8)],
    ]
    await runtime.session.add_items(history)
    completion = AsyncMock(
        return_value="<summary>Earlier discussion without the file body.</summary>"
    )
    monkeypatch.setattr(compact, "llm_completion", completion)
    if post_failure:
        refresh = runtime.scheduler.refresh_context_usage_from_session
        refresh_calls = 0

        async def fail_after_replacement(items=None):
            nonlocal refresh_calls
            refresh_calls += 1
            if refresh_calls == 2:
                raise RuntimeError("synthetic post-commit telemetry failure")
            return await refresh(items)

        monkeypatch.setattr(
            runtime.scheduler, "refresh_context_usage_from_session", fail_after_replacement
        )

    if route == "auto":
        await runtime.scheduler._run_auto_compact()
    else:
        result = await runtime.handler.handle_slash_input("/compact", runtime.scheduler)
        assert result.startswith("compact failed:" if post_failure else "compacted,")

    persisted = await runtime.session.get_items()
    assert cached_file.body not in json.dumps(persisted)
    assert cached_file.body in completion.await_args.args[0][0]["content"]
    assert not cached_file.state.has_been_read(str(cached_file.path))
    assert "has not been read" in cached_file.validate()
    refreshed_output = await cached_file.read()
    assert cached_file.body in refreshed_output
    assert "prior content still in context" not in refreshed_output
    assert cached_file.validate() is None


@pytest.mark.parametrize("route", ["auto", "manual"])
@pytest.mark.parametrize("outcome", ["noop", "empty_summary", "replacement_failure"])
@pytest.mark.asyncio
async def test_unchanged_history_keeps_valid_file_read_state(
    runtime, cached_file, monkeypatch, route, outcome
):
    history = list(cached_file.history)
    if outcome != "noop":
        history += [{"role": "user", "content": f"later intent {index}"} for index in range(8)]
    await runtime.session.add_items(history)
    before = raw_rows(runtime.database)
    completion = AsyncMock(
        return_value="" if outcome == "empty_summary" else "<summary>Old.</summary>"
    )
    monkeypatch.setattr(compact, "llm_completion", completion)
    if outcome == "replacement_failure":
        runtime.scheduler._replace_session_items = AsyncMock(
            side_effect=RuntimeError("synthetic replacement failure")
        )

    if route == "auto":
        await runtime.scheduler._run_auto_compact()
    else:
        await runtime.handler.handle_slash_input("/compact", runtime.scheduler)

    assert raw_rows(runtime.database) == before
    assert cached_file.state.has_been_read(str(cached_file.path))
    assert "prior content still in context" in await cached_file.read()
    if outcome == "noop":
        completion.assert_not_awaited()
