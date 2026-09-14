"""File-read evidence belongs to the actual managed conversation."""

import asyncio
import json
from types import SimpleNamespace

import pytest
import pytest_asyncio

from koder_agent.core.session import EnhancedSQLiteSession
from koder_agent.harness.agents.runtime_context import agent_session_scope
from koder_agent.tools import file as file_tools
from koder_agent.tools.file_state import ReadFileState


@pytest_asyncio.fixture
async def conversations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(file_tools, "_file_state", ReadFileState())
    sessions = []

    def create(session_id, database="sessions.db"):
        session = EnhancedSQLiteSession(session_id, db_path=str(tmp_path / database))
        sessions.append(session)
        return session

    path = tmp_path / "source.txt"
    path.write_text("original file body\n", encoding="utf-8")
    try:
        yield SimpleNamespace(create=create, path=path, root=tmp_path)
    finally:
        for session in sessions:
            session.close()


async def invoke(tool, **arguments):
    return await tool.on_invoke_tool(None, json.dumps(arguments))


@pytest.mark.parametrize(
    ("second_id", "second_database"),
    [
        ("second", "sessions.db"),
        ("first", "another.db"),
        ("first", "sessions.db"),
    ],
    ids=["different_session", "different_database", "distinct_session_object"],
)
@pytest.mark.asyncio
async def test_each_session_receives_its_own_file_contents(
    conversations, second_id, second_database
):
    first = conversations.create("first")
    second = conversations.create(second_id, second_database)
    with agent_session_scope(first):
        assert "original file body" in await invoke(
            file_tools.read_file, path=str(conversations.path)
        )
        assert "prior content still in context" in await invoke(
            file_tools.read_file, path=str(conversations.path)
        )
    with agent_session_scope(second):
        output = await invoke(file_tools.read_file, path=str(conversations.path))
    assert "original file body" in output
    assert "prior content still in context" not in output


@pytest.mark.parametrize("operation", ["write", "append", "edit"])
@pytest.mark.asyncio
async def test_another_sessions_read_cannot_authorize_a_write(conversations, operation):
    first = conversations.create("reader")
    second = conversations.create("writer")
    path = str(conversations.path)
    with agent_session_scope(first):
        await invoke(file_tools.read_file, path=path)

    with agent_session_scope(second):
        if operation == "write":
            result = await invoke(file_tools.write_file, path=path, content="wrong owner")
        elif operation == "append":
            result = await invoke(file_tools.append_file, path=path, content="wrong owner")
        else:
            result = await invoke(
                file_tools.edit_file,
                path=path,
                old_string="original file body",
                new_string="wrong owner",
            )

    assert "has not been read" in result
    assert conversations.path.read_text(encoding="utf-8") == "original file body\n"


@pytest.mark.asyncio
async def test_read_state_survives_a_new_turn_of_the_same_session(conversations):
    session = conversations.create("same-session")
    with agent_session_scope(session):
        await invoke(file_tools.read_file, path=str(conversations.path))
        first_state = file_tools.get_file_state()
    with agent_session_scope(session):
        assert file_tools.get_file_state() is first_state
        assert "prior content still in context" in await invoke(
            file_tools.read_file, path=str(conversations.path)
        )


@pytest.mark.asyncio
async def test_unscoped_reads_do_not_authorize_a_managed_session(conversations):
    await invoke(file_tools.read_file, path=str(conversations.path))
    with agent_session_scope(conversations.create("managed")):
        result = await invoke(
            file_tools.write_file, path=str(conversations.path), content="wrong owner"
        )
    assert "has not been read" in result
    assert conversations.path.read_text(encoding="utf-8") == "original file body\n"


@pytest.mark.asyncio
async def test_concurrent_sessions_do_not_share_read_evidence(conversations):
    first = conversations.create("first")
    second = conversations.create("second")
    first_read = asyncio.Event()

    async def read_first():
        with agent_session_scope(first):
            result = await invoke(file_tools.read_file, path=str(conversations.path))
            first_read.set()
            return result

    async def read_second():
        await first_read.wait()
        with agent_session_scope(second):
            return await invoke(file_tools.read_file, path=str(conversations.path))

    results = await asyncio.gather(read_first(), read_second())
    assert all("original file body" in result for result in results)


@pytest.mark.asyncio
async def test_closed_session_cannot_reuse_file_read_evidence(conversations):
    session = conversations.create("closed")
    with agent_session_scope(session):
        await invoke(file_tools.read_file, path=str(conversations.path))
    session.close()
    with agent_session_scope(session):
        result = await invoke(
            file_tools.write_file, path=str(conversations.path), content="closed owner"
        )
    assert "closed" in result.lower()
    assert conversations.path.read_text(encoding="utf-8") == "original file body\n"


@pytest.mark.parametrize("create_new", [False, True])
@pytest.mark.asyncio
async def test_retired_turn_cannot_write_or_create_files(conversations, create_new):
    session = conversations.create("retired")
    release = asyncio.Event()
    target = conversations.root / "new-dir/new.txt" if create_new else conversations.path

    async def late_write():
        await release.wait()
        return await invoke(file_tools.write_file, path=str(target), content="retired owner")

    with agent_session_scope(session):
        await invoke(file_tools.read_file, path=str(conversations.path))
        task = asyncio.create_task(late_write())
    try:
        release.set()
        result = await task
        assert "session" in result.lower()
        if create_new:
            assert not target.parent.exists()
        else:
            assert target.read_text(encoding="utf-8") == "original file body\n"
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_detached_session_scope_does_not_borrow_legacy_read_evidence(conversations):
    await invoke(file_tools.read_file, path=str(conversations.path))
    with agent_session_scope(None):
        result = await invoke(
            file_tools.write_file, path=str(conversations.path), content="detached owner"
        )
    assert conversations.path.read_text(encoding="utf-8") == "original file body\n"
    assert "session" in result.lower()


@pytest.mark.asyncio
async def test_thread_calls_keep_one_state_per_actual_session(conversations):
    first = conversations.create("first")
    second = conversations.create("second")
    with agent_session_scope(first):
        states = await asyncio.gather(
            *(asyncio.to_thread(file_tools.get_file_state) for _ in range(4))
        )
    with agent_session_scope(second):
        other = await asyncio.to_thread(file_tools.get_file_state)
    assert all(state is states[0] for state in states)
    assert other is not states[0]


@pytest.mark.parametrize("operation", ["replace", "clear", "pop", "micro"])
@pytest.mark.asyncio
async def test_history_changes_invalidate_only_the_owning_sessions_state(
    conversations, monkeypatch, operation
):
    first = conversations.create("first")
    second = conversations.create("second")
    monkeypatch.setenv("KODER_MICRO_COMPACT", "0")
    with agent_session_scope(first):
        output = await invoke(file_tools.read_file, path=str(conversations.path))
        first_state = file_tools.get_file_state()
    with agent_session_scope(second):
        await invoke(file_tools.read_file, path=str(conversations.path))
        second_state = file_tools.get_file_state()

    original = [
        {"role": "user", "content": "read the file"},
        {"type": "function_call", "call_id": "read", "name": "read_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "read", "output": output},
    ]
    await first.add_items(original if operation != "micro" else original[:1])
    # History operations address their explicit Session, not an ambient other
    # owner (as can happen in command handling outside a model turn).
    with agent_session_scope(second):
        if operation == "replace":
            await first.replace_items([{"role": "user", "content": "new context"}])
        elif operation == "clear":
            await first.clear_session()
        elif operation == "pop":
            assert await first.pop_item() == original[-1]
        else:
            monkeypatch.setenv("KODER_MICRO_COMPACT", "1")
            monkeypatch.setenv("KODER_MICRO_COMPACT_MAX_CHARS", "5")
            await first.add_items(original[1:])
            assert (await first.get_items())[-1]["output"] != output

    assert not first_state.has_been_read(str(conversations.path))
    assert second_state.has_been_read(str(conversations.path))
    with agent_session_scope(first):
        assert "original file body" in await invoke(
            file_tools.read_file, path=str(conversations.path)
        )
    with agent_session_scope(second):
        assert "prior content still in context" in await invoke(
            file_tools.read_file, path=str(conversations.path)
        )


@pytest.mark.asyncio
async def test_failed_atomic_replacement_keeps_original_read_evidence(conversations, monkeypatch):
    session = conversations.create("rollback")
    history = [{"role": "user", "content": "original history"}]
    await session.add_items(history)
    with agent_session_scope(session):
        await invoke(file_tools.read_file, path=str(conversations.path))
        state = file_tools.get_file_state()

    def fail_commit(_connection):
        raise RuntimeError("synthetic replacement failure")

    monkeypatch.setattr(session, "_before_replace_commit", fail_commit)
    with pytest.raises(RuntimeError, match="synthetic replacement"):
        await session.replace_items([{"role": "user", "content": "new history"}])

    assert await session.get_items() == history
    assert state.has_been_read(str(conversations.path))


@pytest.mark.asyncio
async def test_close_releases_only_its_own_file_evidence(conversations):
    first = conversations.create("first")
    second = conversations.create("second")
    with agent_session_scope(first):
        await invoke(file_tools.read_file, path=str(conversations.path))
        first_state = file_tools.get_file_state()
    with agent_session_scope(second):
        await invoke(file_tools.read_file, path=str(conversations.path))
        second_state = file_tools.get_file_state()
    first.close()
    assert not first_state.has_been_read(str(conversations.path))
    assert second_state.has_been_read(str(conversations.path))


@pytest.mark.asyncio
async def test_untruncated_append_preserves_existing_read_evidence(conversations, monkeypatch):
    session = conversations.create("append")
    monkeypatch.setenv("KODER_MICRO_COMPACT", "0")
    with agent_session_scope(session):
        await invoke(file_tools.read_file, path=str(conversations.path))
        state = file_tools.get_file_state()
    await session.add_items([{"role": "user", "content": "continue"}])
    assert state.has_been_read(str(conversations.path))
