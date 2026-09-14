"""Legacy-data problems must not take unrelated modern sessions offline."""

import asyncio
import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents import SQLiteSession

from koder_agent.core import scheduler as scheduler_module
from koder_agent.harness.plugins.context import get_plugin_root


def _scheduler(db_path):
    scheduler = scheduler_module.AgentScheduler.__new__(scheduler_module.AgentScheduler)
    scheduler.session = SimpleNamespace(db_path=str(db_path))
    scheduler._migration_done = False
    scheduler._agent_initialized = False
    scheduler.agent_definition = None
    scheduler.instructions_override = None
    scheduler.instructions_append = None
    scheduler.tools = []
    scheduler.plugin_root = get_plugin_root()
    return scheduler


@pytest.fixture
def agent_factory(monkeypatch):
    agent = SimpleNamespace(
        model=SimpleNamespace(context_window=32_768),
        model_settings=SimpleNamespace(max_tokens=1_024),
        _koder_mcp_servers=[],
    )
    factory = AsyncMock(return_value=agent)
    monkeypatch.setattr(scheduler_module, "create_dev_agent", factory)
    monkeypatch.setattr(scheduler_module, "get_model_name", lambda: "fixture-model")
    return factory


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["invalid_json", "conflicting_history"])
async def test_legacy_data_error_defers_import_without_blocking_agent(
    tmp_path, monkeypatch, capsys, agent_factory, problem
):
    db_path = tmp_path / "sessions.db"
    legacy = [{"role": "user", "content": "private legacy marker"}]
    raw = "{private legacy marker" if problem == "invalid_json" else json.dumps(legacy)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("CREATE TABLE ctx (sid TEXT PRIMARY KEY, msgs TEXT, title TEXT)")
        conn.execute("INSERT INTO ctx VALUES ('old', ?, 'old title')", (raw,))
    modern = SQLiteSession("old", db_path)
    current = [{"role": "user", "content": "independent modern history"}]
    try:
        await modern.add_items(current)
        migration = AsyncMock(wraps=scheduler_module.migrate_legacy_sessions)
        monkeypatch.setattr(scheduler_module, "migrate_legacy_sessions", migration)
        scheduler = _scheduler(db_path)

        await scheduler._ensure_agent_initialized()
        await scheduler._ensure_agent_initialized()

        assert scheduler._agent_initialized
        assert scheduler.dev_agent is agent_factory.return_value
        agent_factory.assert_awaited_once()
        migration.assert_awaited_once()
        assert await modern.get_items() == current
        with closing(sqlite3.connect(db_path)) as conn:
            assert conn.execute("SELECT msgs FROM ctx").fetchone()[0] == raw
            assert (
                conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE name = 'migration_status'"
                ).fetchone()
                is None
            )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.count("Legacy session import deferred") == 1
        assert "/backfill-sessions" in captured.err
        assert "private legacy marker" not in captured.err
    finally:
        modern.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        sqlite3.OperationalError("database is locked"),
        sqlite3.DatabaseError("modern database is corrupt"),
        ValueError("unclassified failure"),
        RuntimeError("unexpected failure"),
    ],
)
async def test_storage_and_unclassified_errors_still_abort_initialization(
    tmp_path, monkeypatch, capsys, agent_factory, error
):
    migration = AsyncMock(side_effect=error)
    monkeypatch.setattr(scheduler_module, "migrate_legacy_sessions", migration)
    scheduler = _scheduler(tmp_path / "sessions.db")

    with pytest.raises(type(error), match=str(error)):
        await scheduler._ensure_agent_initialized()

    assert not scheduler._migration_done
    assert not scheduler._agent_initialized
    agent_factory.assert_not_awaited()
    assert not capsys.readouterr().err


@pytest.mark.asyncio
async def test_cancelled_import_does_not_start_an_agent_or_mark_the_attempt_done(
    tmp_path, monkeypatch, agent_factory
):
    migration = AsyncMock(side_effect=asyncio.CancelledError)
    monkeypatch.setattr(scheduler_module, "migrate_legacy_sessions", migration)
    scheduler = _scheduler(tmp_path / "sessions.db")

    with pytest.raises(asyncio.CancelledError):
        await scheduler._ensure_agent_initialized()

    assert not scheduler._migration_done
    assert not scheduler._agent_initialized
    agent_factory.assert_not_awaited()
