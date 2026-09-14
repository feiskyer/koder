from __future__ import annotations

import pytest

from koder_agent.harness import session_flow


class FakeSession:
    """Minimal EnhancedSQLiteSession stand-in for scoping by cwd."""

    _cwd_map: dict[str, str] = {}

    def __init__(self, session_id):
        self.session_id = session_id

    async def get_cwd(self):
        return type(self)._cwd_map.get(self.session_id)


@pytest.mark.asyncio
async def test_resolve_resume_value_exact_id(monkeypatch):
    async def fake_list():
        return [("2026-01-01-abc", "My Session"), ("2026-01-02-def", "Other")]

    monkeypatch.setattr(
        "koder_agent.core.session.EnhancedSQLiteSession.list_sessions_with_titles",
        staticmethod(fake_list),
    )
    resolved = await session_flow._resolve_resume_value("2026-01-01-abc")
    assert resolved == "2026-01-01-abc"


@pytest.mark.asyncio
async def test_resolve_resume_value_title_lookup(monkeypatch):
    async def fake_list():
        return [("2026-01-01-abc", "My Session"), ("2026-01-02-def", "Other")]

    monkeypatch.setattr(
        "koder_agent.core.session.EnhancedSQLiteSession.list_sessions_with_titles",
        staticmethod(fake_list),
    )
    resolved = await session_flow._resolve_resume_value("My Session")
    assert resolved == "2026-01-01-abc"


@pytest.mark.asyncio
async def test_resolve_resume_value_unknown(monkeypatch):
    async def fake_list():
        return [("2026-01-01-abc", "My Session")]

    monkeypatch.setattr(
        "koder_agent.core.session.EnhancedSQLiteSession.list_sessions_with_titles",
        staticmethod(fake_list),
    )
    resolved = await session_flow._resolve_resume_value("does-not-exist")
    assert resolved is None


@pytest.mark.asyncio
async def test_resolve_resume_value_ambiguous_title(monkeypatch):
    async def fake_list():
        return [("id-1", "Dup"), ("id-2", "Dup")]

    monkeypatch.setattr(
        "koder_agent.core.session.EnhancedSQLiteSession.list_sessions_with_titles",
        staticmethod(fake_list),
    )
    resolved = await session_flow._resolve_resume_value("Dup")
    assert resolved is None


@pytest.mark.asyncio
async def test_resolve_resume_value_finds_metadata_free_history(tmp_path, monkeypatch):
    from koder_agent.core.session import EnhancedSQLiteSession

    home = tmp_path / "home"
    (home / ".koder").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    session = EnhancedSQLiteSession("stored-without-title")
    items = [{"role": "user", "content": "Resume the persisted conversation."}]
    try:
        await session.add_items(items)
    finally:
        session.close()

    assert (
        await session_flow._resolve_resume_value("stored-without-title") == "stored-without-title"
    )
    resumed = EnhancedSQLiteSession("stored-without-title")
    try:
        assert await resumed.get_items() == items
    finally:
        resumed.close()
