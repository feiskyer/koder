"""Seed metadata-free history for a real terminal test using explicit targets."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from contextlib import closing
from pathlib import Path


async def seed_history(database: Path, session_id: str) -> None:
    if not session_id.strip():
        raise ValueError("A nonempty fixture session id is required")
    with closing(sqlite3.connect(database)) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for table in ("agent_sessions", "agent_messages", "items", "session_metadata"):
            if (
                table in tables
                and connection.execute(
                    f"SELECT 1 FROM {table} WHERE session_id = ? LIMIT 1", (session_id,)
                ).fetchone()
            ):
                raise ValueError("Refusing to seed an existing session")

    from koder_agent.core.session import EnhancedSQLiteSession

    history = [
        {"role": "user", "content": "Untitled user message"},
        {"role": "assistant", "content": "Untitled assistant reply"},
    ]
    with closing(EnhancedSQLiteSession(session_id, str(database))) as session:
        await session.add_items(history)
        assert await session.get_items() == history

    with closing(sqlite3.connect(database)) as connection:
        has_metadata = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='session_metadata'"
        ).fetchone()
        if has_metadata:
            assert connection.execute(
                "SELECT COUNT(*) FROM session_metadata WHERE session_id = ?", (session_id,)
            ).fetchone() == (0,)
    print(f"untitled-fixture-ready: {session_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    arguments = parser.parse_args()
    asyncio.run(seed_history(arguments.database, arguments.session_id))


if __name__ == "__main__":
    main()
