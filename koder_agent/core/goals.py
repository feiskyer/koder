"""Persistent session goals: state model and SQLite store.

A goal is a long-running objective attached to a session. At most one goal row
exists per session (``session_id`` is the primary key). Goals carry an optional
token budget; accounting flips an active goal to ``budget_limited`` when the
budget is crossed.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import aiosqlite

from ..utils.async_tasks import await_owned_task
from ..utils.sqlite_connections import close_sqlite_connection

MAX_GOAL_OBJECTIVE_CHARS = 4_000


class GoalStatus(str, Enum):
    """Lifecycle status of a session goal."""

    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    USAGE_LIMITED = "usage_limited"
    BUDGET_LIMITED = "budget_limited"
    COMPLETE = "complete"

    @classmethod
    def from_str(cls, value: str) -> "GoalStatus":
        for status in cls:
            if status.value == value:
                return status
        raise ValueError(f"unknown goal status `{value}`")

    def is_active(self) -> bool:
        return self is GoalStatus.ACTIVE

    def is_terminal(self) -> bool:
        return self in (GoalStatus.BUDGET_LIMITED, GoalStatus.COMPLETE)


_ALL_STATUS_VALUES = tuple(status.value for status in GoalStatus)
_STOPPED_STATUS_VALUES = (
    GoalStatus.ACTIVE.value,
    GoalStatus.PAUSED.value,
    GoalStatus.BLOCKED.value,
    GoalStatus.USAGE_LIMITED.value,
    GoalStatus.BUDGET_LIMITED.value,
)


class GoalAccountingMode(Enum):
    """Which goal statuses a usage-accounting write may update."""

    ACTIVE_STATUS_ONLY = "active_status_only"
    ACTIVE_ONLY = "active_only"
    ACTIVE_OR_COMPLETE = "active_or_complete"
    ACTIVE_OR_STOPPED = "active_or_stopped"


@dataclass(frozen=True)
class Goal:
    """A session goal row."""

    session_id: str
    goal_id: str
    objective: str
    status: GoalStatus
    token_budget: Optional[int]
    tokens_used: int
    time_used_seconds: int
    created_at_ms: int
    updated_at_ms: int

    @property
    def remaining_tokens(self) -> Optional[int]:
        if self.token_budget is None:
            return None
        return max(self.token_budget - self.tokens_used, 0)


_UNSET = object()


@dataclass
class GoalUpdate:
    """Partial update; unset fields are left unchanged.

    ``token_budget`` distinguishes "leave unchanged" (default sentinel) from
    "clear the budget" (``None``) and "set to n" (an int).
    """

    objective: Optional[str] = None
    status: Optional[GoalStatus] = None
    token_budget: object = _UNSET
    expected_goal_id: Optional[str] = None

    @property
    def has_budget_update(self) -> bool:
        return self.token_budget is not _UNSET


class GoalAccountingOutcome:
    """Result of a usage-accounting write."""

    __slots__ = ("updated", "goal")

    def __init__(self, updated: bool, goal: Optional[Goal]):
        self.updated = updated
        self.goal = goal


def validate_goal_objective(value: str) -> None:
    """Raise ``ValueError`` when the objective is empty or oversized."""
    if not value:
        raise ValueError("goal objective must not be empty")
    if len(value) > MAX_GOAL_OBJECTIVE_CHARS:
        raise ValueError(f"goal objective must be at most {MAX_GOAL_OBJECTIVE_CHARS} characters")


def validate_goal_budget(value: Optional[int]) -> None:
    """Raise ``ValueError`` when a provided budget is not positive."""
    if value is not None and value <= 0:
        raise ValueError("goal budgets must be positive when provided")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _status_after_budget_limit(
    status: GoalStatus, tokens_used: int, token_budget: Optional[int]
) -> GoalStatus:
    """Active goals already at/over budget are stored as budget_limited."""
    if status is GoalStatus.ACTIVE and token_budget is not None and tokens_used >= token_budget:
        return GoalStatus.BUDGET_LIMITED
    return status


_GOAL_COLUMNS = (
    "session_id, goal_id, objective, status, token_budget, "
    "tokens_used, time_used_seconds, created_at_ms, updated_at_ms"
)


def _goal_from_row(row) -> Goal:
    return Goal(
        session_id=row[0],
        goal_id=row[1],
        objective=row[2],
        status=GoalStatus.from_str(row[3]),
        token_budget=row[4],
        tokens_used=row[5],
        time_used_seconds=row[6],
        created_at_ms=row[7],
        updated_at_ms=row[8],
    )


class GoalStore:
    """aiosqlite-backed store for session goals.

    One row per session (``session_id`` primary key). Uses a single lazily
    opened connection so ``:memory:`` databases keep their contents across
    calls; an asyncio lock serializes access.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            koder_dir = Path.home() / ".koder"
            koder_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(koder_dir / "koder.db")
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def _connection(self) -> aiosqlite.Connection:
        if self._conn is None:
            conn = aiosqlite.connect(self.db_path)
            status_list = ", ".join(f"'{value}'" for value in _ALL_STATUS_VALUES)
            try:
                # Retain ownership before the first await: cancelling connect()
                # itself can lose a connection still being opened by its worker.
                await await_owned_task(asyncio.ensure_future(conn))
                cursor = await conn.execute(f"""CREATE TABLE IF NOT EXISTS session_goals (
                    session_id TEXT PRIMARY KEY NOT NULL,
                    goal_id TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ({status_list})),
                    token_budget INTEGER,
                    tokens_used INTEGER NOT NULL DEFAULT 0,
                    time_used_seconds INTEGER NOT NULL DEFAULT 0,
                    created_at_ms INTEGER NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                )""")
                await cursor.close()
                await conn.commit()
            except BaseException:
                await await_owned_task(asyncio.create_task(close_sqlite_connection(conn)))
                raise
            self._conn = conn
        return self._conn

    async def close(self) -> None:
        async with self._lock:
            if self._conn is not None:
                conn = self._conn
                self._conn = None
                await await_owned_task(asyncio.create_task(close_sqlite_connection(conn)))

    async def _rollback_or_close(self, conn: aiosqlite.Connection) -> None:
        try:
            await conn.rollback()
        except BaseException:
            # A connection with an unresolvable transaction cannot be reused.
            self._conn = None
            await close_sqlite_connection(conn)
            raise

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Keep writes, their result snapshot, and failure cleanup under one lock.

        Rollback is queued after any interrupted SQLite work and joined before
        the next caller can use the connection. A commit already executed by
        SQLite cannot be undone if cancellation races its completion.
        """
        async with self._lock:
            conn = await self._connection()
            try:
                yield conn
                await conn.commit()
            except BaseException:
                await await_owned_task(asyncio.create_task(self._rollback_or_close(conn)))
                raise

    async def _fetch_goal(self, conn: aiosqlite.Connection, session_id: str) -> Optional[Goal]:
        async with conn.execute(
            f"SELECT {_GOAL_COLUMNS} FROM session_goals WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return _goal_from_row(row) if row else None

    async def get_goal(self, session_id: str) -> Optional[Goal]:
        async with self._lock:
            conn = await self._connection()
            return await self._fetch_goal(conn, session_id)

    async def replace_goal(
        self,
        session_id: str,
        objective: str,
        status: GoalStatus,
        token_budget: Optional[int],
    ) -> Goal:
        """Create or unconditionally replace the session goal.

        The replacement gets a fresh goal_id and zeroed usage counters.
        """
        goal_id = str(uuid.uuid4())
        now_ms = _now_ms()
        status = _status_after_budget_limit(status, 0, token_budget)
        async with self._transaction() as conn:
            cursor = await conn.execute(
                f"""INSERT INTO session_goals (
                    session_id, goal_id, objective, status, token_budget,
                    tokens_used, time_used_seconds, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, 0, 0, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    goal_id = excluded.goal_id,
                    objective = excluded.objective,
                    status = excluded.status,
                    token_budget = excluded.token_budget,
                    tokens_used = 0,
                    time_used_seconds = 0,
                    created_at_ms = excluded.created_at_ms,
                    updated_at_ms = excluded.updated_at_ms
                RETURNING {_GOAL_COLUMNS}""",
                (session_id, goal_id, objective, status.value, token_budget, now_ms, now_ms),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return _goal_from_row(row)

    async def insert_goal(
        self,
        session_id: str,
        objective: str,
        status: GoalStatus,
        token_budget: Optional[int],
    ) -> Optional[Goal]:
        """Insert a goal, replacing only completed goals.

        Returns ``None`` (and leaves the row untouched) when an unfinished goal
        already exists for the session.
        """
        goal_id = str(uuid.uuid4())
        now_ms = _now_ms()
        status = _status_after_budget_limit(status, 0, token_budget)
        async with self._transaction() as conn:
            cursor = await conn.execute(
                f"""INSERT INTO session_goals (
                    session_id, goal_id, objective, status, token_budget,
                    tokens_used, time_used_seconds, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, 0, 0, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    goal_id = excluded.goal_id,
                    objective = excluded.objective,
                    status = excluded.status,
                    token_budget = excluded.token_budget,
                    tokens_used = 0,
                    time_used_seconds = 0,
                    created_at_ms = excluded.created_at_ms,
                    updated_at_ms = excluded.updated_at_ms
                WHERE session_goals.status = 'complete'
                RETURNING {_GOAL_COLUMNS}""",
                (session_id, goal_id, objective, status.value, token_budget, now_ms, now_ms),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return _goal_from_row(row) if row else None

    async def update_goal(self, session_id: str, update: GoalUpdate) -> Optional[Goal]:
        """Apply a partial update with optional goal_id CAS.

        Status rules:
        - Setting ``paused``/``blocked`` on a ``budget_limited`` goal keeps
          ``budget_limited`` (terminal budget status is preserved).
        - Setting ``active`` while ``tokens_used >= token_budget`` stores
          ``budget_limited`` instead.
        - Lowering the budget below ``tokens_used`` flips an ``active`` goal to
          ``budget_limited`` immediately.

        Returns ``None`` when no row matched (missing goal or CAS mismatch).
        """
        objective = update.objective
        status = update.status
        expected = update.expected_goal_id
        now_ms = _now_ms()

        async with self._transaction() as conn:
            if status is not None and update.has_budget_update:
                token_budget = update.token_budget
                cursor = await conn.execute(
                    """UPDATE session_goals
                    SET
                        objective = COALESCE(?, objective),
                        status = CASE
                            WHEN status = 'budget_limited' AND ? IN ('paused', 'blocked')
                                THEN status
                            WHEN ? = 'active' AND ? IS NOT NULL AND tokens_used >= ?
                                THEN 'budget_limited'
                            ELSE ?
                        END,
                        token_budget = ?,
                        updated_at_ms = ?
                    WHERE session_id = ?
                      AND (? IS NULL OR goal_id = ?)""",
                    (
                        objective,
                        status.value,
                        status.value,
                        token_budget,
                        token_budget,
                        status.value,
                        token_budget,
                        now_ms,
                        session_id,
                        expected,
                        expected,
                    ),
                )
            elif status is not None:
                cursor = await conn.execute(
                    """UPDATE session_goals
                    SET
                        objective = COALESCE(?, objective),
                        status = CASE
                            WHEN status = 'budget_limited' AND ? IN ('paused', 'blocked')
                                THEN status
                            WHEN ? = 'active' AND token_budget IS NOT NULL
                                 AND tokens_used >= token_budget
                                THEN 'budget_limited'
                            ELSE ?
                        END,
                        updated_at_ms = ?
                    WHERE session_id = ?
                      AND (? IS NULL OR goal_id = ?)""",
                    (
                        objective,
                        status.value,
                        status.value,
                        status.value,
                        now_ms,
                        session_id,
                        expected,
                        expected,
                    ),
                )
            elif update.has_budget_update:
                token_budget = update.token_budget
                cursor = await conn.execute(
                    """UPDATE session_goals
                    SET
                        objective = COALESCE(?, objective),
                        token_budget = ?,
                        status = CASE
                            WHEN status = 'active' AND ? IS NOT NULL AND tokens_used >= ?
                                THEN 'budget_limited'
                            ELSE status
                        END,
                        updated_at_ms = ?
                    WHERE session_id = ?
                      AND (? IS NULL OR goal_id = ?)""",
                    (
                        objective,
                        token_budget,
                        token_budget,
                        token_budget,
                        now_ms,
                        session_id,
                        expected,
                        expected,
                    ),
                )
            elif objective is not None:
                cursor = await conn.execute(
                    """UPDATE session_goals
                    SET objective = ?, updated_at_ms = ?
                    WHERE session_id = ?
                      AND (? IS NULL OR goal_id = ?)""",
                    (objective, now_ms, session_id, expected, expected),
                )
            else:
                goal = await self._fetch_goal(conn, session_id)
                if goal is not None and expected is not None and goal.goal_id != expected:
                    return None
                return goal

            rows_affected = cursor.rowcount
            await cursor.close()
            if rows_affected == 0:
                return None
            return await self._fetch_goal(conn, session_id)

    async def pause_active_goal(self, session_id: str) -> Optional[Goal]:
        return await self._update_active_goal_status(session_id, GoalStatus.PAUSED)

    async def usage_limit_active_goal(self, session_id: str) -> Optional[Goal]:
        return await self._update_active_goal_status(session_id, GoalStatus.USAGE_LIMITED)

    async def _update_active_goal_status(
        self, session_id: str, status: GoalStatus
    ) -> Optional[Goal]:
        """Set the status of the active goal only.

        ``usage_limited`` may additionally supersede ``budget_limited`` so the
        stronger stop reason stays visible.
        """
        now_ms = _now_ms()
        async with self._transaction() as conn:
            cursor = await conn.execute(
                """UPDATE session_goals
                SET status = ?, updated_at_ms = ?
                WHERE session_id = ?
                  AND (
                      status = 'active'
                      OR (? = 'usage_limited' AND status = 'budget_limited')
                  )""",
                (status.value, now_ms, session_id, status.value),
            )
            rows_affected = cursor.rowcount
            await cursor.close()
            if rows_affected == 0:
                return None
            return await self._fetch_goal(conn, session_id)

    async def delete_goal(self, session_id: str) -> Optional[Goal]:
        async with self._transaction() as conn:
            cursor = await conn.execute(
                f"""DELETE FROM session_goals
                WHERE session_id = ?
                RETURNING {_GOAL_COLUMNS}""",
                (session_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return _goal_from_row(row) if row else None

    async def account_usage(
        self,
        session_id: str,
        time_delta_seconds: int,
        token_delta: int,
        mode: GoalAccountingMode,
        expected_goal_id: Optional[str] = None,
    ) -> GoalAccountingOutcome:
        """Add usage deltas to the goal, flipping to budget_limited on crossing.

        Negative deltas are clamped to zero. Which statuses may be charged (and
        which may trip the budget limit) depends on ``mode``.
        """
        time_delta_seconds = max(time_delta_seconds, 0)
        token_delta = max(token_delta, 0)
        if time_delta_seconds == 0 and token_delta == 0:
            return GoalAccountingOutcome(updated=False, goal=await self.get_goal(session_id))

        stopped_values = ", ".join(f"'{value}'" for value in _STOPPED_STATUS_VALUES)
        if mode is GoalAccountingMode.ACTIVE_STATUS_ONLY:
            status_filter = "status = 'active'"
        elif mode is GoalAccountingMode.ACTIVE_ONLY:
            status_filter = "status IN ('active', 'budget_limited')"
        elif mode is GoalAccountingMode.ACTIVE_OR_COMPLETE:
            status_filter = "status IN ('active', 'budget_limited', 'complete')"
        else:
            status_filter = f"status IN ({stopped_values})"

        if mode is GoalAccountingMode.ACTIVE_OR_STOPPED:
            budget_limit_filter = f"status IN ({stopped_values})"
        else:
            budget_limit_filter = "status = 'active'"

        now_ms = _now_ms()
        params: list = [time_delta_seconds, token_delta, token_delta, now_ms, session_id]
        expected_clause = ""
        if expected_goal_id is not None:
            expected_clause = " AND goal_id = ?"
            params.append(expected_goal_id)

        async with self._transaction() as conn:
            cursor = await conn.execute(
                f"""UPDATE session_goals
                SET
                    time_used_seconds = time_used_seconds + ?,
                    tokens_used = tokens_used + ?,
                    status = CASE
                        WHEN {budget_limit_filter}
                             AND token_budget IS NOT NULL
                             AND tokens_used + ? >= token_budget
                            THEN 'budget_limited'
                        ELSE status
                    END,
                    updated_at_ms = ?
                WHERE session_id = ? AND {status_filter}{expected_clause}
                RETURNING {_GOAL_COLUMNS}""",
                tuple(params),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                current = await self._fetch_goal(conn, session_id)
                return GoalAccountingOutcome(updated=False, goal=current)
        return GoalAccountingOutcome(updated=True, goal=_goal_from_row(row))


__all__ = [
    "Goal",
    "GoalAccountingMode",
    "GoalAccountingOutcome",
    "GoalStatus",
    "GoalStore",
    "GoalUpdate",
    "MAX_GOAL_OBJECTIVE_CHARS",
    "validate_goal_budget",
    "validate_goal_objective",
]
