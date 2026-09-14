"""Goal tool tests (get_goal / create_goal / update_goal)."""

import json

import pytest
import pytest_asyncio

from koder_agent.core.goal_runtime import GoalRuntime
from koder_agent.core.goals import GoalStatus, GoalStore, GoalUpdate
from koder_agent.tools.goal import (
    COMPLETION_BUDGET_REPORT,
    create_goal,
    get_goal,
    reset_goal_context,
    set_goal_context,
    update_goal,
)

SESSION = "session-tools"


@pytest_asyncio.fixture
async def store(tmp_path):
    instance = GoalStore(db_path=str(tmp_path / "goals.db"))
    try:
        yield instance
    finally:
        await instance.close()


@pytest.fixture
def runtime(store):
    runtime = GoalRuntime(session_id=SESSION, store=store)
    token = set_goal_context(runtime)
    yield runtime
    reset_goal_context(token)


async def invoke(tool, payload: dict) -> str:
    return await tool.on_invoke_tool(None, json.dumps(payload))


class TestGetGoal:
    @pytest.mark.asyncio
    async def test_get_goal_without_goal_returns_null_goal(self, runtime):
        result = json.loads(await invoke(get_goal, {}))
        assert result == {
            "goal": None,
            "remainingTokens": None,
            "completionBudgetReport": None,
        }

    @pytest.mark.asyncio
    async def test_get_goal_reports_remaining_tokens(self, store, runtime):
        await store.replace_goal(SESSION, "do the thing", GoalStatus.ACTIVE, token_budget=10_000)
        from koder_agent.core.goals import GoalAccountingMode

        await store.account_usage(
            SESSION, time_delta_seconds=5, token_delta=1_500, mode=GoalAccountingMode.ACTIVE_ONLY
        )

        result = json.loads(await invoke(get_goal, {}))
        assert result["goal"]["objective"] == "do the thing"
        assert result["goal"]["status"] == "active"
        assert result["goal"]["tokenBudget"] == 10_000
        assert result["goal"]["tokensUsed"] == 1_500
        assert result["goal"]["timeUsedSeconds"] == 5
        assert result["remainingTokens"] == 8_500
        assert result["completionBudgetReport"] is None

    @pytest.mark.asyncio
    async def test_get_goal_without_context_returns_error(self):
        result = await invoke(get_goal, {})
        assert "goals are not available" in result


class TestCreateGoal:
    @pytest.mark.asyncio
    async def test_create_goal_creates_active_goal(self, store, runtime):
        result = json.loads(
            await invoke(create_goal, {"objective": "ship the feature", "token_budget": 50_000})
        )
        assert result["goal"]["objective"] == "ship the feature"
        assert result["goal"]["status"] == "active"
        assert result["goal"]["tokenBudget"] == 50_000
        assert result["remainingTokens"] == 50_000

        goal = await store.get_goal(SESSION)
        assert goal is not None
        assert goal.objective == "ship the feature"
        # The creating turn is marked so it is not charged.
        assert runtime._turn_goal_id == goal.goal_id
        assert runtime._goal_created_this_turn

    @pytest.mark.asyncio
    async def test_create_goal_trims_objective(self, store, runtime):
        await invoke(create_goal, {"objective": "  padded objective  "})
        goal = await store.get_goal(SESSION)
        assert goal.objective == "padded objective"

    @pytest.mark.asyncio
    async def test_create_goal_rejects_unfinished_existing_goal(self, store, runtime):
        # Creating a goal cannot overwrite unfinished work.
        await store.replace_goal(SESSION, "existing", GoalStatus.ACTIVE, token_budget=None)
        result = await invoke(create_goal, {"objective": "replacement"})
        assert "unfinished goal" in result
        assert (await store.get_goal(SESSION)).objective == "existing"

    @pytest.mark.asyncio
    async def test_create_goal_replaces_complete_goal(self, store, runtime):
        await store.replace_goal(SESSION, "old", GoalStatus.COMPLETE, token_budget=None)
        result = json.loads(await invoke(create_goal, {"objective": "new goal"}))
        assert result["goal"]["objective"] == "new goal"
        assert result["goal"]["status"] == "active"

    @pytest.mark.asyncio
    async def test_create_goal_rejects_empty_objective(self, runtime):
        result = await invoke(create_goal, {"objective": "   "})
        assert "must not be empty" in result

    @pytest.mark.asyncio
    async def test_create_goal_rejects_oversized_objective(self, runtime):
        result = await invoke(create_goal, {"objective": "x" * 4_001})
        assert "at most 4000 characters" in result

    @pytest.mark.asyncio
    async def test_create_goal_rejects_non_positive_budget(self, runtime):
        for budget in (0, -5):
            result = await invoke(create_goal, {"objective": "budgeted", "token_budget": budget})
            assert "must be positive" in result


class TestUpdateGoal:
    @pytest.mark.asyncio
    async def test_update_goal_marks_goal_complete_with_budget_report(self, store, runtime):
        await store.replace_goal(SESSION, "finish it", GoalStatus.ACTIVE, token_budget=10_000)
        from koder_agent.core.goals import GoalAccountingMode

        await store.account_usage(
            SESSION, time_delta_seconds=30, token_delta=4_000, mode=GoalAccountingMode.ACTIVE_ONLY
        )

        result = json.loads(await invoke(update_goal, {"status": "complete"}))
        assert result["goal"]["status"] == "complete"
        assert result["goal"]["tokensUsed"] == 4_000
        assert result["completionBudgetReport"] == COMPLETION_BUDGET_REPORT

    @pytest.mark.asyncio
    async def test_update_goal_complete_without_budget_or_time_omits_report(self, store, runtime):
        await store.replace_goal(SESSION, "finish it", GoalStatus.ACTIVE, token_budget=None)
        result = json.loads(await invoke(update_goal, {"status": "complete"}))
        assert result["goal"]["status"] == "complete"
        assert result["completionBudgetReport"] is None

    @pytest.mark.asyncio
    async def test_update_goal_blocked_omits_budget_report(self, store, runtime):
        # Blocking a goal preserves final accounting state.
        await store.replace_goal(SESSION, "stuck", GoalStatus.ACTIVE, token_budget=10_000)
        result = json.loads(await invoke(update_goal, {"status": "blocked"}))
        assert result["goal"]["status"] == "blocked"
        assert result["completionBudgetReport"] is None

    @pytest.mark.asyncio
    async def test_update_goal_rejects_non_terminal_statuses(self, store, runtime):
        await store.replace_goal(SESSION, "task", GoalStatus.ACTIVE, token_budget=None)
        for status in ("active", "paused", "usage_limited", "budget_limited"):
            result = await invoke(update_goal, {"status": status})
            assert "can only mark the existing goal complete or blocked" in result
        goal = await store.get_goal(SESSION)
        assert goal.status is GoalStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_update_goal_rejects_unknown_status(self, store, runtime):
        await store.replace_goal(SESSION, "task", GoalStatus.ACTIVE, token_budget=None)
        result = await invoke(update_goal, {"status": "bogus"})
        assert "unknown goal status" in result

    @pytest.mark.asyncio
    async def test_update_goal_without_goal_returns_error(self, runtime):
        result = await invoke(update_goal, {"status": "complete"})
        assert "cannot update goal because this thread has no goal" in result

    @pytest.mark.asyncio
    async def test_blocking_budget_limited_goal_preserves_terminal_status(self, store, runtime):
        await store.replace_goal(SESSION, "over budget", GoalStatus.BUDGET_LIMITED, token_budget=10)
        result = json.loads(await invoke(update_goal, {"status": "blocked"}))
        assert result["goal"]["status"] == "budget_limited"


class TestToolRegistration:
    def test_goal_tools_are_registered(self):
        from koder_agent.tools import get_all_tools

        names = {tool.name for tool in get_all_tools()}
        assert {"get_goal", "create_goal", "update_goal"} <= names


class TestSchedulerIntegration:
    @pytest.mark.asyncio
    async def test_scheduler_continuation_loop(self, tmp_path, monkeypatch):
        """A goal completed via update_goal stops the continuation loop."""
        from unittest.mock import AsyncMock, patch

        monkeypatch.setenv("HOME", str(tmp_path))

        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as session_cls,
            patch("koder_agent.core.scheduler.get_companion", return_value=None),
        ):
            session = AsyncMock()
            session.db_path = str(tmp_path / "koder.db")
            session.session_id = "goal-loop"
            session.get_items = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
            session_cls.return_value = session

            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="goal-loop")
            scheduler.dev_agent = object()
            scheduler._agent_initialized = True
            scheduler._migration_done = True
            scheduler._capture_usage = AsyncMock()
            scheduler._refresh_magic_docs_after_turn = AsyncMock()
            scheduler._repair_unreplayable_session_items = AsyncMock()
            scheduler._load_memory_context = AsyncMock(return_value="")

            goal = await scheduler.goal_store.replace_goal(
                "goal-loop", "finish everything", GoalStatus.ACTIVE, token_budget=None
            )

            calls = []

            async def fake_run(agent, user_input, **kwargs):
                calls.append(user_input)
                if len(calls) >= 2:
                    # Second (continuation) turn completes the goal.
                    await scheduler.goal_store.update_goal(
                        "goal-loop", GoalUpdate(status=GoalStatus.COMPLETE)
                    )
                result = AsyncMock()
                result.final_output = f"turn {len(calls)}"
                return result

            with patch("koder_agent.core.scheduler.Runner.run", side_effect=fake_run):
                response = await scheduler.handle("start working", render_output=False)

            # Turn 1: user prompt; turn 2: hidden continuation; then complete.
            assert len(calls) == 2
            assert calls[0] == "start working"
            assert calls[1].startswith("[Goal continuation]")
            assert "finish everything" in calls[1]
            assert response == "turn 2"

            final = await scheduler.goal_store.get_goal("goal-loop")
            assert final.status is GoalStatus.COMPLETE
            assert final.goal_id == goal.goal_id
            await scheduler.goal_store.close()

    @pytest.mark.asyncio
    async def test_scheduler_continuation_respects_backstop_cap(self, tmp_path, monkeypatch):
        """The loop stops at the configured cap when a goal never completes."""
        from unittest.mock import AsyncMock, patch

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("KODER_GOAL_MAX_CONTINUATIONS", "3")

        with (
            patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
            patch("koder_agent.core.scheduler.get_display_hooks"),
            patch("koder_agent.core.scheduler.ApprovalHooks"),
            patch("koder_agent.core.scheduler.EnhancedSQLiteSession") as session_cls,
            patch("koder_agent.core.scheduler.get_companion", return_value=None),
        ):
            session = AsyncMock()
            session.db_path = str(tmp_path / "koder.db")
            session.session_id = "goal-cap"
            session.get_items = AsyncMock(return_value=[{"role": "user", "content": "hi"}])
            session_cls.return_value = session

            from koder_agent.core.scheduler import AgentScheduler

            scheduler = AgentScheduler(session_id="goal-cap")
            scheduler.dev_agent = object()
            scheduler._agent_initialized = True
            scheduler._migration_done = True
            scheduler._capture_usage = AsyncMock()
            scheduler._refresh_magic_docs_after_turn = AsyncMock()
            scheduler._repair_unreplayable_session_items = AsyncMock()
            scheduler._load_memory_context = AsyncMock(return_value="")

            await scheduler.goal_store.replace_goal(
                "goal-cap", "never ends", GoalStatus.ACTIVE, token_budget=None
            )

            calls = []

            async def fake_run(agent, user_input, **kwargs):
                calls.append(user_input)
                result = AsyncMock()
                result.final_output = "still working"
                return result

            with patch("koder_agent.core.scheduler.Runner.run", side_effect=fake_run):
                await scheduler.handle("start", render_output=False)

            # 1 user turn + 3 continuations (cap).
            assert len(calls) == 4
            await scheduler.goal_store.close()
