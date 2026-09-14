"""Tests for the /goal slash command."""

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

# Stub litellm before importing koder_agent to avoid optional dependency issues
if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from koder_agent.core.goals import (  # noqa: E402
    MAX_GOAL_OBJECTIVE_CHARS,
    GoalStatus,
    GoalStore,
)
from koder_agent.harness.commands.interactive import (  # noqa: E402
    HarnessInteractiveCommandHandler,
)
from koder_agent.harness.commands.registry import CommandRegistry  # noqa: E402

SESSION = "goal-cmd-session"


class FakeScheduler:
    def __init__(self, tmp_path):
        self.goal_store = GoalStore(db_path=str(tmp_path / "goals.db"))
        self.session = types.SimpleNamespace(session_id=SESSION)
        self.handle = AsyncMock(return_value="turn done")


@pytest.fixture
def handler():
    return HarnessInteractiveCommandHandler(emit_console=False)


@pytest_asyncio.fixture
async def scheduler(tmp_path):
    instance = FakeScheduler(tmp_path)
    try:
        yield instance
    finally:
        await instance.goal_store.close()


def test_goal_command_is_registered():
    registry = CommandRegistry.with_defaults()
    assert "goal" in registry.list_names()
    spec = registry.get("goal")
    assert "goal" in spec.help_text.lower()


def test_goal_command_has_handler(handler):
    assert "goal" in handler.commands
    names = [name for name, _description in handler.get_command_list()]
    assert "goal" in names


class TestBareGoal:
    @pytest.mark.asyncio
    async def test_no_goal_shows_usage(self, handler, scheduler):
        result = await handler.handle_slash_input("/goal", scheduler)
        assert "No goal is currently set." in result
        assert "Usage: /goal" in result

    @pytest.mark.asyncio
    async def test_shows_goal_summary(self, handler, scheduler):
        await scheduler.goal_store.replace_goal(
            SESSION, "improve benchmark coverage", GoalStatus.ACTIVE, token_budget=50_000
        )
        result = await handler.handle_slash_input("/goal", scheduler)
        assert "Status: active" in result
        assert "Objective: improve benchmark coverage" in result
        assert "Token budget: 50K" in result
        assert "Commands: /goal edit, /goal pause, /goal clear" in result

    @pytest.mark.asyncio
    async def test_no_scheduler_reports_temporary_session(self, handler):
        result = await handler.handle_slash_input("/goal", None)
        assert "Goals need a saved session" in result


class TestSetGoal:
    @pytest.mark.asyncio
    async def test_sets_new_goal_and_starts_turn(self, handler, scheduler):
        result = await handler.handle_slash_input("/goal improve benchmark coverage", scheduler)
        assert "Goal active" in result
        assert "Objective: improve benchmark coverage" in result

        goal = await scheduler.goal_store.get_goal(SESSION)
        assert goal.objective == "improve benchmark coverage"
        assert goal.status is GoalStatus.ACTIVE

        # A hidden goal turn is kicked off immediately.
        scheduler.handle.assert_awaited_once()
        prompt = scheduler.handle.await_args.args[0]
        assert prompt.startswith("[Goal continuation]")
        assert "improve benchmark coverage" in prompt

    @pytest.mark.asyncio
    async def test_sets_goal_with_budget(self, handler, scheduler):
        result = await handler.handle_slash_input(
            "/goal port the feature --budget 50000", scheduler
        )
        assert "Goal active" in result
        goal = await scheduler.goal_store.get_goal(SESSION)
        assert goal.objective == "port the feature"
        assert goal.token_budget == 50_000

    @pytest.mark.asyncio
    async def test_rejects_invalid_budget(self, handler, scheduler):
        result = await handler.handle_slash_input("/goal task --budget abc", scheduler)
        assert "Invalid token budget" in result
        assert await scheduler.goal_store.get_goal(SESSION) is None

    @pytest.mark.asyncio
    async def test_rejects_non_positive_budget(self, handler, scheduler):
        result = await handler.handle_slash_input("/goal task --budget 0", scheduler)
        assert "must be positive" in result

    @pytest.mark.asyncio
    async def test_accepts_objective_at_limit(self, handler, scheduler):
        # Objective text at the limit remains valid.
        objective = "y" * MAX_GOAL_OBJECTIVE_CHARS
        result = await handler.handle_slash_input(f"/goal {objective}", scheduler)
        assert "Goal active" in result
        goal = await scheduler.goal_store.get_goal(SESSION)
        assert goal.objective == objective

    @pytest.mark.asyncio
    async def test_rejects_oversized_objective(self, handler, scheduler):
        # Oversized objectives are rejected before persistence.
        objective = "y" * (MAX_GOAL_OBJECTIVE_CHARS + 1)
        result = await handler.handle_slash_input(f"/goal {objective}", scheduler)
        assert "at most 4000 characters" in result
        assert await scheduler.goal_store.get_goal(SESSION) is None

    @pytest.mark.asyncio
    async def test_unfinished_goal_requires_replace_confirmation(self, handler, scheduler):
        await scheduler.goal_store.replace_goal(
            SESSION, "current objective", GoalStatus.ACTIVE, token_budget=None
        )
        result = await handler.handle_slash_input("/goal new objective", scheduler)
        assert "Replace goal?" in result
        assert "Current objective: current objective" in result
        assert "New objective: new objective" in result
        # Goal untouched until confirmation.
        goal = await scheduler.goal_store.get_goal(SESSION)
        assert goal.objective == "current objective"

    @pytest.mark.asyncio
    async def test_replace_subcommand_replaces_unfinished_goal(self, handler, scheduler):
        await scheduler.goal_store.replace_goal(
            SESSION, "current objective", GoalStatus.ACTIVE, token_budget=None
        )
        result = await handler.handle_slash_input("/goal replace new objective", scheduler)
        assert "Goal active" in result
        goal = await scheduler.goal_store.get_goal(SESSION)
        assert goal.objective == "new objective"
        assert goal.tokens_used == 0

    @pytest.mark.asyncio
    async def test_complete_goal_is_replaced_without_confirmation(self, handler, scheduler):
        # Completed goals can be replaced without confirmation.
        await scheduler.goal_store.replace_goal(
            SESSION, "done objective", GoalStatus.COMPLETE, token_budget=None
        )
        result = await handler.handle_slash_input("/goal fresh objective", scheduler)
        assert "Goal active" in result
        goal = await scheduler.goal_store.get_goal(SESSION)
        assert goal.objective == "fresh objective"


class TestGoalControls:
    @pytest.mark.asyncio
    async def test_pause_active_goal(self, handler, scheduler):
        await scheduler.goal_store.replace_goal(
            SESSION, "long task", GoalStatus.ACTIVE, token_budget=None
        )
        result = await handler.handle_slash_input("/goal pause", scheduler)
        assert "Goal paused" in result
        goal = await scheduler.goal_store.get_goal(SESSION)
        assert goal.status is GoalStatus.PAUSED

    @pytest.mark.asyncio
    async def test_pause_without_goal(self, handler, scheduler):
        result = await handler.handle_slash_input("/goal pause", scheduler)
        assert "No goal is currently set." in result

    @pytest.mark.asyncio
    async def test_pause_non_active_goal_reports_status(self, handler, scheduler):
        await scheduler.goal_store.replace_goal(
            SESSION, "done", GoalStatus.COMPLETE, token_budget=None
        )
        result = await handler.handle_slash_input("/goal pause", scheduler)
        assert "only active goals can be paused" in result

    @pytest.mark.asyncio
    async def test_resume_paused_goal_starts_turn(self, handler, scheduler):
        await scheduler.goal_store.replace_goal(
            SESSION, "long task", GoalStatus.ACTIVE, token_budget=None
        )
        await scheduler.goal_store.pause_active_goal(SESSION)

        result = await handler.handle_slash_input("/goal resume", scheduler)
        assert "Goal active" in result
        goal = await scheduler.goal_store.get_goal(SESSION)
        assert goal.status is GoalStatus.ACTIVE
        scheduler.handle.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_resume_budget_limited_goal_stays_budget_limited(self, handler, scheduler):
        # Activating a goal over budget keeps it budget_limited; no turn starts.
        await scheduler.goal_store.replace_goal(
            SESSION, "over budget", GoalStatus.ACTIVE, token_budget=10
        )
        from koder_agent.core.goals import GoalAccountingMode

        await scheduler.goal_store.account_usage(
            SESSION, time_delta_seconds=1, token_delta=20, mode=GoalAccountingMode.ACTIVE_ONLY
        )

        result = await handler.handle_slash_input("/goal resume", scheduler)
        assert "limited by budget" in result
        scheduler.handle.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clear_goal(self, handler, scheduler):
        await scheduler.goal_store.replace_goal(
            SESSION, "long task", GoalStatus.ACTIVE, token_budget=None
        )
        result = await handler.handle_slash_input("/goal clear", scheduler)
        assert result == "Goal cleared"
        assert await scheduler.goal_store.get_goal(SESSION) is None

    @pytest.mark.asyncio
    async def test_clear_without_goal(self, handler, scheduler):
        result = await handler.handle_slash_input("/goal clear", scheduler)
        assert "No goal to clear" in result
        assert "does not currently have a goal" in result


class TestGoalEdit:
    @pytest.mark.asyncio
    async def test_edit_updates_objective_preserving_usage(self, handler, scheduler):
        await scheduler.goal_store.replace_goal(
            SESSION, "draft the report", GoalStatus.ACTIVE, token_budget=100
        )
        from koder_agent.core.goals import GoalAccountingMode

        await scheduler.goal_store.account_usage(
            SESSION, time_delta_seconds=12, token_delta=30, mode=GoalAccountingMode.ACTIVE_ONLY
        )

        result = await handler.handle_slash_input("/goal edit draft the report clearly", scheduler)
        assert "Goal active" in result
        goal = await scheduler.goal_store.get_goal(SESSION)
        assert goal.objective == "draft the report clearly"
        assert goal.tokens_used == 30
        assert goal.time_used_seconds == 12

    @pytest.mark.asyncio
    async def test_edit_without_goal(self, handler, scheduler):
        result = await handler.handle_slash_input("/goal edit new objective", scheduler)
        assert "No goal is currently set." in result
        assert "Create a goal before editing it." in result

    @pytest.mark.asyncio
    async def test_edit_without_objective_shows_current(self, handler, scheduler):
        await scheduler.goal_store.replace_goal(
            SESSION, "current text", GoalStatus.ACTIVE, token_budget=None
        )
        result = await handler.handle_slash_input("/goal edit", scheduler)
        assert "Usage: /goal edit <objective>" in result
        assert "Current objective: current text" in result

    @pytest.mark.asyncio
    async def test_edit_complete_goal_reactivates_it(self, handler, scheduler):
        # Editing a budget-limited or complete goal reactivates it.
        await scheduler.goal_store.replace_goal(
            SESSION, "done", GoalStatus.COMPLETE, token_budget=None
        )
        result = await handler.handle_slash_input("/goal edit keep going", scheduler)
        assert "Goal active" in result
        goal = await scheduler.goal_store.get_goal(SESSION)
        assert goal.status is GoalStatus.ACTIVE
        scheduler.handle.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_edit_paused_goal_stays_paused(self, handler, scheduler):
        # Editing preserves paused, blocked, and usage-limited statuses.
        await scheduler.goal_store.replace_goal(
            SESSION, "long task", GoalStatus.ACTIVE, token_budget=None
        )
        await scheduler.goal_store.pause_active_goal(SESSION)

        result = await handler.handle_slash_input("/goal edit revised objective", scheduler)
        assert "Goal paused" in result
        goal = await scheduler.goal_store.get_goal(SESSION)
        assert goal.status is GoalStatus.PAUSED
        assert goal.objective == "revised objective"
        scheduler.handle.assert_not_awaited()


class TestGoalBudget:
    @pytest.mark.asyncio
    async def test_budget_updates_existing_goal(self, handler, scheduler):
        await scheduler.goal_store.replace_goal(
            SESSION, "long task", GoalStatus.ACTIVE, token_budget=None
        )
        result = await handler.handle_slash_input("/goal budget 75000", scheduler)
        assert "Goal active" in result
        goal = await scheduler.goal_store.get_goal(SESSION)
        assert goal.token_budget == 75_000

    @pytest.mark.asyncio
    async def test_budget_below_usage_stops_goal(self, handler, scheduler):
        # Lowering a budget below existing usage stops an active goal.
        await scheduler.goal_store.replace_goal(
            SESSION, "long task", GoalStatus.ACTIVE, token_budget=100
        )
        from koder_agent.core.goals import GoalAccountingMode

        await scheduler.goal_store.account_usage(
            SESSION, time_delta_seconds=1, token_delta=50, mode=GoalAccountingMode.ACTIVE_ONLY
        )
        result = await handler.handle_slash_input("/goal budget 40", scheduler)
        assert "limited by budget" in result
        goal = await scheduler.goal_store.get_goal(SESSION)
        assert goal.status is GoalStatus.BUDGET_LIMITED

    @pytest.mark.asyncio
    async def test_budget_requires_single_numeric_argument(self, handler, scheduler):
        assert "Usage: /goal budget" in await handler.handle_slash_input("/goal budget", scheduler)
        assert "Invalid token budget" in await handler.handle_slash_input(
            "/goal budget notanumber", scheduler
        )

    @pytest.mark.asyncio
    async def test_budget_without_goal(self, handler, scheduler):
        result = await handler.handle_slash_input("/goal budget 1000", scheduler)
        assert "No goal is currently set." in result
