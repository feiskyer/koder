"""GoalRuntime turn-accounting tests."""

import pytest
import pytest_asyncio

from koder_agent.core.goal_runtime import GoalRuntime
from koder_agent.core.goals import GoalAccountingMode, GoalStatus, GoalStore

SESSION = "session-runtime"


@pytest_asyncio.fixture
async def store(tmp_path):
    instance = GoalStore(db_path=str(tmp_path / "goals.db"))
    try:
        yield instance
    finally:
        await instance.close()


@pytest.fixture
def runtime(store):
    return GoalRuntime(session_id=SESSION, store=store)


class TestTurnAccounting:
    @pytest.mark.asyncio
    async def test_turn_charges_token_delta_against_active_goal(self, store, runtime):
        await store.replace_goal(SESSION, "long task", GoalStatus.ACTIVE, token_budget=1_000)

        await runtime.on_turn_start(cumulative_tokens=100)
        goal = await runtime.on_turn_end(cumulative_tokens=350)

        assert goal is not None
        assert goal.tokens_used == 250
        assert goal.status is GoalStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_turn_without_goal_charges_nothing(self, runtime):
        await runtime.on_turn_start(cumulative_tokens=0)
        assert await runtime.on_turn_end(cumulative_tokens=500) is None

    @pytest.mark.asyncio
    async def test_budget_crossing_flips_goal_to_budget_limited(self, store, runtime):
        await store.replace_goal(SESSION, "long task", GoalStatus.ACTIVE, token_budget=200)

        await runtime.on_turn_start(cumulative_tokens=0)
        goal = await runtime.on_turn_end(cumulative_tokens=250)

        assert goal is not None
        assert goal.status is GoalStatus.BUDGET_LIMITED
        assert goal.tokens_used == 250

    @pytest.mark.asyncio
    async def test_goal_created_this_turn_is_not_charged(self, store, runtime):
        # A goal created mid-turn starts accounting on the next turn.
        # tokens spent in the turn that created the goal are not charged.
        await runtime.on_turn_start(cumulative_tokens=100)
        goal = await store.insert_goal(SESSION, "fresh goal", GoalStatus.ACTIVE, token_budget=None)
        runtime.mark_goal_created(goal.goal_id)

        result = await runtime.on_turn_end(cumulative_tokens=900)
        assert result is not None
        assert result.tokens_used == 0
        assert result.time_used_seconds == 0

    @pytest.mark.asyncio
    async def test_negative_token_delta_is_clamped(self, store, runtime):
        await store.replace_goal(SESSION, "long task", GoalStatus.ACTIVE, token_budget=None)
        await runtime.on_turn_start(cumulative_tokens=500)
        goal = await runtime.on_turn_end(cumulative_tokens=100)
        # No usable delta: nothing charged, goal stays active.
        current = await store.get_goal(SESSION)
        assert current.tokens_used == 0
        assert goal is None or goal.tokens_used == 0

    @pytest.mark.asyncio
    async def test_paused_goal_is_not_bound_to_turn(self, store, runtime):
        await store.replace_goal(SESSION, "long task", GoalStatus.ACTIVE, token_budget=None)
        await store.pause_active_goal(SESSION)

        await runtime.on_turn_start(cumulative_tokens=0)
        assert await runtime.on_turn_end(cumulative_tokens=100) is None
        goal = await store.get_goal(SESSION)
        assert goal.tokens_used == 0

    @pytest.mark.asyncio
    async def test_budget_limited_goal_keeps_accruing_in_flight_usage(self, store, runtime):
        # Budget-limited goals keep accruing until the turn stops.
        await store.replace_goal(SESSION, "long task", GoalStatus.BUDGET_LIMITED, token_budget=100)

        await runtime.on_turn_start(cumulative_tokens=0)
        goal = await runtime.on_turn_end(cumulative_tokens=50)

        assert goal is not None
        assert goal.status is GoalStatus.BUDGET_LIMITED
        assert goal.tokens_used == 50

    @pytest.mark.asyncio
    async def test_goal_completed_mid_turn_still_accounts_final_usage(self, store, runtime):
        # Turn usage lands on the completed goal before accounting stops.
        await store.replace_goal(SESSION, "long task", GoalStatus.ACTIVE, token_budget=1_000)
        await runtime.on_turn_start(cumulative_tokens=0)

        # Model calls update_goal(complete) mid-turn.
        from koder_agent.core.goals import GoalUpdate

        await store.update_goal(SESSION, GoalUpdate(status=GoalStatus.COMPLETE))
        goal = await runtime.on_turn_end(cumulative_tokens=300)

        assert goal is not None
        assert goal.status is GoalStatus.COMPLETE
        assert goal.tokens_used == 300

    @pytest.mark.asyncio
    async def test_cancelled_turn_pauses_active_goal(self, store, runtime):
        await store.replace_goal(SESSION, "long task", GoalStatus.ACTIVE, token_budget=None)

        await runtime.on_turn_start(cumulative_tokens=0)
        goal = await runtime.on_turn_end(cumulative_tokens=100, cancelled=True)

        assert goal is not None
        assert goal.status is GoalStatus.PAUSED
        current = await store.get_goal(SESSION)
        assert current.tokens_used == 100

    @pytest.mark.asyncio
    async def test_turn_error_blocks_active_goal(self, store, runtime):
        # Terminal turn errors block the goal so
        # automatic continuation cannot loop.
        await store.replace_goal(SESSION, "long task", GoalStatus.ACTIVE, token_budget=None)

        await runtime.on_turn_start(cumulative_tokens=0)
        goal = await runtime.on_turn_end(cumulative_tokens=100, error=True)

        assert goal is not None
        assert goal.status is GoalStatus.BLOCKED

    @pytest.mark.asyncio
    async def test_replaced_goal_is_not_charged_for_stale_turn(self, store, runtime):
        original = await store.replace_goal(SESSION, "old", GoalStatus.ACTIVE, token_budget=100)
        await runtime.on_turn_start(cumulative_tokens=0)
        assert runtime._turn_goal_id == original.goal_id

        replacement = await store.replace_goal(SESSION, "new", GoalStatus.ACTIVE, token_budget=100)
        goal = await runtime.on_turn_end(cumulative_tokens=50)

        # CAS on goal_id: the replacement goal keeps zero usage.
        current = await store.get_goal(SESSION)
        assert current.goal_id == replacement.goal_id
        assert current.tokens_used == 0
        assert goal is None or goal.tokens_used == 0


class TestContinuationDecisions:
    @pytest.mark.asyncio
    async def test_active_goal_produces_continuation_prompt(self, store, runtime):
        await store.replace_goal(SESSION, "keep going", GoalStatus.ACTIVE, token_budget=5_000)
        prompt = await runtime.next_continuation_prompt()
        assert prompt is not None
        assert "Continue working toward the active thread goal." in prompt
        assert "keep going" in prompt
        assert "Token budget: 5000" in prompt

    @pytest.mark.asyncio
    async def test_no_goal_produces_no_prompt(self, runtime):
        assert await runtime.next_continuation_prompt() is None

    @pytest.mark.asyncio
    async def test_paused_blocked_usage_limited_complete_produce_no_prompt(self, store, runtime):
        for status in (
            GoalStatus.PAUSED,
            GoalStatus.BLOCKED,
            GoalStatus.USAGE_LIMITED,
            GoalStatus.COMPLETE,
        ):
            await store.replace_goal(SESSION, "stopped", status, token_budget=None)
            assert await runtime.next_continuation_prompt() is None

    @pytest.mark.asyncio
    async def test_budget_limited_goal_produces_one_wrap_up_prompt(self, store, runtime):
        # Budget-limit reporting is deduped per goal_id.
        await store.replace_goal(SESSION, "over budget", GoalStatus.ACTIVE, token_budget=10)
        await store.account_usage(
            SESSION, time_delta_seconds=1, token_delta=20, mode=GoalAccountingMode.ACTIVE_ONLY
        )

        first = await runtime.next_continuation_prompt()
        assert first is not None
        assert "reached its token budget" in first

        second = await runtime.next_continuation_prompt()
        assert second is None

    @pytest.mark.asyncio
    async def test_new_goal_resets_budget_limit_report_dedup(self, store, runtime):
        await store.replace_goal(SESSION, "over budget", GoalStatus.ACTIVE, token_budget=10)
        await store.account_usage(
            SESSION, time_delta_seconds=1, token_delta=20, mode=GoalAccountingMode.ACTIVE_ONLY
        )
        assert await runtime.next_continuation_prompt() is not None
        assert await runtime.next_continuation_prompt() is None

        # A replacement goal crossing its own budget reports again.
        await store.replace_goal(SESSION, "second try", GoalStatus.ACTIVE, token_budget=10)
        await runtime.on_turn_start(cumulative_tokens=0)
        await store.account_usage(
            SESSION, time_delta_seconds=1, token_delta=20, mode=GoalAccountingMode.ACTIVE_ONLY
        )
        assert await runtime.next_continuation_prompt() is not None
