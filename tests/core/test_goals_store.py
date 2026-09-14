"""GoalStore state-machine tests."""

import asyncio

import pytest
import pytest_asyncio

from koder_agent.core.goals import (
    MAX_GOAL_OBJECTIVE_CHARS,
    GoalAccountingMode,
    GoalStatus,
    GoalStore,
    GoalUpdate,
    validate_goal_budget,
    validate_goal_objective,
)

SESSION = "session-123"


@pytest_asyncio.fixture
async def store(tmp_path):
    instance = GoalStore(db_path=str(tmp_path / "goals.db"))
    try:
        yield instance
    finally:
        await instance.close()


class TestGoalCrud:
    @pytest.mark.asyncio
    async def test_replace_update_and_get_goal(self, store):
        goal = await store.replace_goal(
            SESSION, "optimize the benchmark", GoalStatus.ACTIVE, token_budget=100_000
        )
        assert await store.get_goal(SESSION) == goal
        assert goal.objective == "optimize the benchmark"
        assert goal.status is GoalStatus.ACTIVE
        assert goal.token_budget == 100_000
        assert goal.tokens_used == 0
        assert goal.time_used_seconds == 0

        updated = await store.update_goal(
            SESSION,
            GoalUpdate(status=GoalStatus.PAUSED, token_budget=200_000),
        )
        assert updated is not None
        assert updated.status is GoalStatus.PAUSED
        assert updated.token_budget == 200_000
        assert updated.goal_id == goal.goal_id
        assert updated.objective == goal.objective

        replaced = await store.replace_goal(
            SESSION, "ship the new result", GoalStatus.ACTIVE, token_budget=None
        )
        assert replaced.objective == "ship the new result"
        assert replaced.status is GoalStatus.ACTIVE
        assert replaced.token_budget is None
        assert replaced.tokens_used == 0
        assert replaced.time_used_seconds == 0
        assert replaced.goal_id != goal.goal_id

        assert await store.delete_goal(SESSION) == replaced
        assert await store.get_goal(SESSION) is None
        assert await store.delete_goal(SESSION) is None

    @pytest.mark.asyncio
    async def test_replace_goal_applies_budget_limit_immediately(self, store):
        # Budget <= tokens_used(0) at creation is stored budget_limited.
        replaced = await store.replace_goal(
            SESSION, "stay within budget", GoalStatus.ACTIVE, token_budget=0
        )
        assert replaced.status is GoalStatus.BUDGET_LIMITED
        assert replaced.token_budget == 0
        assert replaced.tokens_used == 0
        assert replaced.time_used_seconds == 0

    @pytest.mark.asyncio
    async def test_insert_goal_does_not_replace_existing_goal(self, store):
        inserted = await store.insert_goal(
            SESSION, "optimize the benchmark", GoalStatus.ACTIVE, token_budget=100_000
        )
        assert inserted is not None

        duplicate = await store.insert_goal(
            SESSION, "replace the benchmark", GoalStatus.ACTIVE, token_budget=200_000
        )
        assert duplicate is None
        assert await store.get_goal(SESSION) == inserted

    @pytest.mark.asyncio
    async def test_insert_goal_replaces_complete_goal(self, store):
        first = await store.insert_goal(SESSION, "first", GoalStatus.ACTIVE, token_budget=None)
        assert first is not None
        await store.update_goal(SESSION, GoalUpdate(status=GoalStatus.COMPLETE))

        second = await store.insert_goal(SESSION, "second", GoalStatus.ACTIVE, token_budget=None)
        assert second is not None
        assert second.goal_id != first.goal_id
        assert second.objective == "second"
        assert second.status is GoalStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_insert_goal_applies_budget_limit_immediately(self, store):
        inserted = await store.insert_goal(
            SESSION, "stay within budget", GoalStatus.ACTIVE, token_budget=0
        )
        assert inserted is not None
        assert inserted.status is GoalStatus.BUDGET_LIMITED

    @pytest.mark.asyncio
    async def test_update_goal_ignores_replaced_goal_version(self, store):
        original = await store.replace_goal(
            SESSION, "old objective", GoalStatus.ACTIVE, token_budget=100
        )
        replacement = await store.replace_goal(
            SESSION, "new objective", GoalStatus.ACTIVE, token_budget=10
        )

        stale = await store.update_goal(
            SESSION,
            GoalUpdate(status=GoalStatus.COMPLETE, expected_goal_id=original.goal_id),
        )
        assert stale is None
        assert await store.get_goal(SESSION) == replacement

        fresh = await store.update_goal(
            SESSION,
            GoalUpdate(status=GoalStatus.COMPLETE, expected_goal_id=replacement.goal_id),
        )
        assert fresh is not None
        assert fresh.status is GoalStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_update_goal_objective_preserves_usage_and_created_at(self, store):
        await store.replace_goal(SESSION, "draft the report", GoalStatus.ACTIVE, token_budget=100)
        outcome = await store.account_usage(
            SESSION,
            time_delta_seconds=12,
            token_delta=30,
            mode=GoalAccountingMode.ACTIVE_ONLY,
        )
        assert outcome.updated
        accounted = outcome.goal

        updated = await store.update_goal(
            SESSION,
            GoalUpdate(
                objective="draft the report clearly",
                status=GoalStatus.PAUSED,
                token_budget=200,
                expected_goal_id=accounted.goal_id,
            ),
        )
        assert updated is not None
        assert updated.objective == "draft the report clearly"
        assert updated.status is GoalStatus.PAUSED
        assert updated.token_budget == 200
        assert updated.tokens_used == accounted.tokens_used
        assert updated.time_used_seconds == accounted.time_used_seconds
        assert updated.created_at_ms == accounted.created_at_ms

    @pytest.mark.asyncio
    async def test_concurrent_partial_updates_preserve_independent_fields(self, store):
        await store.replace_goal(
            SESSION, "optimize the benchmark", GoalStatus.ACTIVE, token_budget=100_000
        )
        status_update = store.update_goal(SESSION, GoalUpdate(status=GoalStatus.PAUSED))
        budget_update = store.update_goal(SESSION, GoalUpdate(token_budget=200_000))
        await asyncio.gather(status_update, budget_update)

        goal = await store.get_goal(SESSION)
        assert goal is not None
        assert goal.status is GoalStatus.PAUSED
        assert goal.token_budget == 200_000


class TestActiveStatusTransitions:
    @pytest.mark.asyncio
    async def test_pause_active_goal_does_not_clobber_terminal_status(self, store):
        goal = await store.replace_goal(
            SESSION, "optimize the benchmark", GoalStatus.ACTIVE, token_budget=100_000
        )
        paused = await store.pause_active_goal(SESSION)
        assert paused is not None
        assert paused.status is GoalStatus.PAUSED
        assert paused.goal_id == goal.goal_id

        complete = await store.update_goal(SESSION, GoalUpdate(status=GoalStatus.COMPLETE))
        assert complete is not None
        pause_result = await store.pause_active_goal(SESSION)
        assert pause_result is None
        current = await store.get_goal(SESSION)
        assert current is not None
        assert current.status is GoalStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_usage_limit_active_goal_updates_active_or_budget_limited_goals(self, store):
        await store.replace_goal(
            SESSION, "optimize the benchmark", GoalStatus.ACTIVE, token_budget=None
        )
        usage_limited = await store.usage_limit_active_goal(SESSION)
        assert usage_limited is not None
        assert usage_limited.status is GoalStatus.USAGE_LIMITED

        second = await store.usage_limit_active_goal(SESSION)
        assert second is None

        await store.replace_goal(
            SESSION, "keep the usage failure visible", GoalStatus.BUDGET_LIMITED, token_budget=1
        )
        usage_limited = await store.usage_limit_active_goal(SESSION)
        assert usage_limited is not None
        assert usage_limited.status is GoalStatus.USAGE_LIMITED


class TestUsageAccounting:
    @pytest.mark.asyncio
    async def test_accounting_updates_active_and_accounts_budget_limited_in_flight(self, store):
        await store.replace_goal(SESSION, "stay within budget", GoalStatus.ACTIVE, token_budget=20)

        outcome = await store.account_usage(
            SESSION, time_delta_seconds=7, token_delta=5, mode=GoalAccountingMode.ACTIVE_ONLY
        )
        assert outcome.updated
        assert outcome.goal.status is GoalStatus.ACTIVE
        assert outcome.goal.tokens_used == 5
        assert outcome.goal.time_used_seconds == 7

        outcome = await store.account_usage(
            SESSION, time_delta_seconds=3, token_delta=15, mode=GoalAccountingMode.ACTIVE_ONLY
        )
        assert outcome.updated
        assert outcome.goal.status is GoalStatus.BUDGET_LIMITED
        assert outcome.goal.tokens_used == 20
        assert outcome.goal.time_used_seconds == 10

        # Budget-limited goals still account in-flight active usage.
        outcome = await store.account_usage(
            SESSION, time_delta_seconds=5, token_delta=5, mode=GoalAccountingMode.ACTIVE_ONLY
        )
        assert outcome.updated
        assert outcome.goal.status is GoalStatus.BUDGET_LIMITED
        assert outcome.goal.tokens_used == 25
        assert outcome.goal.time_used_seconds == 15

    @pytest.mark.asyncio
    async def test_active_status_only_does_not_update_budget_limited_goals(self, store):
        await store.replace_goal(
            SESSION, "stay stopped", GoalStatus.BUDGET_LIMITED, token_budget=20
        )
        outcome = await store.account_usage(
            SESSION,
            time_delta_seconds=5,
            token_delta=5,
            mode=GoalAccountingMode.ACTIVE_STATUS_ONLY,
        )
        assert not outcome.updated
        assert outcome.goal.status is GoalStatus.BUDGET_LIMITED
        assert outcome.goal.tokens_used == 0
        assert outcome.goal.time_used_seconds == 0

    @pytest.mark.asyncio
    async def test_stopped_accounting_promotes_paused_goal_over_budget(self, store):
        await store.replace_goal(SESSION, "stop before overrun", GoalStatus.ACTIVE, token_budget=20)
        await store.update_goal(SESSION, GoalUpdate(status=GoalStatus.PAUSED))

        outcome = await store.account_usage(
            SESSION,
            time_delta_seconds=3,
            token_delta=25,
            mode=GoalAccountingMode.ACTIVE_OR_STOPPED,
        )
        assert outcome.updated
        assert outcome.goal.status is GoalStatus.BUDGET_LIMITED
        assert outcome.goal.tokens_used == 25
        assert outcome.goal.time_used_seconds == 3

    @pytest.mark.asyncio
    async def test_budget_updates_immediately_stop_active_goals_already_over_budget(self, store):
        await store.replace_goal(SESSION, "stay within budget", GoalStatus.ACTIVE, token_budget=100)
        await store.account_usage(
            SESSION, time_delta_seconds=1, token_delta=50, mode=GoalAccountingMode.ACTIVE_ONLY
        )

        lowered = await store.update_goal(SESSION, GoalUpdate(token_budget=40))
        assert lowered is not None
        assert lowered.status is GoalStatus.BUDGET_LIMITED
        assert lowered.token_budget == 40
        assert lowered.tokens_used == 50

    @pytest.mark.asyncio
    async def test_activating_goal_already_over_budget_keeps_it_budget_limited(self, store):
        await store.replace_goal(SESSION, "stay within budget", GoalStatus.ACTIVE, token_budget=40)
        await store.account_usage(
            SESSION, time_delta_seconds=1, token_delta=50, mode=GoalAccountingMode.ACTIVE_ONLY
        )

        reactivated = await store.update_goal(
            SESSION,
            GoalUpdate(
                objective="stay within budget, with clearer wording",
                status=GoalStatus.ACTIVE,
            ),
        )
        assert reactivated is not None
        assert reactivated.status is GoalStatus.BUDGET_LIMITED
        assert reactivated.objective == "stay within budget, with clearer wording"
        assert reactivated.token_budget == 40
        assert reactivated.tokens_used == 50

    @pytest.mark.asyncio
    async def test_pausing_budget_limited_goal_preserves_terminal_status(self, store):
        await store.replace_goal(SESSION, "stay within budget", GoalStatus.ACTIVE, token_budget=40)
        await store.account_usage(
            SESSION, time_delta_seconds=1, token_delta=50, mode=GoalAccountingMode.ACTIVE_ONLY
        )

        paused = await store.update_goal(SESSION, GoalUpdate(status=GoalStatus.PAUSED))
        assert paused is not None
        assert paused.status is GoalStatus.BUDGET_LIMITED
        assert paused.token_budget == 40
        assert paused.tokens_used == 50

    @pytest.mark.asyncio
    async def test_blocking_budget_limited_goal_preserves_terminal_status(self, store):
        await store.replace_goal(SESSION, "stay within budget", GoalStatus.ACTIVE, token_budget=40)
        outcome = await store.account_usage(
            SESSION, time_delta_seconds=1, token_delta=50, mode=GoalAccountingMode.ACTIVE_ONLY
        )
        assert outcome.updated
        budget_limited = outcome.goal

        blocked = await store.update_goal(SESSION, GoalUpdate(status=GoalStatus.BLOCKED))
        assert blocked is not None
        assert blocked.status is GoalStatus.BUDGET_LIMITED
        assert blocked.tokens_used == budget_limited.tokens_used

    @pytest.mark.asyncio
    async def test_accounting_can_finalize_completed_goal_for_completing_turn(self, store):
        await store.replace_goal(
            SESSION, "finish the report", GoalStatus.COMPLETE, token_budget=1_000
        )

        active_only = await store.account_usage(
            SESSION, time_delta_seconds=30, token_delta=200, mode=GoalAccountingMode.ACTIVE_ONLY
        )
        assert not active_only.updated
        assert active_only.goal.status is GoalStatus.COMPLETE
        assert active_only.goal.tokens_used == 0

        completing = await store.account_usage(
            SESSION,
            time_delta_seconds=30,
            token_delta=200,
            mode=GoalAccountingMode.ACTIVE_OR_COMPLETE,
        )
        assert completing.updated
        assert completing.goal.status is GoalStatus.COMPLETE
        assert completing.goal.tokens_used == 200
        assert completing.goal.time_used_seconds == 30

    @pytest.mark.asyncio
    async def test_accounting_can_finalize_stopped_goal_for_in_flight_turn(self, store):
        await store.replace_goal(
            SESSION, "finish the report", GoalStatus.ACTIVE, token_budget=1_000
        )
        await store.update_goal(SESSION, GoalUpdate(status=GoalStatus.PAUSED))

        active_only = await store.account_usage(
            SESSION, time_delta_seconds=30, token_delta=200, mode=GoalAccountingMode.ACTIVE_ONLY
        )
        assert not active_only.updated
        assert active_only.goal.status is GoalStatus.PAUSED
        assert active_only.goal.tokens_used == 0

        in_flight = await store.account_usage(
            SESSION,
            time_delta_seconds=30,
            token_delta=200,
            mode=GoalAccountingMode.ACTIVE_OR_STOPPED,
        )
        assert in_flight.updated
        assert in_flight.goal.status is GoalStatus.PAUSED
        assert in_flight.goal.tokens_used == 200
        assert in_flight.goal.time_used_seconds == 30

    @pytest.mark.asyncio
    async def test_accounting_ignores_replaced_goal_version(self, store):
        original = await store.replace_goal(
            SESSION, "old objective", GoalStatus.ACTIVE, token_budget=100
        )
        replacement = await store.replace_goal(
            SESSION, "new objective", GoalStatus.ACTIVE, token_budget=10
        )

        outcome = await store.account_usage(
            SESSION,
            time_delta_seconds=5,
            token_delta=5,
            mode=GoalAccountingMode.ACTIVE_ONLY,
            expected_goal_id=original.goal_id,
        )
        assert not outcome.updated
        assert outcome.goal is not None
        assert outcome.goal.goal_id == replacement.goal_id
        assert outcome.goal.objective == "new objective"
        assert outcome.goal.tokens_used == 0
        assert outcome.goal.time_used_seconds == 0

    @pytest.mark.asyncio
    async def test_accounting_adds_concurrent_token_deltas(self, store):
        await store.replace_goal(
            SESSION, "count every token", GoalStatus.ACTIVE, token_budget=1_000
        )
        first = store.account_usage(
            SESSION, time_delta_seconds=4, token_delta=40, mode=GoalAccountingMode.ACTIVE_ONLY
        )
        second = store.account_usage(
            SESSION, time_delta_seconds=6, token_delta=60, mode=GoalAccountingMode.ACTIVE_ONLY
        )
        await asyncio.gather(first, second)

        goal = await store.get_goal(SESSION)
        assert goal is not None
        assert goal.tokens_used == 100
        assert goal.time_used_seconds == 10

    @pytest.mark.asyncio
    async def test_zero_deltas_are_unchanged(self, store):
        await store.replace_goal(SESSION, "noop", GoalStatus.ACTIVE, token_budget=None)
        outcome = await store.account_usage(
            SESSION, time_delta_seconds=0, token_delta=0, mode=GoalAccountingMode.ACTIVE_ONLY
        )
        assert not outcome.updated
        assert outcome.goal is not None

    @pytest.mark.asyncio
    async def test_negative_deltas_are_clamped(self, store):
        await store.replace_goal(SESSION, "no negative usage", GoalStatus.ACTIVE, token_budget=None)
        outcome = await store.account_usage(
            SESSION, time_delta_seconds=-5, token_delta=-9, mode=GoalAccountingMode.ACTIVE_ONLY
        )
        assert not outcome.updated
        goal = await store.get_goal(SESSION)
        assert goal.tokens_used == 0
        assert goal.time_used_seconds == 0


class TestValidation:
    def test_validate_goal_objective_rejects_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_goal_objective("")

    def test_validate_goal_objective_accepts_limit(self):
        validate_goal_objective("x" * MAX_GOAL_OBJECTIVE_CHARS)

    def test_validate_goal_objective_rejects_oversized(self):
        with pytest.raises(ValueError, match="at most 4000 characters"):
            validate_goal_objective("x" * (MAX_GOAL_OBJECTIVE_CHARS + 1))

    def test_validate_goal_budget_rejects_non_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            validate_goal_budget(0)
        with pytest.raises(ValueError, match="must be positive"):
            validate_goal_budget(-1)

    def test_validate_goal_budget_accepts_positive_and_none(self):
        validate_goal_budget(1)
        validate_goal_budget(None)

    def test_goal_status_terminal_and_active(self):
        assert GoalStatus.ACTIVE.is_active()
        assert not GoalStatus.PAUSED.is_active()
        assert GoalStatus.BUDGET_LIMITED.is_terminal()
        assert GoalStatus.COMPLETE.is_terminal()
        for status in (
            GoalStatus.ACTIVE,
            GoalStatus.PAUSED,
            GoalStatus.BLOCKED,
            GoalStatus.USAGE_LIMITED,
        ):
            assert not status.is_terminal()

    def test_goal_status_round_trips_strings(self):
        for status in GoalStatus:
            assert GoalStatus.from_str(status.value) is status
        with pytest.raises(ValueError, match="unknown goal status"):
            GoalStatus.from_str("bogus")
