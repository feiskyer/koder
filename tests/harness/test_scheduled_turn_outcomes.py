"""Scheduled callers need a success signal, not a rendered error string."""

from types import SimpleNamespace

import pytest

from koder_agent.harness.session_flow import _SchedulerState


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["_last_turn_errored", "_last_turn_cancelled"])
async def test_scheduled_dispatch_does_not_acknowledge_an_unsuccessful_turn(flag):
    scheduler = SimpleNamespace()

    async def handle(_prompt, **_kwargs):
        setattr(scheduler, flag, True)
        return "Rendered failure for the interactive user"

    scheduler.handle = handle
    state = _SchedulerState(SimpleNamespace(), scheduler)

    with pytest.raises(RuntimeError, match="Scheduled turn did not complete"):
        await state.dispatch_handle("scheduled", require_success=True)

    assert await state.dispatch_handle("interactive") == "Rendered failure for the interactive user"
