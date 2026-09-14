"""Tests for draining fired cron prompts into the active scheduler."""

import asyncio

import pytest

from koder_agent.harness.cron.runtime import CronPromptRunner
from koder_agent.harness.cron.storage import CronStorage


class _RecordingScheduler:
    def __init__(self, *, fail: bool = False, delay: float = 0):
        self.prompts: list[tuple[str, bool]] = []
        self.fail = fail
        self.delay = delay

    async def handle(self, prompt: str, render_output: bool = True, multimodal_input=None) -> str:
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("scheduled prompt failed")
        self.prompts.append((prompt, render_output))
        return prompt


def _dispatcher(scheduler):
    async def dispatch(prompt: str, **kwargs):
        return await scheduler.handle(prompt, **kwargs)

    return dispatch


async def _wait_until(assertion):
    deadline = asyncio.get_running_loop().time() + 1
    while True:
        if assertion():
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true")
        await asyncio.sleep(0.01)


def test_cron_prompt_runner_uses_current_scheduler(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    first_scheduler = _RecordingScheduler()
    current_scheduler = first_scheduler

    async def scenario():
        nonlocal current_scheduler

        async def dispatch(prompt: str, **kwargs):
            return await current_scheduler.handle(prompt, **kwargs)

        runner = CronPromptRunner(
            dispatch,
            storage=storage,
            check_interval=60,
        )
        runner.start()
        try:
            runner.enqueue("first scheduled prompt")
            await _wait_until(lambda: first_scheduler.prompts)

            second_scheduler = _RecordingScheduler()
            current_scheduler = second_scheduler
            runner.enqueue("second scheduled prompt")
            await _wait_until(lambda: second_scheduler.prompts)

            assert first_scheduler.prompts == [("first scheduled prompt", True)]
            assert second_scheduler.prompts == [("second scheduled prompt", True)]
        finally:
            await runner.stop()

    asyncio.run(scenario())


def test_cron_prompt_runner_deletes_one_shot_after_success(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    job = storage.create(cron="* * * * *", prompt="once", recurring=False)
    scheduler = _RecordingScheduler()

    async def scenario():
        runner = CronPromptRunner(_dispatcher(scheduler), storage=storage, check_interval=60)
        runner.start()
        try:
            runner.enqueue_job(job)
            await _wait_until(lambda: scheduler.prompts)

            assert storage.get(job["id"]) is None
        finally:
            await runner.stop()

    asyncio.run(scenario())


def test_cron_prompt_runner_keeps_one_shot_after_failure(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    job = storage.create(cron="* * * * *", prompt="once", recurring=False)
    scheduler = _RecordingScheduler(fail=True)

    async def scenario():
        runner = CronPromptRunner(_dispatcher(scheduler), storage=storage, check_interval=60)
        runner.start()
        try:
            runner.enqueue_job(job)
            await _wait_until(lambda: job["id"] not in runner.pending_job_ids)

            assert storage.get(job["id"]) is not None
        finally:
            await runner.stop()

    asyncio.run(scenario())


def test_cron_prompt_runner_deduplicates_pending_job(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    job = storage.create(cron="* * * * *", prompt="slow", recurring=True)
    scheduler = _RecordingScheduler(delay=0.05)

    async def scenario():
        runner = CronPromptRunner(_dispatcher(scheduler), storage=storage, check_interval=60)
        runner.start()
        try:
            runner.enqueue_job(job)
            runner.enqueue_job(job)
            await _wait_until(lambda: scheduler.prompts)

            assert scheduler.prompts == [("slow", True)]
        finally:
            await runner.stop()

    asyncio.run(scenario())


def test_cron_prompt_runner_survives_dispatcher_error(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    job = storage.create(cron="* * * * *", prompt="after getter failure", recurring=True)
    scheduler = _RecordingScheduler()
    calls = 0

    async def dispatch(prompt: str, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("scheduler unavailable")
        return await scheduler.handle(prompt, **kwargs)

    async def scenario():
        runner = CronPromptRunner(dispatch, storage=storage, check_interval=60)
        runner.start()
        try:
            runner.enqueue_job(job)
            await _wait_until(lambda: job["id"] not in runner.pending_job_ids)
            runner.enqueue_job(job)
            await _wait_until(lambda: scheduler.prompts)

            assert scheduler.prompts == [("after getter failure", True)]
        finally:
            await runner.stop()

    asyncio.run(scenario())


def test_cron_prompt_runner_cancels_consumer_when_scheduler_stop_fails(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    scheduler = _RecordingScheduler()

    async def scenario():
        runner = CronPromptRunner(_dispatcher(scheduler), storage=storage, check_interval=60)
        runner.start()
        consumer = runner._consumer_task
        assert consumer is not None

        async def failing_stop():
            raise RuntimeError("poller stop failed")

        runner._cron_scheduler.stop_async = failing_stop

        with pytest.raises(RuntimeError, match="poller stop failed"):
            await runner.stop()

        assert consumer.done()
        assert consumer.cancelled()
        assert runner._consumer_task is None

    asyncio.run(scenario())


def test_cron_prompt_runner_stop_survives_repeated_caller_cancellation(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    scheduler = _RecordingScheduler()

    async def scenario():
        runner = CronPromptRunner(_dispatcher(scheduler), storage=storage, check_interval=60)
        runner.start()
        consumer = runner._consumer_task
        assert consumer is not None
        stop_started = asyncio.Event()
        allow_stop = asyncio.Event()

        async def blocked_stop():
            stop_started.set()
            await allow_stop.wait()

        runner._cron_scheduler.stop_async = blocked_stop
        stop_task = asyncio.create_task(runner.stop())
        await stop_started.wait()
        stop_task.cancel()
        await asyncio.sleep(0)
        stop_task.cancel()
        allow_stop.set()

        with pytest.raises(asyncio.CancelledError):
            await stop_task

        assert consumer.done()
        assert consumer.cancelled()
        assert runner._consumer_task is None
        await runner.stop()

    asyncio.run(scenario())


@pytest.mark.asyncio
async def test_deleted_job_is_not_dispatched_from_an_old_queue_snapshot(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    job = storage.create(cron="* * * * *", prompt="deleted job", recurring=False)
    blocked = asyncio.Event()
    release = asyncio.Event()
    drained = asyncio.Event()
    prompts = []

    async def dispatch(prompt, **_kwargs):
        prompts.append(prompt)
        if prompt == "blocker":
            blocked.set()
            await release.wait()
        elif prompt == "barrier":
            drained.set()
        return prompt

    runner = CronPromptRunner(dispatch, storage=storage)
    runner.enqueue("blocker")
    assert runner.enqueue_job(job)
    runner.start()
    try:
        await asyncio.wait_for(blocked.wait(), timeout=2)
        assert storage.delete(job["id"])
        runner.enqueue("barrier")
        release.set()
        await asyncio.wait_for(drained.wait(), timeout=2)
        assert prompts == ["blocker", "barrier"]
        assert not runner.pending_job_ids
    finally:
        release.set()
        await runner.stop()


@pytest.mark.asyncio
async def test_restarted_runner_stops_its_new_consumer_and_poller(tmp_path):
    runner = CronPromptRunner(
        _dispatcher(_RecordingScheduler()),
        storage=CronStorage(tmp_path / "crons.json"),
    )
    tasks = []
    try:
        for _ in range(2):
            runner.start()
            consumer = runner._consumer_task
            poller = runner._cron_scheduler._task
            tasks.extend((consumer, poller))
            await runner.stop()
            assert consumer.done()
            assert poller.done()
            assert runner._consumer_task is None
            assert runner._cron_scheduler._task is None
    finally:
        # Join even a leaked task when this regression runs against old code.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_drops_queued_deliveries_but_preserves_durable_jobs(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    first = storage.create(cron="* * * * *", prompt="running", recurring=False)
    second = storage.create(cron="* * * * *", prompt="queued", recurring=False)
    dispatch_started = asyncio.Event()
    release_dispatch = asyncio.Event()

    async def dispatch(prompt, **_kwargs):
        dispatch_started.set()
        await release_dispatch.wait()
        return prompt

    runner = CronPromptRunner(dispatch, storage=storage)
    assert runner.enqueue_job(first)
    assert runner.enqueue_job(second)
    runner.start()
    try:
        await asyncio.wait_for(dispatch_started.wait(), timeout=2)
        await runner.stop()
        assert not runner.pending_job_ids
        assert runner._queue.empty()
        assert storage.get(first["id"]) == first
        assert storage.get(second["id"]) == second
    finally:
        release_dispatch.set()
        await runner.stop()


@pytest.mark.asyncio
async def test_stopped_runner_rejects_delivery_until_restarted(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    job = storage.create(cron="* * * * *", prompt="not accepted", recurring=False)
    runner = CronPromptRunner(_dispatcher(_RecordingScheduler()), storage=storage)
    await runner.stop()

    assert runner.enqueue_job(job) is False
    assert runner.enqueue("manual prompt") is False
    assert not runner.pending_job_ids
    assert runner._queue.empty()
    assert storage.get(job["id"]) == job
    runner.start()
    try:
        assert runner.enqueue_job(job) is True
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_runner_cannot_restart_while_shutdown_is_in_progress(tmp_path, monkeypatch):
    runner = CronPromptRunner(
        _dispatcher(_RecordingScheduler()),
        storage=CronStorage(tmp_path / "crons.json"),
    )
    runner.start()
    stop_started = asyncio.Event()
    release_stop = asyncio.Event()
    real_stop = runner._cron_scheduler.stop_async

    async def blocked_stop():
        stop_started.set()
        await release_stop.wait()
        await real_stop()

    monkeypatch.setattr(runner._cron_scheduler, "stop_async", blocked_stop)
    stopping = asyncio.create_task(runner.stop())
    try:
        await asyncio.wait_for(stop_started.wait(), timeout=2)
        with pytest.raises(RuntimeError, match="shutdown"):
            runner.start()
    finally:
        release_stop.set()
        await stopping
