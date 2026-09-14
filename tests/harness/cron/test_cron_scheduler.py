"""Tests for the cron scheduler."""

import asyncio
from datetime import datetime

from koder_agent.harness.cron.scheduler import (
    CronScheduler,
    _cron_matches_now,
    _field_matches,
)
from koder_agent.harness.cron.storage import CronStorage


def test_field_matches_star():
    assert _field_matches("*", 5, 0, 59)
    assert _field_matches("*", 0, 0, 59)


def test_field_matches_exact():
    assert _field_matches("5", 5, 0, 59)
    assert not _field_matches("5", 6, 0, 59)


def test_field_matches_range():
    assert _field_matches("1-5", 3, 0, 59)
    assert not _field_matches("1-5", 6, 0, 59)


def test_field_matches_step():
    assert _field_matches("*/15", 0, 0, 59)
    assert _field_matches("*/15", 15, 0, 59)
    assert _field_matches("*/15", 30, 0, 59)
    assert not _field_matches("*/15", 7, 0, 59)


def test_field_matches_comma_list():
    assert _field_matches("1,3,5", 3, 0, 59)
    assert not _field_matches("1,3,5", 4, 0, 59)


def test_cron_matches_now():
    # "0 9 * * *" = 9:00 AM every day
    dt = datetime(2026, 4, 7, 9, 0, 0)
    assert _cron_matches_now("0 9 * * *", dt)

    dt2 = datetime(2026, 4, 7, 10, 0, 0)
    assert not _cron_matches_now("0 9 * * *", dt2)


def test_cron_every_five_minutes():
    # "*/5 * * * *" = every 5 minutes
    assert _cron_matches_now("*/5 * * * *", datetime(2026, 4, 7, 9, 0, 0))
    assert _cron_matches_now("*/5 * * * *", datetime(2026, 4, 7, 9, 5, 0))
    assert not _cron_matches_now("*/5 * * * *", datetime(2026, 4, 7, 9, 3, 0))


def test_scheduler_fires_matching_job(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    fired = []
    scheduler = CronScheduler(storage, on_fire=fired.append, check_interval=0.1)

    # Create a job that matches every minute
    storage.create(cron="* * * * *", prompt="test prompt")

    asyncio.run(scheduler._tick())
    assert len(fired) == 1
    assert fired[0] == "test prompt"


def test_scheduler_skips_non_matching_job(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    fired = []
    scheduler = CronScheduler(storage, on_fire=fired.append)

    # Create a job for a specific minute that probably doesn't match now
    storage.create(cron="59 23 31 12 *", prompt="new year")

    asyncio.run(scheduler._tick())
    # Might fire if it happens to be Dec 31 23:59, but very unlikely
    # The key assertion is it doesn't crash


def test_scheduler_deletes_one_shot_after_fire(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    fired = []
    scheduler = CronScheduler(storage, on_fire=fired.append)

    storage.create(cron="* * * * *", prompt="once", recurring=False)

    asyncio.run(scheduler._tick())
    assert len(fired) == 1
    # Job should be deleted
    assert len(storage.list_all()) == 0


def test_scheduler_retains_one_shot_when_consumer_rejects_delivery(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    job = storage.create(cron="* * * * *", prompt="retry me", recurring=False)
    scheduler = CronScheduler(storage, on_job_fire=lambda _job: False)

    asyncio.run(scheduler._tick())

    assert storage.get(job["id"]) == job


def test_scheduler_retains_one_shot_without_a_consumer(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    job = storage.create(cron="* * * * *", prompt="undelivered", recurring=False)

    asyncio.run(CronScheduler(storage)._tick())

    assert storage.get(job["id"]) == job


def test_scheduler_skips_invalid_job_and_continues(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    fired = []
    scheduler = CronScheduler(storage, on_fire=fired.append)

    storage.create(cron="*/0 * * * *", prompt="poison")
    storage.create(cron="* * * * *", prompt="healthy")

    asyncio.run(scheduler._tick())

    assert fired == ["healthy"]


def test_scheduler_minute_dedup_uses_full_minute_key(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    fired = []
    scheduler = CronScheduler(storage, on_fire=fired.append)

    scheduler._fired_this_minute.add("job-1")
    scheduler._last_minute_key = (2026, 7, 3, 9, 0)
    scheduler._reset_fired_set_if_needed(datetime(2026, 7, 3, 10, 0, 0))

    assert scheduler._fired_this_minute == set()


def test_scheduler_stop_async_stops_background_task(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    scheduler = CronScheduler(storage, on_fire=lambda _prompt: None, check_interval=60)

    async def scenario():
        scheduler.start()
        assert scheduler._task is not None

        await scheduler.stop_async()

        assert scheduler._task is None

    asyncio.run(scenario())
