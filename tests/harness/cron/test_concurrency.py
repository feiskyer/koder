"""Regression tests for shared cron ownership, receipts, and crash recovery."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from koder_agent.harness.cron.runtime import CronPromptRunner
from koder_agent.harness.cron.scheduler import CronScheduler
from koder_agent.harness.cron.storage import CronStorage


def _probe(path, minute="2026-09-06T09:00:00", *, mode="fire"):
    return subprocess.Popen(
        [
            "uv",
            "run",
            "--no-project",
            "--no-config",
            sys.executable,
            str(Path(__file__).with_name("_process_probe.py")),
            str(path),
            minute,
            "--mode",
            mode,
        ],
        cwd=path.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _finish(process, *, expected_code=0):
    try:
        output, error = process.communicate(timeout=25)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
    assert process.returncode == expected_code, error
    if expected_code:
        return []
    result = next(line for line in output.splitlines() if line.startswith("CRON_EVENTS:"))
    return json.loads(result.removeprefix("CRON_EVENTS:"))


def test_recurring_occurrence_is_not_repeated_by_a_new_process(tmp_path):
    path = tmp_path / "crons.json"
    CronStorage(path).create(cron="0 9 * * *", prompt="one occurrence")

    first = _finish(_probe(path))
    second = _finish(_probe(path))

    assert first == ["one occurrence"]
    assert second == []


@pytest.mark.parametrize("recurring", [False, True])
def test_crashed_claim_is_recovered_after_its_matching_minute(tmp_path, recurring):
    path = tmp_path / "crons.json"
    storage = CronStorage(path)
    job = storage.create(cron="0 9 * * *", prompt="recover me", recurring=recurring)
    _finish(_probe(path, mode="crash"), expected_code=23)

    assert _finish(_probe(path, "2026-09-06T09:01:00")) == ["recover me"]
    assert (storage.get(job["id"]) is not None) is recurring
    assert _finish(_probe(path, "2026-09-06T09:01:00")) == []


def test_live_process_keeps_ownership_across_matching_minutes(tmp_path):
    path = tmp_path / "crons.json"
    CronStorage(path).create(cron="* * * * *", prompt="long running")
    owner = _probe(path, mode="hold")
    try:
        deadline = time.monotonic() + 20
        while not (tmp_path / "entered").exists():
            assert owner.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        contender_events = _finish(_probe(path, "2026-09-06T09:02:00"))
    finally:
        (tmp_path / "release").write_text("finish", encoding="utf-8")
        owner_events = _finish(owner)

    assert owner_events == ["long running"]
    assert contender_events == []


def test_deleted_snapshot_cannot_fire_in_scheduler(tmp_path, monkeypatch):
    storage = CronStorage(tmp_path / "crons.json")
    job = storage.create(cron="* * * * *", prompt="deleted")
    assert storage.delete(job["id"])
    monkeypatch.setattr(storage, "list_all", lambda: [job])
    fired = []

    asyncio.run(CronScheduler(storage, on_fire=fired.append)._tick())

    assert fired == []


@pytest.mark.asyncio
async def test_independent_runners_do_not_dispatch_the_same_queued_job(tmp_path):
    path = tmp_path / "crons.json"
    storage = CronStorage(path)
    job = storage.create(cron="* * * * *", prompt="shared", recurring=False)
    fired = []

    async def dispatch(prompt, **_kwargs):
        fired.append(prompt)
        await asyncio.sleep(0.01)
        return prompt

    runners = [CronPromptRunner(dispatch, storage=CronStorage(path)) for _ in range(2)]
    try:
        for runner in runners:
            runner.enqueue_job(job)
            runner.start()
        await asyncio.wait_for(
            asyncio.gather(*(runner._queue.join() for runner in runners)), timeout=2
        )
        assert fired == ["shared"]
        assert storage.get(job["id"]) is None
    finally:
        await asyncio.gather(*(runner.stop() for runner in runners))


@pytest.mark.asyncio
async def test_cancelled_one_shot_is_due_again_after_its_matching_minute(tmp_path, monkeypatch):
    storage = CronStorage(tmp_path / "crons.json")
    job = storage.create(cron="0 9 * * *", prompt="interrupted", recurring=False)
    entered = asyncio.Event()

    async def dispatch(_prompt, **_kwargs):
        entered.set()
        await asyncio.Event().wait()

    runner = CronPromptRunner(dispatch, storage=storage)
    runner.enqueue_job(job)
    runner.start()
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
    finally:
        await runner.stop()

    class Clock:
        @staticmethod
        def now():
            return datetime(2026, 9, 6, 9, 1)

    fired = []
    monkeypatch.setattr("koder_agent.harness.cron.scheduler.datetime", Clock)
    await CronScheduler(storage, on_fire=fired.append)._tick()
    assert fired == ["interrupted"]
    assert storage.get(job["id"]) is None


def test_released_claim_cannot_acknowledge_a_new_owner(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    job = storage.create(cron="* * * * *", prompt="still running", recurring=False)
    stale = storage.claim(job["id"], minute=100)
    assert stale is not None
    stale.release()
    current = storage.claim(job["id"], minute=101)
    assert current is not None
    try:
        with pytest.raises(RuntimeError, match="active"):
            stale.complete()
        assert current.is_current()
        assert storage.get(job["id"]) == job
    finally:
        current.release()


def test_empty_legacy_document_remains_usable(tmp_path):
    path = tmp_path / "crons.json"
    path.write_text("{}", encoding="utf-8")
    storage = CronStorage(path)
    assert storage.list_all() == []
    assert storage.create(cron="* * * * *", prompt="legacy") in storage.list_all()


def test_failed_acknowledgement_retains_recoverable_occurrence(tmp_path, monkeypatch):
    storage = CronStorage(tmp_path / "crons.json")
    job = storage.create(cron="* * * * *", prompt="retry acknowledgement")
    claim = storage.claim(job["id"], minute=100)
    assert claim is not None

    def fail_write(_data):
        raise OSError("injected acknowledgement failure")

    with monkeypatch.context() as context:
        context.setattr(storage, "_write_document", fail_write)
        try:
            with pytest.raises(OSError, match="acknowledgement"):
                claim.complete()
        finally:
            claim.release()
    retry = storage.claim(job["id"], minute=105)
    assert retry is not None
    try:
        assert retry.minute == 100
        retry.complete()
    finally:
        retry.release()
    assert storage.pending_ids() == set()
    assert storage.claim(job["id"], minute=100) is None


def test_different_jobs_can_be_owned_together_and_recur_next_minute(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    first = storage.create(cron="* * * * *", prompt="first")
    second = storage.create(cron="* * * * *", prompt="second")
    claims = []
    try:
        for job in (first, second):
            claim = storage.claim(job["id"], minute=100)
            assert claim is not None
            claims.append(claim)
        for claim in claims:
            claim.complete()
    finally:
        for claim in claims:
            claim.release()
    for job in (first, second):
        assert storage.claim(job["id"], minute=100) is None
        claim = storage.claim(job["id"], minute=101)
        assert claim is not None
        claim.release()


def test_deleted_pending_job_cannot_be_recovered(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    job = storage.create(cron="0 9 * * *", prompt="cancelled")
    claim = storage.claim(job["id"], minute=100)
    assert claim is not None
    claim.release()
    storage.delete(job["id"])
    assert not storage.pending_ids()
    assert storage.claim(job["id"], minute=101) is None


@pytest.mark.asyncio
async def test_deletion_during_dispatch_cannot_resurrect_the_job(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    job = storage.create(cron="* * * * *", prompt="in flight", recurring=False)
    started = asyncio.Event()
    release = asyncio.Event()

    async def dispatch(_prompt, **_kwargs):
        started.set()
        await release.wait()

    runner = CronPromptRunner(dispatch, storage=storage)
    runner.enqueue_job(job)
    runner.start()
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        assert storage.delete(job["id"])
        release.set()
        await asyncio.wait_for(runner._queue.join(), timeout=2)
        assert storage.get(job["id"]) is None
        assert not storage.pending_ids()
    finally:
        release.set()
        await runner.stop()


@pytest.mark.asyncio
async def test_stop_releases_both_running_and_queued_claims(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    jobs = [storage.create(cron="* * * * *", prompt=str(index)) for index in range(2)]
    started = asyncio.Event()

    async def dispatch(_prompt, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    runner = CronPromptRunner(dispatch, storage=storage)
    for job in jobs:
        assert runner.enqueue_job(job)
    runner.start()
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
    finally:
        await runner.stop()
    for job in jobs:
        claim = CronStorage(storage._path).claim(job["id"])
        assert claim is not None
        claim.release()
