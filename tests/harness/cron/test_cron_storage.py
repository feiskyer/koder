"""Tests for cron job persistence."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import pytest

from koder_agent.harness.cron.storage import CronStorage


def test_create_cron_job(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    job = storage.create(cron="0 9 * * *", prompt="daily standup", recurring=True)
    assert job["id"]
    assert job["cron"] == "0 9 * * *"
    assert job["prompt"] == "daily standup"
    assert job["recurring"] is True


def test_list_jobs(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    storage.create(cron="0 9 * * *", prompt="morning")
    storage.create(cron="0 17 * * *", prompt="evening")
    jobs = storage.list_all()
    assert len(jobs) == 2


def test_delete_job(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    job = storage.create(cron="0 9 * * *", prompt="temp")
    assert storage.delete(job["id"]) is True
    assert len(storage.list_all()) == 0


def test_delete_nonexistent(tmp_path):
    storage = CronStorage(tmp_path / "crons.json")
    assert storage.delete("nonexistent") is False


def test_max_jobs_limit(tmp_path):
    storage = CronStorage(tmp_path / "crons.json", max_jobs=3)
    storage.create(cron="0 1 * * *", prompt="1")
    storage.create(cron="0 2 * * *", prompt="2")
    storage.create(cron="0 3 * * *", prompt="3")
    with pytest.raises(ValueError, match="limit"):
        storage.create(cron="0 4 * * *", prompt="4")


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "crons.json"
    s1 = CronStorage(path)
    s1.create(cron="0 9 * * *", prompt="persist me")
    s2 = CronStorage(path)
    jobs = s2.list_all()
    assert len(jobs) == 1
    assert jobs[0]["prompt"] == "persist me"


@pytest.mark.parametrize("max_jobs", [1, 50])
def test_concurrent_storage_instances_do_not_lose_jobs_or_bypass_limit(
    tmp_path, monkeypatch, max_jobs
):
    path = tmp_path / "crons.json"
    first = CronStorage(path, max_jobs=max_jobs)
    second = CronStorage(path, max_jobs=max_jobs)
    first_read = threading.Event()
    release_first = threading.Event()
    original_read = first._read

    def hold_first_snapshot():
        jobs = original_read()
        first_read.set()
        assert release_first.wait(5)
        return jobs

    monkeypatch.setattr(first, "_read", hold_first_snapshot)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_write = pool.submit(first.create, cron="* * * * *", prompt="first")
        try:
            assert first_read.wait(5)
            second_write = pool.submit(second.create, cron="* * * * *", prompt="second")
            try:
                second_write.result(timeout=0.1)
            except TimeoutError:
                pass
        finally:
            release_first.set()
        first_job = first_write.result(timeout=5)
        if max_jobs == 1:
            with pytest.raises(ValueError, match="limit"):
                second_write.result(timeout=5)
        else:
            second_job = second_write.result(timeout=5)
            assert {job["id"] for job in second.list_all()} == {
                first_job["id"],
                second_job["id"],
            }
    assert len(second.list_all()) == min(max_jobs, 2)


@pytest.mark.parametrize("payload", [[], {"tasks": {}}, {"tasks": ["not a job"]}])
def test_malformed_storage_is_not_silently_replaced(tmp_path, payload):
    path = tmp_path / "crons.json"
    original = json.dumps(payload)
    path.write_text(original, encoding="utf-8")
    storage = CronStorage(path)

    with pytest.raises(ValueError, match="Invalid cron storage"):
        storage.create(cron="* * * * *", prompt="new")

    assert path.read_text(encoding="utf-8") == original
