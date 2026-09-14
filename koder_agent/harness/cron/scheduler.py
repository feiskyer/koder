"""Basic cron scheduler that checks and fires jobs."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Optional

from .expression import cron_matches_now, field_matches
from .storage import CronClaim, CronStorage

logger = logging.getLogger(__name__)


def _cron_matches_now(cron_expr: str, now: datetime) -> bool:
    return cron_matches_now(cron_expr, now)


def _field_matches(expr: str, value: int, min_val: int, max_val: int) -> bool:
    return field_matches(expr, value, min_val, max_val)


class CronScheduler:
    """Tick-based scheduler that fires cron jobs.

    Call `start()` to begin polling. Call `stop()` to shut down.
    The `on_fire` callback receives the prompt string when a job fires.
    Synchronous callbacks are acknowledged on return. An `on_claim_fire` consumer
    instead owns the claim until it explicitly completes/releases the delivery.
    """

    def __init__(
        self,
        storage: CronStorage,
        on_fire: Callable[[str], None] | None = None,
        *,
        on_job_fire: Callable[[dict[str, Any]], object] | None = None,
        on_claim_fire: Callable[[CronClaim], object] | None = None,
        delete_one_shot_after_fire: bool = True,
        check_interval: float = 60.0,
    ):
        self._storage = storage
        self._on_fire = on_fire
        self._on_job_fire = on_job_fire
        self._on_claim_fire = on_claim_fire
        self._delete_one_shot_after_fire = delete_one_shot_after_fire
        self._check_interval = check_interval
        self._task: Optional[asyncio.Task] = None
        self._stopped = False
        self._fired_this_minute: set[str] = set()
        self._last_minute_key: tuple[int, int, int, int, int] | None = None

    def _reset_fired_set_if_needed(self, now: datetime) -> None:
        minute_key = (now.year, now.month, now.day, now.hour, now.minute)
        if minute_key != self._last_minute_key:
            self._fired_this_minute.clear()
            self._last_minute_key = minute_key

    async def _tick(self) -> None:
        """Check all jobs and fire any that match current time."""
        now = datetime.now()
        self._reset_fired_set_if_needed(now)

        jobs = self._storage.list_all()
        pending_ids = self._storage.pending_ids()
        for job in jobs:
            job_id = str(job.get("id") or "")
            if not job_id or job_id in self._fired_this_minute:
                continue
            try:
                matches = job_id in pending_ids or _cron_matches_now(
                    str(job.get("cron") or ""), now
                )
            except Exception:
                logger.exception("Invalid cron job %s skipped", job_id)
                continue
            if matches:
                if (
                    self._on_claim_fire is None
                    and self._on_job_fire is None
                    and self._on_fire is None
                ):
                    continue
                claim = self._storage.claim(job_id, minute=int(now.timestamp()) // 60)
                if claim is None:
                    continue
                self._fired_this_minute.add(job_id)
                logger.info("Firing cron job %s: %s", job_id, str(job.get("prompt", ""))[:50])
                delivered = False
                transferred = False
                try:
                    if self._on_claim_fire is not None:
                        # A queued consumer now owns the lock and acknowledgement.
                        transferred = self._on_claim_fire(claim) is not False
                    elif self._on_job_fire is not None:
                        delivered = self._on_job_fire(dict(claim.job)) is not False
                    elif self._on_fire is not None:
                        self._on_fire(str(claim.job.get("prompt") or ""))
                        delivered = True
                    if delivered:
                        claim.complete(delete_one_shot=self._delete_one_shot_after_fire)
                except Exception:
                    logger.exception("Error firing cron job %s", job_id)
                finally:
                    if not transferred:
                        claim.release()

    async def _loop(self) -> None:
        """Main scheduler loop."""
        while not self._stopped:
            try:
                await self._tick()
            except Exception:
                logger.exception("Cron scheduler tick error")
            await asyncio.sleep(self._check_interval)

    def start(self) -> None:
        """Start the scheduler in the background."""
        if self._task is not None:
            return
        self._stopped = False
        self._task = asyncio.ensure_future(self._loop())
        logger.info("Cron scheduler started (interval=%ss)", self._check_interval)

    def stop(self) -> None:
        """Stop the scheduler."""
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            self._task = None
        logger.info("Cron scheduler stopped")

    async def stop_async(self) -> None:
        """Stop the scheduler and wait for the background task to settle."""

        task = self._task
        self.stop()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
