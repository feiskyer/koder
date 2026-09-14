"""Runtime bridge between durable cron jobs and the active agent scheduler."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

from ...utils.async_tasks import await_owned_task
from .scheduler import CronScheduler
from .storage import CronClaim, CronStorage, default_cron_storage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueuedCronJob:
    id: str | None
    prompt: str
    claim: CronClaim | None = None


class CronPromptRunner:
    """Drain fired cron prompts into whichever scheduler is currently active."""

    def __init__(
        self,
        prompt_dispatcher: Callable[..., Awaitable[str]],
        *,
        storage: CronStorage | None = None,
        check_interval: float = 60.0,
    ):
        if storage is None:
            storage = default_cron_storage()
        self._prompt_dispatcher = prompt_dispatcher
        self._storage = storage
        self._queue: asyncio.Queue[QueuedCronJob] = asyncio.Queue()
        self._pending_job_ids: set[str] = set()
        self._cron_scheduler = CronScheduler(
            storage,
            on_claim_fire=self._enqueue_claim,
            check_interval=check_interval,
        )
        self._consumer_task: asyncio.Task | None = None
        self._stop_task: asyncio.Task | None = None

    @property
    def pending_job_ids(self) -> set[str]:
        return set(self._pending_job_ids)

    def enqueue(self, prompt: str) -> bool:
        """Queue a manual prompt unless shutdown has started."""
        if self._stop_task is not None:
            return False
        self._queue.put_nowait(
            QueuedCronJob(
                id=None,
                prompt=prompt,
            )
        )
        return True

    def enqueue_job(self, job: dict) -> bool:
        """Queue a stored cron job, skipping duplicates while it is pending/running."""
        if self._stop_task is not None:
            return False
        job_id = str(job.get("id") or "")
        prompt = str(job.get("prompt") or "")
        if not job_id or not prompt:
            logger.warning("Cron job missing id or prompt: %s", job)
            return False
        if job_id in self._pending_job_ids:
            return False
        claim = self._storage.claim(job_id)
        if claim is None:
            return False
        accepted = False
        try:
            accepted = self._enqueue_claim(claim)
            return accepted
        finally:
            if not accepted:
                claim.release()

    def _enqueue_claim(self, claim: CronClaim) -> bool:
        """Accept ownership of a claim from the poller or a direct enqueue."""
        job = claim.job
        job_id = str(job["id"])
        prompt = str(job.get("prompt") or "")
        if self._stop_task is not None or job_id in self._pending_job_ids or not prompt:
            return False
        self._pending_job_ids.add(job_id)
        self._queue.put_nowait(
            QueuedCronJob(
                id=job_id,
                prompt=prompt,
                claim=claim,
            )
        )
        return True

    def start(self) -> None:
        """Start polling cron storage and consuming fired prompts."""
        if self._stop_task is not None and not self._stop_task.done():
            raise RuntimeError("Cron prompt runner shutdown is still in progress")
        if self._consumer_task is not None:
            return
        self._stop_task = None
        self._cron_scheduler.start()
        self._consumer_task = asyncio.create_task(self._consume())

    async def stop(self) -> None:
        """Stop background polling and prompt consumption."""
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._stop_owned())
        await await_owned_task(self._stop_task)

    async def _stop_owned(self) -> None:
        try:
            await self._cron_scheduler.stop_async()
        except Exception:
            logger.exception("Cron scheduler shutdown failed")
            raise
        finally:
            if self._consumer_task is not None:
                self._consumer_task.cancel()
                results = await asyncio.gather(
                    self._consumer_task,
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, BaseException) and not isinstance(
                        result,
                        asyncio.CancelledError,
                    ):
                        logger.debug(
                            "Cron prompt consumer shutdown failed",
                            exc_info=(type(result), result, result.__traceback__),
                        )
                self._consumer_task = None
            # Stop is a delivery boundary, not a durable job deletion. Pending
            # receipts are retried after a fresh start, even outside their
            # original matching minute.
            while not self._queue.empty():
                queued = self._queue.get_nowait()
                if queued.claim is not None:
                    queued.claim.release()
                self._queue.task_done()
            self._pending_job_ids.clear()

    async def _consume(self) -> None:
        while True:
            queued = await self._queue.get()
            try:
                if queued.claim is not None and not queued.claim.is_current():
                    # /loop delete may have run while another prompt was busy.
                    continue
                await self._prompt_dispatcher(queued.prompt, render_output=True)
                if queued.claim is not None:
                    queued.claim.complete()
            except Exception:
                logger.exception("Cron prompt failed: %s", queued.prompt[:80])
            finally:
                if queued.claim is not None:
                    queued.claim.release()
                if queued.id is not None:
                    self._pending_job_ids.discard(queued.id)
                self._queue.task_done()
