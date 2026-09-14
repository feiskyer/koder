"""Background termination must retain and join its exact monitor task."""

from __future__ import annotations

import asyncio

import pytest

from koder_agent.tools.shell import BackgroundShell, BackgroundShellManager


class _Reader:
    def __init__(self):
        self.started = asyncio.Event()
        self.closing = asyncio.Event()
        self.release = asyncio.Event()

    async def readline(self):
        self.started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.closing.set()
            await self.release.wait()


class _ExitedProcess:
    returncode = 0

    def __init__(self, reader):
        self.stdout = reader

    async def wait(self):
        return self.returncode


@pytest.fixture(autouse=True)
def isolated_manager(monkeypatch):
    monkeypatch.setattr(BackgroundShellManager, "_shells", {})
    monkeypatch.setattr(BackgroundShellManager, "_monitor_tasks", {})


async def _start(shell_id):
    reader = _Reader()
    shell = BackgroundShell(
        shell_id=shell_id,
        command="synthetic completed process",
        process=_ExitedProcess(reader),
        start_time=0,
    )
    BackgroundShellManager.add(shell)
    await BackgroundShellManager.start_monitor(shell_id)
    monitor = BackgroundShellManager._monitor_tasks[shell_id]
    await asyncio.wait_for(reader.started.wait(), timeout=2)
    return shell, monitor, reader


async def _finish(baseline, *owned):
    tasks = (asyncio.all_tasks() - baseline) | set(owned)
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2)


@pytest.mark.asyncio
async def test_terminate_joins_cancelled_monitor_before_removing_shell():
    baseline = asyncio.all_tasks()
    shell, monitor, reader = await _start("join-monitor")
    owner = asyncio.create_task(BackgroundShellManager.terminate(shell.shell_id))
    try:
        await asyncio.wait_for(reader.closing.wait(), timeout=2)
        assert not owner.done(), "termination returned while monitor cleanup was pending"
        assert BackgroundShellManager.get(shell.shell_id) is shell
        reader.release.set()
        assert await owner is shell
        assert monitor.done()
        assert shell.shell_id not in BackgroundShellManager._monitor_tasks
        assert BackgroundShellManager.get(shell.shell_id) is None
    finally:
        reader.release.set()
        await _finish(baseline, owner, monitor)


@pytest.mark.asyncio
async def test_concurrent_terminate_and_cancel_wait_for_one_owned_cleanup(cancellation_observer):
    observe, cancellations = cancellation_observer
    baseline = asyncio.all_tasks()
    shell, monitor, reader = await _start("shared-cleanup")
    first = asyncio.create_task(observe(BackgroundShellManager.terminate(shell.shell_id)))
    second = None
    try:
        await asyncio.wait_for(reader.closing.wait(), timeout=2)
        second = asyncio.create_task(BackgroundShellManager.terminate(shell.shell_id))
        first.cancel("cancel one waiter")
        await asyncio.sleep(0)
        first.cancel("cancel again")
        await asyncio.sleep(0.02)
        assert not first.done()
        assert not second.done()
        assert not monitor.done()
        reader.release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await second is shell
        assert monitor.done()
        assert [error.args for error in cancellations] == [("cancel one waiter",)]
        assert BackgroundShellManager.get(shell.shell_id) is None
    finally:
        reader.release.set()
        await _finish(baseline, first, monitor, *((second,) if second is not None else ()))


@pytest.mark.asyncio
async def test_old_termination_preserves_replacement_shell_and_monitor():
    baseline = asyncio.all_tasks()
    old, old_monitor, old_reader = await _start("reused-id")
    owner = asyncio.create_task(BackgroundShellManager.terminate(old.shell_id))
    new_reader = None
    new_monitor = None
    try:
        await asyncio.wait_for(old_reader.closing.wait(), timeout=2)
        new, new_monitor, new_reader = await _start(old.shell_id)
        old_reader.release.set()
        assert await owner is old
        await asyncio.gather(old_monitor, return_exceptions=True)
        assert BackgroundShellManager.get(old.shell_id) is new
        assert BackgroundShellManager._monitor_tasks.get(old.shell_id) is new_monitor
        assert not new_monitor.done()
    finally:
        old_reader.release.set()
        if new_reader is not None:
            new_reader.release.set()
        await _finish(
            baseline, owner, old_monitor, *((new_monitor,) if new_monitor is not None else ())
        )


@pytest.mark.asyncio
async def test_failed_process_termination_keeps_record_and_allows_retry(monkeypatch):
    baseline = asyncio.all_tasks()
    shell, monitor, reader = await _start("retry-cleanup")
    original = shell.terminate
    attempts = 0

    async def terminate():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("controlled process termination failure")
        await original()

    monkeypatch.setattr(shell, "terminate", terminate)
    owner = None
    try:
        with pytest.raises(OSError, match="controlled process termination failure"):
            await BackgroundShellManager.terminate(shell.shell_id)
        assert BackgroundShellManager.get(shell.shell_id) is shell
        assert BackgroundShellManager._monitor_tasks.get(shell.shell_id) is monitor
        assert not monitor.done()
        owner = asyncio.create_task(BackgroundShellManager.terminate(shell.shell_id))
        await asyncio.wait_for(reader.closing.wait(), timeout=2)
        reader.release.set()
        assert await owner is shell
        await asyncio.gather(monitor, return_exceptions=True)
        assert attempts == 2
        assert BackgroundShellManager.get(shell.shell_id) is None
    finally:
        reader.release.set()
        await _finish(baseline, monitor, *((owner,) if owner is not None else ()))
