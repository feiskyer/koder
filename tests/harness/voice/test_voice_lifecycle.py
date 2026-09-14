"""Deterministic voice lifecycle probes; no audio device or provider access."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from koder_agent.harness.voice.service import (
    SoundDeviceRecorder,
    VoiceDictationController,
    VoiceDictationError,
)


class Recorder:
    def __init__(self, *, fail_start=False, fail_stop=False, fail_cancel=False):
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.fail_cancel = fail_cancel
        self.cancelled = False
        self.active = False

    def start(self):
        if self.fail_start:
            raise VoiceDictationError("synthetic start failure")
        self.active = True

    def stop(self):
        if self.fail_stop:
            raise VoiceDictationError("synthetic stop failure")
        self.active = False
        return b"synthetic audio"

    def cancel(self):
        self.cancelled = True
        self.active = False
        if self.fail_cancel:
            raise VoiceDictationError("synthetic cleanup failure")


def controller_for(recorder, transcriber=None):
    return VoiceDictationController(
        config_getter=lambda: SimpleNamespace(
            voice=SimpleNamespace(enabled=True, provider="openai")
        ),
        model_provider_getter=lambda: "openai",
        recorder_factory=lambda: recorder,
        transcriber=transcriber or SimpleNamespace(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["start", "stop", "status"])
async def test_capture_failure_releases_recorder_and_controller(failure):
    recorder = Recorder(fail_start=failure == "start", fail_stop=failure == "stop")
    controller = controller_for(recorder)

    def status(value):
        if failure == "status" and value == "transcribing":
            raise RuntimeError("synthetic status failure")

    with pytest.raises((RuntimeError, VoiceDictationError)):
        await controller.start_recording(on_status=status)
        await controller.stop_recording(on_status=status)
    assert not controller.is_busy
    assert recorder.cancelled


@pytest.mark.asyncio
async def test_cancel_transcription_drains_it_and_preserves_new_recording():
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    partials = []
    statuses = []

    class Transcriber:
        async def transcribe(self, *, on_partial, **_kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                # A late callback during provider cleanup must not overwrite the
                # cancellation notice or the next recording.
                on_partial("stale partial")
                cleaned.set()

    recorder = Recorder()
    controller = controller_for(recorder, Transcriber())
    await controller.start_recording(on_status=statuses.append)
    task = asyncio.create_task(
        controller.stop_recording(on_status=statuses.append, on_partial=partials.append)
    )
    await asyncio.wait_for(entered.wait(), timeout=2)
    controller.cancel()
    await controller.start_recording(on_status=statuses.append)
    try:
        await asyncio.sleep(0)
        assert task.done(), "cancel left the provider request running"
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleaned.is_set()
        assert controller.is_recording
        assert recorder.active
        assert partials == []
        assert statuses == ["recording", "transcribing", "recording"]
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        controller.cancel()


@pytest.mark.asyncio
async def test_suppressed_provider_cancellation_never_returns_stale_text():
    entered = asyncio.Event()

    class Transcriber:
        async def transcribe(self, **_kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return "stale final transcript"

    controller = controller_for(Recorder(), Transcriber())
    await controller.start_recording(on_status=lambda _value: None)
    task = asyncio.create_task(controller.stop_recording(on_status=lambda _value: None))
    await asyncio.wait_for(entered.wait(), timeout=2)
    controller.cancel()
    try:
        await asyncio.sleep(0)
        assert task.done()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_resets_even_if_recorder_cleanup_fails():
    controller = controller_for(Recorder(fail_cancel=True))
    await controller.start_recording(on_status=lambda _value: None)
    controller.cancel()
    assert not controller.is_busy


@pytest.mark.parametrize("failure", ["start", "stop", "cancel"])
def test_sounddevice_stream_closes_on_failure(monkeypatch, failure):
    class Stream:
        closed = False

        def start(self):
            if failure == "start":
                raise RuntimeError("synthetic hardware start error")

        def stop(self):
            raise RuntimeError("synthetic hardware stop error")

        def close(self):
            self.closed = True

    stream = Stream()
    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(RawInputStream=lambda **_kw: stream)
    )
    recorder = SoundDeviceRecorder()
    with pytest.raises((RuntimeError, VoiceDictationError)):
        recorder.start()
        getattr(recorder, failure)()
    assert stream.closed
    assert recorder._stream is None
    recorder.cancel()


@pytest.mark.asyncio
async def test_external_task_cancellation_releases_controller():
    entered = asyncio.Event()

    class Transcriber:
        async def transcribe(self, **_kwargs):
            entered.set()
            await asyncio.Event().wait()

    recorder = Recorder()
    controller = controller_for(recorder, Transcriber())
    statuses = []
    await controller.start_recording(on_status=statuses.append)
    task = asyncio.create_task(controller.stop_recording(on_status=statuses.append))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not controller.is_busy
    assert recorder.cancelled
    assert statuses[-1] is None
