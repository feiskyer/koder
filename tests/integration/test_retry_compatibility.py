"""Retry consumers keep their contracts without deprecated asyncio detection."""

import asyncio
import contextvars
import inspect
import logging
import random
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import litellm.exceptions
import pytest
from agents import ModelSettings
from agents.models.interface import ModelTracing

from koder_agent.agentic.agent import RetryingLitellmModel
from koder_agent.auth import oauth_routing
from koder_agent.utils import client

pytestmark = pytest.mark.filterwarnings(
    r"error:.*asyncio\.iscoroutinefunction.*:DeprecationWarning"
)


def _transient_error():
    return litellm.exceptions.InternalServerError(
        message="synthetic retry failure",
        model="fixture",
        llm_provider="fixture",
    )


def _consumer(monkeypatch, kind, request):
    """Substitute only provider I/O and auxiliary config resolution."""
    if kind == "auxiliary":
        monkeypatch.setattr(
            client,
            "_resolve_completion_settings",
            lambda _model: (None, None, "openai", "fixture", False),
        )
        monkeypatch.setattr(client, "_setup_provider_env_vars", lambda *_args: None)
        monkeypatch.setattr(
            client,
            "_compute_effective_model",
            lambda *_args: ("openai/fixture", False, "synthetic-key"),
        )
        monkeypatch.setattr(client, "get_configured_context_window", lambda *_a, **_k: 4096)
        monkeypatch.setattr(client, "_resolve_base_url", lambda *_args: None)
        monkeypatch.setattr(
            client, "estimate_model_request_preflight", lambda **_kwargs: SimpleNamespace(fits=True)
        )
        monkeypatch.setattr(oauth_routing, "acompletion", request)

        async def invoke():
            return await client.llm_completion(
                [{"role": "user", "content": "fixture"}],
                model="openai/fixture",
                response_reserve=16,
            )

        return invoke

    monkeypatch.setattr(client.litellm, "aresponses", request)
    model = RetryingLitellmModel(model="github_copilot/gpt-5.1-codex", context_window=4096)

    async def invoke():
        args = (None, "fixture", ModelSettings(max_tokens=16), [], None, [], ModelTracing.DISABLED)
        if kind == "stream":
            return [event async for event in model.stream_response(*args)]
        return await model.get_response(*args)

    return invoke


@pytest.mark.parametrize("remove_legacy_api", [False, True])
def test_retry_imports_need_no_deprecated_api_or_global_patch(
    tmp_path, python_child_environment, remove_legacy_api
):
    source = textwrap.dedent(f"""\
        import asyncio
        import inspect
        import warnings

        if {remove_legacy_api!r} and hasattr(asyncio, "iscoroutinefunction"):
            del asyncio.iscoroutinefunction
        original = getattr(asyncio, "iscoroutinefunction", None)
        warnings.filterwarnings(
            "error", message=r".*asyncio\\.iscoroutinefunction.*", category=DeprecationWarning
        )
        from koder_agent.agentic.agent import RetryingLitellmModel
        from koder_agent.utils.client import llm_completion

        assert getattr(asyncio, "iscoroutinefunction", None) is original
        assert inspect.iscoroutinefunction(llm_completion)
        assert inspect.iscoroutinefunction(RetryingLitellmModel.get_response)
        print("retry-imports-ok")
        """)
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env=python_child_environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "retry-imports-ok"


@pytest.mark.asyncio
@pytest.mark.parametrize(("kind", "limit"), [("auxiliary", 3), ("model", 3), ("stream", 5)])
async def test_retry_limit_jitter_and_original_exception(monkeypatch, caplog, kind, limit):
    error = _transient_error()
    attempts = []
    waits = []

    async def fail(**_kwargs):
        attempts.append(True)
        raise error

    async def sleep(seconds):
        waits.append(float(seconds))

    monkeypatch.setattr(asyncio, "sleep", sleep)
    monkeypatch.setattr(random, "uniform", lambda _low, high: high)
    caplog.set_level(logging.INFO, logger="koder_agent.agentic.agent")
    invoke = _consumer(monkeypatch, kind, fail)
    with pytest.raises(type(error)) as raised:
        await invoke()

    assert raised.value is error
    assert len(attempts) == limit
    assert waits == [float(2**index) for index in range(limit - 1)]
    if kind != "auxiliary":
        messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == "koder_agent.agentic.agent" and "will retry" in record.getMessage()
        ]
        assert len(messages) == limit - 1
        assert all(
            message.endswith(f"[attempt {attempt}/{limit}]")
            for attempt, message in enumerate(messages, start=1)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["auxiliary", "model", "stream"])
async def test_nonretryable_error_is_not_wrapped_or_delayed(monkeypatch, kind):
    error = ValueError("synthetic nonretryable failure")
    attempts = []

    async def fail(**_kwargs):
        attempts.append(True)
        raise error

    async def unexpected_wait(_seconds):
        raise AssertionError("nonretryable failure waited")

    monkeypatch.setattr(asyncio, "sleep", unexpected_wait)
    invoke = _consumer(monkeypatch, kind, fail)
    with pytest.raises(ValueError) as raised:
        await invoke()
    assert raised.value is error
    assert len(attempts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["auxiliary", "model", "stream"])
@pytest.mark.parametrize("phase", ["request", "wait"])
async def test_cancellation_stops_retry_and_joins_active_operation(monkeypatch, kind, phase):
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    attempts = []

    async def hold():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    async def request(**_kwargs):
        attempts.append(True)
        if phase == "request":
            await hold()
        raise _transient_error()

    async def sleep(_seconds):
        await hold()

    monkeypatch.setattr(asyncio, "sleep", sleep)
    invoke = _consumer(monkeypatch, kind, request)
    task = asyncio.create_task(invoke())
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()
    assert len(attempts) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["auxiliary", "model"])
async def test_concurrent_retry_invocations_keep_independent_state(monkeypatch, kind):
    label = contextvars.ContextVar("retry-test-label")
    attempts = {"alpha": 0, "beta": 0}
    yield_control = asyncio.sleep

    async def sleep(_seconds):
        await yield_control(0)

    async def request(**_kwargs):
        name = label.get()
        attempts[name] += 1
        await yield_control(0)
        if attempts[name] <= (1 if name == "alpha" else 2):
            raise _transient_error()
        return SimpleNamespace(
            id=name,
            status="completed",
            usage=None,
            output=[],
            choices=[SimpleNamespace(message=SimpleNamespace(content=name))],
        )

    monkeypatch.setattr(asyncio, "sleep", sleep)
    invoke = _consumer(monkeypatch, kind, request)

    async def run(name):
        token = label.set(name)
        try:
            result = await invoke()
            return result if kind == "auxiliary" else result.response_id
        finally:
            label.reset(token)

    assert inspect.iscoroutinefunction(invoke)
    assert await asyncio.gather(run("alpha"), run("beta")) == ["alpha", "beta"]
    assert attempts == {"alpha": 2, "beta": 3}
