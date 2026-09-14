"""Global pytest test-harness setup for the repository."""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def _reuse_test_environment_for_child_clis():
    """Keep isolated CLI homes from rebuilding the checkout's virtualenv.

    Tests change the child CLI's home to isolate application state. Without an
    explicit interpreter/environment, uv can replace .venv with an interpreter
    downloaded into that disposable home, leaving .venv broken after cleanup.
    """
    if sys.prefix == sys.base_prefix:
        yield
        return
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("UV_PROJECT_ENVIRONMENT", sys.prefix)
        patch.setenv("UV_PYTHON", sys.executable)
        patch.setenv("UV_NO_SYNC", "1")
        patch.setenv("UV_NO_ENV_FILE", "1")
        yield


@pytest.fixture
def python_child_environment(tmp_path):
    """Import checkout code from a synthetic child profile and working directory."""
    child_home = tmp_path / "python-home"
    child_home.mkdir()
    environment = dict(os.environ)
    environment.update(
        HOME=str(child_home),
        USERPROFILE=str(child_home),
        PYTHONPATH=str(PROJECT_ROOT),
        UV_NO_ENV_FILE="1",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        OPENAI_AGENTS_DISABLE_TRACING="1",
    )
    return environment


@pytest.fixture
def cancellation_observer():
    """Capture the application exception before older Tasks drop its message.

    Await the wrapped coroutine in the same task: cancellation and resource
    ownership assertions must not gain another task boundary from this fixture.
    """
    observed = []

    async def observe(awaitable):
        try:
            return await awaitable
        except asyncio.CancelledError as error:
            observed.append(error)
            raise

    return observe, observed


@pytest.fixture
def event_loop_progress_probe():
    """Wrap a synchronous call and observe a loop heartbeat before it returns."""
    observed = []

    def wrap(function):
        loop = asyncio.get_running_loop()

        def blocking(*args, **kwargs):
            heartbeat = threading.Event()
            loop.call_soon_threadsafe(heartbeat.set)
            observed.append(heartbeat.wait(timeout=1))
            return function(*args, **kwargs)

        return blocking

    return wrap, observed


# Override anyio's default ``anyio_backend`` fixture so that @pytest.mark.anyio
# tests only run under asyncio (trio is not installed).
@pytest.fixture(scope="module", params=["asyncio"])
def anyio_backend(request):
    return request.param


if "ddgs" not in sys.modules:
    ddgs_stub = types.ModuleType("ddgs")

    class _StubDDGS:
        def text(self, *_args, **_kwargs):
            return []

    ddgs_stub.DDGS = _StubDDGS
    sys.modules["ddgs"] = ddgs_stub

    ddgs_exceptions = types.ModuleType("ddgs.exceptions")

    class DDGSException(Exception):
        pass

    ddgs_exceptions.DDGSException = DDGSException
    sys.modules["ddgs.exceptions"] = ddgs_exceptions
