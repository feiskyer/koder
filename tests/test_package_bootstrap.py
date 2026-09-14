"""Package metadata and low-level imports must not initialize model providers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib.metadata import version as package_version
from pathlib import Path

import pytest


def _run(script, tmp_path, python_child_environment, *arguments):
    result = subprocess.run(
        [sys.executable, "-c", script, *arguments],
        cwd=tmp_path,
        env=python_child_environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize(
    "module",
    [
        "koder_agent",
        "koder_agent.core.goals",
        "koder_agent.core.turn_cancellation",
        "koder_agent.utils.async_tasks",
        "koder_agent.utils.image_input",
        "koder_agent.harness.version_info",
    ],
)
def test_low_level_import_does_not_initialize_provider_sdk(
    tmp_path, python_child_environment, module
):
    script = """
import importlib
import sys

forbidden = {"litellm", "agents", "openai", "dotenv"}
assert forbidden.isdisjoint(sys.modules)

class RejectProviderImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in forbidden:
            raise AssertionError("Unexpected provider initialization: " + fullname)
        return None

sys.meta_path.insert(0, RejectProviderImports())
importlib.import_module(sys.argv[1])
assert forbidden.isdisjoint(sys.modules)
from koder_agent.litellm_cost_map import load_vendored_model_cost_map
assert load_vendored_model_cost_map.cache_info().currsize == 0
print("LIGHT_IMPORT_OK")
"""
    assert _run(script, tmp_path, python_child_environment, module) == "LIGHT_IMPORT_OK"


def test_package_version_matches_distribution_and_runtime(tmp_path, python_child_environment):
    script = """
from importlib.metadata import version
import koder_agent
from koder_agent.harness.version_info import resolve_runtime_version

assert koder_agent.__version__ == version("koder"), (koder_agent.__version__, version("koder"))
assert resolve_runtime_version() == koder_agent.__version__
print("VERSION_PARITY_OK")
"""
    assert _run(script, tmp_path, python_child_environment) == "VERSION_PARITY_OK"


def test_headless_cli_reports_the_same_distribution_version(tmp_path, python_child_environment):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("the declared CLI entrypoint requires uv")
    result = subprocess.run(
        [
            uv,
            "run",
            "--project",
            str(Path(__file__).resolve().parents[1]),
            "--no-sync",
            "--no-env-file",
            "koder",
            "--version",
        ],
        cwd=tmp_path,
        env=python_child_environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{package_version('koder')} (Koder)"


def test_package_import_does_not_load_a_synthetic_dotenv(tmp_path, python_child_environment):
    (tmp_path / ".env").write_text("KODER_BOOTSTRAP_SENTINEL=synthetic-dotenv\n", encoding="utf-8")
    script = """
import os
assert "KODER_BOOTSTRAP_SENTINEL" not in os.environ
import koder_agent
assert "KODER_BOOTSTRAP_SENTINEL" not in os.environ, "package import loaded dotenv"
print("NO_DOTENV_IMPORT")
"""
    assert _run(script, tmp_path, python_child_environment) == "NO_DOTENV_IMPORT"


@pytest.mark.parametrize(
    "module",
    [
        "koder_agent.utils.model_info",
        "koder_agent.utils.client",
        "koder_agent.core.usage_tracker",
        "koder_agent.auth.providers",
        "koder_agent.agentic.agent",
    ],
)
def test_model_consumer_installs_vendored_map_with_real_sdk(
    tmp_path, python_child_environment, module
):
    script = """
import importlib
import sys

importlib.import_module(sys.argv[1])
import litellm
from koder_agent.litellm_cost_map import load_vendored_model_cost_map
vendored = load_vendored_model_cost_map()
assert len(litellm.model_cost) >= len(vendored)
assert litellm.model_cost["gpt-4o"] == vendored["gpt-4o"]
assert litellm.get_model_info("gpt-4o")["max_input_tokens"] == vendored["gpt-4o"]["max_input_tokens"]
print("COST_MAP_READY")
"""
    assert _run(script, tmp_path, python_child_environment, module) == "COST_MAP_READY"


def test_preimported_sdk_custom_model_survives_consumer_import(tmp_path, python_child_environment):
    script = """
import litellm
litellm.model_cost["synthetic-custom-model"] = {"max_input_tokens": 12345}
import koder_agent.utils.model_info
from koder_agent.litellm_cost_map import load_vendored_model_cost_map
assert litellm.model_cost["synthetic-custom-model"]["max_input_tokens"] == 12345
assert litellm.model_cost["gpt-4o"] == load_vendored_model_cost_map()["gpt-4o"]
print("CUSTOM_MODEL_PRESERVED")
"""
    assert _run(script, tmp_path, python_child_environment) == "CUSTOM_MODEL_PRESERVED"


def test_vision_lookup_initializes_its_model_registry_when_used(tmp_path, python_child_environment):
    script = """
import sys
from koder_agent.utils.image_input import model_supports_vision
assert "litellm" not in sys.modules
assert model_supports_vision("gpt-4o")
import litellm
from koder_agent.litellm_cost_map import load_vendored_model_cost_map
assert litellm.model_cost["gpt-4o"] == load_vendored_model_cost_map()["gpt-4o"]
print("VISION_REGISTRY_READY")
"""
    assert _run(script, tmp_path, python_child_environment) == "VISION_REGISTRY_READY"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork lock-reset contract")
def test_cost_map_lock_is_reset_in_a_single_threaded_fork(tmp_path, python_child_environment):
    script = """
import multiprocessing
import os
import threading
import types
from koder_agent import litellm_cost_map as costs

assert threading.active_count() == 1, "fork test requires an otherwise idle parent"
original = costs._INSTALL_LOCK
parent_pid = os.getpid()

def child():
    assert os.getpid() != parent_pid
    assert costs._INSTALL_LOCK is not original
    module = types.SimpleNamespace(model_cost={"custom": {"max_input_tokens": 12}})
    assert costs.install_vendored_litellm_model_cost_map(module)["custom"]["max_input_tokens"] == 12

original.acquire()
process = multiprocessing.get_context("fork").Process(target=child)
try:
    process.start()
    process.join(timeout=5)
    assert process.exitcode == 0, process.exitcode
finally:
    original.release()
    if process.is_alive():
        process.terminate()
        process.join(timeout=3)
    process.close()
print("FORK_LOCK_RESET")
"""
    assert _run(script, tmp_path, python_child_environment) == "FORK_LOCK_RESET"
