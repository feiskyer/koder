"""Nested uv invocations must reuse the environment running the test suite."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import tmux_feature_scenarios as scenarios


@pytest.mark.skipif(sys.prefix == sys.base_prefix, reason="requires the uv test virtualenv")
def test_child_cli_environment_is_pinned_outside_disposable_application_homes():
    assert os.environ["UV_PROJECT_ENVIRONMENT"] == sys.prefix
    assert os.environ["UV_PYTHON"] == sys.executable
    assert os.environ["UV_NO_SYNC"] == "1"


def test_pytest_bootstrap_disables_child_uv_dotenv_loading(tmp_path):
    """Exercise the checked-in fixture without inheriting the outer runner's flag."""
    probe = tmp_path / "probe"
    probe.mkdir()
    (probe / "conftest.py").write_text(
        (scenarios.PROJECT_ROOT / "tests/conftest.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (probe / "test_environment.py").write_text(
        "import os\n"
        "os.environ.pop('UV_NO_ENV_FILE', None)\n"
        "def test_nested_uv_has_no_dotenv():\n"
        "    assert os.environ.get('UV_NO_ENV_FILE') == '1'\n",
        encoding="utf-8",
    )
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path / "synthetic-home"),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(probe)],
        cwd=probe,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_tmux_launch_passes_test_environment_to_the_server(tmp_path, monkeypatch):
    launches = []

    def fake_tmux(*args, **_kwargs):
        if args[0] == "new-session":
            launches.append(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(scenarios, "_tmux", fake_tmux)
    monkeypatch.setattr(scenarios, "_wait_for_prompt", lambda *_a, **_kw: "| ⚡ Koder | │>")
    monkeypatch.setattr(scenarios, "_session_exists", lambda _session: True)
    monkeypatch.setattr(scenarios.sys, "prefix", "/test/virtual environment")
    monkeypatch.setattr(scenarios.sys, "base_prefix", "/test/base")
    monkeypatch.setattr(scenarios.sys, "executable", "/test/virtual environment/bin/python")
    scenario = SimpleNamespace(suite="test", name="environment", payload={})

    scenarios._launch_session(tmp_path, tmp_path, scenario, fake_openai_url=None)

    tokens = launches[0]
    assert "UV_PROJECT_ENVIRONMENT=/test/virtual environment" in tokens
    assert "UV_PYTHON=/test/virtual environment/bin/python" in tokens


def test_tmux_launch_passes_paths_and_cli_arguments_without_a_shell(tmp_path, monkeypatch):
    launches = []

    def fake_tmux(*args, **_kwargs):
        launches.append(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(scenarios, "_tmux", fake_tmux)
    monkeypatch.setattr(scenarios, "_wait_for_prompt", lambda *_a, **_kw: "| ⚡ Koder | │>")
    monkeypatch.setattr(scenarios, "_session_exists", lambda _session: True)
    monkeypatch.setattr(scenarios.shutil, "which", lambda _name: "/tools with spaces/uv")
    repo = tmp_path / "repo with spaces"
    home = tmp_path / "isolated home"
    argument = "literal $(do-not-execute); 'quoted' input\nsecond line"
    scenario = SimpleNamespace(
        suite="test",
        name="argv",
        payload={
            "cli_args": ["--append-system-prompt", argument],
            "env": {"KODER_SMALL_MODEL": "fixture/model"},
        },
    )

    scenarios._launch_session(home, repo, scenario, fake_openai_url=None)

    argv = launches[0]
    assert argv[argv.index("-c") + 1] == str(repo)
    assert f"HOME={home}" in argv
    assert "KODER_SMALL_MODEL=fixture/model" in argv
    assert argv[argv.index("/tools with spaces/uv") :] == (
        "/tools with spaces/uv",
        "--project",
        str(scenarios.PROJECT_ROOT),
        "run",
        "--no-sync",
        "--no-env-file",
        "koder",
        "--teammate-mode",
        "tmux",
        "--append-system-prompt",
        argument,
    )


@pytest.mark.parametrize("raise_on_wait", [False, True])
def test_failed_tmux_startup_retires_the_session_it_created(tmp_path, monkeypatch, raise_on_wait):
    calls = []

    def fake_tmux(*args, **_kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0)

    def wait_for_prompt(*_args, **_kwargs):
        if raise_on_wait:
            raise RuntimeError("capture failed")
        return "not a Koder prompt"

    monkeypatch.setattr(scenarios, "_tmux", fake_tmux)
    monkeypatch.setattr(scenarios, "_wait_for_prompt", wait_for_prompt)
    monkeypatch.setattr(scenarios, "_session_exists", lambda _session: True)
    scenario = SimpleNamespace(suite="test", name="failed-start", payload={})

    with pytest.raises(RuntimeError, match="capture failed|prompt did not appear"):
        scenarios._launch_session(tmp_path, tmp_path, scenario, fake_openai_url=None)

    created = calls[0][calls[0].index("-s") + 1]
    assert ("kill-session", "-t", created) in calls
