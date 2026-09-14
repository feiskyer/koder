"""Scenario failures must retain evidence and release acquired resources."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import tmux_feature_scenarios as scenarios
from scripts.tmux_feature_scenarios import _launch_session as launch_real_session


@pytest.fixture
def scenario_run(tmp_path, monkeypatch):
    calls = []
    provider = object()

    def prepare(root):
        profile, repo = root / "home", root / "repo"
        profile.mkdir()
        repo.mkdir()
        return profile, repo

    def tmux(*args, **_kwargs):
        calls.append(("tmux", args))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(scenarios, "_prepare_workspace", prepare)
    monkeypatch.setattr(scenarios, "_write_prelaunch_files", lambda *_a, **_kw: None)
    monkeypatch.setattr(scenarios, "_start_fake_openai", lambda *_a, **_kw: (provider, None))
    monkeypatch.setattr(scenarios, "_launch_session", lambda *_a, **_kw: "owned-session")
    monkeypatch.setattr(scenarios, "_tmux", tmux)
    monkeypatch.setattr(scenarios, "_stop_fake_openai", lambda proc: calls.append(("stop", proc)))
    monkeypatch.setattr(
        scenarios, "_copy_fake_openai_log", lambda *_a, **_kw: calls.append(("copy", None))
    )
    monkeypatch.setattr(scenarios, "_dispatch_turn_input_actions", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        scenarios, "_wait_for_assertions", lambda *_a, **_kw: (True, "fixture capture")
    )
    monkeypatch.setattr(scenarios, "_run_post_assertions", lambda *_a, **_kw: [])
    monkeypatch.setattr(scenarios.time, "sleep", lambda _seconds: None)
    ref = scenarios.ScenarioRef(
        suite="features",
        name="lifecycle",
        payload={"turns": [{"send": "/status", "expect_all": ["fixture"]}]},
    )
    return SimpleNamespace(ref=ref, output=tmp_path / "captures", calls=calls, provider=provider)


@pytest.mark.parametrize(
    "target,phase",
    [
        ("_prepare_workspace", "workspace_setup"),
        ("_write_prelaunch_files", "prelaunch"),
        ("_start_fake_openai", "provider_startup"),
        ("_launch_session", "cli_startup"),
        ("_wait_for_assertions", "turn_1"),
        ("_run_post_assertions", "post_assertions"),
    ],
)
def test_phase_failure_is_recorded_instead_of_aborting_run(
    scenario_run, monkeypatch, target, phase
):
    def fail(*_args, **_kwargs):
        raise RuntimeError("injected scenario failure")

    monkeypatch.setattr(scenarios, target, fail)

    assert scenarios.run_scenario(scenario_run.ref, output_dir=scenario_run.output) is False
    receipt = (scenario_run.output / "features-lifecycle-error.txt").read_text()
    assert phase in receipt
    assert "RuntimeError" in receipt
    if target in {"_launch_session", "_wait_for_assertions", "_run_post_assertions"}:
        assert ("stop", scenario_run.provider) in scenario_run.calls


def test_temporary_workspace_allocation_failure_is_recorded(scenario_run, monkeypatch):
    monkeypatch.setattr(
        scenarios,
        "tempfile",
        SimpleNamespace(TemporaryDirectory=Mock(side_effect=OSError("workspace unavailable"))),
    )
    assert scenarios.run_scenario(scenario_run.ref, output_dir=scenario_run.output) is False
    receipt = (scenario_run.output / "features-lifecycle-error.txt").read_text()
    assert "workspace_setup" in receipt
    assert "workspace unavailable" in receipt


def test_temporary_workspace_cleanup_failure_is_recorded(scenario_run, monkeypatch):
    original = scenarios.tempfile.TemporaryDirectory

    class FailingWorkspace:
        def __init__(self, **kwargs):
            self.inner = original(**kwargs)
            self.name = self.inner.name

        def __enter__(self):
            return self.name

        def __exit__(self, *_args):
            self.cleanup()

        def cleanup(self):
            self.inner.cleanup()
            raise OSError("workspace cleanup failed")

    monkeypatch.setattr(scenarios, "tempfile", SimpleNamespace(TemporaryDirectory=FailingWorkspace))
    assert scenarios.run_scenario(scenario_run.ref, output_dir=scenario_run.output) is False
    receipt = (scenario_run.output / "features-lifecycle-error.txt").read_text()
    assert "workspace_cleanup" in receipt
    assert [name for name, _ in scenario_run.calls] == ["tmux", "stop", "copy"]


def test_main_continues_to_later_scenario_after_startup_failure(scenario_run, monkeypatch):
    second = scenarios.ScenarioRef(
        suite="features", name="second", payload=scenario_run.ref.payload
    )
    selected = [scenario_run.ref, second]
    launches = []

    def launch(_profile, _repo, ref, **_kwargs):
        launches.append(ref.name)
        if ref.name == "lifecycle":
            raise RuntimeError("first startup failed")
        return "second-session"

    monkeypatch.setattr(scenarios, "_launch_session", launch)
    monkeypatch.setattr(scenarios, "_load_manifest", lambda _path: {})
    monkeypatch.setattr(scenarios, "validate_manifest", lambda *_a, **_kw: [])
    monkeypatch.setattr(scenarios, "select_scenarios", lambda *_a, **_kw: selected)
    monkeypatch.setattr(scenarios.shutil, "which", lambda _name: "/fixture/tmux")
    monkeypatch.setattr(
        scenarios,
        "parse_args",
        lambda: SimpleNamespace(
            manifest=Path("unused.json"),
            strict_acceptance=False,
            check=False,
            list=False,
            run=["first", "second"],
            run_all=False,
            output_dir=scenario_run.output,
        ),
    )

    assert scenarios.main() == 1
    assert launches == ["lifecycle", "second"]
    assert (scenario_run.output / "features-lifecycle-error.txt").is_file()
    assert (scenario_run.output / "features-second-turn-1.txt").read_text() == "fixture capture"


@pytest.mark.parametrize(
    "target,label",
    [
        ("_tmux", "tmux_cleanup"),
        ("_stop_fake_openai", "provider_cleanup"),
        ("_copy_fake_openai_log", "provider_log"),
    ],
)
def test_cleanup_failure_does_not_skip_other_cleanup(scenario_run, monkeypatch, target, label):
    original = getattr(scenarios, target)

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected cleanup failure")

    monkeypatch.setattr(scenarios, target, fail)

    assert scenarios.run_scenario(scenario_run.ref, output_dir=scenario_run.output) is False
    assert [name for name, _ in scenario_run.calls] == ["tmux", "stop", "copy"]
    assert label in (scenario_run.output / "features-lifecycle-error.txt").read_text()


@pytest.mark.parametrize(
    "error",
    [
        subprocess.TimeoutExpired(["tmux", "-e", "API_KEY=synthetic-secret"], 5),
        subprocess.CalledProcessError(
            7,
            ["tmux", "-e", "API_KEY=synthetic-secret"],
            output="synthetic-secret",
            stderr="synthetic-secret",
        ),
    ],
)
def test_subprocess_error_receipts_do_not_dump_command_credentials(
    scenario_run, monkeypatch, capsys, error
):
    monkeypatch.setattr(scenarios, "_launch_session", Mock(side_effect=error))

    assert scenarios.run_scenario(scenario_run.ref, output_dir=scenario_run.output) is False
    receipt = (scenario_run.output / "features-lifecycle-error.txt").read_text()
    assert "cli_startup" in receipt
    assert type(error).__name__ in receipt
    assert "synthetic-secret" not in receipt
    captured = capsys.readouterr()
    assert "synthetic-secret" not in captured.out + captured.err


def test_failed_tmux_launch_does_not_copy_stderr_into_the_receipt(
    scenario_run, monkeypatch, capsys
):
    monkeypatch.setattr(scenarios, "_launch_session", launch_real_session)
    monkeypatch.setattr(scenarios.shutil, "which", lambda _name: "/fixture/uv")
    monkeypatch.setattr(
        scenarios,
        "_tmux",
        lambda *_a, **_kw: SimpleNamespace(returncode=7, stderr="API_KEY=synthetic-secret"),
    )

    assert scenarios.run_scenario(scenario_run.ref, output_dir=scenario_run.output) is False
    receipt = (scenario_run.output / "features-lifecycle-error.txt").read_text()
    assert "cli_startup" in receipt
    assert "status 7" in receipt
    assert "synthetic-secret" not in receipt
    assert "synthetic-secret" not in capsys.readouterr().err


def test_user_interrupt_still_propagates_after_cleanup(scenario_run, monkeypatch):
    monkeypatch.setattr(
        scenarios, "_dispatch_turn_input_actions", Mock(side_effect=KeyboardInterrupt)
    )

    with pytest.raises(KeyboardInterrupt):
        scenarios.run_scenario(scenario_run.ref, output_dir=scenario_run.output)

    assert [name for name, _ in scenario_run.calls] == ["tmux", "stop", "copy"]


def test_successful_scenario_remains_successful(scenario_run):
    assert scenarios.run_scenario(scenario_run.ref, output_dir=scenario_run.output) is True
    assert not (scenario_run.output / "features-lifecycle-error.txt").exists()
    assert [name for name, _ in scenario_run.calls] == ["tmux", "stop", "copy"]


@pytest.mark.parametrize(
    ("command", "explicit_timeout", "expected"),
    [
        ("!uv run python fixture.py", None, 45.0),
        ("/status", None, 12.0),
        ("!uv run python fixture.py", 0.8, 0.8),
    ],
)
def test_shell_turn_budget_preserves_explicit_timing_contracts(
    scenario_run, monkeypatch, command, explicit_timeout, expected
):
    turn = {"send": command, "expect_all": ["fixture"]}
    if explicit_timeout is not None:
        turn["timeout"] = explicit_timeout
    ref = scenarios.ScenarioRef("features", "lifecycle", {"turns": [turn]})
    timeouts = []

    def wait(_session, _turn, *, timeout, **_kwargs):
        timeouts.append(timeout)
        return True, "fixture capture"

    monkeypatch.setattr(scenarios, "_wait_for_assertions", wait)

    assert scenarios.run_scenario(ref, output_dir=scenario_run.output) is True
    assert timeouts == [expected]


def test_successful_rerun_removes_its_stale_error_receipt(scenario_run, monkeypatch):
    monkeypatch.setattr(
        scenarios, "_launch_session", Mock(side_effect=RuntimeError("first attempt failed"))
    )
    assert scenarios.run_scenario(scenario_run.ref, output_dir=scenario_run.output) is False
    receipt = scenario_run.output / "features-lifecycle-error.txt"
    assert receipt.exists()

    monkeypatch.setattr(scenarios, "_launch_session", lambda *_a, **_kw: "owned-session")
    assert scenarios.run_scenario(scenario_run.ref, output_dir=scenario_run.output) is True
    assert not receipt.exists()


def _provider_scenario():
    return scenarios.ScenarioRef(
        suite="features",
        name="provider",
        payload={
            "fake_openai": {
                "port": 0,
                "response": "fixture",
                "log_file": "$HOME/provider.log",
                "ready_file": "$HOME/provider.ready",
            }
        },
    )


class _ProviderProcess:
    def __init__(self, *, ignore_terminate=False):
        self.calls = []
        self.returncode = None
        self.ignore_terminate = ignore_terminate

    def poll(self):
        return self.returncode

    def terminate(self):
        self.calls.append("terminate")

    def wait(self, timeout):
        self.calls.append("wait")
        if self.ignore_terminate and "kill" not in self.calls:
            raise subprocess.TimeoutExpired(["fixture-provider"], timeout)
        self.returncode = 0
        return 0

    def kill(self):
        self.calls.append("kill")


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("ignore_terminate", [False, True])
def test_unready_provider_is_joined_before_startup_failure_propagates(
    tmp_path, monkeypatch, interrupted, ignore_terminate
):
    process = _ProviderProcess(ignore_terminate=ignore_terminate)
    monkeypatch.setattr(scenarios.subprocess, "Popen", lambda *_a, **_kw: process)
    if interrupted:
        clock = SimpleNamespace(time=lambda: 0, sleep=Mock(side_effect=KeyboardInterrupt))
        expected_error = KeyboardInterrupt
    else:
        ticks = iter([0, 6])
        clock = SimpleNamespace(time=lambda: next(ticks, 6), sleep=lambda _seconds: None)
        expected_error = RuntimeError
    monkeypatch.setattr(scenarios, "time", clock)

    with pytest.raises(expected_error):
        scenarios._start_fake_openai(_provider_scenario(), home=tmp_path, repo=tmp_path)

    expected = ["terminate", "wait"]
    if ignore_terminate:
        expected.extend(["kill", "wait"])
    assert process.calls == expected
    assert process.returncode is not None


def test_ready_provider_transfers_ownership_to_caller(tmp_path, monkeypatch):
    process = _ProviderProcess()

    def start(*_args, **_kwargs):
        (tmp_path / "provider.ready").write_text("ready http://127.0.0.1:43210/v1")
        return process

    monkeypatch.setattr(scenarios.subprocess, "Popen", start)
    owner, url = scenarios._start_fake_openai(_provider_scenario(), home=tmp_path, repo=tmp_path)
    assert owner is process
    assert url == "http://127.0.0.1:43210/v1"
    assert not process.calls


def test_startup_timeout_reaps_a_real_child_process(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    children = []

    def start(*_args, **_kwargs):
        child = real_popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        children.append(child)
        return child

    ticks = iter([0, 6])
    monkeypatch.setattr(scenarios.subprocess, "Popen", start)
    monkeypatch.setattr(
        scenarios,
        "time",
        SimpleNamespace(time=lambda: next(ticks, 6), sleep=lambda _seconds: None),
    )
    try:
        with pytest.raises(RuntimeError, match="did not become ready"):
            scenarios._start_fake_openai(_provider_scenario(), home=tmp_path, repo=tmp_path)
        assert children[0].returncode is not None
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
