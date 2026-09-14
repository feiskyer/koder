"""Primary local validation must follow the installed application's dependencies."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import run_runtime_tests as runner


def snapshot(**packages):
    return {"python": [3, 13, 1], "implementation": "cpython", "packages": packages}


def test_runtime_versions_not_historical_lock_determine_parity():
    actual = snapshot(mcp="2.2.0", **{"openai-agents": "0.20.0", "pytest": "9.1.1"})
    assert runner.version_differences(snapshot(mcp="2.2.0"), actual) == []
    assert runner.version_differences(snapshot(mcp="1.27.1"), actual) == [
        "mcp: runtime=1.27.1, tests=2.2.0"
    ]


def test_all_runtime_dependencies_and_interpreter_must_match():
    actual = snapshot(mcp="2.2.0")
    actual["python"] = [3, 11, 11]
    differences = runner.version_differences(snapshot(mcp="2.2.0", anyio="4.15.1"), actual)
    assert len(differences) == 2
    assert "python:" in differences[0]
    assert "anyio: runtime=4.15.1, tests=missing" in differences


def test_mismatch_stops_before_test_collection(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps(snapshot(mcp="2.2.0")))
    monkeypatch.setattr(runner, "runtime_snapshot", lambda: snapshot(mcp="1.27.1"))
    run_tests = Mock(side_effect=AssertionError("must not collect tests"))
    from scripts import run_isolated_tests

    monkeypatch.setattr(run_isolated_tests, "main", run_tests)
    assert runner.main(["--verify-runtime", str(manifest)]) == 2
    run_tests.assert_not_called()
    assert "tests were NOT collected" in capsys.readouterr().err


def test_resolve_installed_launcher_preserves_virtualenv_symlink(tmp_path, monkeypatch):
    interpreter = tmp_path / "runtime" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    launcher = interpreter.with_name("koder")
    launcher.write_text(f"#!{interpreter}\n")
    command = tmp_path / "koder"
    command.symlink_to(launcher)
    monkeypatch.setattr(runner.shutil, "which", lambda _name: str(command))
    assert runner.resolve_runtime_python(None, None) == str(interpreter)
    assert runner.resolve_runtime_python(None, str(interpreter)) == str(interpreter)


@pytest.mark.parametrize("shebang", ["#!/usr/bin/env python", "#!/bin/sh\n", "not a launcher"])
def test_unknown_launchers_do_not_silently_use_project_python(tmp_path, monkeypatch, shebang):
    launcher = tmp_path / "koder"
    launcher.write_text(shebang)
    monkeypatch.setattr(runner.shutil, "which", lambda _name: str(launcher))
    with pytest.raises(ValueError, match="supply --runtime-python"):
        runner.resolve_runtime_python(None, None)


def test_missing_launcher_requires_explicit_runtime(monkeypatch):
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)
    with pytest.raises(ValueError, match="not on PATH"):
        runner.resolve_runtime_python(None, None)


def test_each_launch_snapshots_runtime_and_preserves_test_failure(monkeypatch):
    calls = []
    installed = snapshot(koder="0.6.3", mcp="2.2.0")

    def run(command, **kwargs):
        calls.append(command)
        assert "--frozen" not in command
        assert "--no-project" in command and "--no-env-file" in command
        assert command[command.index("--python") + 1] == sys.executable
        if "--describe-runtime" in command:
            return SimpleNamespace(returncode=0, stdout=json.dumps(installed))
        manifest = Path(command[command.index("--verify-runtime") + 1])
        assert json.loads(manifest.read_text()) == installed
        tools = Path(command[command.index("--with-requirements") + 1]).read_text()
        constraints = Path(tools.splitlines()[0][3:]).read_text()
        assert "mcp==2.2.0\n" in constraints
        assert "1.27.1" not in constraints
        assert command[-3:] == ["--test", "tests/mcp", "-q"]
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner.shutil, "which", lambda _name: "/synthetic/uv")
    assert runner.main(["--runtime-python", sys.executable, "--test", "tests/mcp", "-q"]) == 1
    assert len(calls) == 2
