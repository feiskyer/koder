"""Scenario source paths are not Python import-path lists or shared tmp files."""

import json
import os
from types import SimpleNamespace

import pytest

from scripts import tmux_feature_scenarios as scenarios


@pytest.mark.parametrize("key", ["KODER_SCENARIO_SOURCE_ROOT", "PYTHONPATH"])
def test_scenario_launch_keeps_source_root_separate_from_import_paths(tmp_path, monkeypatch, key):
    root = str(scenarios.PROJECT_ROOT)
    pythonpath = os.pathsep.join((str(tmp_path / "bootstrap"), root))
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    captured = []

    def tmux(*args, **_kwargs):
        captured.append(args)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(scenarios, "_tmux", tmux)
    monkeypatch.setattr(scenarios, "_wait_for_prompt", lambda *_a, **_k: "| ⚡ Koder |\n│>")
    monkeypatch.setattr(scenarios, "_session_exists", lambda _session: True)
    monkeypatch.setattr(scenarios.shutil, "which", lambda _command: "/synthetic/uv")
    ref = scenarios.ScenarioRef("features", "env-test", {"turns": []})
    scenarios._launch_session(tmp_path / "home", tmp_path / "repo", ref, fake_openai_url=None)
    arguments = captured[0]
    environment = {
        arguments[index + 1].partition("=")[0]: arguments[index + 1].partition("=")[2]
        for index, argument in enumerate(arguments[:-1])
        if argument == "-e"
    }
    assert environment[key] == (root if key == "KODER_SCENARIO_SOURCE_ROOT" else pythonpath)


def test_manifest_never_treats_pythonpath_as_a_single_source_directory():
    manifest = scenarios._load_manifest(scenarios.DEFAULT_MANIFEST)
    text = json.dumps(manifest)
    assert "$PYTHONPATH" not in text
    assert "$KODER_SCENARIO_SOURCE_ROOT" in text


def test_sandbox_escape_probe_has_a_per_scenario_target():
    manifest = scenarios._load_manifest(scenarios.DEFAULT_MANIFEST)
    scenario = manifest["features"]["sandbox-unix-local-shell"]
    assert "/tmp/koder-sandbox-scenario-escape" not in json.dumps(scenario)
    assert {"path_not_exists": "$HOME/koder-sandbox-scenario-escape"} in scenario["post_assertions"]
