"""Deterministic terminal drivers must not count echoed commands as outcomes."""

import re
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import tmux_feature_scenarios as scenarios

PROMPT = "┌──| ⚡ Koder |──┐\n│> !echo fixture-ready │\n└────────────────┘\n"


def test_unattached_tmux_client_receives_the_scenario_path(tmp_path, monkeypatch):
    captured = []

    def tmux(*args, **kwargs):
        captured.append((args, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(scenarios, "_tmux", tmux)
    monkeypatch.setattr(scenarios, "_wait_for_prompt", lambda *_a, **_kw: "| ⚡ Koder |\n│>")
    monkeypatch.setattr(scenarios, "_session_exists", lambda _session: True)
    monkeypatch.setattr(scenarios.shutil, "which", lambda _name: "/synthetic/uv")
    ref = scenarios.ScenarioRef("features", "path-test", {"env": {"PATH": "$REPO/bin:$PATH"}})
    scenarios._launch_session(tmp_path / "home", tmp_path / "repo", ref, fake_openai_url=None)
    client_env = captured[0][1]["env"]
    assert client_env["PATH"].startswith(str(tmp_path / "repo/bin") + scenarios.os.pathsep)


def test_fake_provider_url_is_bound_without_mutating_the_manifest():
    original = scenarios.ScenarioRef(
        "features",
        "fake-test",
        {
            "env": {"KODER_BASE_URL": "$FAKE_OPENAI_URL"},
            "turns": [{"expect_all": ["ready $FAKE_OPENAI_URL"]}],
        },
    )
    result = scenarios._bind_fake_provider_url(original, "http://127.0.0.1:12345/v1")
    assert result.payload["env"]["KODER_BASE_URL"] == "http://127.0.0.1:12345/v1"
    assert result.payload["turns"][0]["expect_all"] == ["ready http://127.0.0.1:12345/v1"]
    assert original.payload["env"]["KODER_BASE_URL"] == "$FAKE_OPENAI_URL"


def test_manifest_fake_providers_use_ephemeral_ports():
    manifest = scenarios._load_manifest(scenarios.DEFAULT_MANIFEST)
    for ref in scenarios._scenario_refs(manifest):
        if "fake_openai" in ref.payload:
            assert ref.payload["fake_openai"]["port"] == 0, ref.name


def test_command_echo_does_not_satisfy_execution_assertions(tmp_path, monkeypatch):
    captures = []
    output = PROMPT + "fixture-ready\n"

    def capture(*_args):
        captures.append(True)
        return PROMPT if len(captures) == 1 else output

    monkeypatch.setattr(scenarios, "_capture_for_turn", capture)
    monkeypatch.setattr(scenarios, "_session_exists", lambda _s: True)
    passed, actual = scenarios._wait_for_assertions(
        "test",
        {"send": "!echo fixture-ready", "expect_all": ["fixture-ready"]},
        home=tmp_path,
        repo=tmp_path,
        timeout=2,
    )
    assert passed and actual == output
    assert len(captures) == 2


def test_explicit_input_assertions_can_still_see_the_input_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(scenarios, "_capture_for_turn", lambda *_a: PROMPT)
    monkeypatch.setattr(scenarios, "_session_exists", lambda _s: True)
    passed, _ = scenarios._wait_for_assertions(
        "test",
        {"type": "!echo fixture-ready", "expect_all": ["fixture-ready"]},
        home=tmp_path,
        repo=tmp_path,
        timeout=2,
    )
    assert passed


def test_prompt_filter_preserves_layout_markers_but_not_typed_text():
    filtered = scenarios._without_prompt_input(PROMPT)
    assert "| ⚡ Koder |" in filtered and "│>" in filtered
    assert "fixture-ready" not in filtered


def test_runtime_python_pattern_accepts_only_the_selected_environment(tmp_path):
    pattern = scenarios._expand_scenario_text(
        "$RUNTIME_PYTHON_PATTERN", home=tmp_path, repo=tmp_path
    )
    root = Path(sysconfig.get_path("scripts"))
    suffix = ".exe" if scenarios.os.name == "nt" else ""
    assert re.fullmatch(pattern, str(root / f"python{suffix}"))
    assert re.fullmatch(pattern, str(root / f"python3{suffix}"))
    assert not re.search(pattern, str(root / "python-other"))
    assert not re.search(pattern, "/different/environment/bin/python")


def test_reviewed_interpreter_placeholder_resolves_the_environment_symlink(tmp_path, monkeypatch):
    executable = tmp_path / "python-real"
    executable.write_text("fixture executable", encoding="utf-8")
    alias = tmp_path / "python"
    alias.symlink_to(executable)
    monkeypatch.setattr(scenarios.sys, "executable", str(alias))

    expanded = scenarios._expand_scenario_text(
        "$RUNTIME_PYTHON_RESOLVED", home=tmp_path, repo=tmp_path
    )

    assert expanded == str(executable)
    assert not Path(expanded).is_symlink()


@pytest.mark.parametrize("ready_after", [25.0, None])
def test_cold_cli_startup_has_a_bounded_readiness_budget(tmp_path, monkeypatch, ready_after):
    clock = [0.0]
    calls = []

    def tick(seconds):
        clock[0] += seconds

    def capture(_session):
        if ready_after is not None and clock[0] >= ready_after:
            return "| ⚡ Koder |\n│>"
        return ""

    def tmux(*args, **_kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(scenarios.time, "time", lambda: clock[0])
    monkeypatch.setattr(scenarios.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(scenarios.time, "sleep", tick)
    monkeypatch.setattr(scenarios, "_tmux", tmux)
    monkeypatch.setattr(scenarios, "_session_exists", lambda _session: True)
    monkeypatch.setattr(scenarios, "_capture", capture)
    monkeypatch.setattr(scenarios.shutil, "which", lambda _name: "/fixture/uv")
    ref = scenarios.ScenarioRef("features", "cold-start", {})

    if ready_after is None:
        with pytest.raises(RuntimeError, match="prompt did not appear"):
            scenarios._launch_session(
                tmp_path / "home", tmp_path / "repo", ref, fake_openai_url=None
            )
        assert clock[0] == 60
        assert calls[-1][0] == "kill-session"
    else:
        session = scenarios._launch_session(
            tmp_path / "home", tmp_path / "repo", ref, fake_openai_url=None
        )
        assert session.startswith("koder-scenario-")
        assert clock[0] == ready_after
        assert all(call[0] != "kill-session" for call in calls)
