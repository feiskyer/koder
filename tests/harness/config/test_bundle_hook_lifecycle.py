"""Bundle transactions must be judged by their pre-import hook configuration."""

import hashlib
import json
import shlex
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from koder_agent.harness.config import settings_bundle as bundles
from koder_agent.harness.config.service import RuntimeConfigService
from koder_agent.harness.hooks import runtime as hooks
from koder_agent.harness.hooks.project_approval import (
    approve_project_hooks,
    project_hooks_approval_error,
)


def _settings(command="trusted", matcher="*"):
    return json.dumps(
        {
            "hooks": {
                "ConfigChange": [
                    {"matcher": matcher, "hooks": [{"type": "command", "command": command}]}
                ]
            }
        }
    )


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _bundle(tmp_path, entries):
    return _write(
        tmp_path / "bundle.json",
        json.dumps(
            {
                "format": bundles.BUNDLE_FORMAT,
                "version": bundles.BUNDLE_VERSION,
                "files": [
                    {
                        "role": role,
                        "scope": "user" if role.startswith("user_") else "project",
                        "relative_path": "note.md",
                        "content": text,
                        "sha256": hashlib.sha256(text.encode()).hexdigest(),
                    }
                    for role, text in entries
                ],
            }
        ),
    )


@pytest.fixture
def profile(tmp_path, monkeypatch, real_project_hook_trust):
    home = tmp_path / "home"
    project = home / "project"  # Role, not HOME containment, determines matcher.
    project.mkdir(parents=True)
    (project / ".git").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(project)
    monkeypatch.setattr(hooks, "managed_settings_path", lambda: tmp_path / "absent-policy")
    return home, project


@pytest.mark.parametrize("outcome", ["accept", "block", "exception"])
def test_import_dispatches_config_change_and_rolls_back(profile, tmp_path, monkeypatch, outcome):
    home, project = profile
    target = _write(home / ".koder/config.yaml", "model:\n  name: original\n")
    _write(home / ".koder/settings.json", _settings())
    content = "model:\n  name: imported\n"
    source = _bundle(tmp_path, [("user_config", content)])
    seen = []

    def run(**kwargs):
        seen.append(json.loads(kwargs["payload_text"]))
        assert target.read_text() == content
        if outcome == "exception":
            raise OSError("controlled hook failure")
        return 0, json.dumps({"decision": outcome, "reason": "trusted veto"}), ""

    monkeypatch.setattr(hooks, "_run_command_hook", run)
    if outcome == "accept":
        assert bundles.import_settings_bundle(source, home=home, cwd=project).written == 1
        assert target.read_text() == content
    else:
        with pytest.raises((RuntimeError, OSError), match="trusted veto|controlled hook failure"):
            bundles.import_settings_bundle(source, home=home, cwd=project)
        assert target.read_text() == "model:\n  name: original\n"
    assert seen == [{"event": "ConfigChange", "source": "user_settings", "file_path": str(target)}]


@pytest.mark.parametrize("replacement", [_settings("imported"), '{"disableAllHooks": true}', "{}"])
def test_import_cannot_replace_or_disable_its_veto(profile, tmp_path, monkeypatch, replacement):
    home, project = profile
    target = _write(home / ".koder/settings.json", _settings())
    source = _bundle(
        tmp_path,
        [
            ("user_config", "model:\n  name: imported\n"),
            ("user_settings", replacement),
        ],
    )
    seen = []

    def run(**kwargs):
        seen.append(kwargs["command"])
        assert target.read_text() == replacement
        return 2, "", "trusted veto"

    monkeypatch.setattr(hooks, "_run_command_hook", run)
    with pytest.raises(RuntimeError, match="trusted veto"):
        bundles.import_settings_bundle(source, home=home, cwd=project)
    assert seen == ["trusted"]
    assert target.read_text() == _settings()
    assert not (home / ".koder/config.yaml").exists()


@pytest.mark.parametrize("scope", ["all", "user", "project"])
def test_selected_roles_and_preimport_project_approval(profile, tmp_path, monkeypatch, scope):
    home, project = profile
    _write(project / ".koder/settings.json", _settings("old-project"))
    approve_project_hooks(project)
    source = _bundle(
        tmp_path,
        [
            ("user_config", "model:\n  name: imported\n"),
            ("user_settings", _settings("new-user")),
            ("user_keybindings", "{}"),
            ("project_settings", _settings("new-project")),
            ("project_local_settings", _settings("new-local")),
        ],
    )
    seen = []

    def run(**kwargs):
        seen.append((kwargs["command"], json.loads(kwargs["payload_text"])))
        return 0, "", ""

    monkeypatch.setattr(hooks, "_run_command_hook", run)
    bundles.import_settings_bundle(source, home=home, cwd=project, scope=scope)
    expected = []
    if scope in {"all", "user"}:
        expected += [
            ("user_settings", home / ".koder/config.yaml"),
            ("user_settings", home / ".koder/settings.json"),
            ("user_settings", home / ".koder/keybindings.json"),
        ]
    if scope in {"all", "project"}:
        expected += [
            ("project_settings", project / ".koder/settings.json"),
            ("local_settings", project / ".koder/settings.local.json"),
        ]
    assert seen == [
        ("old-project", {"event": "ConfigChange", "source": role, "file_path": str(path)})
        for role, path in expected
    ]
    # Import does not extend a previous approval to new executable hook content.
    if scope == "user":
        assert project_hooks_approval_error(project) is None
    else:
        assert "changed" in project_hooks_approval_error(project)


@pytest.mark.parametrize("mode", ["noop", "dry_run", "invalid"])
def test_no_hooks_or_mutation_without_valid_changes(profile, tmp_path, monkeypatch, mode):
    home, project = profile
    original = _settings()
    target = _write(home / ".koder/settings.json", original)
    entries = [("user_settings", original if mode == "noop" else "{}")]
    if mode == "invalid":
        entries.append(("project_settings", "[]"))
    source = _bundle(tmp_path, entries)

    def unexpected(**kwargs):
        pytest.fail("hook must not execute")

    monkeypatch.setattr(hooks, "_run_command_hook", unexpected)
    if mode == "invalid":
        with pytest.raises(ValueError):
            bundles.import_settings_bundle(source, home=home, cwd=project)
    else:
        result = bundles.import_settings_bundle(
            source, home=home, cwd=project, dry_run=mode == "dry_run"
        )
        assert result.unchanged == (1 if mode == "noop" else 0)
    assert target.read_text() == original
    assert sorted(p.name for p in target.parent.iterdir()) == ["settings.json"]


@pytest.mark.parametrize("blocked", [False, True])
def test_real_synthetic_command_receives_candidate(profile, tmp_path, blocked):
    home, project = profile
    log = tmp_path / "hook-payload.json"
    candidate_log = tmp_path / "candidate.yaml"
    script = _write(
        tmp_path / "hook.sh",
        f"cat > {shlex.quote(str(log))}\n"
        f"cat {shlex.quote(str(home / '.koder/config.yaml'))} > {shlex.quote(str(candidate_log))}\n"
        + ('printf \'{"decision":"block","reason":"script veto"}\'\n' if blocked else "exit 0\n"),
    )
    _write(home / ".koder/settings.json", _settings(f"sh {shlex.quote(str(script))}"))
    source = _bundle(tmp_path, [("user_config", "model:\n  name: imported\n")])
    if blocked:
        with pytest.raises(RuntimeError, match="script veto"):
            bundles.import_settings_bundle(source, home=home, cwd=project)
    else:
        bundles.import_settings_bundle(source, home=home, cwd=project)
    assert json.loads(log.read_text())["source"] == "user_settings"
    assert candidate_log.read_text() == "model:\n  name: imported\n"
    assert (home / ".koder/config.yaml").exists() is not blocked


def test_late_scope_veto_rolls_back_whole_bundle(profile, tmp_path, monkeypatch):
    home, project = profile
    old = _settings(matcher="^local_settings$")
    settings = _write(home / ".koder/settings.json", old)
    project_settings = _write(project / ".koder/settings.json", '{"old": true}')
    source = _bundle(
        tmp_path,
        [
            ("user_settings", "{}"),
            ("project_settings", '{"new": true}'),
            ("project_local_settings", "{}"),
        ],
    )
    seen = []

    def veto(**kwargs):
        seen.append(json.loads(kwargs["payload_text"])["source"])
        assert settings.read_text() == "{}"
        assert project_settings.read_text() == '{"new": true}'
        assert (project / ".koder/settings.local.json").read_text() == "{}"
        return 2, "", "local veto"

    monkeypatch.setattr(hooks, "_run_command_hook", veto)
    with pytest.raises(RuntimeError, match="local veto"):
        bundles.import_settings_bundle(source, home=home, cwd=project)
    assert seen == ["local_settings"]
    assert settings.read_text() == old
    assert project_settings.read_text() == '{"old": true}'
    assert not (project / ".koder/settings.local.json").exists()


@pytest.mark.parametrize("old_disabled", [False, True])
def test_no_imported_hook_runs_without_preimport_hooks(
    profile, tmp_path, monkeypatch, old_disabled
):
    home, project = profile
    if old_disabled:
        _write(home / ".koder/settings.json", '{"disableAllHooks": true}')
    _write(project / ".koder/settings.json", _settings("unapproved"))
    source = _bundle(tmp_path, [("user_settings", _settings("imported"))])

    def unexpected(**kwargs):
        pytest.fail("Neither imported nor unapproved hooks may execute")

    monkeypatch.setattr(hooks, "_run_command_hook", unexpected)
    assert bundles.import_settings_bundle(source, home=home, cwd=project).written == 1


def test_explicit_home_does_not_load_ambient_user_hooks(profile, tmp_path, monkeypatch):
    ambient_home, project = profile
    _write(ambient_home / ".koder/settings.json", _settings("ambient"))
    destination_home = tmp_path / "destination-home"
    _write(destination_home / ".koder/settings.json", _settings("destination"))
    source = _bundle(tmp_path, [("user_config", "{}")])
    seen = []

    def run(**kwargs):
        seen.append(kwargs["command"])
        return 0, "", ""

    monkeypatch.setattr(hooks, "_run_command_hook", run)
    bundles.import_settings_bundle(source, home=destination_home, cwd=project)
    assert seen == ["destination"]


@pytest.mark.parametrize("outcome", ["accept", "block"])
def test_external_winner_is_preserved_and_other_files_rolled_back(
    profile, tmp_path, monkeypatch, outcome
):
    home, project = profile
    config = _write(home / ".koder/config.yaml", "model:\n  name: original\n")
    _write(home / ".koder/settings.json", _settings())
    source = _bundle(
        tmp_path,
        [
            ("user_config", "model:\n  name: imported\n"),
            ("project_settings", "{}"),
        ],
    )

    def run(**kwargs):
        config.write_text("model:\n  name: external\n")
        return 0, json.dumps({"decision": outcome}), ""

    monkeypatch.setattr(hooks, "_run_command_hook", run)
    with pytest.raises(OSError, match="rollback was incomplete"):
        bundles.import_settings_bundle(source, home=home, cwd=project)
    assert config.read_text() == "model:\n  name: external\n"
    assert not (project / ".koder/settings.json").exists()


def test_rejected_import_holds_lock_until_rollback(profile, tmp_path, monkeypatch):
    home, project = profile
    config = _write(home / ".koder/config.yaml", "model:\n  name: original\n")
    _write(home / ".koder/settings.json", _settings())
    source = _bundle(tmp_path, [("user_config", "model:\n  name: imported\n")])
    entered = threading.Event()
    release = threading.Event()
    writer_started = threading.Event()
    writer_finished = threading.Event()

    def run(**kwargs):
        if config.read_text() == "model:\n  name: imported\n":
            entered.set()
            assert release.wait(3)
            return 2, "", "veto"
        return 0, "", ""

    def write_winner():
        writer_started.set()
        RuntimeConfigService(config).save_text("model:\n  name: winner\n")
        writer_finished.set()

    monkeypatch.setattr(hooks, "_run_command_hook", run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        imported = pool.submit(bundles.import_settings_bundle, source, home=home, cwd=project)
        writer = None
        try:
            assert entered.wait(3)
            writer = pool.submit(write_winner)
            assert writer_started.wait(3)
            assert not writer_finished.wait(0.15)
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="veto"):
            imported.result(timeout=3)
        assert writer is not None
        writer.result(timeout=3)
    assert config.read_text() == "model:\n  name: winner\n"


@pytest.mark.parametrize("replacement", [_settings("new"), '{"disableAllHooks": true}'])
def test_accepted_replacement_keeps_old_user_policy_for_every_event(
    profile, tmp_path, monkeypatch, replacement
):
    home, project = profile
    _write(home / ".koder/settings.json", _settings("old"))
    source = _bundle(
        tmp_path,
        [
            ("user_settings", replacement),
            ("project_settings", "{}"),
        ],
    )
    seen = []

    def run(**kwargs):
        seen.append(kwargs["command"])
        return 0, "", ""

    monkeypatch.setattr(hooks, "_run_command_hook", run)
    assert bundles.import_settings_bundle(source, home=home, cwd=project).written == 2
    assert seen == ["old", "old"]


def test_destination_managed_policy_cannot_be_disabled_by_import(profile, tmp_path, monkeypatch):
    ambient_home, project = profile
    destination_home = tmp_path / "destination-home"
    _write(destination_home / ".koder/managed-settings.json", _settings("policy"))
    source = _bundle(tmp_path, [("user_settings", '{"disableAllHooks": true}')])
    seen = []

    def veto(**kwargs):
        seen.append(kwargs["command"])
        return 2, "", "managed veto"

    monkeypatch.setattr(hooks, "_run_command_hook", veto)
    with pytest.raises(RuntimeError, match="managed veto"):
        bundles.import_settings_bundle(source, home=destination_home, cwd=project)
    assert seen == ["policy"]
    assert not (destination_home / ".koder/settings.json").exists()
