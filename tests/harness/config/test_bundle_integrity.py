"""Synthetic bundle failure cases must leave configuration intact."""

import hashlib
import json

import pytest

from koder_agent.harness.config import settings_bundle


def entry(role, scope, content, relative_path="settings.json"):
    return {
        "role": role,
        "scope": scope,
        "content": content,
        "relative_path": relative_path,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
    }


def bundle(tmp_path, entries):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({"format": "koder-settings-bundle", "version": 1, "files": entries}))
    return path


def test_late_validation_failure_does_not_apply_earlier_entries(tmp_path):
    home = tmp_path / "home"
    target = home / ".koder" / "settings.json"
    target.parent.mkdir(parents=True)
    target.write_text('{"old": true}')
    path = bundle(
        tmp_path,
        [entry("user_settings", "user", "{}"), entry("project_settings", "project", "{invalid")],
    )
    with pytest.raises(ValueError):
        settings_bundle.import_settings_bundle(path, home=home, cwd=tmp_path / "project")
    assert target.read_text() == '{"old": true}'
    assert list(target.parent.glob("*.bak-*")) == []


def test_scope_cannot_be_forged_to_write_user_profile(tmp_path):
    path = bundle(tmp_path, [entry("user_config", "project", "model:\n  name: forged\n")])
    home = tmp_path / "home"
    with pytest.raises(ValueError, match="scope"):
        settings_bundle.import_settings_bundle(
            path, home=home, cwd=tmp_path / "project", scope="project"
        )
    assert not (home / ".koder").exists()


def test_import_rejects_symlink_parent(tmp_path):
    home = tmp_path / "home"
    outside = tmp_path / "outside"
    home.mkdir()
    outside.mkdir()
    (home / ".koder").symlink_to(outside, target_is_directory=True)
    path = bundle(tmp_path, [entry("user_settings", "user", "{}")])
    with pytest.raises(ValueError, match="symlink"):
        settings_bundle.import_settings_bundle(path, home=home, cwd=tmp_path / "project")
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("role", ["project_local_settings", "user_settings", "user_keybindings"])
def test_all_settings_roles_require_json_object(tmp_path, role):
    scope = "user" if role.startswith("user_") else "project"
    path = bundle(tmp_path, [entry(role, scope, "[]")])
    with pytest.raises(ValueError):
        settings_bundle.import_settings_bundle(
            path, home=tmp_path / "home", cwd=tmp_path / "project"
        )


def test_export_does_not_follow_symlink_profile_directory(tmp_path):
    home = tmp_path / "home"
    outside = tmp_path / "outside"
    home.mkdir()
    outside.mkdir()
    (outside / "config.yaml").write_text("model:\n  api_key: synthetic-canary\n")
    (home / ".koder").symlink_to(outside, target_is_directory=True)
    target = tmp_path / "export.json"
    result = settings_bundle.export_settings_bundle(target, home=home, cwd=tmp_path / "project")
    assert result.file_count == 0
    assert "synthetic-canary" not in target.read_text()


@pytest.mark.parametrize("existing", [False, True])
def test_export_file_is_private(tmp_path, existing):
    target = tmp_path / "export.json"
    if existing:
        target.write_text("old bundle")
        target.chmod(0o644)
    settings_bundle.export_settings_bundle(target, home=tmp_path / "home", cwd=tmp_path / "project")
    assert target.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("existing_first", [True, False])
def test_write_failure_rolls_back_previously_published_files(tmp_path, monkeypatch, existing_first):
    home = tmp_path / "home"
    first = home / ".koder" / "settings.json"
    if existing_first:
        first.parent.mkdir(parents=True)
        first.write_text('{"original": true}')
    path = bundle(
        tmp_path,
        [entry("user_settings", "user", "{}"), entry("project_settings", "project", "{}")],
    )
    project = tmp_path / "project"
    second = project / ".koder" / "settings.json"
    original_write = settings_bundle.write_text_atomic

    def fail_second(target, content):
        if target == second:
            raise OSError("synthetic disk failure")
        original_write(target, content)

    monkeypatch.setattr(settings_bundle, "write_text_atomic", fail_second)
    with pytest.raises(OSError, match="synthetic disk failure"):
        settings_bundle.import_settings_bundle(path, home=home, cwd=project)
    assert first.exists() is existing_first
    if existing_first:
        assert first.read_text() == '{"original": true}'
    assert not second.exists()


def test_duplicate_targets_rejected_before_writing(tmp_path):
    path = bundle(
        tmp_path,
        [entry("user_settings", "user", "{}"), entry("user_settings", "user", '{"new": true}')],
    )
    with pytest.raises(ValueError, match="Duplicate"):
        settings_bundle.import_settings_bundle(
            path, home=tmp_path / "home", cwd=tmp_path / "project"
        )
    assert not (tmp_path / "home" / ".koder").exists()


def test_import_accepts_source_config_before_env_relation_validation(tmp_path):
    path = bundle(
        tmp_path, [entry("user_config", "user", "harness:\n  task_delegate_max_batch_size: 3\n")]
    )
    result = settings_bundle.import_settings_bundle(
        path, home=tmp_path / "home", cwd=tmp_path / "project"
    )
    assert result.written == 1
