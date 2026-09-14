"""Public config mutations use synthetic profiles and controlled subprocesses."""

import asyncio
import hashlib
import json
import multiprocessing
import os
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from koder_agent.config.manager import ConfigManager
from koder_agent.harness.config import commands, service, settings_bundle
from koder_agent.harness.config.migration import migrate_config_file
from koder_agent.harness.config.schema import RuntimeConfig
from koder_agent.harness.config.service import RuntimeConfigService

REAL_HOOK_DISPATCH = service.dispatch_command_hooks


def _config_writer_process(path, home, project, started, finished, output):
    """Only disk state and synchronization objects cross the spawn boundary."""
    assert multiprocessing.get_start_method() == "spawn"
    assert Path.home() == home
    assert Path.cwd() == project
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(service, "dispatch_command_hooks", lambda **_: SimpleNamespace(blocked=False))
        started.set()
        RuntimeConfigService(path).save(RuntimeConfig(model={"name": "process-winner"}))
        output.put(os.getpid())
        finished.set()


@pytest.fixture
def threaded_parent(recwarn):
    """Exercise process startup with a live, controlled parent thread."""
    started, stop = Event(), Event()

    def background():
        started.set()
        stop.wait()

    thread = Thread(target=background, name="config-process-test-parent")
    thread.start()
    try:
        assert started.wait(5)
        yield
    finally:
        stop.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        # On the pinned runtime, -W error alone does not fail os.fork().
        assert not recwarn.list, [str(warning.message) for warning in recwarn]


@pytest.fixture
def profile(tmp_path, monkeypatch):
    home = tmp_path / "home with spaces & punctuation"
    project = tmp_path / "project"
    path = home / ".koder" / "config.yaml"
    path.parent.mkdir(parents=True)
    project.mkdir()
    path.write_text("model:\n  name: original\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    manager = ConfigManager(path)
    monkeypatch.setattr(commands, "get_config_manager", lambda: manager)
    hooks = []

    def accept(**kwargs):
        hooks.append(kwargs)
        return SimpleNamespace(blocked=False)

    monkeypatch.setattr(service, "dispatch_command_hooks", accept)
    return path, manager, hooks


@pytest.fixture(params=["arguments", "executable"])
def editor(tmp_path, monkeypatch, request):
    script = tmp_path / (
        "controlled editor.py" if request.param == "arguments" else "controlled-editor"
    )
    script.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "target = pathlib.Path(sys.argv[-1])\n"
        "assert len(sys.argv) == 2\n"
        "target.write_text('model:\\n  name: edited\\n')\n"
    )
    script.chmod(0o700)
    monkeypatch.setenv(
        "EDITOR",
        shlex.join([sys.executable, str(script)]) if request.param == "arguments" else str(script),
    )
    return script


def edit():
    return asyncio.run(commands.handle_config_subcommand(SimpleNamespace(config_action="edit")))


def test_editor_arguments_publish_once_and_refresh_loaded_cache(profile, editor):
    path, manager, hooks = profile
    manager.load()
    assert edit() == 0
    assert manager.load().model.name == "edited"
    assert len(hooks) == 1
    assert hooks[0]["payload"]["file_path"] == str(path.resolve())
    assert hooks[0]["match_value"] == "user_settings"


@pytest.mark.parametrize("behavior", ["invalid", "exit", "block"])
def test_editor_failure_keeps_original(profile, editor, monkeypatch, behavior):
    path, manager, hooks = profile
    original = path.read_bytes()
    manager.load()
    header = f"#!{sys.executable}\n"
    if behavior == "invalid":
        editor.write_text(
            header + "import pathlib, sys\npathlib.Path(sys.argv[-1]).write_text('[]')\n"
        )
    elif behavior == "exit":
        editor.write_text(
            header + "import pathlib, sys\n"
            "pathlib.Path(sys.argv[-1]).write_text('model:\\n  name: rejected\\n')\n"
            "sys.exit(7)\n"
        )
    else:
        monkeypatch.setattr(
            service,
            "dispatch_command_hooks",
            lambda **_: SimpleNamespace(blocked=True, block_reason="controlled rejection"),
        )
    assert edit() == 1
    assert path.read_bytes() == original
    assert manager.load().model.name == "original"
    assert not list(path.parent.glob(".config.*.yaml"))


def test_editor_noop_does_not_publish_or_dispatch(profile, editor):
    path, _, hooks = profile
    original = path.read_bytes()
    editor.write_text(f"#!{sys.executable}\nimport sys\nassert len(sys.argv) == 2\n")
    assert edit() == 0
    assert path.read_bytes() == original
    assert hooks == []


def test_editor_cannot_overwrite_an_intervening_writer(profile, editor):
    path, manager, _ = profile
    editor.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        f"pathlib.Path({str(path)!r}).write_text('model:\\n  name: winner\\n')\n"
        "pathlib.Path(sys.argv[-1]).write_text('model:\\n  name: stale\\n')\n"
    )
    assert edit() == 1
    assert manager.load().model.name == "winner"


@pytest.mark.parametrize("manager_type", [ConfigManager, RuntimeConfigService])
def test_loaded_snapshot_cannot_overwrite_a_newer_save(profile, manager_type):
    path, _, _ = profile
    stale = manager_type(path)
    candidate = stale.load()
    candidate.model.name = "stale"
    RuntimeConfigService(path).save(RuntimeConfig(model={"name": "winner"}))
    with pytest.raises(RuntimeError, match="changed"):
        stale.save(candidate)
    assert stale.load().model.name == "winner"


@pytest.mark.parametrize("manager_type", [ConfigManager, RuntimeConfigService])
def test_load_refreshes_after_external_publication(profile, manager_type):
    path, _, _ = profile
    reader = manager_type(path)
    reader.load()
    RuntimeConfigService(path).save(RuntimeConfig(model={"name": "winner"}))
    assert reader.load().model.name == "winner"


def test_rejected_save_does_not_erase_concurrent_success(profile, monkeypatch):
    path, _, _ = profile
    in_hook = Event()
    release_hook = Event()
    second_started = Event()
    second_hook = Event()

    def dispatch(**kwargs):
        name = RuntimeConfigService(path).load().model.name
        if name == "rejected":
            in_hook.set()
            assert release_hook.wait(4)
            return SimpleNamespace(blocked=True, block_reason="controlled rejection")
        second_hook.set()
        return SimpleNamespace(blocked=False)

    def first_writer():
        with pytest.raises(RuntimeError, match="controlled rejection"):
            RuntimeConfigService(path).save(RuntimeConfig(model={"name": "rejected"}))

    def second_writer():
        second_started.set()
        RuntimeConfigService(path).save(RuntimeConfig(model={"name": "winner"}))

    monkeypatch.setattr(service, "dispatch_command_hooks", dispatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_writer)
        assert in_hook.wait(4)
        second = pool.submit(second_writer)
        assert second_started.wait(4)
        # An uncoordinated writer reaches its hook while the first is undecided.
        overlapped = second_hook.wait(0.2)
        release_hook.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert RuntimeConfigService(path).load().model.name == "winner"
    assert not overlapped


@pytest.mark.parametrize("source", ["[]", "false", "0", '""'])
def test_migration_prevalidates_before_backup(profile, source):
    path, _, hooks = profile
    path.write_text(source)
    with pytest.raises(ValueError):
        migrate_config_file(path)
    assert path.read_text() == source
    assert not path.with_suffix(".yaml.bak").exists()
    assert hooks == []


def test_migration_uses_source_parser_and_hook(profile):
    path, manager, hooks = profile
    manager.load()
    source = (
        "harness:\n  voice_enabled: true\n  voice_provider: google\n"
        "  task_delegate_max_batch_size: 3\n"
    )
    path.write_text(source)
    result = migrate_config_file(path)
    assert result.backup_path.read_text() == source
    assert result.backup_path.stat().st_mode & 0o777 == 0o600
    assert manager.load().voice.enabled is True
    assert manager.load().voice.provider == "google"
    assert manager.load().harness.task_delegate_max_batch_size == 3
    assert len(hooks) == 1


def test_migration_rejection_restores_original(profile, monkeypatch):
    path, manager, _ = profile
    original = path.read_bytes()
    monkeypatch.setattr(
        service,
        "dispatch_command_hooks",
        lambda **_: SimpleNamespace(blocked=True, block_reason="controlled rejection"),
    )
    with pytest.raises(RuntimeError, match="controlled rejection"):
        migrate_config_file(path)
    assert path.read_bytes() == original
    assert manager.load().model.name == "original"


def make_bundle(tmp_path, *entries):
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "format": "koder-settings-bundle",
                "version": 1,
                "files": [
                    {
                        "role": role,
                        "scope": scope,
                        "content": content,
                        "sha256": hashlib.sha256(content.encode()).hexdigest(),
                    }
                    for role, scope, content in entries
                ],
            }
        )
    )
    return bundle


def test_import_refreshes_cached_manager(profile, tmp_path):
    _, manager, _ = profile
    manager.load()
    bundle = make_bundle(tmp_path, ("user_config", "user", "model:\n  name: imported\n"))
    settings_bundle.import_settings_bundle(bundle)
    assert manager.load().model.name == "imported"


def test_import_failure_cannot_roll_back_another_service_writer(profile, tmp_path, monkeypatch):
    path, _, hooks = profile
    second = Path.cwd() / ".koder" / "settings.json"
    bundle = make_bundle(
        tmp_path,
        ("user_config", "user", "model:\n  name: imported\n"),
        ("project_settings", "project", "{}"),
    )
    import_published = Event()
    release_import = Event()
    writer_started = Event()
    writer_finished = Event()
    original_write = settings_bundle.write_text_atomic

    def fail_second(target, content):
        if target == second:
            import_published.set()
            assert release_import.wait(4)
            raise OSError("controlled disk failure")
        original_write(target, content)

    def import_bundle():
        with pytest.raises(OSError, match="controlled disk failure"):
            settings_bundle.import_settings_bundle(bundle)

    def save_config():
        writer_started.set()
        RuntimeConfigService(path).save(RuntimeConfig(model={"name": "winner"}))
        writer_finished.set()

    monkeypatch.setattr(settings_bundle, "write_text_atomic", fail_second)
    with ThreadPoolExecutor(max_workers=2) as pool:
        importing = pool.submit(import_bundle)
        assert import_published.wait(4)
        saving = pool.submit(save_config)
        assert writer_started.wait(4)
        overlapped = writer_finished.wait(0.2)
        release_import.set()
        importing.result(timeout=5)
        saving.result(timeout=5)
    assert RuntimeConfigService(path).load().model.name == "winner"
    assert not overlapped
    assert len(hooks) == 1


def test_import_detects_changed_snapshot_before_backup(profile, tmp_path, monkeypatch):
    path, _, _ = profile
    bundle = make_bundle(tmp_path, ("user_config", "user", "model:\n  name: imported\n"))
    original_lock = settings_bundle.config_write_lock

    def intervene(target):
        RuntimeConfigService(path).save(RuntimeConfig(model={"name": "winner"}))
        return original_lock(target)

    monkeypatch.setattr(settings_bundle, "config_write_lock", intervene)
    with pytest.raises(RuntimeError, match="changed"):
        settings_bundle.import_settings_bundle(bundle)
    assert RuntimeConfigService(path).load().model.name == "winner"
    assert not list(path.parent.glob("*.bak-*"))


def test_service_rollback_preserves_hook_owned_external_write(profile, monkeypatch):
    path, manager, _ = profile

    def write_then_reject(**kwargs):
        path.write_text("model:\n  name: external-winner\n")
        return SimpleNamespace(blocked=True, block_reason="controlled rejection")

    monkeypatch.setattr(service, "dispatch_command_hooks", write_then_reject)
    with pytest.raises(RuntimeError, match="newer file preserved"):
        manager.save(RuntimeConfig(model={"name": "rejected"}))
    assert manager.load().model.name == "external-winner"


@pytest.mark.parametrize("outcome", ["block", "raise"])
def test_missing_editor_target_stays_missing_on_hook_failure(profile, editor, monkeypatch, outcome):
    path, _, _ = profile
    path.unlink()

    def reject(**kwargs):
        if outcome == "raise":
            raise RuntimeError("controlled hook exception")
        return SimpleNamespace(blocked=True, block_reason="controlled rejection")

    monkeypatch.setattr(service, "dispatch_command_hooks", reject)
    assert edit() == 1
    assert not path.exists()


@pytest.mark.parametrize("editor_value", ["  ", "'unterminated", "/missing-controlled-editor"])
def test_invalid_editor_configuration_keeps_missing_target_missing(
    profile, monkeypatch, editor_value
):
    path, _, hooks = profile
    path.unlink()
    monkeypatch.setenv("EDITOR", editor_value)
    assert edit() == 1
    assert not path.exists()
    assert hooks == []


def test_editor_keeps_comments_and_candidate_private(profile, editor):
    path, manager, _ = profile
    editor.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "candidate = pathlib.Path(sys.argv[-1])\n"
        "assert candidate.stat().st_mode & 0o777 == 0o600\n"
        f"assert pathlib.Path({str(path)!r}).read_text() == 'model:\\n  name: original\\n'\n"
        "candidate.write_text('# retained comment\\nmodel:\\n  name: edited\\n')\n"
    )
    assert edit() == 0
    assert path.read_text().startswith("# retained comment\n")
    assert manager.load().model.name == "edited"


def test_migration_backup_symlink_cannot_overwrite_unrelated_file(profile, tmp_path):
    path, _, hooks = profile
    outside = tmp_path / "outside"
    outside.write_text("untouched")
    path.with_suffix(".yaml.bak").symlink_to(outside)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="backup symlink"):
        migrate_config_file(path)
    assert outside.read_text() == "untouched"
    assert path.read_bytes() == original
    assert hooks == []


def test_process_writer_waits_for_existing_config_transaction(
    profile, monkeypatch, threaded_parent
):
    path, _, _ = profile
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    finished = context.Event()
    output = context.Queue()

    child = context.Process(
        target=_config_writer_process,
        args=(path, Path.home(), Path.cwd(), started, finished, output),
    )
    try:
        with service.config_write_lock(path):
            child.start()
            assert started.wait(10)
            overlapped = finished.wait(0.2)
            assert path.read_text() == "model:\n  name: original\n"
        assert finished.wait(10)
        worker_pid = output.get(timeout=5)
        assert worker_pid == child.pid
        assert worker_pid != os.getpid()
        child.join(timeout=5)
        assert child.exitcode == 0
        assert not overlapped
        assert RuntimeConfigService(path).load().model.name == "process-winner"
    finally:
        if child.is_alive():
            child.terminate()
            child.join(timeout=3)
        assert not child.is_alive()
        child.close()
        output.close()
        output.join_thread()


def test_migration_preserves_symlink_and_exact_backup_bytes(profile, tmp_path):
    path, _, _ = profile
    original = b"# original\r\nmodel:\r\n  name: original\r\n"
    path.write_bytes(original)
    alias = tmp_path / "alias.yml"
    alias.symlink_to(path)
    result = migrate_config_file(alias)
    assert alias.is_symlink()
    assert result.backup_path == path.with_suffix(".yaml.bak")
    assert result.backup_path.read_bytes() == original
    assert RuntimeConfigService(alias).load().model.name == "original"


def test_rejected_save_restores_exact_crlf_bytes(profile, monkeypatch):
    path, manager, _ = profile
    original = b"model:\r\n  name: original\r\n"
    path.write_bytes(original)
    monkeypatch.setattr(
        service,
        "dispatch_command_hooks",
        lambda **_: SimpleNamespace(blocked=True, block_reason="controlled rejection"),
    )
    with pytest.raises(RuntimeError):
        manager.save(RuntimeConfig(model={"name": "rejected"}))
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("platform", "editor_value", "expected_argv"),
    [
        ("darwin", None, ["open", "-W", "-e"]),
        ("linux", None, ["nano"]),
        ("win32", None, ["notepad"]),
        (
            "win32",
            '"C:\\Program Files\\Editor\\edit.exe" --wait',
            ["C:\\Program Files\\Editor\\edit.exe", "--wait"],
        ),
    ],
)
def test_editor_platform_argv_without_launching_native_apps(
    profile, monkeypatch, platform, editor_value, expected_argv
):
    _, _, hooks = profile
    monkeypatch.setattr(commands.sys, "platform", platform)
    if editor_value is None:
        monkeypatch.delenv("EDITOR", raising=False)
    else:
        monkeypatch.setenv("EDITOR", editor_value)
    calls = []

    def controlled_run(argv, *, check):
        calls.append(argv)
        assert check is True
        assert argv[:-1] == expected_argv
        assert Path(argv[-1]).is_file()

    monkeypatch.setattr(commands.subprocess, "run", controlled_run)
    assert edit() == 0
    assert len(calls) == 1
    assert hooks == []


def test_import_rejects_aliases_to_same_target_before_any_write(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(home, target_is_directory=True)
    bundle = make_bundle(
        tmp_path,
        ("user_settings", "user", "{}"),
        ("project_settings", "project", '{"other": true}'),
    )
    with pytest.raises(ValueError, match="Duplicate"):
        settings_bundle.import_settings_bundle(bundle, home=alias, cwd=home)
    assert not (home / ".koder").exists()


@pytest.mark.parametrize("blocked", [False, True])
def test_editor_through_real_dispatcher_and_controlled_hook_subprocess(
    profile, editor, tmp_path, monkeypatch, blocked
):
    path, manager, _ = profile
    marker = tmp_path / "hook-payload.json"
    hook_script = tmp_path / "controlled-hook.py"
    response = json.dumps(
        {"decision": "block" if blocked else "allow", "reason": "controlled rejection"}
    )
    hook_script.write_text(
        "import json, pathlib, sys\n"
        "payload = json.load(sys.stdin)\n"
        "assert 'name: edited' in pathlib.Path(payload['file_path']).read_text()\n"
        f"pathlib.Path({str(marker)!r}).write_text(json.dumps(payload))\n"
        f"print({response!r})\n"
    )
    hook_command = shlex.join(
        ["uv", "run", "--no-project", "--no-env-file", sys.executable, str(hook_script)]
    )
    (path.parent / "settings.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "ConfigChange": [
                        {
                            "matcher": "user_settings",
                            "hooks": [{"type": "command", "command": hook_command, "timeout": 5}],
                        }
                    ]
                },
            }
        )
    )
    monkeypatch.setattr(service, "dispatch_command_hooks", REAL_HOOK_DISPATCH)
    manager.load()
    assert edit() == (1 if blocked else 0)
    assert manager.load().model.name == ("original" if blocked else "edited")
    payload = json.loads(marker.read_text())
    assert payload["source"] == "user_settings"
    assert payload["file_path"] == str(path.resolve())


def test_import_rollback_does_not_overwrite_uncoordinated_winner(profile, tmp_path, monkeypatch):
    path, _, _ = profile
    second = Path.cwd() / ".koder" / "settings.json"
    bundle = make_bundle(
        tmp_path,
        ("user_config", "user", "model:\n  name: imported\n"),
        ("project_settings", "project", "{}"),
    )
    original_write = settings_bundle.write_text_atomic

    def fail_second(target, content):
        if target == second:
            path.write_text("model:\n  name: external-winner\n")
            raise OSError("controlled disk failure")
        original_write(target, content)

    monkeypatch.setattr(settings_bundle, "write_text_atomic", fail_second)
    with pytest.raises(OSError, match="rollback was incomplete"):
        settings_bundle.import_settings_bundle(bundle)
    assert RuntimeConfigService(path).load().model.name == "external-winner"


def test_service_publish_failure_invalidates_mutated_cache(profile, monkeypatch):
    path, manager, hooks = profile
    original = path.read_bytes()
    candidate = manager.load()
    candidate.model.name = "rejected"

    def fail_write(*args):
        raise OSError("controlled disk failure")

    monkeypatch.setattr(service, "write_text_atomic", fail_write)
    with pytest.raises(OSError, match="controlled disk failure"):
        manager.save(candidate)
    assert path.read_bytes() == original
    assert manager.load().model.name == "original"
    assert hooks == []
