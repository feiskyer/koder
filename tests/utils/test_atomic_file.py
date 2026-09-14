"""Failure-path coverage for runtime-owned state snapshots."""

import os
import stat

import pytest

from koder_agent.utils.atomic_file import write_text_atomic


def test_new_state_file_is_private_and_utf8(tmp_path):
    path = tmp_path / "state" / "tasks.json"
    write_text_atomic(path, "持久化状态\n")

    assert path.read_text(encoding="utf-8") == "持久化状态\n"
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_replacement_preserves_symlink_and_target_mode(tmp_path):
    target = tmp_path / "target.yaml"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)
    alias = tmp_path / "config.yaml"
    alias.symlink_to(target)

    write_text_atomic(alias, "new")

    assert alias.is_symlink()
    assert target.read_text(encoding="utf-8") == "new"
    if os.name != "nt":
        assert stat.S_IMODE(target.stat().st_mode) == 0o640


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_failed_publish_preserves_original_and_cleans_temporary_file(
    tmp_path, monkeypatch, operation
):
    path = tmp_path / "tasks.json"
    path.write_text("original", encoding="utf-8")

    def fail(*_args, **_kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(f"koder_agent.utils.atomic_file.os.{operation}", fail)
    with pytest.raises(OSError, match="simulated disk failure"):
        write_text_atomic(path, "replacement")

    assert path.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [path]
