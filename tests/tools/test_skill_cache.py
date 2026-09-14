"""Model-facing skill loads must observe filesystem changes without restarting."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from koder_agent.tools import skill as skills
from koder_agent.tools.skill_context import get_active_restrictions, skill_invocation_scope


def _write_skill(path, name, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: Synthetic skill\n---\n{body}\n",
        encoding="utf-8",
    )


@pytest.fixture
def skill_directory(tmp_path, monkeypatch):
    project = tmp_path / "project"
    directory = project / ".koder" / "skills"
    directory.mkdir(parents=True)
    (project / ".git").mkdir()
    monkeypatch.chdir(project)
    config = SimpleNamespace(
        skills=SimpleNamespace(
            enabled=True,
            user_skills_dir=str(tmp_path / "empty-user"),
            project_skills_dir=str(directory),
        )
    )
    monkeypatch.setattr(skills, "get_config", lambda: config)
    monkeypatch.setattr(skills, "get_plugin_root", lambda: tmp_path / "empty-plugins")
    monkeypatch.setattr(skills, "_additional_dirs_from_env", lambda: [])
    monkeypatch.setattr(skills, "_merged_skills", None)
    monkeypatch.setattr(skills, "_merged_skills_key", None)
    # A future-dated unrelated file makes the old max-mtime cache insensitive
    # to ordinary edits, insertion or removal of other skills in this directory.
    anchor = directory / "anchor.md"
    _write_skill(anchor, "cache-anchor", "anchor instructions")
    os.utime(anchor, (3_000_000_000, 3_000_000_000))
    return directory


async def _load(name="cache-target"):
    return await skills.get_skill.on_invoke_tool(None, json.dumps({"skill_name": name}))


@pytest.mark.asyncio
@pytest.mark.parametrize("layout", ["standard", "legacy"])
@pytest.mark.parametrize("change", ["edit", "remove", "add"])
async def test_non_maximum_skill_changes_refresh_model_facing_cache(
    skill_directory, layout, change
):
    path = (
        skill_directory / "target" / "SKILL.md"
        if layout == "standard"
        else skill_directory / "target.md"
    )
    if change != "add":
        _write_skill(path, "cache-target", "original instructions")
        os.utime(path, (1_000_000_000, 1_000_000_000))
    before = await _load()
    assert ("not found" in before) == (change == "add")

    if change == "remove":
        path.unlink()
    else:
        _write_skill(path, "cache-target", "updated instructions")
    after = await _load()

    if change == "remove":
        assert "not found" in after
        assert "original instructions" not in after
    else:
        assert "updated instructions" in after
        assert "original instructions" not in after


@pytest.mark.asyncio
async def test_skill_rename_with_identical_mtime_refreshes_name_and_base_directory(skill_directory):
    old_path = skill_directory / "before" / "SKILL.md"
    _write_skill(old_path, "cache-target", "instructions")
    os.utime(old_path, (1_000_000_000, 1_000_000_000))
    await _load()

    old_path.parent.rename(skill_directory / "after")
    await _load()

    assert skills._get_merged_skills()["cache-target"].base_dir == skill_directory / "after"


@pytest.mark.asyncio
async def test_atomic_skill_replacement_with_same_size_and_mtime_refreshes_content(skill_directory):
    path = skill_directory / "target.md"
    _write_skill(path, "cache-target", "old body")
    os.utime(path, (1_000_000_000, 1_000_000_000))
    previous_stat = path.stat()
    assert "old body" in await _load()

    replacement = skill_directory / "replacement.tmp"
    _write_skill(replacement, "cache-target", "new body")
    os.utime(replacement, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
    assert replacement.stat().st_size == previous_stat.st_size
    replacement.replace(path)

    assert "new body" in await _load()


@pytest.mark.asyncio
async def test_unchanged_skills_reuse_cached_objects_and_ignore_reference_files(skill_directory):
    path = skill_directory / "target" / "SKILL.md"
    _write_skill(path, "cache-target", "instructions")
    await _load()
    cached = skills._get_merged_skills()

    reference = path.parent / "reference.md"
    reference.write_text("non-skill reference", encoding="utf-8")
    await _load()
    assert skills._get_merged_skills() is cached

    reference.write_text("changed reference", encoding="utf-8")
    await _load()
    assert skills._get_merged_skills() is cached


@pytest.mark.asyncio
async def test_skill_stat_failure_does_not_serve_previously_cached_instructions(
    skill_directory, monkeypatch
):
    path = skill_directory / "target.md"
    _write_skill(path, "cache-target", "cached instructions")
    assert "cached instructions" in await _load()
    original_stat = Path.stat

    def unavailable(target, *args, **kwargs):
        if target == path:
            raise PermissionError("synthetic unavailable skill")
        return original_stat(target, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", unavailable)
    result = await _load()
    assert "cached instructions" not in result
    assert "synthetic unavailable skill" in result


@pytest.mark.asyncio
async def test_future_invocation_uses_changed_skill_tool_policy(skill_directory):
    path = skill_directory / "target.md"

    def write_policy(tool):
        path.write_text(
            "---\nname: cache-target\ndescription: Synthetic policy\n"
            f"allowed_tools:\n  - {tool}\n---\nInstructions\n",
            encoding="utf-8",
        )
        os.utime(path, (1_000_000_000, 1_000_000_000))

    write_policy("run_shell")
    with skill_invocation_scope():
        await _load()
        assert get_active_restrictions().is_tool_allowed("run_shell")

    write_policy("read_file")
    with skill_invocation_scope():
        await _load()
        assert get_active_restrictions().is_tool_allowed("read_file")
        assert not get_active_restrictions().is_tool_allowed("run_shell")
