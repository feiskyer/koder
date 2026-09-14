"""Discovery must not undo the public loader's repository boundary."""

from koder_agent.harness.skills.discovery import discover_skills_for_paths


def test_directory_input_includes_its_own_skills(tmp_path):
    skills = tmp_path / ".koder" / "skills"
    skills.mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    assert discover_skills_for_paths([str(tmp_path)], set()) == [skills.resolve()]


def test_git_worktree_file_stops_ancestor_discovery(tmp_path):
    outer = tmp_path / ".koder" / "skills"
    outer.mkdir(parents=True)
    repo = tmp_path / "repo"
    inner = repo / ".koder" / "skills"
    inner.mkdir(parents=True)
    (repo / ".git").write_text("gitdir: synthetic-worktree-metadata")
    assert discover_skills_for_paths([str(repo / "module.py")], set()) == [inner.resolve()]


def test_public_loader_does_not_import_skills_above_repository(tmp_path):
    from koder_agent.tools.skill import discover_merged_skills

    outer = tmp_path / ".koder" / "skills"
    outer.mkdir(parents=True)
    (outer / "outside.md").write_text(
        "---\nname: outside-repository\ndescription: Should not be discovered\n---\nBody\n"
    )
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    skills = discover_merged_skills(
        cwd=repo,
        user_dir=tmp_path / "empty-user",
        plugin_root=tmp_path / "empty-plugins",
        additional_dirs=[],
    )
    assert "outside-repository" not in skills


def test_discovery_order_is_nearest_first_and_stable(tmp_path):
    repo = tmp_path / "repo"
    child = repo / "src"
    root_skills = repo / ".koder" / "skills"
    child_skills = child / ".koder" / "skills"
    root_skills.mkdir(parents=True)
    child_skills.mkdir(parents=True)
    (repo / ".git").mkdir()
    assert discover_skills_for_paths([str(child / "module.py"), str(repo / "other.py")], set()) == [
        child_skills.resolve(),
        root_skills.resolve(),
    ]
