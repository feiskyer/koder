import asyncio
import json
import sys
import textwrap
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from koder_agent.tools import skill as skill_module  # noqa: E402
from koder_agent.tools.skill import SkillLoader  # noqa: E402


@pytest.fixture(autouse=True)
def reset_skill_cache():
    skill_module._merged_skills = None
    yield
    skill_module._merged_skills = None


def invoke_get_skill(payload: dict) -> str:
    """Invoke the FunctionTool wrapper for get_skill with JSON input."""
    return asyncio.run(skill_module.get_skill.on_invoke_tool(None, json.dumps(payload)))


def test_load_skill_with_invalid_yaml_warns_and_defaults(tmp_path, capsys):
    skill_file = tmp_path / "bad" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        textwrap.dedent(
            """\
            ---
            name: bad-skill
            : missing-value
            ---
            Body text that should still load.
            """
        ),
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)
    skill = loader.load_skill(skill_file)
    output = capsys.readouterr().out

    assert "invalid YAML" in output
    assert skill is not None
    assert skill.name == "SKILL"
    assert "Body text that should still load." in skill.content
    assert skill.description == ""
    assert skill.allowed_tools is None


def test_load_skill_without_frontmatter_warns_and_uses_body(tmp_path, capsys):
    skill_file = tmp_path / "no-frontmatter" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("Plain body only", encoding="utf-8")

    loader = SkillLoader(tmp_path)
    skill = loader.load_skill(skill_file)
    output = capsys.readouterr().out

    assert "no frontmatter" in output
    assert skill is not None
    assert skill.name == "SKILL"
    assert skill.description == ""
    assert skill.content.startswith("Plain body only")


def test_allowed_tools_scalar_and_metadata_are_preserved(tmp_path):
    skill_file = tmp_path / "meta-skill" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        textwrap.dedent(
            """\
            ---
            name: meta-skill
            description: Tests allowed tools
            allowed_tools: write_file
            priority: 5
            ---
            Small body content.
            """
        ),
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)
    skill = loader.load_skill(skill_file)

    assert skill.allowed_tools == ["write_file"]
    assert skill.metadata == {"priority": 5}
    assert skill.description == "Tests allowed tools"
    assert "Small body content." in skill.content


def test_relative_links_are_resolved_to_absolute_paths(tmp_path):
    skill_dir = tmp_path / "links"
    docs_dir = skill_dir / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "info.md").write_text("extra details", encoding="utf-8")

    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        textwrap.dedent(
            """\
            ---
            name: link-skill
            description: Tests link rewriting
            ---
            See the [notes](docs/info.md) and the [site](https://example.com).
            """
        ),
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)
    skill = loader.load_skill(skill_file)

    abs_path = (docs_dir / "info.md").resolve()
    assert f"[notes]({abs_path})" in skill.content
    assert "[site](https://example.com)" in skill.content


def test_duplicate_skill_names_emit_warning_and_keep_first(tmp_path, capsys):
    first = tmp_path / "first" / "SKILL.md"
    second = tmp_path / "second" / "SKILL.md"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)

    first.write_text(
        textwrap.dedent(
            """\
            ---
            name: shared-skill
            description: primary copy
            ---
            Primary content.
            """
        ),
        encoding="utf-8",
    )
    second.write_text(
        textwrap.dedent(
            """\
            ---
            name: shared-skill
            description: duplicate copy
            ---
            Duplicate content that should be ignored.
            """
        ),
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)
    skills = loader.discover_skills()
    output = capsys.readouterr().out

    assert "duplicate skill name 'shared-skill'" in output
    assert len(skills) == 1
    prompt = loader.get_skill("shared-skill").to_prompt()
    assert "Primary content." in prompt
    assert "Duplicate content" not in prompt


def test_get_skill_rejects_empty_name():
    assert invoke_get_skill({"skill_name": ""}) == (
        "Invalid skill name: skill_name cannot be empty"
    )


def test_get_skill_reports_missing_when_no_skills(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(workspace)

    result = invoke_get_skill({"skill_name": "unknown-skill"})

    assert result == "Skill 'unknown-skill' not found. No skills are currently available."


def test_project_skills_override_user_skills(tmp_path, monkeypatch):
    user_home = tmp_path / "home"
    project_root = tmp_path / "project"
    user_home.mkdir()
    project_root.mkdir()

    monkeypatch.setattr(Path, "home", lambda: user_home)
    monkeypatch.chdir(project_root)

    user_skill = user_home / ".koder/skills/shared/SKILL.md"
    user_skill.parent.mkdir(parents=True)
    user_skill.write_text(
        textwrap.dedent(
            """\
            ---
            name: shared
            description: user copy
            ---
            User content.
            """
        ),
        encoding="utf-8",
    )

    project_skill = project_root / ".koder/skills/shared/SKILL.md"
    project_skill.parent.mkdir(parents=True)
    project_skill.write_text(
        textwrap.dedent(
            """\
            ---
            name: shared
            description: project copy
            ---
            Project content.
            """
        ),
        encoding="utf-8",
    )

    result = invoke_get_skill({"skill_name": "shared"})

    assert "Project content." in result
    assert "project copy" in result
    assert "User content." not in result


# ============================================================================
# Tests for hyphenated allowed-tools (Claude Code compatibility)
# ============================================================================


def test_allowed_tools_hyphenated_format(tmp_path):
    """Test that allowed-tools (hyphenated, Claude Code format) is parsed correctly."""
    skill_file = tmp_path / "hyphen-skill" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        textwrap.dedent(
            """\
            ---
            name: hyphen-skill
            description: Tests hyphenated allowed-tools
            allowed-tools:
              - read_file
              - glob_search
            ---
            Content here.
            """
        ),
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)
    skill = loader.load_skill(skill_file)

    assert skill.allowed_tools == ["read_file", "glob_search"]
    assert skill.description == "Tests hyphenated allowed-tools"


def test_allowed_tools_hyphenated_scalar(tmp_path):
    """Test that a single hyphenated allowed-tools value is parsed as a list."""
    skill_file = tmp_path / "single-tool" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        textwrap.dedent(
            """\
            ---
            name: single-tool
            description: Single tool allowed
            allowed-tools: read_file
            ---
            Content.
            """
        ),
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)
    skill = loader.load_skill(skill_file)

    assert skill.allowed_tools == ["read_file"]


def test_allowed_tools_underscore_still_works(tmp_path):
    """Test that allowed_tools (underscored) still works for backwards compatibility."""
    skill_file = tmp_path / "underscore-skill" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        textwrap.dedent(
            """\
            ---
            name: underscore-skill
            description: Tests underscored allowed_tools
            allowed_tools:
              - write_file
              - edit_file
            ---
            Content.
            """
        ),
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)
    skill = loader.load_skill(skill_file)

    assert skill.allowed_tools == ["write_file", "edit_file"]


def test_allowed_tools_hyphenated_takes_priority(tmp_path):
    """Test that allowed-tools takes priority over allowed_tools if both present."""
    skill_file = tmp_path / "both-formats" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        textwrap.dedent(
            """\
            ---
            name: both-formats
            description: Both formats present
            allowed-tools:
              - read_file
            allowed_tools:
              - write_file
            ---
            Content.
            """
        ),
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)
    skill = loader.load_skill(skill_file)

    # allowed-tools (hyphenated) should take priority
    assert skill.allowed_tools == ["read_file"]


def test_allowed_tools_empty_hyphenated_takes_priority(tmp_path):
    """Test that empty allowed-tools takes priority over non-empty allowed_tools.

    This is an important edge case: if the Claude Code format (allowed-tools)
    is present but empty, it should still win over the underscored format.
    Empty allowed-tools means "no restrictions" (functionally - see skill_context.py).
    The allowed_tools field stores the empty list, which is treated as "no restrictions"
    by the enforcement system (add_skill_restrictions returns early for falsy values).
    """
    skill_file = tmp_path / "empty-hyphen" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        textwrap.dedent(
            """\
            ---
            name: empty-hyphen
            description: Empty hyphenated takes priority
            allowed-tools: []
            allowed_tools:
              - write_file
            ---
            Content.
            """
        ),
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)
    skill = loader.load_skill(skill_file)

    # Empty allowed-tools should win (empty list from hyphenated key, not from underscore)
    # Empty list is functionally treated as "no restrictions" by skill_context
    assert skill.allowed_tools == []  # Not ["write_file"] from allowed_tools


# ============================================================================
# Tests for name/description validation
# ============================================================================


def test_skill_name_validation_warns_on_uppercase(tmp_path, capsys):
    """Test that uppercase in name triggers a warning."""
    skill_file = tmp_path / "bad-name" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        textwrap.dedent(
            """\
            ---
            name: BadName
            description: Invalid name format
            ---
            Content
            """
        ),
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)
    skill = loader.load_skill(skill_file)
    output = capsys.readouterr().out

    assert skill is not None  # Should still load (graceful degradation)
    assert "lowercase letters, numbers, and hyphens" in output


def test_skill_name_validation_warns_on_underscores(tmp_path, capsys):
    """Test that underscores in name trigger a warning."""
    skill_file = tmp_path / "bad-name" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        textwrap.dedent(
            """\
            ---
            name: bad_name_here
            description: Name with underscores
            ---
            Content
            """
        ),
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)
    skill = loader.load_skill(skill_file)
    output = capsys.readouterr().out

    assert skill is not None
    assert "lowercase letters, numbers, and hyphens" in output


def test_skill_name_too_long_warns(tmp_path, capsys):
    """Test that name exceeding 64 characters triggers a warning."""
    long_name = "a" * 65
    skill_file = tmp_path / "long" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        f"---\nname: {long_name}\ndescription: Test\n---\nContent",
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)
    skill = loader.load_skill(skill_file)
    output = capsys.readouterr().out

    assert skill is not None
    assert "exceeds 64 characters" in output


def test_skill_name_valid_does_not_warn(tmp_path, capsys):
    """Test that valid names don't trigger warnings."""
    skill_file = tmp_path / "valid" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        textwrap.dedent(
            """\
            ---
            name: valid-skill-name123
            description: A valid skill name
            ---
            Content
            """
        ),
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)
    skill = loader.load_skill(skill_file)
    output = capsys.readouterr().out

    assert skill is not None
    assert skill.name == "valid-skill-name123"
    assert "lowercase letters" not in output
    assert "exceeds" not in output


def test_skill_description_too_long_warns(tmp_path, capsys):
    """Test that description exceeding 1024 characters triggers a warning."""
    long_desc = "x" * 1025
    skill_file = tmp_path / "longdesc" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        f"---\nname: valid\ndescription: {long_desc}\n---\nContent",
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)
    skill = loader.load_skill(skill_file)
    output = capsys.readouterr().out

    assert skill is not None
    assert "exceeds 1024 characters" in output


def test_skill_description_valid_does_not_warn(tmp_path, capsys):
    """Test that valid descriptions don't trigger warnings."""
    skill_file = tmp_path / "valid-desc" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text(
        textwrap.dedent(
            """\
            ---
            name: valid-desc
            description: A perfectly normal description that is well under the limit.
            ---
            Content
            """
        ),
        encoding="utf-8",
    )

    loader = SkillLoader(tmp_path)
    skill = loader.load_skill(skill_file)
    output = capsys.readouterr().out

    assert skill is not None
    assert "exceeds" not in output
