"""Token efficiency tests for skills with progressive disclosure."""

import sys
import types
from pathlib import Path

import pytest
import tiktoken

# Stub litellm before importing koder_agent to avoid optional dependency issues
if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

# Stub ddgs so that web tools can be imported without the optional dependency
if "ddgs" not in sys.modules:
    ddgs_stub = types.ModuleType("ddgs")

    class _DummyDDGS:
        def __init__(self, *_, **__):
            pass

    ddgs_stub.DDGS = _DummyDDGS

    ddgs_exceptions_stub = types.ModuleType("ddgs.exceptions")

    class _DummyDDGSError(Exception):
        pass

    ddgs_exceptions_stub.DDGSException = _DummyDDGSError

    ddgs_stub.exceptions = ddgs_exceptions_stub

    sys.modules["ddgs"] = ddgs_stub
    sys.modules["ddgs.exceptions"] = ddgs_exceptions_stub

# Ensure project root is on sys.path when running tests directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from koder_agent.agentic.agent import _get_skills_metadata  # noqa: E402
from koder_agent.config.models import KoderConfig, SkillsConfig  # noqa: E402
from koder_agent.tools.skill import SkillLoader  # noqa: E402

ENCODING = tiktoken.get_encoding("cl100k_base")
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "skills"


def count_tokens(text: str) -> int:
    """Count tokens using the same encoding as the main codebase."""
    return len(ENCODING.encode(text or ""))


def parse_metadata_lines(metadata: str) -> dict[str, str]:
    """Parse '- name: description' lines from a metadata prompt."""
    mapping: dict[str, str] = {}
    for line in metadata.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        body = line[2:]
        if ":" not in body:
            continue
        name, _ = body.split(":", 1)
        mapping[name.strip()] = line
    return mapping


@pytest.fixture(scope="module")
def skill_loader() -> SkillLoader:
    """Loader bound to the test fixtures directory."""
    loader = SkillLoader(FIXTURES_DIR)
    loader.discover_skills()
    return loader


@pytest.fixture(scope="module")
def skills_by_name(skill_loader: SkillLoader) -> dict[str, object]:
    """Mapping of skill name -> Skill instance."""
    return {name: skill_loader.get_skill(name) for name in skill_loader.list_skills()}


@pytest.fixture(scope="module")
def metadata_prompt(skill_loader: SkillLoader) -> str:
    """Level 1 metadata prompt produced by the loader."""
    return skill_loader.get_skills_metadata_prompt()


class TestTokenCounting:
    """Basic verification that token counting behaves as expected."""

    def test_longer_text_has_more_tokens(self):
        short = "small example"
        medium = " ".join(["small example"] * 10)
        long = " ".join(["small example"] * 100)

        assert count_tokens(short) < count_tokens(medium) < count_tokens(long)

    def test_empty_string_has_zero_tokens(self):
        assert count_tokens("") == 0
        assert count_tokens(None or "") == 0


class TestMetadataVsFullContent:
    """Compare Level 1 metadata to full skill content."""

    def test_each_skill_metadata_under_ten_percent(self, skills_by_name, metadata_prompt):
        meta_map = parse_metadata_lines(metadata_prompt)

        for name, skill in skills_by_name.items():
            assert name in meta_map, f"Missing metadata line for skill {name}"

            meta_tokens = count_tokens(meta_map[name])
            full_tokens = count_tokens(skill.to_prompt())

            # Per-skill metadata should be <10% of full prompt tokens
            ratio = meta_tokens / full_tokens
            assert ratio < 0.10, f"Metadata for {name} is too large: {ratio:.2%}"

    def test_overall_metadata_under_five_percent(self, skills_by_name, metadata_prompt):
        meta_map = parse_metadata_lines(metadata_prompt)

        total_meta_tokens = sum(count_tokens(line) for line in meta_map.values())
        total_full_tokens = sum(
            count_tokens(skill.to_prompt()) for skill in skills_by_name.values()
        )

        ratio = total_meta_tokens / total_full_tokens
        # Level 1 metadata should be a very small slice of the total
        assert ratio < 0.05, f"Level 1 metadata too large overall: {ratio:.2%}"


class TestProgressiveDisclosureSavings:
    """Measure token savings from using Level 1 metadata instead of full prompts."""

    def test_level1_metadata_saves_over_ninety_percent(self, skills_by_name, metadata_prompt):
        baseline_full_tokens = sum(
            count_tokens(skill.to_prompt()) for skill in skills_by_name.values()
        )
        level1_tokens = count_tokens(metadata_prompt)

        savings = 1.0 - (level1_tokens / baseline_full_tokens)
        # Progressive disclosure should give very large savings at Level 1
        assert savings > 0.90, f"Expected >90% savings, got {savings:.2%}"

    def test_agent_system_prompt_uses_metadata(self):
        """_get_skills_metadata should mirror the loader's metadata semantics."""
        config = KoderConfig(
            skills=SkillsConfig(
                enabled=True,
                project_skills_dir=str(FIXTURES_DIR),
                user_skills_dir="/nonexistent-user-skills",
            )
        )
        metadata = _get_skills_metadata(config)

        assert "Available skills:" in metadata
        # Names and short descriptions should be present, but not detailed body sections
        assert "large-skill: Large skill with extensive content and supplementary files" in metadata
        assert "System Design Field Guide" not in metadata
        assert "Resilience and Operations" not in metadata


class TestSelectiveLoading:
    """Ensure only requested content is loaded at Level 2."""

    def test_single_skill_prompt_does_not_include_other_skills(self, skill_loader):
        small_skill = skill_loader.get_skill("small-skill")
        assert small_skill is not None
        prompt = small_skill.to_prompt()

        # The prompt should focus on the requested skill only
        assert "small-skill" in prompt
        assert "medium-skill" not in prompt
        assert "large-skill" not in prompt

    def test_template_skill_is_much_smaller_than_large_skill(self, skill_loader):
        template = skill_loader.get_skill("template-skill")
        large = skill_loader.get_skill("large-skill")
        assert template is not None and large is not None

        template_tokens = count_tokens(template.to_prompt())
        large_tokens = count_tokens(large.to_prompt())

        assert large_tokens > template_tokens * 3


class TestLevel3LazyLoading:
    """Verify that supplementary files are not auto-loaded as content."""

    def test_reference_file_is_not_inlined(self, skill_loader):
        large = skill_loader.get_skill("large-skill")
        assert large is not None

        content = large.content
        reference_path = (FIXTURES_DIR / "large-skill" / "reference.md").resolve()

        # Link path should be rewritten to an absolute path
        assert str(reference_path) in content

        # But the actual reference file contents should not be inlined
        ref_text = reference_path.read_text(encoding="utf-8")
        # Use a small prefix to avoid brittle length assumptions
        assert ref_text.splitlines()[0] not in content


class TestRealWorldScenarios:
    """Simulate realistic usage patterns and compare token costs."""

    def test_single_skill_session_vs_eager_loading(self, skills_by_name, metadata_prompt):
        # Baseline: system prompt eagerly includes full content for all skills
        eager_tokens = sum(count_tokens(skill.to_prompt()) for skill in skills_by_name.values())

        # Realistic scenario: agent starts with Level 1 metadata,
        # then loads only the small skill during the session.
        small_skill = skills_by_name["small-skill"]
        session_tokens = count_tokens(metadata_prompt) + count_tokens(small_skill.to_prompt())

        savings = 1.0 - (session_tokens / eager_tokens)
        assert savings > 0.70, f"Expected >70% savings, got {savings:.2%}"

    def test_heavy_session_loading_multiple_skills(self, skills_by_name, metadata_prompt):
        # Scenario: a longer session that ends up loading the medium and large skills.
        eager_tokens = sum(count_tokens(skill.to_prompt()) for skill in skills_by_name.values())

        medium_skill = skills_by_name["medium-skill"]
        large_skill = skills_by_name["large-skill"]

        session_tokens = (
            count_tokens(metadata_prompt)
            + count_tokens(medium_skill.to_prompt())
            + count_tokens(large_skill.to_prompt())
        )

        # Even when several skills are loaded, there should still be
        # a meaningful reduction compared with eager loading everything
        assert session_tokens <= eager_tokens
        savings = 1.0 - (session_tokens / eager_tokens)
        assert savings > 0.20, f"Expected noticeable savings, got {savings:.2%}"
