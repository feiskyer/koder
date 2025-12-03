"""Tests for skill-based tool restriction enforcement."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from koder_agent.tools.skill import Skill  # noqa: E402
from koder_agent.tools.skill_context import (  # noqa: E402
    SkillRestrictions,
    add_skill_restrictions,
    clear_restrictions,
    get_active_restrictions,
    has_active_restrictions,
)


@pytest.fixture(autouse=True)
def reset_restrictions():
    """Clear restrictions before and after each test."""
    clear_restrictions()
    yield
    clear_restrictions()


class TestSkillRestrictions:
    """Tests for the SkillRestrictions dataclass."""

    def test_always_allowed_tools_bypass_restrictions(self):
        """Test that always-allowed tools work regardless of restrictions."""
        restrictions = SkillRestrictions(
            loaded_skills=["test-skill"],
            allowed_tools={"read_file"},
        )

        # Always-allowed tools should pass
        assert restrictions.is_tool_allowed("get_skill") is True
        assert restrictions.is_tool_allowed("todo_read") is True
        assert restrictions.is_tool_allowed("todo_write") is True

    def test_allowed_tools_are_permitted(self):
        """Test that tools in the allowed set are permitted."""
        restrictions = SkillRestrictions(
            loaded_skills=["test-skill"],
            allowed_tools={"read_file", "glob_search"},
        )

        assert restrictions.is_tool_allowed("read_file") is True
        assert restrictions.is_tool_allowed("glob_search") is True

    def test_non_allowed_tools_are_blocked(self):
        """Test that tools not in the allowed set are blocked."""
        restrictions = SkillRestrictions(
            loaded_skills=["test-skill"],
            allowed_tools={"read_file"},
        )

        assert restrictions.is_tool_allowed("write_file") is False
        assert restrictions.is_tool_allowed("run_shell") is False

    def test_empty_allowed_tools_permits_all(self):
        """Test that empty allowed_tools means no restrictions."""
        restrictions = SkillRestrictions(
            loaded_skills=["test-skill"],
            allowed_tools=set(),
        )

        # Should allow any tool when no restrictions defined
        assert restrictions.is_tool_allowed("read_file") is True
        assert restrictions.is_tool_allowed("write_file") is True
        assert restrictions.is_tool_allowed("run_shell") is True

    def test_add_skill_accumulates_tools(self):
        """Test that adding skills accumulates allowed tools (union)."""
        restrictions = SkillRestrictions()

        restrictions.add_skill("skill1", ["read_file", "glob_search"])
        assert restrictions.allowed_tools == {"read_file", "glob_search"}
        assert restrictions.loaded_skills == ["skill1"]

        restrictions.add_skill("skill2", ["write_file", "edit_file"])
        assert restrictions.allowed_tools == {
            "read_file",
            "glob_search",
            "write_file",
            "edit_file",
        }
        assert restrictions.loaded_skills == ["skill1", "skill2"]

    def test_add_same_skill_twice_no_duplicates(self):
        """Test that adding the same skill twice doesn't create duplicates."""
        restrictions = SkillRestrictions()

        restrictions.add_skill("skill1", ["read_file"])
        restrictions.add_skill("skill1", ["write_file"])

        assert restrictions.loaded_skills == ["skill1"]
        assert restrictions.allowed_tools == {"read_file", "write_file"}


class TestSkillContextFunctions:
    """Tests for the skill context management functions."""

    def test_get_active_restrictions_returns_none_initially(self):
        """Test that no restrictions are active initially."""
        assert get_active_restrictions() is None
        assert has_active_restrictions() is False

    def test_add_skill_restrictions_activates_restrictions(self):
        """Test that adding skill restrictions activates them."""
        skill = Skill(
            name="test-skill",
            description="Test skill",
            content="Content",
            allowed_tools=["read_file", "glob_search"],
        )

        add_skill_restrictions(skill)

        restrictions = get_active_restrictions()
        assert restrictions is not None
        assert has_active_restrictions() is True
        assert "test-skill" in restrictions.loaded_skills
        assert restrictions.allowed_tools == {"read_file", "glob_search"}

    def test_add_skill_without_allowed_tools_does_nothing(self):
        """Test that adding a skill without allowed_tools doesn't create restrictions."""
        skill = Skill(
            name="unrestricted-skill",
            description="No restrictions",
            content="Content",
            allowed_tools=None,
        )

        add_skill_restrictions(skill)

        assert get_active_restrictions() is None
        assert has_active_restrictions() is False

    def test_clear_restrictions_removes_all(self):
        """Test that clear_restrictions removes all active restrictions."""
        skill = Skill(
            name="test-skill",
            description="Test skill",
            content="Content",
            allowed_tools=["read_file"],
        )

        add_skill_restrictions(skill)
        assert has_active_restrictions() is True

        clear_restrictions()
        assert get_active_restrictions() is None
        assert has_active_restrictions() is False

    def test_multiple_skills_union_behavior(self):
        """Test that multiple skills with restrictions combine (union)."""
        skill1 = Skill(
            name="skill1",
            description="First skill",
            content="Content",
            allowed_tools=["read_file", "glob_search"],
        )
        skill2 = Skill(
            name="skill2",
            description="Second skill",
            content="Content",
            allowed_tools=["write_file", "edit_file"],
        )

        add_skill_restrictions(skill1)
        add_skill_restrictions(skill2)

        restrictions = get_active_restrictions()
        assert restrictions is not None
        assert restrictions.loaded_skills == ["skill1", "skill2"]
        assert restrictions.allowed_tools == {
            "read_file",
            "glob_search",
            "write_file",
            "edit_file",
        }

        # All tools from both skills should be allowed
        assert restrictions.is_tool_allowed("read_file") is True
        assert restrictions.is_tool_allowed("write_file") is True
        # Tools not in either skill should be blocked
        assert restrictions.is_tool_allowed("run_shell") is False


class TestSkillGuardrail:
    """Tests for the skill tool restriction guardrail."""

    def test_guardrail_allows_when_no_restrictions(self):
        """Test that guardrail allows all tools when no restrictions active."""
        from agents import ToolInputGuardrailData

        from koder_agent.agentic.skill_guardrail import skill_tool_restriction_guardrail

        # Create mock data
        mock_context = MagicMock()
        mock_context.tool_name = "run_shell"
        data = MagicMock(spec=ToolInputGuardrailData)
        data.context = mock_context

        result = skill_tool_restriction_guardrail(data)

        assert result.behavior["type"] == "allow"

    def test_guardrail_allows_permitted_tools(self):
        """Test that guardrail allows tools in the allowed set."""
        from agents import ToolInputGuardrailData

        from koder_agent.agentic.skill_guardrail import skill_tool_restriction_guardrail

        # Set up restrictions
        skill = Skill(
            name="read-only-skill",
            description="Read only",
            content="Content",
            allowed_tools=["read_file", "glob_search"],
        )
        add_skill_restrictions(skill)

        # Create mock data
        mock_context = MagicMock()
        mock_context.tool_name = "read_file"
        data = MagicMock(spec=ToolInputGuardrailData)
        data.context = mock_context

        result = skill_tool_restriction_guardrail(data)

        assert result.behavior["type"] == "allow"

    def test_guardrail_blocks_unpermitted_tools(self):
        """Test that guardrail blocks tools not in the allowed set."""
        from agents import ToolInputGuardrailData

        from koder_agent.agentic.skill_guardrail import skill_tool_restriction_guardrail

        # Set up restrictions
        skill = Skill(
            name="read-only-skill",
            description="Read only",
            content="Content",
            allowed_tools=["read_file"],
        )
        add_skill_restrictions(skill)

        # Create mock data for a blocked tool
        mock_context = MagicMock()
        mock_context.tool_name = "write_file"
        data = MagicMock(spec=ToolInputGuardrailData)
        data.context = mock_context

        result = skill_tool_restriction_guardrail(data)

        assert result.behavior["type"] == "reject_content"
        assert result.output_info.get("blocked_tool") == "write_file"

    def test_guardrail_always_allows_escape_tools(self):
        """Test that always-allowed tools work even with restrictions."""
        from agents import ToolInputGuardrailData

        from koder_agent.agentic.skill_guardrail import skill_tool_restriction_guardrail

        # Set up restrictions
        skill = Skill(
            name="restrictive-skill",
            description="Very restrictive",
            content="Content",
            allowed_tools=["read_file"],  # Only read_file allowed
        )
        add_skill_restrictions(skill)

        # get_skill should still work (escape hatch)
        mock_context = MagicMock()
        mock_context.tool_name = "get_skill"
        data = MagicMock(spec=ToolInputGuardrailData)
        data.context = mock_context

        result = skill_tool_restriction_guardrail(data)

        assert result.behavior["type"] == "allow"

    def test_guardrail_rejects_missing_tool_name(self):
        """Test that missing tool_name is handled gracefully and rejected."""
        from agents import ToolInputGuardrailData

        from koder_agent.agentic.skill_guardrail import skill_tool_restriction_guardrail

        # Set up restrictions
        skill = Skill(
            name="restrictive-skill",
            description="Very restrictive",
            content="Content",
            allowed_tools=["read_file"],
        )
        add_skill_restrictions(skill)

        # Mock context without tool_name attribute
        mock_context = MagicMock(spec=[])  # Empty spec - no attributes
        data = MagicMock(spec=ToolInputGuardrailData)
        data.context = mock_context

        result = skill_tool_restriction_guardrail(data)

        assert result.behavior["type"] == "reject_content"
        assert result.output_info.get("error") == "missing_tool_name"

    def test_guardrail_rejects_empty_tool_name(self):
        """Test that empty tool_name string is handled gracefully and rejected."""
        from agents import ToolInputGuardrailData

        from koder_agent.agentic.skill_guardrail import skill_tool_restriction_guardrail

        # Set up restrictions
        skill = Skill(
            name="restrictive-skill",
            description="Very restrictive",
            content="Content",
            allowed_tools=["read_file"],
        )
        add_skill_restrictions(skill)

        # Mock context with empty tool_name
        mock_context = MagicMock()
        mock_context.tool_name = ""
        data = MagicMock(spec=ToolInputGuardrailData)
        data.context = mock_context

        result = skill_tool_restriction_guardrail(data)

        assert result.behavior["type"] == "reject_content"
        assert result.output_info.get("error") == "missing_tool_name"
