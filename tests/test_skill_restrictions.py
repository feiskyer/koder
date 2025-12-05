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


class TestToolGuardrailIntegration:
    """Tests for proper guardrail integration with tools and agent.

    These tests ensure that:
    1. ToolInputGuardrail is attached to tools (not agent's input_guardrails)
    2. Agent creation works without AttributeError for run_in_parallel
    3. The guardrail type is correct for the SDK's expectations
    """

    def test_all_tools_have_skill_guardrail_attached(self):
        """Test that get_all_tools() attaches skill_restriction_guardrail to each tool."""
        from agents import FunctionTool

        from koder_agent.agentic.skill_guardrail import skill_restriction_guardrail
        from koder_agent.tools import get_all_tools

        tools = get_all_tools()

        # Verify we have tools
        assert len(tools) > 0, "Expected at least one tool"

        # Verify each FunctionTool has the guardrail attached
        for tool in tools:
            if isinstance(tool, FunctionTool):
                assert tool.tool_input_guardrails is not None, (
                    f"Tool '{tool.name}' should have tool_input_guardrails"
                )
                assert len(tool.tool_input_guardrails) > 0, (
                    f"Tool '{tool.name}' should have at least one guardrail"
                )
                assert skill_restriction_guardrail in tool.tool_input_guardrails, (
                    f"Tool '{tool.name}' should have skill_restriction_guardrail"
                )

    def test_skill_restriction_guardrail_is_correct_type(self):
        """Test that skill_restriction_guardrail is a ToolInputGuardrail, not InputGuardrail."""
        from agents import ToolInputGuardrail

        from koder_agent.agentic.skill_guardrail import skill_restriction_guardrail

        # The guardrail must be ToolInputGuardrail (for per-tool validation)
        # NOT InputGuardrail (which has run_in_parallel and is for agent-level)
        assert isinstance(skill_restriction_guardrail, ToolInputGuardrail), (
            "skill_restriction_guardrail must be a ToolInputGuardrail instance"
        )

        # ToolInputGuardrail should NOT have run_in_parallel attribute
        # (that's only on InputGuardrail for agent-level guardrails)
        assert not hasattr(skill_restriction_guardrail, "run_in_parallel"), (
            "ToolInputGuardrail should not have run_in_parallel attribute"
        )

    def test_agent_creation_with_tools_no_attribute_error(self):
        """Test that Agent can be created with tools without run_in_parallel AttributeError.

        This is a regression test for the bug where ToolInputGuardrail was incorrectly
        passed to Agent's input_guardrails (which expects InputGuardrail with run_in_parallel).
        """
        from agents import Agent

        from koder_agent.tools import get_all_tools

        tools = get_all_tools()

        # This should NOT raise AttributeError: 'ToolInputGuardrail' has no attribute 'run_in_parallel'
        agent = Agent(
            name="test-agent",
            instructions="Test agent for guardrail integration",
            tools=tools,
        )

        assert agent is not None
        assert len(agent.tools) == len(tools)
        # Agent should NOT have input_guardrails set (guardrails are on tools now)
        assert len(agent.input_guardrails) == 0

    def test_tool_guardrails_not_duplicated_on_repeated_calls(self):
        """Test that calling get_all_tools() multiple times doesn't duplicate guardrails."""
        from koder_agent.agentic.skill_guardrail import skill_restriction_guardrail
        from koder_agent.tools import get_all_tools

        # Call get_all_tools multiple times to simulate repeated usage
        get_all_tools()
        get_all_tools()
        tools = get_all_tools()

        # Check that guardrails aren't duplicated
        for tool in tools:
            if hasattr(tool, "tool_input_guardrails") and tool.tool_input_guardrails:
                guardrail_count = tool.tool_input_guardrails.count(skill_restriction_guardrail)
                assert guardrail_count == 1, (
                    f"Tool '{tool.name}' has {guardrail_count} copies of skill_restriction_guardrail, expected 1"
                )


class TestPatternBasedRestrictions:
    """Tests for pattern-based tool restriction matching.

    Pattern syntax:
    - "read_file"           - Exact tool name match
    - "run_shell:git *"     - Shell commands matching glob pattern
    - "run_shell:*"         - All shell commands allowed
    - "*"                   - Wildcard, all tools allowed
    """

    def test_exact_tool_name_match(self):
        """Test that exact tool names still work."""
        restrictions = SkillRestrictions(
            loaded_skills=["test-skill"],
            allowed_tools={"read_file", "write_file"},
        )

        assert restrictions.is_tool_allowed("read_file") is True
        assert restrictions.is_tool_allowed("write_file") is True
        assert restrictions.is_tool_allowed("run_shell") is False

    def test_wildcard_allows_all_tools(self):
        """Test that '*' pattern allows all tools."""
        restrictions = SkillRestrictions(
            loaded_skills=["permissive-skill"],
            allowed_tools={"*"},
        )

        assert restrictions.is_tool_allowed("read_file") is True
        assert restrictions.is_tool_allowed("write_file") is True
        assert restrictions.is_tool_allowed("run_shell") is True
        assert restrictions.is_tool_allowed("any_tool_name") is True

    def test_shell_command_pattern_allows_matching_commands(self):
        """Test that 'run_shell:pattern' allows matching shell commands."""
        import json

        restrictions = SkillRestrictions(
            loaded_skills=["git-skill"],
            allowed_tools={"run_shell:git *"},
        )

        # Should allow git commands
        git_status_args = json.dumps({"command": "git status"})
        assert restrictions.is_tool_allowed("run_shell", git_status_args) is True

        git_commit_args = json.dumps({"command": "git commit -m 'test'"})
        assert restrictions.is_tool_allowed("run_shell", git_commit_args) is True

        # Should block non-git commands
        cat_args = json.dumps({"command": "cat /etc/passwd"})
        assert restrictions.is_tool_allowed("run_shell", cat_args) is False

        rm_args = json.dumps({"command": "rm -rf /"})
        assert restrictions.is_tool_allowed("run_shell", rm_args) is False

    def test_shell_command_pattern_with_wildcard(self):
        """Test that 'run_shell:*' allows all shell commands."""
        import json

        restrictions = SkillRestrictions(
            loaded_skills=["shell-skill"],
            allowed_tools={"run_shell:*"},
        )

        # Should allow any command
        assert (
            restrictions.is_tool_allowed("run_shell", json.dumps({"command": "cat foo.txt"}))
            is True
        )
        assert restrictions.is_tool_allowed("run_shell", json.dumps({"command": "ls -la"})) is True
        assert (
            restrictions.is_tool_allowed("run_shell", json.dumps({"command": "rm -rf /"})) is True
        )

    def test_shell_pattern_blocks_run_shell_without_args(self):
        """Test that shell patterns require tool_args to match."""
        restrictions = SkillRestrictions(
            loaded_skills=["git-skill"],
            allowed_tools={"run_shell:git *"},
        )

        # Without tool_args, pattern can't match
        assert restrictions.is_tool_allowed("run_shell") is False
        assert restrictions.is_tool_allowed("run_shell", None) is False
        assert restrictions.is_tool_allowed("run_shell", "") is False

    def test_multiple_shell_patterns(self):
        """Test multiple shell command patterns work together."""
        import json

        restrictions = SkillRestrictions(
            loaded_skills=["dev-skill"],
            allowed_tools={"run_shell:git *", "run_shell:npm *", "run_shell:cat *"},
        )

        # All patterns should work
        assert (
            restrictions.is_tool_allowed("run_shell", json.dumps({"command": "git status"})) is True
        )
        assert (
            restrictions.is_tool_allowed("run_shell", json.dumps({"command": "npm install"}))
            is True
        )
        assert (
            restrictions.is_tool_allowed("run_shell", json.dumps({"command": "cat README.md"}))
            is True
        )

        # Non-matching commands should be blocked
        assert (
            restrictions.is_tool_allowed("run_shell", json.dumps({"command": "rm -rf /"})) is False
        )

    def test_git_command_pattern_matching(self):
        """Test that 'git_command:pattern' matches git command args."""
        import json

        restrictions = SkillRestrictions(
            loaded_skills=["git-readonly"],
            allowed_tools={"git_command:status", "git_command:log *", "git_command:diff *"},
        )

        # Exact match
        assert restrictions.is_tool_allowed("git_command", json.dumps({"args": "status"})) is True

        # Pattern match
        assert (
            restrictions.is_tool_allowed("git_command", json.dumps({"args": "log --oneline"}))
            is True
        )
        assert (
            restrictions.is_tool_allowed("git_command", json.dumps({"args": "diff HEAD~1"})) is True
        )

        # Non-matching
        assert (
            restrictions.is_tool_allowed("git_command", json.dumps({"args": "push origin main"}))
            is False
        )
        assert (
            restrictions.is_tool_allowed("git_command", json.dumps({"args": "commit -m 'test'"}))
            is False
        )

    def test_mixed_exact_and_pattern_restrictions(self):
        """Test combining exact tool names with patterns."""
        import json

        restrictions = SkillRestrictions(
            loaded_skills=["mixed-skill"],
            allowed_tools={"read_file", "glob_search", "run_shell:cat *"},
        )

        # Exact matches work
        assert restrictions.is_tool_allowed("read_file") is True
        assert restrictions.is_tool_allowed("glob_search") is True

        # Pattern matches work
        assert (
            restrictions.is_tool_allowed("run_shell", json.dumps({"command": "cat foo.txt"}))
            is True
        )

        # Non-allowed tools blocked
        assert restrictions.is_tool_allowed("write_file") is False
        assert (
            restrictions.is_tool_allowed("run_shell", json.dumps({"command": "rm foo.txt"}))
            is False
        )

    def test_glob_pattern_on_tool_names(self):
        """Test glob patterns on tool names themselves."""
        restrictions = SkillRestrictions(
            loaded_skills=["file-skill"],
            allowed_tools={"*_file", "glob_*"},
        )

        # Matching patterns
        assert restrictions.is_tool_allowed("read_file") is True
        assert restrictions.is_tool_allowed("write_file") is True
        assert restrictions.is_tool_allowed("edit_file") is True
        assert restrictions.is_tool_allowed("glob_search") is True

        # Non-matching
        assert restrictions.is_tool_allowed("run_shell") is False
        assert restrictions.is_tool_allowed("web_search") is False

    def test_invalid_json_in_tool_args_is_safe(self):
        """Test that invalid JSON in tool_args doesn't crash."""
        restrictions = SkillRestrictions(
            loaded_skills=["test-skill"],
            allowed_tools={"run_shell:git *"},
        )

        # Invalid JSON should not match (but also not crash)
        assert restrictions.is_tool_allowed("run_shell", "not valid json") is False
        assert restrictions.is_tool_allowed("run_shell", "{broken") is False
        assert restrictions.is_tool_allowed("run_shell", "null") is False

    def test_always_allowed_tools_bypass_patterns(self):
        """Test that always-allowed tools work even with restrictive patterns."""
        restrictions = SkillRestrictions(
            loaded_skills=["restrictive-skill"],
            allowed_tools={"read_file"},  # Very restrictive
        )

        # Always-allowed should still work
        assert restrictions.is_tool_allowed("get_skill") is True
        assert restrictions.is_tool_allowed("todo_read") is True
        assert restrictions.is_tool_allowed("todo_write") is True


class TestPatternGuardrailIntegration:
    """Tests for pattern matching through the guardrail."""

    def test_guardrail_with_shell_pattern(self):
        """Test that guardrail correctly enforces shell command patterns."""
        import json

        from agents import ToolInputGuardrailData

        from koder_agent.agentic.skill_guardrail import skill_tool_restriction_guardrail

        # Set up restrictions with shell pattern
        skill = Skill(
            name="git-only-skill",
            description="Only allows git commands",
            content="Content",
            allowed_tools=["run_shell:git *", "read_file"],
        )
        add_skill_restrictions(skill)

        # Test allowed git command
        mock_context = MagicMock()
        mock_context.tool_name = "run_shell"
        mock_context.tool_arguments = json.dumps({"command": "git status"})
        data = MagicMock(spec=ToolInputGuardrailData)
        data.context = mock_context

        result = skill_tool_restriction_guardrail(data)
        assert result.behavior["type"] == "allow"

        # Test blocked command
        mock_context.tool_arguments = json.dumps({"command": "rm -rf /"})
        result = skill_tool_restriction_guardrail(data)
        assert result.behavior["type"] == "reject_content"

    def test_guardrail_with_wildcard_pattern(self):
        """Test that guardrail correctly handles wildcard pattern."""
        from agents import ToolInputGuardrailData

        from koder_agent.agentic.skill_guardrail import skill_tool_restriction_guardrail

        # Set up restrictions with wildcard
        skill = Skill(
            name="permissive-skill",
            description="Allows everything",
            content="Content",
            allowed_tools=["*"],
        )
        add_skill_restrictions(skill)

        # Any tool should be allowed
        mock_context = MagicMock()
        mock_context.tool_name = "any_tool"
        mock_context.tool_arguments = None
        data = MagicMock(spec=ToolInputGuardrailData)
        data.context = mock_context

        result = skill_tool_restriction_guardrail(data)
        assert result.behavior["type"] == "allow"

        mock_context.tool_name = "another_tool"
        result = skill_tool_restriction_guardrail(data)
        assert result.behavior["type"] == "allow"
