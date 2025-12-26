"""Tests for streaming display functionality, particularly diff extraction and formatting."""

import sys
import types
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

# Stub litellm before importing koder_agent to avoid optional dependency issues
if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

# Ensure project root is on sys.path when running tests directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import koder_agent streaming display
try:
    from koder_agent.core.streaming_display import StreamingDisplayManager
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(f"Failed to import koder_agent modules: {e}") from e


@pytest.fixture
def display_manager():
    """Create a StreamingDisplayManager for testing."""
    console = Console(force_terminal=True, width=120)
    return StreamingDisplayManager(console)


# =============================================================================
# Tests for _extract_diff_content
# =============================================================================


def test_extract_diff_content_edit_file(display_manager):
    """_extract_diff_content extracts diff from edit_file output."""
    output = """Successfully applied diff to test.txt
---DIFF---
@@ -1,3 +1,3 @@
 line1
-old_line
+new_line
 line3"""

    renderables = display_manager._extract_diff_content("edit_file", output)

    # Should have renderables (summary + diff lines)
    assert len(renderables) > 0

    # Convert to strings for assertion
    texts = [str(r) if isinstance(r, Text) else r for r in renderables]
    text_content = "\n".join(texts)

    # Check for diff content
    assert "-old_line" in text_content or "old_line" in text_content
    assert "+new_line" in text_content or "new_line" in text_content


def test_extract_diff_content_write_file(display_manager):
    """_extract_diff_content extracts diff from write_file output."""
    output = """Created /path/test.txt (50 bytes)
---DIFF---
--- /dev/null
+++ b/test.txt
@@ -0,0 +1,3 @@
+line1
+line2
+line3"""

    renderables = display_manager._extract_diff_content("write_file", output)

    assert len(renderables) > 0

    texts = [str(r) if isinstance(r, Text) else r for r in renderables]
    text_content = "\n".join(texts)

    assert "+line1" in text_content or "line1" in text_content
    assert "+line2" in text_content or "line2" in text_content


def test_extract_diff_content_append_file(display_manager):
    """_extract_diff_content extracts diff from append_file output."""
    output = """Appended 20 bytes to /path/test.txt
---DIFF---
@@ -1,2 +1,4 @@
 existing1
 existing2
+appended1
+appended2"""

    renderables = display_manager._extract_diff_content("append_file", output)

    assert len(renderables) > 0

    texts = [str(r) if isinstance(r, Text) else r for r in renderables]
    text_content = "\n".join(texts)

    assert "+appended1" in text_content or "appended1" in text_content
    assert "+appended2" in text_content or "appended2" in text_content


def test_extract_diff_content_ignores_non_file_tools(display_manager):
    """_extract_diff_content returns empty for non-file tools."""
    output = """Some output with +additions and -deletions
@@ -1,1 +1,1 @@
-old
+new"""

    # These tools should not trigger diff extraction
    for tool in ["read_file", "list_directory", "run_shell", "git_command"]:
        renderables = display_manager._extract_diff_content(tool, output)
        assert len(renderables) == 0, f"{tool} should not extract diff"


def test_extract_diff_content_no_diff_marker(display_manager):
    """_extract_diff_content handles output without ---DIFF--- marker."""
    output = "Simple success message"

    renderables = display_manager._extract_diff_content("edit_file", output)

    # Should return empty or minimal content
    assert len(renderables) == 0


def test_extract_diff_content_counts_additions_deletions(display_manager):
    """_extract_diff_content correctly counts additions and deletions."""
    output = """Updated /path/test.txt
---DIFF---
@@ -1,4 +1,3 @@
 context
-deleted1
-deleted2
+added1
 context"""

    renderables = display_manager._extract_diff_content("write_file", output)

    # First renderable should be the summary with counts
    assert len(renderables) > 0

    # Convert first item to string and check for counts
    summary = str(renderables[0])
    # Should contain +1 (one addition) and -2 (two deletions)
    assert "+1" in summary or "1" in summary
    assert "-2" in summary or "2" in summary


def test_extract_diff_content_multiple_hunks(display_manager):
    """_extract_diff_content handles multiple hunks."""
    output = """Updated /path/test.txt
---DIFF---
@@ -1,2 +1,2 @@
-old1
+new1
 context
@@ -10,2 +10,2 @@
-old2
+new2
 context2"""

    renderables = display_manager._extract_diff_content("write_file", output)

    texts = [str(r) if isinstance(r, Text) else r for r in renderables]
    text_content = "\n".join(texts)

    # Should contain both hunks
    assert "new1" in text_content
    assert "new2" in text_content


def test_extract_diff_content_file_headers(display_manager):
    """_extract_diff_content includes file headers."""
    output = """Updated /path/test.txt
---DIFF---
--- a/test.txt
+++ b/test.txt
@@ -1,1 +1,1 @@
-old
+new"""

    renderables = display_manager._extract_diff_content("write_file", output)

    texts = [str(r) if isinstance(r, Text) else r for r in renderables]
    text_content = "\n".join(texts)

    # Should include file headers
    assert "--- a/test.txt" in text_content or "a/test.txt" in text_content


# =============================================================================
# Tests for _generate_smart_summary
# =============================================================================


def test_smart_summary_edit_file_success(display_manager):
    """_generate_smart_summary returns correct summary for successful edit_file."""
    output = """Successfully applied diff to /path/file.txt
---DIFF---
@@ -1,1 +1,1 @@
-old
+new"""

    summary = display_manager._generate_smart_summary("edit_file", output)

    assert summary == "Diff applied"


def test_smart_summary_edit_file_failure(display_manager):
    """_generate_smart_summary returns error message for failed edit_file."""
    output = "Failed to apply diff: patch does not match file content"

    summary = display_manager._generate_smart_summary("edit_file", output)

    assert "Failed to apply diff" in summary


def test_smart_summary_edit_file_not_found(display_manager):
    """_generate_smart_summary returns error for file not found."""
    output = "File not found: /path/nonexistent.txt"

    summary = display_manager._generate_smart_summary("edit_file", output)

    assert "File not found" in summary


def test_smart_summary_write_file_created(display_manager):
    """_generate_smart_summary returns correct summary for new file creation."""
    output = """Created /path/new.txt (100 bytes)
---DIFF---
+content"""

    summary = display_manager._generate_smart_summary("write_file", output)

    assert "Created" in summary
    assert "100 bytes" in summary


def test_smart_summary_write_file_updated(display_manager):
    """_generate_smart_summary returns correct summary for file update."""
    output = """Updated /path/existing.txt (50 bytes)
---DIFF---
-old
+new"""

    summary = display_manager._generate_smart_summary("write_file", output)

    assert "Updated" in summary


def test_smart_summary_append_file(display_manager):
    """_generate_smart_summary returns correct summary for append_file."""
    output = """Appended 25 bytes to /path/file.txt
---DIFF---
+appended content"""

    summary = display_manager._generate_smart_summary("append_file", output)

    assert "Appended" in summary
    assert "25 bytes" in summary


# =============================================================================
# Tests for _is_error_output
# =============================================================================


def test_is_error_output_edit_file_success(display_manager):
    """_is_error_output returns False for successful edit_file."""
    output = """Successfully applied diff to /path/file.txt
---DIFF---
-old
+new"""

    is_error = display_manager._is_error_output(output, "edit_file")

    assert is_error is False


def test_is_error_output_edit_file_failure(display_manager):
    """_is_error_output returns True for failed edit_file with error indicator."""
    # Use an output with a recognized error pattern
    output = "Error: Failed to apply diff: patch does not match"

    is_error = display_manager._is_error_output(output, "edit_file")

    # The generic error detection should catch "Error:"
    assert is_error is True


def test_is_error_output_file_not_found(display_manager):
    """_is_error_output returns True for file not found."""
    output = "File not found: /path/missing.txt"

    is_error = display_manager._is_error_output(output, "edit_file")

    assert is_error is True


# =============================================================================
# Integration tests
# =============================================================================


def test_full_display_flow_edit_file(display_manager):
    """Test complete display flow for edit_file tool."""

    # Simulate tool call
    class MockToolCallItem:
        class RawItem:
            name = "edit_file"
            arguments = '{"path": "/test.txt", "diff": "@@ -1 +1 @@\\n-old\\n+new"}'
            call_id = "test-123"

        raw_item = RawItem()

    display_manager.handle_tool_called(MockToolCallItem())

    # Simulate tool output
    class MockToolOutputItem:
        output = """Successfully applied diff to /test.txt
---DIFF---
@@ -1,1 +1,1 @@
-old
+new"""
        tool_call_id = "test-123"

        class RawItem:
            pass

        raw_item = RawItem()

    display_manager.handle_tool_output(MockToolOutputItem())

    # Get display content
    content = display_manager.get_display_content()

    # Should have content to display
    assert content is not None


def test_full_display_flow_write_file(display_manager):
    """Test complete display flow for write_file tool."""

    # Simulate tool call
    class MockToolCallItem:
        class RawItem:
            name = "write_file"
            arguments = '{"path": "/new.txt", "content": "hello"}'
            call_id = "test-456"

        raw_item = RawItem()

    display_manager.handle_tool_called(MockToolCallItem())

    # Simulate tool output
    class MockToolOutputItem:
        output = """Created /new.txt (5 bytes)
---DIFF---
--- /dev/null
+++ b/new.txt
@@ -0,0 +1,1 @@
+hello"""
        tool_call_id = "test-456"

        class RawItem:
            pass

        raw_item = RawItem()

    display_manager.handle_tool_output(MockToolOutputItem())

    # Get display content
    content = display_manager.get_display_content()

    assert content is not None
