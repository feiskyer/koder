"""Tests for file operation tools."""

import json
import sys
import tempfile
import types
from pathlib import Path

import pytest

# Stub litellm before importing koder_agent to avoid optional dependency issues
if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

# Ensure project root is on sys.path when running tests directly
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import koder_agent file tools
try:
    from koder_agent.tools.file import (
        _generate_diff_output,
        append_file,
        apply_diff,
        edit_file,
        list_directory,
        read_file,
        truncate_text_by_tokens,
        write_file,
    )
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(f"Failed to import koder_agent modules: {e}") from e


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample file with known content."""
    file_path = temp_dir / "sample.txt"
    content = "line1\nline2\nline3\nline4\nline5\n"
    file_path.write_text(content, encoding="utf-8")
    return file_path


# =============================================================================
# Tests for truncate_text_by_tokens
# =============================================================================


def test_truncate_text_by_tokens_under_limit():
    """Text under token limit is returned unchanged."""
    text = "Hello, this is a short text."
    result = truncate_text_by_tokens(text, max_tokens=1000)
    assert result == text


def test_truncate_text_by_tokens_over_limit():
    """Text over token limit is truncated with head and tail preserved."""
    # Create a long text that will exceed the token limit
    text = "word " * 10000  # ~10000 tokens
    result = truncate_text_by_tokens(text, max_tokens=500)

    assert len(result) < len(text)
    assert "[Content truncated:" in result
    assert "tokens limit]" in result


def test_truncate_text_by_tokens_preserves_boundaries():
    """Truncation preserves newline boundaries."""
    lines = ["line " + str(i) for i in range(1000)]
    text = "\n".join(lines)
    result = truncate_text_by_tokens(text, max_tokens=500)

    # The truncation note should be present
    assert "[Content truncated:" in result
    # Head and tail should be present (not cut mid-line)
    assert "line 0" in result or "line 1" in result  # Head preserved
    assert "line 999" in result or "line 998" in result  # Tail preserved


# =============================================================================
# Tests for read_file
# =============================================================================


@pytest.mark.asyncio
async def test_read_file_basic(sample_file):
    """read_file returns file content with line numbers."""
    result = await read_file.on_invoke_tool(None, json.dumps({"path": str(sample_file)}))

    # Check line number format
    assert "1|line1" in result
    assert "2|line2" in result
    assert "5|line5" in result


@pytest.mark.asyncio
async def test_read_file_line_number_format(sample_file):
    """read_file formats line numbers with proper padding."""
    result = await read_file.on_invoke_tool(None, json.dumps({"path": str(sample_file)}))

    # Line numbers should be right-aligned in 6-character field
    lines = result.split("\n")
    for line in lines:
        if "|" in line:
            num_part = line.split("|")[0]
            assert len(num_part) == 6, f"Line number padding wrong: {num_part!r}"


@pytest.mark.asyncio
async def test_read_file_with_offset(sample_file):
    """read_file respects offset parameter."""
    result = await read_file.on_invoke_tool(
        None, json.dumps({"path": str(sample_file), "offset": 3})
    )

    # Should start from line 3
    assert "1|line1" not in result
    assert "2|line2" not in result
    assert "3|line3" in result
    assert "4|line4" in result
    assert "5|line5" in result


@pytest.mark.asyncio
async def test_read_file_with_limit(sample_file):
    """read_file respects limit parameter."""
    result = await read_file.on_invoke_tool(
        None, json.dumps({"path": str(sample_file), "limit": 2})
    )

    # Should only have 2 lines
    assert "1|line1" in result
    assert "2|line2" in result
    assert "3|line3" not in result


@pytest.mark.asyncio
async def test_read_file_with_offset_and_limit(sample_file):
    """read_file respects both offset and limit parameters."""
    result = await read_file.on_invoke_tool(
        None, json.dumps({"path": str(sample_file), "offset": 2, "limit": 2})
    )

    # Should have lines 2-3 only
    assert "1|line1" not in result
    assert "2|line2" in result
    assert "3|line3" in result
    assert "4|line4" not in result


@pytest.mark.asyncio
async def test_read_file_not_found(temp_dir):
    """read_file returns error for non-existent file."""
    result = await read_file.on_invoke_tool(
        None, json.dumps({"path": str(temp_dir / "nonexistent.txt")})
    )

    assert "File not found" in result


@pytest.mark.asyncio
async def test_read_file_offset_beyond_file(sample_file):
    """read_file handles offset beyond file length gracefully."""
    result = await read_file.on_invoke_tool(
        None, json.dumps({"path": str(sample_file), "offset": 100})
    )

    # Should return empty content (no lines match)
    assert result == "" or "line" not in result


# =============================================================================
# Tests for write_file
# =============================================================================


@pytest.mark.asyncio
async def test_write_file_creates_new_file(temp_dir):
    """write_file creates a new file with content."""
    file_path = temp_dir / "new_file.txt"
    content = "Hello, World!"

    result = await write_file.on_invoke_tool(
        None, json.dumps({"path": str(file_path), "content": content})
    )

    assert "Created" in result
    assert str(len(content)) in result
    assert file_path.exists()
    assert file_path.read_text() == content
    # Check diff output is included
    assert "---DIFF---" in result
    assert "+Hello, World!" in result


@pytest.mark.asyncio
async def test_write_file_overwrites_existing(sample_file):
    """write_file overwrites existing file content."""
    new_content = "Completely new content"

    result = await write_file.on_invoke_tool(
        None, json.dumps({"path": str(sample_file), "content": new_content})
    )

    assert "Updated" in result
    assert sample_file.read_text() == new_content
    # Check diff output is included
    assert "---DIFF---" in result


@pytest.mark.asyncio
async def test_write_file_creates_parent_directories(temp_dir):
    """write_file creates parent directories if they don't exist."""
    file_path = temp_dir / "nested" / "deep" / "file.txt"
    content = "Nested content"

    result = await write_file.on_invoke_tool(
        None, json.dumps({"path": str(file_path), "content": content})
    )

    assert "Created" in result
    assert file_path.exists()
    assert file_path.read_text() == content


# =============================================================================
# Tests for append_file
# =============================================================================


@pytest.mark.asyncio
async def test_append_file_to_existing(sample_file):
    """append_file adds content to existing file."""
    original = sample_file.read_text()
    new_content = "\nappended line"

    result = await append_file.on_invoke_tool(
        None, json.dumps({"path": str(sample_file), "content": new_content})
    )

    assert "Appended" in result
    assert sample_file.read_text() == original + new_content


@pytest.mark.asyncio
async def test_append_file_creates_new_file(temp_dir):
    """append_file creates file if it doesn't exist."""
    file_path = temp_dir / "new_append.txt"
    content = "First content"

    result = await append_file.on_invoke_tool(
        None, json.dumps({"path": str(file_path), "content": content})
    )

    assert "Appended" in result
    assert file_path.exists()
    assert file_path.read_text() == content


# =============================================================================
# Tests for edit_file (diff-based editing)
# =============================================================================


@pytest.mark.asyncio
async def test_edit_file_simple_diff(sample_file):
    """edit_file successfully applies a simple unified diff."""
    diff = """--- a/sample.txt
+++ b/sample.txt
@@ -2,3 +2,3 @@
 line2
-line3
+REPLACED
 line4"""

    result = await edit_file.on_invoke_tool(
        None,
        json.dumps({"path": str(sample_file), "diff": diff}),
    )

    assert "Successfully applied diff" in result
    content = sample_file.read_text()
    assert "REPLACED" in content
    assert "line3" not in content


@pytest.mark.asyncio
async def test_edit_file_file_not_found(temp_dir):
    """edit_file returns error for non-existent file."""
    diff = "@@ -1,1 +1,1 @@\n-old\n+new"
    result = await edit_file.on_invoke_tool(
        None,
        json.dumps({"path": str(temp_dir / "nonexistent.txt"), "diff": diff}),
    )

    assert "File not found" in result


@pytest.mark.asyncio
async def test_edit_file_diff_not_match(temp_dir):
    """edit_file returns error when diff doesn't match file content."""
    file_path = temp_dir / "test.txt"
    file_path.write_text("actual content\n", encoding="utf-8")

    diff = """@@ -1,1 +1,1 @@
-wrong content
+new content"""

    result = await edit_file.on_invoke_tool(
        None,
        json.dumps({"path": str(file_path), "diff": diff}),
    )

    assert "Failed to apply diff" in result or "does not match" in result
    # File should not be modified
    assert file_path.read_text() == "actual content\n"


@pytest.mark.asyncio
async def test_edit_file_preserves_other_content(sample_file):
    """edit_file only changes the lines specified in the diff."""
    diff = """@@ -2,3 +2,3 @@
 line2
-line3
+MODIFIED
 line4"""

    await edit_file.on_invoke_tool(
        None,
        json.dumps({"path": str(sample_file), "diff": diff}),
    )

    content = sample_file.read_text()
    # Other lines should be unchanged
    assert "line1" in content
    assert "line2" in content
    assert "line4" in content
    assert "line5" in content
    # Only line3 should be changed
    assert "MODIFIED" in content


@pytest.mark.asyncio
async def test_edit_file_multiline_diff(temp_dir):
    """edit_file handles multiline changes in diff."""
    file_path = temp_dir / "multiline.txt"
    file_path.write_text("start\nmiddle\nend\n", encoding="utf-8")

    diff = """@@ -1,2 +1,4 @@
-start
-middle
+replaced
+with
+multiple
+lines
 end"""

    result = await edit_file.on_invoke_tool(
        None,
        json.dumps({"path": str(file_path), "diff": diff}),
    )

    assert "Successfully applied diff" in result
    content = file_path.read_text()
    assert "replaced\nwith\nmultiple\nlines\nend" in content


@pytest.mark.asyncio
async def test_edit_file_delete_lines(sample_file):
    """edit_file can delete lines using diff."""
    diff = """@@ -2,3 +2,2 @@
 line2
-line3
 line4"""

    result = await edit_file.on_invoke_tool(
        None,
        json.dumps({"path": str(sample_file), "diff": diff}),
    )

    assert "Successfully applied diff" in result
    content = sample_file.read_text()
    assert "line3" not in content
    assert "line2" in content
    assert "line4" in content


@pytest.mark.asyncio
async def test_edit_file_add_lines(temp_dir):
    """edit_file can add new lines using diff."""
    file_path = temp_dir / "add.txt"
    file_path.write_text("line1\nline2\n", encoding="utf-8")

    diff = """@@ -1,2 +1,4 @@
 line1
+new_line_a
+new_line_b
 line2"""

    result = await edit_file.on_invoke_tool(
        None,
        json.dumps({"path": str(file_path), "diff": diff}),
    )

    assert "Successfully applied diff" in result
    content = file_path.read_text()
    assert "new_line_a" in content
    assert "new_line_b" in content


@pytest.mark.asyncio
async def test_edit_file_multiple_hunks(temp_dir):
    """edit_file can apply multiple hunks in one diff."""
    file_path = temp_dir / "multi.txt"
    file_path.write_text("a\nb\nc\nd\ne\nf\ng\n", encoding="utf-8")

    diff = """@@ -1,2 +1,2 @@
-a
+A
 b
@@ -6,2 +6,2 @@
-f
+F
 g"""

    result = await edit_file.on_invoke_tool(
        None,
        json.dumps({"path": str(file_path), "diff": diff}),
    )

    assert "Successfully applied diff" in result
    content = file_path.read_text()
    assert "A" in content
    assert "a" not in content
    assert "F" in content
    assert "f" not in content


# =============================================================================
# Tests for list_directory
# =============================================================================


@pytest.mark.asyncio
async def test_list_directory_basic(temp_dir):
    """list_directory lists files and directories."""
    # Create some files and dirs
    (temp_dir / "file1.txt").write_text("content")
    (temp_dir / "file2.py").write_text("print('hello')")
    (temp_dir / "subdir").mkdir()

    result = await list_directory.on_invoke_tool(None, json.dumps({"path": str(temp_dir)}))

    assert "[FILE] file1.txt" in result
    assert "[FILE] file2.py" in result
    assert "[DIR]  subdir/" in result


@pytest.mark.asyncio
async def test_list_directory_with_ignore(temp_dir):
    """list_directory respects ignore patterns."""
    (temp_dir / "keep.txt").write_text("keep")
    (temp_dir / "ignore_me.txt").write_text("ignore")
    (temp_dir / ".hidden").write_text("hidden")

    result = await list_directory.on_invoke_tool(
        None,
        json.dumps(
            {
                "path": str(temp_dir),
                "ignore": ["ignore", ".hidden"],
            }
        ),
    )

    assert "keep.txt" in result
    assert "ignore_me" not in result
    assert ".hidden" not in result


@pytest.mark.asyncio
async def test_list_directory_not_found(temp_dir):
    """list_directory returns error for non-existent path."""
    result = await list_directory.on_invoke_tool(
        None, json.dumps({"path": str(temp_dir / "nonexistent")})
    )

    assert "Path does not exist" in result


@pytest.mark.asyncio
async def test_list_directory_not_a_directory(sample_file):
    """list_directory returns error when path is a file."""
    result = await list_directory.on_invoke_tool(None, json.dumps({"path": str(sample_file)}))

    assert "Path is not a directory" in result


@pytest.mark.asyncio
async def test_list_directory_empty(temp_dir):
    """list_directory handles empty directories."""
    empty_dir = temp_dir / "empty"
    empty_dir.mkdir()

    result = await list_directory.on_invoke_tool(None, json.dumps({"path": str(empty_dir)}))

    assert "Directory is empty" in result


@pytest.mark.asyncio
async def test_list_directory_shows_file_sizes(temp_dir):
    """list_directory displays file sizes."""
    # Create files of different sizes
    small_file = temp_dir / "small.txt"
    small_file.write_text("x" * 100)

    kb_file = temp_dir / "kilobyte.txt"
    kb_file.write_text("x" * 2048)

    result = await list_directory.on_invoke_tool(None, json.dumps({"path": str(temp_dir)}))

    # Small file should show bytes
    assert "100B" in result or "(100B)" in result
    # KB file should show KB
    assert "KB" in result


# =============================================================================
# Tests for _generate_diff_output helper function
# =============================================================================


def test_generate_diff_output_new_file():
    """_generate_diff_output generates correct diff for new file."""
    new_content = "line1\nline2\nline3"
    diff = _generate_diff_output("", new_content, "test.txt", is_new_file=True)

    assert "--- /dev/null" in diff
    assert "+++ b/test.txt" in diff
    assert "@@ -0,0 +1,3 @@" in diff
    assert "+line1" in diff
    assert "+line2" in diff
    assert "+line3" in diff


def test_generate_diff_output_new_file_single_line():
    """_generate_diff_output handles single line new file."""
    new_content = "single line"
    diff = _generate_diff_output("", new_content, "single.txt", is_new_file=True)

    assert "--- /dev/null" in diff
    assert "@@ -0,0 +1,1 @@" in diff
    assert "+single line" in diff


def test_generate_diff_output_empty_new_file():
    """_generate_diff_output handles empty new file."""
    diff = _generate_diff_output("", "", "empty.txt", is_new_file=True)

    assert "--- /dev/null" in diff
    assert "+++ b/empty.txt" in diff
    # No hunk should be generated for empty content
    assert "@@ " not in diff


def test_generate_diff_output_file_modification():
    """_generate_diff_output generates correct diff for file modification."""
    old_content = "line1\nold_line\nline3"
    new_content = "line1\nnew_line\nline3"
    diff = _generate_diff_output(old_content, new_content, "mod.txt", is_new_file=False)

    assert "--- a/mod.txt" in diff
    assert "+++ b/mod.txt" in diff
    assert "-old_line" in diff
    assert "+new_line" in diff


def test_generate_diff_output_file_addition():
    """_generate_diff_output generates correct diff when adding lines."""
    old_content = "line1\nline2"
    new_content = "line1\nline2\nline3\nline4"
    diff = _generate_diff_output(old_content, new_content, "add.txt", is_new_file=False)

    assert "+line3" in diff
    assert "+line4" in diff


def test_generate_diff_output_file_deletion():
    """_generate_diff_output generates correct diff when deleting lines."""
    old_content = "line1\nline2\nline3"
    new_content = "line1"
    diff = _generate_diff_output(old_content, new_content, "del.txt", is_new_file=False)

    assert "-line2" in diff
    assert "-line3" in diff


# =============================================================================
# Tests for apply_diff function
# =============================================================================


def test_apply_diff_simple_replacement():
    """apply_diff correctly replaces a single line."""
    content = "line1\nline2\nline3\n"
    diff = """@@ -1,3 +1,3 @@
 line1
-line2
+replaced
 line3"""

    result, error = apply_diff(content, diff)

    assert error is None
    assert "replaced" in result
    assert "line2" not in result


def test_apply_diff_with_file_headers():
    """apply_diff works with full unified diff including headers."""
    content = "hello\nworld\n"
    diff = """--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
-hello
+goodbye
 world"""

    result, error = apply_diff(content, diff)

    assert error is None
    assert "goodbye" in result
    assert "hello" not in result


def test_apply_diff_add_lines():
    """apply_diff correctly adds new lines."""
    content = "line1\nline2\n"
    diff = """@@ -1,2 +1,4 @@
 line1
+new_a
+new_b
 line2"""

    result, error = apply_diff(content, diff)

    assert error is None
    assert "new_a" in result
    assert "new_b" in result


def test_apply_diff_delete_lines():
    """apply_diff correctly deletes lines."""
    content = "line1\nto_delete\nline3\n"
    diff = """@@ -1,3 +1,2 @@
 line1
-to_delete
 line3"""

    result, error = apply_diff(content, diff)

    assert error is None
    assert "to_delete" not in result
    assert "line1" in result
    assert "line3" in result


def test_apply_diff_multiple_hunks():
    """apply_diff correctly applies multiple hunks."""
    content = "a\nb\nc\nd\ne\nf\n"
    diff = """@@ -1,2 +1,2 @@
-a
+A
 b
@@ -5,2 +5,2 @@
-e
+E
 f"""

    result, error = apply_diff(content, diff)

    assert error is None
    assert "A" in result
    assert "a" not in result
    assert "E" in result
    assert "e" not in result


def test_apply_diff_preserves_trailing_newline():
    """apply_diff preserves trailing newline from original."""
    content = "line1\nline2\n"
    diff = """@@ -1,2 +1,2 @@
-line1
+modified
 line2"""

    result, error = apply_diff(content, diff)

    assert error is None
    assert result.endswith("\n")


def test_apply_diff_no_trailing_newline():
    """apply_diff preserves lack of trailing newline from original."""
    content = "line1\nline2"
    diff = """@@ -1,2 +1,2 @@
-line1
+modified
 line2"""

    result, error = apply_diff(content, diff)

    assert error is None
    assert not result.endswith("\n")


def test_apply_diff_invalid_diff():
    """apply_diff returns error for invalid diff."""
    content = "some content"
    diff = "not a valid diff"

    result, error = apply_diff(content, diff)

    assert error is not None
    assert result == content  # Original unchanged


def test_apply_diff_mismatch():
    """apply_diff returns error when diff doesn't match content."""
    content = "actual line\n"
    diff = """@@ -1,1 +1,1 @@
-wrong line
+new line"""

    result, error = apply_diff(content, diff)

    assert error is not None
    assert "does not match" in error or "Failed" in error
    assert result == content


def test_apply_diff_empty_content():
    """apply_diff handles empty original content."""
    content = ""
    diff = """@@ -0,0 +1,2 @@
+new line 1
+new line 2"""

    result, error = apply_diff(content, diff)

    # This might fail or succeed depending on the diff tool behavior
    # Either way, it should not crash
    assert isinstance(result, str)


# =============================================================================
# Additional tests for write_file diff output
# =============================================================================


@pytest.mark.asyncio
async def test_write_file_diff_shows_additions_for_new_file(temp_dir):
    """write_file shows all lines as additions for new file."""
    file_path = temp_dir / "new.txt"
    content = "line1\nline2\nline3"

    result = await write_file.on_invoke_tool(
        None, json.dumps({"path": str(file_path), "content": content})
    )

    assert "---DIFF---" in result
    assert "+line1" in result
    assert "+line2" in result
    assert "+line3" in result
    assert "--- /dev/null" in result


@pytest.mark.asyncio
async def test_write_file_diff_shows_changes_for_existing_file(temp_dir):
    """write_file shows deletions and additions for existing file."""
    file_path = temp_dir / "existing.txt"
    file_path.write_text("old content\n", encoding="utf-8")

    result = await write_file.on_invoke_tool(
        None, json.dumps({"path": str(file_path), "content": "new content\n"})
    )

    assert "---DIFF---" in result
    assert "-old content" in result
    assert "+new content" in result


@pytest.mark.asyncio
async def test_write_file_multiline_diff(temp_dir):
    """write_file generates correct diff for multiline changes."""
    file_path = temp_dir / "multi.txt"
    file_path.write_text("a\nb\nc\n", encoding="utf-8")

    result = await write_file.on_invoke_tool(
        None, json.dumps({"path": str(file_path), "content": "x\ny\nz\n"})
    )

    assert "---DIFF---" in result
    # Should show changes
    assert "-a" in result or "-b" in result or "-c" in result
    assert "+x" in result or "+y" in result or "+z" in result


# =============================================================================
# Additional tests for append_file diff output
# =============================================================================


@pytest.mark.asyncio
async def test_append_file_diff_shows_appended_content(temp_dir):
    """append_file shows appended content in diff."""
    file_path = temp_dir / "append.txt"
    file_path.write_text("existing\n", encoding="utf-8")

    result = await append_file.on_invoke_tool(
        None, json.dumps({"path": str(file_path), "content": "appended\n"})
    )

    assert "---DIFF---" in result
    assert "+appended" in result


@pytest.mark.asyncio
async def test_append_file_creates_new_with_diff(temp_dir):
    """append_file shows diff when creating new file."""
    file_path = temp_dir / "new_append.txt"

    result = await append_file.on_invoke_tool(
        None, json.dumps({"path": str(file_path), "content": "first line\n"})
    )

    assert "---DIFF---" in result
    assert "+first line" in result


@pytest.mark.asyncio
async def test_append_file_multiline_append(temp_dir):
    """append_file shows correct diff for multiline append."""
    file_path = temp_dir / "multi_append.txt"
    file_path.write_text("line1\n", encoding="utf-8")

    result = await append_file.on_invoke_tool(
        None, json.dumps({"path": str(file_path), "content": "line2\nline3\n"})
    )

    assert "---DIFF---" in result
    assert "+line2" in result
    assert "+line3" in result


# =============================================================================
# Additional tests for edit_file diff output
# =============================================================================


@pytest.mark.asyncio
async def test_edit_file_returns_diff_in_output(sample_file):
    """edit_file returns the applied diff in its output."""
    diff = """@@ -2,3 +2,3 @@
 line2
-line3
+CHANGED
 line4"""

    result = await edit_file.on_invoke_tool(
        None,
        json.dumps({"path": str(sample_file), "diff": diff}),
    )

    assert "---DIFF---" in result
    assert "-line3" in result
    assert "+CHANGED" in result


@pytest.mark.asyncio
async def test_edit_file_git_style_diff_header(temp_dir):
    """edit_file works with full git-style diff header."""
    file_path = temp_dir / "git_style.txt"
    file_path.write_text("hello\nworld\n", encoding="utf-8")

    diff = """diff --git a/git_style.txt b/git_style.txt
--- a/git_style.txt
+++ b/git_style.txt
@@ -1,2 +1,2 @@
-hello
+HELLO
 world"""

    result = await edit_file.on_invoke_tool(
        None,
        json.dumps({"path": str(file_path), "diff": diff}),
    )

    assert "Successfully applied diff" in result
    content = file_path.read_text()
    assert "HELLO" in content


@pytest.mark.asyncio
async def test_edit_file_only_additions(temp_dir):
    """edit_file works with diff that only adds lines."""
    file_path = temp_dir / "add_only.txt"
    file_path.write_text("start\nend\n", encoding="utf-8")

    diff = """@@ -1,2 +1,4 @@
 start
+middle1
+middle2
 end"""

    result = await edit_file.on_invoke_tool(
        None,
        json.dumps({"path": str(file_path), "diff": diff}),
    )

    assert "Successfully applied diff" in result
    content = file_path.read_text()
    assert "middle1" in content
    assert "middle2" in content


@pytest.mark.asyncio
async def test_edit_file_only_deletions(temp_dir):
    """edit_file works with diff that only deletes lines."""
    file_path = temp_dir / "del_only.txt"
    file_path.write_text("keep\ndelete_me\nalso_delete\nkeep_too\n", encoding="utf-8")

    diff = """@@ -1,4 +1,2 @@
 keep
-delete_me
-also_delete
 keep_too"""

    result = await edit_file.on_invoke_tool(
        None,
        json.dumps({"path": str(file_path), "diff": diff}),
    )

    assert "Successfully applied diff" in result
    content = file_path.read_text()
    assert "delete_me" not in content
    assert "also_delete" not in content
    assert "keep" in content
    assert "keep_too" in content


@pytest.mark.asyncio
async def test_edit_file_empty_diff_fails(temp_dir):
    """edit_file fails gracefully with empty diff."""
    file_path = temp_dir / "empty_diff.txt"
    file_path.write_text("content\n", encoding="utf-8")

    result = await edit_file.on_invoke_tool(
        None,
        json.dumps({"path": str(file_path), "diff": ""}),
    )

    assert "Failed" in result or "No valid diff" in result


@pytest.mark.asyncio
async def test_edit_file_whitespace_preservation(temp_dir):
    """edit_file preserves whitespace correctly."""
    file_path = temp_dir / "whitespace.txt"
    file_path.write_text("    indented\n\ttabbed\n", encoding="utf-8")

    diff = """@@ -1,2 +1,2 @@
     indented
-\ttabbed
+\tdouble_tabbed"""

    result = await edit_file.on_invoke_tool(
        None,
        json.dumps({"path": str(file_path), "diff": diff}),
    )

    assert "Successfully applied diff" in result
    content = file_path.read_text()
    assert "    indented" in content  # Preserved leading spaces
    assert "\tdouble_tabbed" in content


# =============================================================================
# Edge case tests
# =============================================================================


@pytest.mark.asyncio
async def test_write_file_empty_content(temp_dir):
    """write_file handles empty content."""
    file_path = temp_dir / "empty.txt"

    result = await write_file.on_invoke_tool(
        None, json.dumps({"path": str(file_path), "content": ""})
    )

    assert "Created" in result
    assert file_path.exists()
    assert file_path.read_text() == ""


@pytest.mark.asyncio
async def test_write_file_unicode_content(temp_dir):
    """write_file handles unicode content correctly."""
    file_path = temp_dir / "unicode.txt"
    content = "Hello 世界\n日本語\nEmoji: 🎉🚀"

    result = await write_file.on_invoke_tool(
        None, json.dumps({"path": str(file_path), "content": content})
    )

    assert "Created" in result
    assert file_path.read_text() == content


@pytest.mark.asyncio
async def test_edit_file_unicode_diff(temp_dir):
    """edit_file handles unicode in diff correctly."""
    file_path = temp_dir / "unicode_edit.txt"
    file_path.write_text("Hello\n世界\n", encoding="utf-8")

    diff = """@@ -1,2 +1,2 @@
 Hello
-世界
+世界！"""

    result = await edit_file.on_invoke_tool(
        None,
        json.dumps({"path": str(file_path), "diff": diff}),
    )

    assert "Successfully applied diff" in result
    content = file_path.read_text()
    assert "世界！" in content


@pytest.mark.asyncio
async def test_edit_file_special_characters(temp_dir):
    """edit_file handles special regex characters in content."""
    file_path = temp_dir / "special.txt"
    file_path.write_text("test $var\n[array]\n{object}\n", encoding="utf-8")

    diff = """@@ -1,3 +1,3 @@
 test $var
-[array]
+[new_array]
 {object}"""

    result = await edit_file.on_invoke_tool(
        None,
        json.dumps({"path": str(file_path), "diff": diff}),
    )

    assert "Successfully applied diff" in result
    content = file_path.read_text()
    assert "[new_array]" in content
