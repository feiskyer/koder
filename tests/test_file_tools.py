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
        append_file,
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

    assert "Wrote" in result
    assert str(len(content)) in result
    assert file_path.exists()
    assert file_path.read_text() == content


@pytest.mark.asyncio
async def test_write_file_overwrites_existing(sample_file):
    """write_file overwrites existing file content."""
    new_content = "Completely new content"

    result = await write_file.on_invoke_tool(
        None, json.dumps({"path": str(sample_file), "content": new_content})
    )

    assert "Wrote" in result
    assert sample_file.read_text() == new_content


@pytest.mark.asyncio
async def test_write_file_creates_parent_directories(temp_dir):
    """write_file creates parent directories if they don't exist."""
    file_path = temp_dir / "nested" / "deep" / "file.txt"
    content = "Nested content"

    result = await write_file.on_invoke_tool(
        None, json.dumps({"path": str(file_path), "content": content})
    )

    assert "Wrote" in result
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
# Tests for edit_file
# =============================================================================


@pytest.mark.asyncio
async def test_edit_file_successful_replacement(sample_file):
    """edit_file successfully replaces unique text."""
    result = await edit_file.on_invoke_tool(
        None,
        json.dumps(
            {
                "path": str(sample_file),
                "old_str": "line3",
                "new_str": "REPLACED",
            }
        ),
    )

    assert "Successfully edited" in result
    content = sample_file.read_text()
    assert "REPLACED" in content
    assert "line3" not in content


@pytest.mark.asyncio
async def test_edit_file_file_not_found(temp_dir):
    """edit_file returns error for non-existent file."""
    result = await edit_file.on_invoke_tool(
        None,
        json.dumps(
            {
                "path": str(temp_dir / "nonexistent.txt"),
                "old_str": "old",
                "new_str": "new",
            }
        ),
    )

    assert "File not found" in result


@pytest.mark.asyncio
async def test_edit_file_text_not_found(sample_file):
    """edit_file returns error when old_str not found."""
    result = await edit_file.on_invoke_tool(
        None,
        json.dumps(
            {
                "path": str(sample_file),
                "old_str": "nonexistent_text",
                "new_str": "replacement",
            }
        ),
    )

    assert "Text not found in file" in result


@pytest.mark.asyncio
async def test_edit_file_non_unique_text(temp_dir):
    """edit_file returns error when old_str appears multiple times."""
    file_path = temp_dir / "duplicate.txt"
    file_path.write_text("word word word", encoding="utf-8")

    result = await edit_file.on_invoke_tool(
        None,
        json.dumps(
            {
                "path": str(file_path),
                "old_str": "word",
                "new_str": "replacement",
            }
        ),
    )

    assert "appears 3 times" in result
    assert "must be unique" in result
    # File should not be modified
    assert file_path.read_text() == "word word word"


@pytest.mark.asyncio
async def test_edit_file_preserves_other_content(sample_file):
    """edit_file only changes the matched text."""
    await edit_file.on_invoke_tool(
        None,
        json.dumps(
            {
                "path": str(sample_file),
                "old_str": "line3",
                "new_str": "MODIFIED",
            }
        ),
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
async def test_edit_file_multiline_replacement(temp_dir):
    """edit_file handles multiline old_str and new_str."""
    file_path = temp_dir / "multiline.txt"
    file_path.write_text("start\nmiddle\nend", encoding="utf-8")

    result = await edit_file.on_invoke_tool(
        None,
        json.dumps(
            {
                "path": str(file_path),
                "old_str": "start\nmiddle",
                "new_str": "replaced\nwith\nmultiple\nlines",
            }
        ),
    )

    assert "Successfully edited" in result
    content = file_path.read_text()
    assert "replaced\nwith\nmultiple\nlines\nend" == content


@pytest.mark.asyncio
async def test_edit_file_empty_new_str(sample_file):
    """edit_file can delete text by replacing with empty string."""
    result = await edit_file.on_invoke_tool(
        None,
        json.dumps(
            {
                "path": str(sample_file),
                "old_str": "line3\n",
                "new_str": "",
            }
        ),
    )

    assert "Successfully edited" in result
    content = sample_file.read_text()
    assert "line3" not in content


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
