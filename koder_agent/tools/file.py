"""File operation tools."""

from pathlib import Path
from typing import List, Optional, Tuple

import tiktoken
import whatthepatch
from agents import function_tool
from pydantic import BaseModel

from ..core.security import SecurityGuard


class FileWriteModel(BaseModel):
    path: str
    content: str


class FileReadModel(BaseModel):
    path: str
    offset: Optional[int] = None
    limit: Optional[int] = None


class FileEditModel(BaseModel):
    path: str
    diff: str


def apply_diff(content: str, diff_text: str) -> Tuple[str, Optional[str]]:
    """Apply a unified diff to file content using whatthepatch library.

    Args:
        content: The original file content.
        diff_text: The unified diff to apply.

    Returns:
        Tuple of (new_content, error_message).
        If successful, error_message is None.
    """
    try:
        # Parse the diff
        diffs = list(whatthepatch.parse_patch(diff_text))

        if not diffs:
            return content, "No valid diff found in input"

        # Split content into lines
        original_lines = content.splitlines(keepends=False)

        # Apply each diff (usually just one for a single file)
        result_lines: Optional[List[str]] = original_lines
        for diff in diffs:
            if diff.changes is None:
                continue

            # Apply the diff using pure Python implementation (use_patch=False)
            result_lines = whatthepatch.apply_diff(diff, result_lines, use_patch=False)

            if result_lines is None:
                return content, "Failed to apply diff: patch does not match file content"

        if result_lines is None:
            return content, "No changes were applied"

        # Reconstruct the content, preserving original line ending style
        result = "\n".join(result_lines)

        # Preserve trailing newline if original had one
        if content.endswith("\n"):
            result += "\n"

        return result, None

    except Exception as e:
        return content, f"Error applying diff: {str(e)}"


class LSModel(BaseModel):
    path: str
    ignore: Optional[List[str]] = None


def truncate_text_by_tokens(text: str, max_tokens: int = 32000) -> str:
    """Truncate text by token count if it exceeds the limit.

    When text exceeds the specified token limit, performs intelligent truncation
    by keeping the front and back parts while truncating the middle.

    Args:
        text: Text to be truncated
        max_tokens: Maximum token limit

    Returns:
        str: Truncated text if it exceeds the limit, otherwise the original text.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoding.encode(text))

    # Return original text if under limit
    if token_count <= max_tokens:
        return text

    # Calculate token/character ratio for approximation
    char_count = len(text)
    ratio = token_count / char_count

    # Keep head and tail mode: allocate half space for each (with 5% safety margin)
    chars_per_half = int((max_tokens / 2) / ratio * 0.95)

    # Truncate front part: find nearest newline
    head_part = text[:chars_per_half]
    last_newline_head = head_part.rfind("\n")
    if last_newline_head > 0:
        head_part = head_part[:last_newline_head]

    # Truncate back part: find nearest newline
    tail_part = text[-chars_per_half:]
    first_newline_tail = tail_part.find("\n")
    if first_newline_tail > 0:
        tail_part = tail_part[first_newline_tail + 1 :]

    # Combine result
    truncation_note = (
        f"\n\n... [Content truncated: {token_count} tokens -> ~{max_tokens} tokens limit] ...\n\n"
    )
    return head_part + truncation_note + tail_part


@function_tool
def read_file(path: str, offset: Optional[int] = None, limit: Optional[int] = None) -> str:
    """Read file contents from the filesystem.

    Output always includes line numbers in format 'LINE_NUMBER|LINE_CONTENT' (1-indexed).
    Supports reading partial content by specifying line offset and limit for large files.
    You can call this tool multiple times in parallel to read different files simultaneously.
    """
    try:
        p = Path(path).resolve()
        if not p.exists():
            return "File not found"

        # Check file size
        error = SecurityGuard.check_file_size(str(p))
        if error:
            return error

        # Read file content with line numbers
        with open(p, encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        # Apply offset and limit
        start = (offset - 1) if offset else 0
        end = (start + limit) if limit else len(lines)
        if start < 0:
            start = 0
        if end > len(lines):
            end = len(lines)

        selected_lines = lines[start:end]

        # Format with line numbers (1-indexed)
        numbered_lines = []
        for i, line in enumerate(selected_lines, start=start + 1):
            # Remove trailing newline for formatting
            line_content = line.rstrip("\n")
            numbered_lines.append(f"{i:6d}|{line_content}")

        content = "\n".join(numbered_lines)

        # Apply token truncation if needed
        content = truncate_text_by_tokens(content)

        return content
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error reading file: {str(e)}"


def _generate_diff_output(
    old_content: str, new_content: str, file_path: str, is_new_file: bool = False
) -> str:
    """Generate unified diff output for display.

    Args:
        old_content: Original file content (empty string for new files).
        new_content: New file content.
        file_path: Path to the file for the diff header.
        is_new_file: Whether this is a new file creation.

    Returns:
        Unified diff formatted string.
    """
    old_lines = old_content.splitlines(keepends=False) if old_content else []
    new_lines = new_content.splitlines(keepends=False) if new_content else []

    diff_lines = []

    # Add file header
    if is_new_file:
        diff_lines.append("--- /dev/null")
        diff_lines.append(f"+++ b/{file_path}")
    else:
        diff_lines.append(f"--- a/{file_path}")
        diff_lines.append(f"+++ b/{file_path}")

    # For simple cases, generate a basic unified diff
    if is_new_file or not old_lines:
        # New file: all lines are additions
        if new_lines:
            diff_lines.append(f"@@ -0,0 +1,{len(new_lines)} @@")
            for line in new_lines:
                diff_lines.append(f"+{line}")
    else:
        # File modification: show full diff using difflib
        import difflib

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )
        # Skip the first two lines (we already added headers)
        diff_list = list(diff)
        if len(diff_list) > 2:
            diff_lines = diff_list

    return "\n".join(diff_lines)


@function_tool
def write_file(path: str, content: str) -> str:
    """Write content to a file.

    Will overwrite existing files completely. For existing files, you should read the file
    first using read_file. Prefer editing existing files over creating new ones unless
    explicitly needed.
    """
    try:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists and get old content for diff
        is_new_file = not p.exists()
        old_content = ""
        if not is_new_file:
            try:
                old_content = p.read_text(encoding="utf-8")
            except Exception:
                old_content = ""

        # Write the new content
        p.write_text(content, "utf-8")

        # Generate diff for display
        filename = p.name
        diff_output = _generate_diff_output(old_content, content, filename, is_new_file)

        if is_new_file:
            return f"Created {path} ({len(content)} bytes)\n---DIFF---\n{diff_output}"
        else:
            return f"Updated {path} ({len(content)} bytes)\n---DIFF---\n{diff_output}"

    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error writing file: {str(e)}"


@function_tool
def append_file(path: str, content: str) -> str:
    """Append content to a file."""
    try:
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        # Get old content for diff (if file exists)
        is_new_file = not p.exists()
        old_content = ""
        if not is_new_file:
            try:
                old_content = p.read_text(encoding="utf-8")
            except Exception:
                old_content = ""

        # Append the content
        with p.open("a", encoding="utf-8") as f:
            f.write(content)

        # Generate diff for display (showing appended content)
        new_content = old_content + content
        filename = p.name
        diff_output = _generate_diff_output(old_content, new_content, filename, is_new_file)

        return f"Appended {len(content)} bytes to {path}\n---DIFF---\n{diff_output}"

    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error appending to file: {str(e)}"


@function_tool
def edit_file(path: str, diff: str) -> str:
    """Apply a unified diff patch to a file.

    The diff parameter should contain a unified diff format patch. This is useful
    for making targeted changes to specific lines in a file. The diff format uses:
    - Lines starting with '-' are removed
    - Lines starting with '+' are added
    - Lines starting with ' ' (space) are context (unchanged)
    - @@ -start,count +start,count @@ marks each hunk

    Example diff:
        @@ -1,3 +1,3 @@
         line1
        -old line2
        +new line2
         line3

    You must read the file first before editing.

    Args:
        path: Path to the file to edit.
        diff: Unified diff to apply.
    """
    try:
        p = Path(path).resolve()
        if not p.exists():
            return f"File not found: {path}"

        content = p.read_text(encoding="utf-8")

        new_content, error = apply_diff(content, diff)
        if error:
            return f"Failed to apply diff: {error}"

        p.write_text(new_content, encoding="utf-8")

        # Return success message with diff for display
        # Format: success message followed by the diff content
        return f"Successfully applied diff to {path}\n---DIFF---\n{diff}"

    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error editing file: {str(e)}"


@function_tool
def list_directory(path: str, ignore: Optional[List[str]] = None) -> str:
    """List contents of a directory."""
    try:
        p = Path(path).resolve()
        if not p.exists():
            return "Path does not exist"
        if not p.is_dir():
            return "Path is not a directory"

        ignore = ignore or []
        items = []

        for item in sorted(p.iterdir()):
            # Skip ignored patterns
            if any(pattern in item.name for pattern in ignore):
                continue

            if item.is_dir():
                items.append(f"[DIR]  {item.name}/")
            else:
                size = item.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f}MB"
                items.append(f"[FILE] {item.name} ({size_str})")

        return "\n".join(items) if items else "Directory is empty"
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error listing directory: {str(e)}"
