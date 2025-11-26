"""File operation tools."""

from pathlib import Path
from typing import List, Optional

import tiktoken
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
    old_str: str
    new_str: str


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
        p.write_text(content, "utf-8")
        return f"Wrote {len(content)} bytes to {path}"
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
        with p.open("a", encoding="utf-8") as f:
            f.write(content)
        return f"Appended {len(content)} bytes to {path}"
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error appending to file: {str(e)}"


@function_tool
def edit_file(path: str, old_str: str, new_str: str) -> str:
    """Perform exact string replacement in a file.

    The old_str must match exactly and appear uniquely in the file, otherwise
    the operation will fail. You must read the file first before editing.
    Preserve exact indentation from the source.
    """
    try:
        p = Path(path).resolve()
        if not p.exists():
            return f"File not found: {path}"

        content = p.read_text(encoding="utf-8")

        # Count occurrences
        count = content.count(old_str)

        if count == 0:
            return f"Text not found in file: {old_str[:100]}{'...' if len(old_str) > 100 else ''}"

        if count > 1:
            return f"Text appears {count} times in file - must be unique. Provide more context to make the match unique."

        # Perform single replacement
        new_content = content.replace(old_str, new_str, 1)
        p.write_text(new_content, encoding="utf-8")

        return f"Successfully edited {path}"
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
