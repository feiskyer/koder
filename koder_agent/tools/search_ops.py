"""File search operation tools."""

import fnmatch
import re
from pathlib import Path
from typing import Optional

from agents import function_tool
from pydantic import BaseModel


class GlobModel(BaseModel):
    pattern: str
    path: Optional[str] = None


class GrepModel(BaseModel):
    pattern: str
    path: Optional[str] = None
    include: Optional[str] = None


@function_tool
def glob_search(pattern: str, path: Optional[str] = None) -> str:
    """Search for files matching a glob pattern."""
    try:
        base_path = Path(path) if path else Path.cwd()

        # Validate base path
        if not base_path.exists():
            return f"Path does not exist: {base_path}"

        if not base_path.is_dir():
            return f"Path is not a directory: {base_path}"

        # Find matches
        matches = []

        # Use rglob for recursive search if pattern contains **
        if "**" in pattern:
            # For patterns like **/*, remove the leading **/
            if pattern.startswith("**/"):
                actual_pattern = pattern[3:]  # Remove "**/"
            else:
                actual_pattern = pattern
            all_matches = base_path.rglob(actual_pattern)
        else:
            all_matches = base_path.glob(pattern)

        # Filter out virtual environments and common ignore patterns
        matches = []
        for match in all_matches:
            # Skip hidden directories and common ignore patterns
            parts = match.parts
            if any(part.startswith(".") and part not in {".github", ".vscode"} for part in parts):
                continue
            if any(
                part in {"__pycache__", "node_modules", ".venv", "venv", ".git"} for part in parts
            ):
                continue
            matches.append(match)

        # Sort by modification time (newest first)
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Limit results
        matches = matches[:100]

        if not matches:
            return "No matches found"

        # Format results
        results = []
        for match in matches:
            try:
                rel_path = match.relative_to(base_path)
                if match.is_dir():
                    results.append(f"[DIR]  {rel_path}/")
                else:
                    size = match.stat().st_size
                    results.append(f"[FILE] {rel_path} ({size} bytes)")
            except Exception:
                results.append(str(match))

        return "\n".join(results)

    except Exception as e:
        return f"Glob search error: {str(e)}"


@function_tool
def grep_search(pattern: str, path: Optional[str] = None, include: Optional[str] = None) -> str:
    """Search for pattern in file contents."""
    try:
        base_path = Path(path) if path else Path.cwd()

        # Validate base path
        if not base_path.exists():
            return f"Path does not exist: {base_path}"

        # Compile regex pattern
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {str(e)}"

        matches = []
        files_searched = 0

        # Search in files
        for file_path in base_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Apply include filter if specified
            if include and not fnmatch.fnmatch(file_path.name, include):
                continue

            # Skip binary files and large files
            if file_path.stat().st_size > 1024 * 1024:  # 1MB limit
                continue

            files_searched += 1
            if files_searched > 1000:  # Limit files searched
                matches.append("... (search limited to 1000 files)")
                break

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                if regex.search(content):
                    rel_path = file_path.relative_to(base_path)

                    # Find matching lines
                    lines = content.splitlines()
                    matching_lines = []
                    for i, line in enumerate(lines, 1):
                        if regex.search(line):
                            matching_lines.append(f"  {i}: {line.strip()}")
                            if len(matching_lines) >= 3:  # Show max 3 lines per file
                                matching_lines.append("  ...")
                                break

                    matches.append(f"\n{rel_path}:\n" + "\n".join(matching_lines))

                    if len(matches) >= 50:  # Limit total results
                        matches.append("\n... (results limited to 50 files)")
                        break

            except (UnicodeDecodeError, PermissionError):
                # Skip files that can't be read
                continue

        if not matches:
            return f"No matches found (searched {files_searched} files)"

        return f"Pattern '{pattern}' found in {len(matches)} files:\n" + "\n".join(matches)

    except Exception as e:
        return f"Grep search error: {str(e)}"
