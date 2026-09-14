"""File search operation tools."""

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from koder_agent.harness.execution_context import execution_path, get_execution_cwd

from .compat import function_tool


class GlobModel(BaseModel):
    pattern: str
    path: Optional[str] = None


class GrepModel(BaseModel):
    pattern: str
    path: Optional[str] = None
    include: Optional[str] = None


@function_tool
def glob_search(pattern: str, path: Optional[str] = None) -> str:
    """Find files by name or path using a glob pattern.

    Matches file NAMES and paths only; it never looks inside files (use
    grep_search for contents). Prefer this over `find` or `ls` via
    run_shell: it is faster and skips noise directories automatically.

    When to use:
      - Locate files by name, extension, or path (e.g. "**/*test*.py").
      - Discover a codebase's layout before diving in.
      - Find the most recently modified files matching a pattern.

    Routing: file names/paths -> glob_search; file contents -> grep_search.
    grep_search accepts its own glob filter, so "search inside these file
    types" usually needs a single grep_search call, not both. For broad,
    open-ended exploration likely to take many searches, prefer
    task_delegate with an exploration prompt.

    Output: paths relative to the base directory, sorted by modification
    time (newest first), capped at 100 entries. Directories appear as
    [DIR] and files as [FILE] with size. Hidden directories (except
    .github and .vscode) plus __pycache__, node_modules, .venv, venv, and
    .git are skipped. Returns "No matches found" (not an error) when
    nothing matches.

    Tips: `*` does not cross directory separators, `**` does. Omit path
    to search the working directory; do not pass "" or a placeholder.

    Args:
        pattern: Glob pattern matched against file names, e.g. "**/*test*" or "src/**/*.py"
        path: Base directory to search in (defaults to cwd); a plain directory, not a pattern

    Returns:
        Matching files sorted by modification time (newest first)
    """
    try:
        base_path = execution_path(path) if path else get_execution_cwd()

        # Validate base path
        if not base_path.exists():
            return f"Path does not exist: {base_path}"

        if not base_path.is_dir():
            return f"Path is not a directory: {base_path}"

        # Use rglob for recursive search if pattern contains **
        if "**" in pattern:
            # For patterns like **/*, remove the leading **/
            actual_pattern = pattern[3:] if pattern.startswith("**/") else pattern
            all_matches = base_path.rglob(actual_pattern)
        else:
            all_matches = base_path.glob(pattern)

        # Filter out virtual environments and common ignore patterns
        matches = []
        for match in all_matches:
            # Skip hidden directories and common ignore patterns
            parts = match.relative_to(base_path).parts
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
def grep_search(
    pattern: str,
    path: Optional[str] = None,
    glob: Optional[str] = None,
    include: Optional[str] = None,
    output_mode: Optional[str] = None,
    context: Optional[int] = None,
    type: Optional[str] = None,
    head_limit: Optional[int] = None,
    offset: Optional[int] = None,
    multiline: Optional[bool] = None,
    case_insensitive: Optional[bool] = None,
    context_after: Optional[int] = None,
    context_before: Optional[int] = None,
    line_numbers: Optional[bool] = None,
) -> str:
    """Search file CONTENTS for a regex pattern using ripgrep.

    Use to find where a function, class, string, or pattern is defined or
    used. Always prefer this over `grep`/`rg` via run_shell: it is
    purpose-built, respects .gitignore, and returns clean relative paths.
    (To find files by NAME, use glob_search instead.)

    Output modes: "files_with_matches" (default) lists matching files
    sorted by modification time, newest first; "content" returns the
    matching lines (supports line numbers and context flags); "count"
    returns per-file match counts plus a total. Results are capped at
    head_limit (default 250); use offset to paginate. Searches time out
    after 30 seconds and lines are clipped at 500 columns.

    Tips: pattern is a regex, not a literal string - escape ., (, [, etc.
    Filter early with glob or type instead of searching everything; that
    replaces a separate glob_search + grep_search chain. Patterns match
    per line unless multiline=True. When context is set it overrides
    context_after/context_before.

    Args:
        pattern: Regex pattern to search for
        path: Base directory to search in (defaults to cwd)
        glob: File pattern filter (e.g., "*.py")
        include: Backward compat alias for glob
        output_mode: "files_with_matches" (default), "content", or "count"
        context: Number of context lines (-C flag)
        type: File type filter (e.g., "py", "js")
        head_limit: Maximum results to return (default 250)
        offset: Skip first N results (for pagination)
        multiline: Enable multiline mode
        case_insensitive: Case-insensitive search (-i flag)
        context_after: Lines of context after match (-A flag)
        context_before: Lines of context before match (-B flag)
        line_numbers: Show line numbers in content mode (default True)

    Returns:
        Formatted search results
    """
    # Strict JSON schema marks every parameter as required, so providers send
    # explicit nulls for omitted arguments; normalize them to real defaults here.
    output_mode = output_mode or "files_with_matches"
    head_limit = 250 if head_limit is None else head_limit
    offset = 0 if offset is None else offset
    multiline = bool(multiline)
    case_insensitive = bool(case_insensitive)
    line_numbers = True if line_numbers is None else line_numbers
    try:
        # Find ripgrep
        rg_path = shutil.which("rg")
        if not rg_path:
            return (
                "Error: ripgrep (rg) is not installed.\n"
                "Please install ripgrep: https://github.com/BurntSushi/ripgrep#installation\n"
                "  macOS: brew install ripgrep\n"
                "  Ubuntu/Debian: apt install ripgrep\n"
                "  Windows: choco install ripgrep"
            )

        base_path = execution_path(path) if path else get_execution_cwd()

        # Validate base path
        if not base_path.exists():
            return f"Path does not exist: {base_path}"

        # Build ripgrep command
        cmd = [rg_path]

        # Basic flags
        cmd.extend(["--hidden", "--max-columns", "500"])

        # Exclude VCS directories
        for vcs_dir in [".git", ".svn", ".hg", ".bzr", ".jj", ".sl"]:
            cmd.extend(["--glob", f"!{vcs_dir}"])

        # Multiline mode
        if multiline:
            cmd.extend(["-U", "--multiline-dotall"])

        # Case sensitivity
        if case_insensitive:
            cmd.append("-i")

        # Output mode
        if output_mode == "files_with_matches":
            cmd.append("-l")
        elif output_mode == "count":
            cmd.append("-c")
        elif output_mode == "content":
            # Line numbers enabled by default in content mode unless explicitly disabled
            if line_numbers:
                cmd.append("-n")

        # Context lines - -C takes precedence over -A/-B
        if context is not None:
            cmd.extend(["-C", str(context)])
        else:
            if context_after is not None:
                cmd.extend(["-A", str(context_after)])
            if context_before is not None:
                cmd.extend(["-B", str(context_before)])

        # Type filter
        if type:
            cmd.extend(["--type", type])

        # Glob filter (include is backward compat)
        glob_pattern = glob or include
        if glob_pattern:
            cmd.extend(["--glob", glob_pattern])

        # Pattern - use -e flag to prevent pattern injection when pattern starts with dash
        if pattern.startswith("-"):
            cmd.extend(["-e", pattern])
        else:
            cmd.append(pattern)

        # Search path
        cmd.append(str(base_path))

        # Run ripgrep
        result = subprocess.run(
            cmd, cwd=get_execution_cwd(), capture_output=True, text=True, timeout=30
        )

        # Handle exit codes: 0 = matches, 1 = no matches, 2+ = error
        if result.returncode >= 2:
            return f"Grep search error: {result.stderr.strip()}"

        if result.returncode == 1 or not result.stdout.strip():
            return "No matches found"

        output = result.stdout

        # Process output based on mode
        if output_mode == "files_with_matches":
            # Sort by mtime (newest first) and relativize paths
            file_lines = output.strip().split("\n")
            files = []
            for line in file_lines:
                if line:
                    file_path = Path(line)
                    try:
                        rel_path = file_path.relative_to(base_path)
                        mtime = file_path.stat().st_mtime
                        files.append((rel_path, mtime))
                    except (ValueError, OSError):
                        files.append((Path(line), 0))

            # Sort by mtime, newest first
            files.sort(key=lambda x: x[1], reverse=True)

            # Apply offset and head_limit
            files = files[offset : offset + head_limit]

            if not files:
                return "No matches found"

            return "\n".join(str(f[0]) for f in files)

        elif output_mode == "content":
            # Relativize paths in content output
            lines = output.split("\n")
            relativized_lines = []
            for line in lines:
                if not line:
                    continue
                # Try to extract and relativize file path
                if ":" in line:
                    parts = line.split(":", 1)
                    file_part = parts[0]
                    try:
                        file_path = Path(file_part)
                        if file_path.exists() and file_path.is_absolute():
                            rel_path = file_path.relative_to(base_path)
                            line = f"{rel_path}:{parts[1]}" if len(parts) > 1 else str(rel_path)
                    except (ValueError, OSError):
                        pass
                relativized_lines.append(line)

            # Apply offset and head_limit
            output_lines = relativized_lines[offset : offset + head_limit]
            return "\n".join(output_lines) if output_lines else "No matches found"

        elif output_mode == "count":
            # Parse count output (file:count format)
            lines = output.strip().split("\n")
            count_data = []

            for line in lines:
                if ":" in line:
                    file_part, count_part = line.rsplit(":", 1)
                    try:
                        count = int(count_part)
                        file_path = Path(file_part)
                        try:
                            rel_path = file_path.relative_to(base_path)
                            count_data.append((str(rel_path), count))
                        except ValueError:
                            count_data.append((file_part, count))
                    except ValueError:
                        continue

            # Apply offset and head_limit
            count_data = count_data[offset : offset + head_limit]

            if not count_data:
                return "No matches found"

            # Calculate total from sliced data only
            total = sum(count for _, count in count_data)

            result_lines = [f"{file}: {count}" for file, count in count_data]
            result_lines.append(f"\nTotal matches: {total}")

            return "\n".join(result_lines)

        return output.strip()

    except subprocess.TimeoutExpired:
        return "Grep search error: Search timed out after 30 seconds"
    except Exception as e:
        return f"Grep search error: {str(e)}"
