"""File operation tools."""

import errno
import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import tiktoken
import whatthepatch
from pydantic import BaseModel

from ..core.security import SecurityGuard
from ..harness.checkpoint import record_pre_edit
from ..harness.execution_context import execution_path
from .compat import function_tool
from .file_state import ReadFileState

logger = logging.getLogger(__name__)
_file_state = ReadFileState()

# ---------------------------------------------------------------------------
# Curly quote normalization for file editing
# ---------------------------------------------------------------------------
_LEFT_SINGLE_CURLY = "\u2018"  # '
_RIGHT_SINGLE_CURLY = "\u2019"  # '
_LEFT_DOUBLE_CURLY = "\u201c"  # "
_RIGHT_DOUBLE_CURLY = "\u201d"  # "


# O_NOFOLLOW makes open() refuse to traverse a symlink at the final path
# component, closing the check-then-write (TOCTOU) window where the resolved
# target could be swapped for a symlink pointing outside the workspace after the
# permission check passed. It is a no-op on platforms lacking the flag (Windows),
# which also lack the POSIX symlink semantics this guards against.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

_SYMLINK_WRITE_MSG = "Refusing to write through a symlink"


def _no_follow_target(path: str) -> Path:
    """Return the write target with the parent resolved but the leaf kept literal.

    Resolving only the parent lets ``O_NOFOLLOW`` guard the final component; a
    full ``Path.resolve()`` would silently redirect a symlinked leaf to its
    target — the bypass this closes. Never pass a fully ``resolve()``-d path here.
    """
    raw = execution_path(path)
    return raw.parent.resolve() / raw.name


def _write_bytes_no_follow(path: str, data: bytes, *, append: bool) -> None:
    """Write ``data`` to ``path``'s leaf without following a leaf symlink.

    The parent is resolved but the final component is kept literal, so ``open``
    sees the leaf as the caller named it. With ``O_NOFOLLOW`` a symlinked leaf
    raises ``OSError`` (``ELOOP``) instead of silently redirecting the write to
    the symlink's target — the case a full ``Path.resolve()`` would mask.
    """
    target = _no_follow_target(path)
    flags = os.O_WRONLY | os.O_CREAT | _O_NOFOLLOW
    flags |= os.O_APPEND if append else os.O_TRUNC
    fd = os.open(str(target), flags, 0o644)
    try:
        remaining = memoryview(data)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError(errno.EIO, "write made no progress")
            remaining = remaining[written:]
    finally:
        os.close(fd)


def _atomic_write_no_follow(path: str, data: bytes) -> None:
    """Atomically replace ``path``'s leaf, refusing to write through a symlink.

    Writes ``data`` to a temp file in the leaf's (resolved) parent directory then
    ``os.replace()``s it over the target, so a crash mid-write cannot leave a
    truncated/corrupt file. Before writing, the leaf is probed with
    ``O_NOFOLLOW`` so a symlinked leaf raises ``ELOOP`` (same no-follow semantics
    as ``_write_bytes_no_follow``) rather than having the link silently swapped
    for a regular file by ``os.replace``.
    """
    target = _no_follow_target(path)
    # Refuse a symlinked leaf: opening with O_NOFOLLOW raises ELOOP for a symlink.
    # This keeps a single consistent error surface with the other write paths and
    # prevents os.replace() from clobbering the link (rather than its target).
    if _O_NOFOLLOW:
        probe = os.open(str(target), os.O_WRONLY | os.O_CREAT | _O_NOFOLLOW, 0o644)
        os.close(probe)
    # Preserve the existing file's permission bits: mkstemp creates the temp file
    # 0600, so os.replace() would otherwise silently strip an executable/group
    # bit (e.g. a 0755 script becomes 0600 after an edit). A brand-new file keeps
    # the temp default (adjusted to the usual 0644 minus umask) so we don't leak
    # 0600 for freshly created files either.
    existing_mode: int | None = None
    try:
        existing_mode = stat.S_IMODE(os.lstat(str(target)).st_mode)
    except OSError:
        existing_mode = None
    parent = target.parent
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(parent))
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(data)
        if existing_mode is not None:
            # Restore the original file's mode onto the replacement.
            os.chmod(tmp_name, existing_mode)
        else:
            # New file: apply the process umask to the 0666 default so it lands
            # at the conventional 0644 (not the private 0600 mkstemp uses).
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp_name, 0o666 & ~umask)
        os.replace(tmp_name, str(target))
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _normalize_quotes(s: str) -> str:
    """Convert curly quotes to straight quotes."""
    return (
        s.replace(_LEFT_SINGLE_CURLY, "'")
        .replace(_RIGHT_SINGLE_CURLY, "'")
        .replace(_LEFT_DOUBLE_CURLY, '"')
        .replace(_RIGHT_DOUBLE_CURLY, '"')
    )


def _find_actual_string(file_content: str, search_string: str) -> str | None:
    """Find the actual string in file content, accounting for quote normalization."""
    if search_string in file_content:
        return search_string
    normalized_search = _normalize_quotes(search_string)
    normalized_file = _normalize_quotes(file_content)
    idx = normalized_file.find(normalized_search)
    if idx != -1:
        return file_content[idx : idx + len(search_string)]
    return None


# ---------------------------------------------------------------------------
# Line-ending preservation
#
# read_file exposes files to the model in universal-newline (LF) form, and the
# model supplies edits with LF. But writing that LF text straight back rewrites a
# CRLF file's every line ending to LF — a whole-file diff from a one-line intent,
# and broken CRLF-sensitive tooling. So tools operate on LF-normalized text
# internally but detect and RE-APPLY the file's dominant on-disk ending on write.
# ---------------------------------------------------------------------------


def _normalize_newlines(text: str) -> str:
    """Normalize all line endings (CRLF, lone CR) to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _detect_line_ending(raw: str) -> str:
    """Detect the dominant line ending of raw (untranslated) file text.

    Returns "\\r\\n", "\\n", or "\\r" -- whichever occurs most often. Ties break
    in favor of "\\n" first, then "\\r\\n", then "\\r". Text with no line endings
    defaults to "\\n".
    """
    crlf = raw.count("\r\n")
    lf = raw.count("\n") - crlf
    cr = raw.count("\r") - crlf
    best = max(lf, crlf, cr)
    if best == 0 or lf == best:
        return "\n"
    if crlf == best:
        return "\r\n"
    return "\r"


def _read_text_preserving_ending(p: Path) -> Tuple[str, str]:
    """Read a file without newline translation.

    Returns (LF-normalized text, dominant line ending). The normalized text is
    what tools operate on internally (matching what read_file exposed to the
    model); the ending is re-applied when writing back to disk.
    """
    with open(p, encoding="utf-8", newline="") as f:
        raw = f.read()
    return _normalize_newlines(raw), _detect_line_ending(raw)


def _apply_line_ending(text: str, ending: str) -> bytes:
    """Re-apply ``ending`` to LF-normalized ``text`` and return UTF-8 bytes."""
    normalized = _normalize_newlines(text)
    if ending != "\n":
        normalized = normalized.replace("\n", ending)
    return normalized.encode("utf-8")


def get_file_state() -> ReadFileState:
    """Get the active Session's read evidence, or the unscoped legacy tracker."""
    from ..harness.agents.runtime_context import (
        get_runtime_agent_session,
        has_agent_session_scope,
    )

    session = get_runtime_agent_session()
    if session is not None:
        file_state = getattr(session, "file_read_state", None)
        if not isinstance(file_state, ReadFileState):
            raise RuntimeError("Managed session does not support file-read state")
        return file_state
    if has_agent_session_scope():
        raise RuntimeError("File tools require an active managed session")
    return _file_state


def validate_read_file_for_edit(path: str) -> Optional[str]:
    """Return an edit guard error unless ``path`` has a fresh, complete read."""
    try:
        file_state = get_file_state()
    except RuntimeError as exc:
        return f"Error accessing file-read state: {exc}"
    p = execution_path(path).resolve()
    if not p.exists():
        return f"File not found: {path}"
    if not file_state.has_been_read(str(p)):
        return "File has not been read yet. Read it first before editing it."
    if file_state.is_partial_view(str(p)):
        return "File was only partially read. Read the full file before editing."
    if file_state.is_stale(str(p)):
        return (
            "File has been modified since it was last read. "
            "Read it again before attempting to edit it."
        )
    return None


def replace_read_file_contents(path: str, content: str) -> Optional[str]:
    """Atomically replace a fully-read existing file using edit-file invariants.

    Returns ``None`` on success, otherwise the same model-facing error used by
    normal file edits. Callers retain responsibility for computing ``content``.
    """
    validation_error = validate_read_file_for_edit(path)
    if validation_error is not None:
        return validation_error

    p = execution_path(path).resolve()
    try:
        file_state = get_file_state()
        _, line_ending = _read_text_preserving_ending(p)
        record_pre_edit(str(p))
        validation_error = validate_read_file_for_edit(path)
        if validation_error is not None:
            return validation_error
        _atomic_write_no_follow(path, _apply_line_ending(content, line_ending))
    except OSError as e:
        if e.errno == errno.ELOOP:
            return f"{_SYMLINK_WRITE_MSG}: {path}"
        return str(e)
    except Exception as e:
        return f"Error editing file: {str(e)}"

    file_state.record_read(str(p), content=_normalize_newlines(content))
    return None


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
    These line-number prefixes are display-only: never copy them into edit_file's
    old_string/new_string or into write_file content.
    Supports reading partial content by specifying line offset and limit for large files.
    You can call this tool multiple times in parallel to read different files simultaneously.

    Do NOT re-read a file right after editing it just to verify the change:
    edit_file and write_file return an error if the change failed, so a successful
    result means the file already matches what you wrote.

    Args:
        path: Path to the file to read (absolute or relative to cwd)
        offset: 1-indexed line number to start reading from (for large files)
        limit: Maximum number of lines to read (for large files)
    """
    try:
        file_state = get_file_state()
        p = execution_path(path).resolve()
        if not p.exists():
            return "File not found"

        # Check file size
        error = SecurityGuard.check_file_size(str(p))
        if error:
            return error

        # Whole-file re-read caching/dedup: if the caller requests the entire
        # file (no offset/limit) and we already have its full (non-partial)
        # content in context and it has not changed on disk, avoid re-emitting
        # the body to save context tokens. Partial/offset/limit reads are never
        # short-circuited.
        is_whole_file = offset is None and limit is None
        if is_whole_file and file_state.has_been_read(str(p)) and not file_state.is_stale(str(p)):
            prior_content = file_state.get_full_content(str(p))
            if prior_content is not None:
                prior_line_count = prior_content.count("\n") + (
                    0 if prior_content.endswith("\n") or prior_content == "" else 1
                )
                # Refresh mtime/size/hash so subsequent staleness checks stay accurate.
                file_state.record_read(str(p), content=prior_content, is_partial=False)
                return (
                    f"File unchanged since last read ({prior_line_count} lines); "
                    "prior content still in context"
                )

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

        if p.suffix.lower() == ".md":
            try:
                from koder_agent.harness.magic_docs import register_magic_doc

                register_magic_doc(p, "".join(lines))
            except Exception:
                logger.debug("Failed to register magic doc", exc_info=True)

        # Apply token truncation if needed. If the display content is actually
        # truncated, the model never saw the whole file even for a full read, so
        # the recorded read MUST be marked partial — otherwise edit_file's
        # is_partial_view guard would wrongly allow a diff/string edit against a
        # file the model only partially saw.
        display_content = truncate_text_by_tokens(content)
        was_truncated = display_content != content

        # Track the read for staleness detection
        is_partial = offset is not None or limit is not None or was_truncated
        full_content = "".join(lines) if not is_partial else None
        file_state.record_read(str(p), content=full_content, is_partial=is_partial)

        return display_content
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

    OVERWRITES the target file completely: any existing content is replaced with
    `content`. Use this for creating new files or full-file rewrites only.
    For changes to an existing file, prefer edit_file (targeted string replacement)
    over rewriting the whole file. Before overwriting an existing file you must
    read it with read_file first; the write is rejected otherwise. Do not include
    read_file's 'LINE_NUMBER|' prefixes in the content.

    Args:
        path: Path to the file to write (parent directories are created as needed)
        content: Full file content to write (replaces any existing content)
    """
    try:
        file_state = get_file_state()
        p = execution_path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists and get old content for diff
        is_new_file = not p.exists()

        # Enforce read-before-write for existing files
        if not is_new_file:
            if not file_state.has_been_read(str(p)):
                return "File has not been read yet. Read it first before writing to it."
            if file_state.is_stale(str(p)):
                return (
                    "File has been modified since it was last read. "
                    "Read it again before attempting to write it."
                )

        old_content = ""
        line_ending = "\n"
        if not is_new_file:
            try:
                old_content, line_ending = _read_text_preserving_ending(p)
            except Exception:
                old_content = ""

        # Snapshot pre-edit content for /rewind code restoration (no-op-safe).
        record_pre_edit(str(p))

        # Write atomically (symlink-safe), preserving the existing file's dominant
        # line ending; a brand-new file keeps the content's own endings.
        disk_bytes = (
            content.encode("utf-8") if is_new_file else _apply_line_ending(content, line_ending)
        )
        _atomic_write_no_follow(path, disk_bytes)
        file_state.record_read(str(p), content=_normalize_newlines(content))

        # Generate diff for display
        filename = p.name
        diff_output = _generate_diff_output(old_content, content, filename, is_new_file)

        if is_new_file:
            return f"Created {path} ({len(content)} bytes)\n---DIFF---\n{diff_output}"
        else:
            return f"Updated {path} ({len(content)} bytes)\n---DIFF---\n{diff_output}"

    except OSError as e:
        if e.errno == errno.ELOOP:
            return f"{_SYMLINK_WRITE_MSG}: {path}"
        return str(e)
    except Exception as e:
        return f"Error writing file: {str(e)}"


@function_tool
def append_file(path: str, content: str) -> str:
    """Append content to a file.

    Args:
        path: Path to the file to append to (created if it does not exist)
        content: Text to append at the end of the file
    """
    try:
        file_state = get_file_state()
        p = execution_path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        # Get old content for diff (if file exists)
        is_new_file = not p.exists()

        # Enforce read-before-write for existing files, matching write_file
        if not is_new_file:
            if not file_state.has_been_read(str(p)):
                return "File has not been read yet. Read it first before appending to it."
            if file_state.is_stale(str(p)):
                return (
                    "File has been modified since it was last read. "
                    "Read it again before attempting to append to it."
                )

        old_content = ""
        line_ending = "\n"
        if not is_new_file:
            try:
                old_content, line_ending = _read_text_preserving_ending(p)
            except Exception:
                old_content = ""

        # Snapshot pre-edit content for /rewind code restoration (no-op-safe).
        record_pre_edit(str(p))

        # Append the content (refusing to follow a leaf symlink), re-encoding the
        # appended text to the existing file's dominant ending so we never splice a
        # lone LF into an otherwise-CRLF file. A new file keeps the content's own.
        append_bytes = (
            content.encode("utf-8") if is_new_file else _apply_line_ending(content, line_ending)
        )
        _write_bytes_no_follow(path, append_bytes, append=True)

        # Generate diff for display (showing appended content)
        new_content = old_content + content
        # Record LF-normalized content so a subsequent append/edit doesn't
        # spuriously fail the staleness check: is_stale() re-reads the file with
        # universal newlines (LF), so the stored copy must be LF too (mirrors
        # write_file). Storing raw CR-bearing content would look "modified".
        file_state.record_read(str(p), content=_normalize_newlines(new_content))
        filename = p.name
        diff_output = _generate_diff_output(old_content, new_content, filename, is_new_file)

        return f"Appended {len(content)} bytes to {path}\n---DIFF---\n{diff_output}"

    except OSError as e:
        if e.errno == errno.ELOOP:
            return f"Refusing to append through a symlink: {path}"
        return str(e)
    except Exception as e:
        return f"Error appending to file: {str(e)}"


def edit_file_by_replacement(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """Edit a file using old_string/new_string replacement.

    String replacement edit mode -- find old_string in the file
    and replace it with new_string.  Supports curly quote normalization and
    replace_all for multiple occurrences.
    """
    try:
        file_state = get_file_state()
    except RuntimeError as exc:
        return f"Error editing file: {exc}"
    p = execution_path(path).resolve()

    # Reject no-op edits
    if old_string == new_string:
        return "No changes to make: old_string and new_string are the same."

    # Empty old_string = file creation. A file that exists on disk with ANY
    # bytes (including whitespace-only) counts as existing content, so we never
    # silently clobber it. This is a create, not an edit, so refusing to
    # overwrite is the correct answer regardless of read state (the earlier
    # ".strip()" check let whitespace-only files be silently clobbered).
    if old_string == "":
        if p.exists():
            try:
                existing = p.read_text(encoding="utf-8")
            except Exception:
                existing = ""
            if existing != "":
                return "Cannot create new file — file already exists with content."
        p.parent.mkdir(parents=True, exist_ok=True)
        # Snapshot pre-edit content for /rewind code restoration (no-op-safe).
        record_pre_edit(str(p))
        try:
            _atomic_write_no_follow(path, new_string.encode("utf-8"))
        except OSError as e:
            if e.errno == errno.ELOOP:
                return f"{_SYMLINK_WRITE_MSG}: {path}"
            return str(e)
        file_state.record_read(str(p), content=_normalize_newlines(new_string))
        return f"Created {path} ({len(new_string)} bytes)"

    # File must exist
    if not p.exists():
        return f"File not found: {path}"

    # Enforce read-before-edit, matching write_file and the diff-mode path
    if not file_state.has_been_read(str(p)):
        return "File has not been read yet. Read it first before editing it."
    if file_state.is_partial_view(str(p)):
        return "File was only partially read. Read the full file before editing."
    if file_state.is_stale(str(p)):
        return (
            "File has been modified since it was last read. "
            "Read it again before attempting to edit it."
        )

    # Read LF-normalized (so the model's LF old_string matches, even for a CRLF
    # file), remembering the dominant on-disk ending to re-apply on write.
    content, line_ending = _read_text_preserving_ending(p)

    # Find with quote normalization
    actual_old = _find_actual_string(content, old_string)
    if actual_old is None:
        return f"String not found in file.\nString: {old_string}"

    # Uniqueness check
    match_count = content.count(actual_old)
    if match_count > 1 and not replace_all:
        return (
            f"Found {match_count} matches of the string to replace, but "
            f"replace_all is false. To replace all occurrences, set replace_all "
            f"to true. To replace only one, provide more context to uniquely "
            f"identify the instance.\nString: {old_string}"
        )

    # Apply the edit
    if replace_all:
        new_content = content.replace(actual_old, new_string)
    else:
        # When deleting (new_string is empty), also strip the trailing newline
        if new_string == "" and actual_old + "\n" in content:
            new_content = content.replace(actual_old + "\n", "", 1)
        else:
            new_content = content.replace(actual_old, new_string, 1)

    if new_content == content:
        return "No changes were applied."

    # Snapshot pre-edit content for /rewind code restoration (no-op-safe).
    record_pre_edit(str(p))

    try:
        _atomic_write_no_follow(path, _apply_line_ending(new_content, line_ending))
    except OSError as e:
        if e.errno == errno.ELOOP:
            return f"{_SYMLINK_WRITE_MSG}: {path}"
        return str(e)
    # Store LF-normalized content so the staleness check (which re-reads with
    # universal newlines) stays consistent even if new_string carried literal \r.
    file_state.record_read(str(p), content=_normalize_newlines(new_content))

    diff_output = _generate_diff_output(content, new_content, p.name)
    return f"Successfully edited {path}\n---DIFF---\n{diff_output}"


@function_tool
def edit_file(
    path: str,
    old_string: Optional[str] = None,
    new_string: Optional[str] = None,
    replace_all: bool = False,
    diff: Optional[str] = None,
) -> str:
    """Edit a file using string replacement or unified diff.

    You must read the file with read_file earlier in the conversation before editing it.

    Two modes:
    1. String replacement (preferred): Provide old_string and new_string.
       - old_string must match the file contents EXACTLY, including indentation
         and whitespace (curly/straight quote differences are tolerated).
       - old_string must be unique in the file, or the edit fails. To fix a
         non-unique match, either expand old_string with surrounding context to
         make it unique, or set replace_all=true to change every occurrence.
       - NEVER include the 'LINE_NUMBER|' prefixes from read_file output in
         old_string or new_string; they are not part of the file.
    2. Unified diff: Provide a diff parameter with a unified diff patch. Use only
       when string replacement is impractical; string replacement is preferred.

    A successful result means the edit was applied; no need to re-read the file
    to verify. If the edit failed, the tool returns an error explaining why.

    Args:
        path: Path to the file to edit
        old_string: Exact text to find in the file (string replacement mode)
        new_string: Replacement text (string replacement mode)
        replace_all: Replace every occurrence of old_string instead of requiring uniqueness
        diff: Unified diff patch to apply (diff mode; mutually exclusive with old/new_string)
    """
    if old_string is not None and new_string is not None:
        return edit_file_by_replacement(path, old_string, new_string, replace_all)

    if diff is not None:
        # Existing diff-based path (keep current logic)
        try:
            file_state = get_file_state()
            p = execution_path(path).resolve()
            if not p.exists():
                return f"File not found: {path}"

            # Read-before-write enforcement
            if not file_state.has_been_read(str(p)):
                return "File has not been read yet. Read it first before editing."
            if file_state.is_partial_view(str(p)):
                return "File was only partially read. Read the full file before editing."
            if file_state.is_stale(str(p)):
                return (
                    "File has been modified since it was last read. "
                    "Read it again before attempting to edit it."
                )

            content, line_ending = _read_text_preserving_ending(p)
            new_content, error = apply_diff(content, diff)
            if error:
                return f"Failed to apply diff: {error}"
            # Snapshot pre-edit content for /rewind code restoration (no-op-safe).
            record_pre_edit(str(p))
            _atomic_write_no_follow(path, _apply_line_ending(new_content, line_ending))
            file_state.record_read(str(p), content=_normalize_newlines(new_content))
            return f"Successfully applied diff to {path}\n---DIFF---\n{diff}"
        except OSError as e:
            if e.errno == errno.ELOOP:
                return f"{_SYMLINK_WRITE_MSG}: {path}"
            return str(e)
        except Exception as e:
            return f"Error editing file: {str(e)}"

    return "Either (old_string + new_string) or diff must be provided."


@function_tool
def list_directory(path: str, ignore: Optional[List[str]] = None) -> str:
    """List contents of a directory.

    Args:
        path: Directory to list
        ignore: Glob patterns for entries to skip (e.g. ["*.pyc", "node_modules"])
    """
    try:
        p = execution_path(path).resolve()
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
