"""File operation tools."""

from pathlib import Path
from typing import List, Optional

from agents import function_tool
from pydantic import BaseModel

from ..core.security import SecurityGuard


class FileWriteModel(BaseModel):
    path: str
    content: str


class FileReadModel(BaseModel):
    path: str


class LSModel(BaseModel):
    path: str
    ignore: Optional[List[str]] = None


@function_tool
def read_file(path: str) -> str:
    """Read a file from the filesystem."""
    try:
        p = Path(path).resolve()
        if not p.exists():
            return "File not found"

        # Check file size
        error = SecurityGuard.check_file_size(str(p))
        if error:
            return error

        return p.read_text("utf-8", errors="ignore")
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"Error reading file: {str(e)}"


@function_tool
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
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
