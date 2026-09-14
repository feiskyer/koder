"""Jupyter notebook cell editing tool."""

from __future__ import annotations

import json
import uuid
from typing import Optional

from koder_agent.harness.execution_context import execution_path

from .compat import function_tool
from .file import replace_read_file_contents, validate_read_file_for_edit


@function_tool
def notebook_edit(
    notebook_path: str,
    cell_index: int,
    operation: str,
    new_source: Optional[str] = None,
    cell_type: Optional[str] = None,
) -> str:
    """Edit cells in a Jupyter notebook (.ipynb file).

    Operations:
    - replace: Replace the source of an existing cell
    - insert: Insert a new cell at the given index
    - delete: Delete the cell at the given index

    Args:
        notebook_path: Path to the .ipynb file.
        cell_index: 0-based index of the cell to operate on.
        operation: One of 'replace', 'insert', 'delete'.
        new_source: New cell source (required for replace/insert).
        cell_type: Cell type for insert: 'code' or 'markdown'. Defaults to 'code'.
    """
    path = execution_path(notebook_path).resolve()
    if not path.exists():
        return f"Error: notebook not found: {notebook_path}"
    if not path.suffix == ".ipynb":
        return f"Error: not a notebook file: {notebook_path}"

    validation_error = validate_read_file_for_edit(notebook_path)
    if validation_error is not None:
        return validation_error

    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return f"Error: invalid notebook JSON: {e}"

    cells = nb.get("cells", [])

    success_message: str
    if operation == "replace":
        if cell_index < 0 or cell_index >= len(cells):
            return f"Error: invalid cell index {cell_index} (notebook has {len(cells)} cells)"
        if new_source is None:
            return "Error: new_source required for replace operation"
        cells[cell_index]["source"] = new_source
        success_message = f"Cell {cell_index} replaced successfully."

    elif operation == "insert":
        if cell_index < 0 or cell_index > len(cells):
            return f"Error: invalid cell index {cell_index} for insert (notebook has {len(cells)} cells)"
        if new_source is None:
            return "Error: new_source required for insert operation"
        ct = cell_type or "code"
        new_cell = {
            "cell_type": ct,
            "id": str(uuid.uuid4())[:8],
            "source": new_source,
            "metadata": {},
        }
        if ct == "code":
            new_cell["outputs"] = []
            new_cell["execution_count"] = None
        cells.insert(cell_index, new_cell)
        success_message = f"Cell inserted at index {cell_index} (type: {ct})."

    elif operation == "delete":
        if cell_index < 0 or cell_index >= len(cells):
            return f"Error: invalid cell index {cell_index} (notebook has {len(cells)} cells)"
        deleted = cells.pop(cell_index)
        success_message = (
            f"Cell {cell_index} deleted (was {deleted.get('cell_type', 'unknown')} cell)."
        )

    else:
        return f"Error: unknown operation '{operation}'. Use 'replace', 'insert', or 'delete'."

    write_error = replace_read_file_contents(
        notebook_path, json.dumps(nb, indent=1, ensure_ascii=False)
    )
    if write_error is not None:
        return write_error
    return success_message
