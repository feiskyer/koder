"""Local code-intelligence helpers for Koder tools."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .execution_context import execution_path, get_execution_cwd

MAX_INDEXED_FILE_SIZE = 1_000_000
MAX_WORKSPACE_FILES = 2_000

SOURCE_EXTENSIONS = {
    ".bash",
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".swift",
    ".ts",
    ".tsx",
    ".zsh",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".sl",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
}

SYMBOL_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "class",
        "class",
        re.compile(r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][\w$]*)"),
    ),
    (
        "interface",
        "interface",
        re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)"),
    ),
    (
        "enum",
        "enum",
        re.compile(r"^\s*(?:export\s+)?enum\s+([A-Za-z_$][\w$]*)"),
    ),
    (
        "type",
        "type",
        re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)"),
    ),
    (
        "function",
        "function",
        re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\("),
    ),
    (
        "function",
        "function",
        re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\("
        ),
    ),
    ("function", "function", re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_]\w*)\s*\(")),
    ("function", "function", re.compile(r"^\s*func\s+(?:\([^)]+\)\s*)?([A-Za-z_]\w*)\s*\(")),
    ("function", "function", re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(")),
    (
        "variable",
        "variable",
        re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)"),
    ),
)

OPERATION_ALIASES = {
    "documentsymbol": "document_symbols",
    "document_symbol": "document_symbols",
    "documentsymbols": "document_symbols",
    "document_symbols": "document_symbols",
    "workspacesymbol": "workspace_symbols",
    "workspace_symbol": "workspace_symbols",
    "workspacesymbols": "workspace_symbols",
    "workspace_symbols": "workspace_symbols",
    "gotodefinition": "definition",
    "go_to_definition": "definition",
    "definition": "definition",
    "findreferences": "references",
    "find_references": "references",
    "references": "references",
    "diagnostic": "diagnostics",
    "diagnostics": "diagnostics",
}


@dataclass(frozen=True)
class CodeSymbol:
    """A symbol discovered by Koder's local scanner."""

    name: str
    kind: str
    path: Path
    line: int
    column: int
    container: str = ""
    signature: str = ""


@dataclass(frozen=True)
class CodeReference:
    """A textual reference discovered in the workspace."""

    path: Path
    line: int
    column: int
    text: str


@dataclass(frozen=True)
class CodeDiagnostic:
    """A local syntax diagnostic."""

    path: Path
    line: int
    column: int
    message: str


def run_code_intelligence(
    operation: str,
    path: str | None = None,
    query: str | None = None,
    line: int | None = None,
    character: int | None = None,
    limit: int = 50,
) -> str:
    """Run a local code-intelligence operation and return formatted text."""
    normalized = _normalize_operation(operation)
    if normalized is None:
        return (
            f"Unsupported operation: {operation}. Supported operations: "
            "document_symbols, workspace_symbols, definition, references, diagnostics."
        )

    limit = _clamp_limit(limit)

    if normalized == "document_symbols":
        target = _require_file(path)
        if isinstance(target, str):
            return target
        return _format_symbols(_document_symbols(target), "Document symbols", limit)

    if normalized == "workspace_symbols":
        root = _workspace_root(path)
        symbols = _workspace_symbols(root, query=query)
        return _format_symbols(symbols, "Workspace symbols", limit)

    if normalized == "definition":
        target_query, source_file = _resolve_query(path, query, line, character)
        if isinstance(target_query, str) and target_query.startswith("Error:"):
            return target_query
        root = get_execution_cwd().resolve()
        symbols = _workspace_symbols(root, query=target_query)
        matches = _rank_definition_matches(symbols, target_query, source_file)
        return _format_symbols(matches, f"Definitions for {target_query}", limit)

    if normalized == "references":
        target_query, source_file = _resolve_query(path, query, line, character)
        if isinstance(target_query, str) and target_query.startswith("Error:"):
            return target_query
        root = get_execution_cwd().resolve() if source_file is None else _workspace_root(path)
        references = _find_references(root, target_query, limit=limit)
        return _format_references(references, target_query, limit)

    target = _require_path(path)
    if isinstance(target, str):
        return target
    return _format_diagnostics(_diagnostics(target), target)


def _normalize_operation(operation: str) -> str | None:
    compact = operation.strip().replace("-", "_")
    return OPERATION_ALIASES.get(compact.lower())


def _clamp_limit(limit: int) -> int:
    if limit < 1:
        return 1
    return min(limit, 200)


def _resolve_path(path: str | None) -> Path:
    return (execution_path(path) if path else get_execution_cwd()).resolve()


def _require_path(path: str | None) -> Path | str:
    if not path:
        return "Error: path is required for this operation."
    target = _resolve_path(path)
    if not target.exists():
        return f"Error: path does not exist: {path}"
    return target


def _require_file(path: str | None) -> Path | str:
    target = _require_path(path)
    if isinstance(target, str):
        return target
    if not target.is_file():
        return f"Error: path is not a file: {path}"
    return target


def _workspace_root(path: str | None) -> Path:
    target = _resolve_path(path)
    if target.is_file():
        return target.parent
    if target.is_dir():
        return target
    return get_execution_cwd().resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(get_execution_cwd().resolve()))
    except ValueError:
        return str(path)


def _read_text(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_INDEXED_FILE_SIZE:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def _iter_source_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if _is_source_file(root):
            yield root
        return

    count = 0
    for candidate in root.rglob("*"):
        if count >= MAX_WORKSPACE_FILES:
            return
        if any(part in SKIP_DIRS for part in candidate.parts):
            continue
        if not candidate.is_file() or not _is_source_file(candidate):
            continue
        count += 1
        yield candidate


def _is_source_file(path: Path) -> bool:
    return path.suffix.lower() in SOURCE_EXTENSIONS


def _document_symbols(path: Path) -> list[CodeSymbol]:
    if path.suffix.lower() == ".py":
        return _python_symbols(path)
    return _regex_symbols(path)


def _workspace_symbols(root: Path, query: str | None = None) -> list[CodeSymbol]:
    query_lower = query.lower() if query else None
    symbols: list[CodeSymbol] = []
    for source_file in _iter_source_files(root):
        for symbol in _document_symbols(source_file):
            if query_lower and query_lower not in symbol.name.lower():
                continue
            symbols.append(symbol)
    symbols.sort(key=lambda symbol: (_display_path(symbol.path), symbol.line, symbol.name))
    return symbols


def _python_symbols(path: Path) -> list[CodeSymbol]:
    text = _read_text(path)
    if text is None:
        return []
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return _regex_symbols(path)
    visitor = _PythonSymbolVisitor(path)
    visitor.visit(tree)
    return visitor.symbols


class _PythonSymbolVisitor(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.symbols: list[CodeSymbol] = []
        self._container_stack: list[tuple[str, str]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add_symbol("class", node.name, node, _class_signature(node))
        self._container_stack.append((node.name, "class"))
        self.generic_visit(node)
        self._container_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, async_function=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, async_function=True)

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._inside_function():
            return
        for target in node.targets:
            name = _target_name(target)
            if name:
                self._add_symbol("variable", name, node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self._inside_function():
            return
        name = _target_name(node.target)
        if name:
            self._add_symbol("variable", name, node)

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, async_function: bool
    ) -> None:
        kind = "method" if self._inside_class() else "function"
        signature = _function_signature(node, async_function=async_function)
        self._add_symbol(kind, node.name, node, signature)
        self._container_stack.append((node.name, kind))
        self.generic_visit(node)
        self._container_stack.pop()

    def _add_symbol(
        self,
        kind: str,
        name: str,
        node: ast.AST,
        signature: str = "",
    ) -> None:
        container = self._container_stack[-1][0] if self._container_stack else ""
        self.symbols.append(
            CodeSymbol(
                name=name,
                kind=kind,
                path=self.path,
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0) + 1,
                container=container,
                signature=signature,
            )
        )

    def _inside_class(self) -> bool:
        return bool(self._container_stack and self._container_stack[-1][1] == "class")

    def _inside_function(self) -> bool:
        return bool(
            self._container_stack and self._container_stack[-1][1] in {"function", "method"}
        )


def _target_name(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _class_signature(node: ast.ClassDef) -> str:
    if not node.bases:
        return ""
    bases = [ast.unparse(base) for base in node.bases]
    return f"class {node.name}({', '.join(bases)})"


def _function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef, *, async_function: bool
) -> str:
    prefix = "async def" if async_function else "def"
    args = [arg.arg for arg in node.args.posonlyargs + node.args.args]
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    args.extend(arg.arg for arg in node.args.kwonlyargs)
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    return f"{prefix} {node.name}({', '.join(args)})"


def _regex_symbols(path: Path) -> list[CodeSymbol]:
    text = _read_text(path)
    if text is None:
        return []
    symbols: list[CodeSymbol] = []
    for line_number, text_line in enumerate(text.splitlines(), start=1):
        for kind, _label, pattern in SYMBOL_PATTERNS:
            match = pattern.search(text_line)
            if not match:
                continue
            symbols.append(
                CodeSymbol(
                    name=match.group(1),
                    kind=kind,
                    path=path,
                    line=line_number,
                    column=match.start(1) + 1,
                    signature=text_line.strip(),
                )
            )
            break
    return symbols


def _resolve_query(
    path: str | None,
    query: str | None,
    line: int | None,
    character: int | None,
) -> tuple[str, Path | None]:
    if query and query.strip():
        return query.strip(), _resolve_path(path) if path else None
    source = _require_file(path)
    if isinstance(source, str):
        return source, None
    if line is None or character is None:
        return "Error: query or file position (path, line, character) is required.", source
    word = _word_at_position(source, line, character)
    if not word:
        return f"Error: no symbol found at {path}:{line}:{character}.", source
    return word, source


def _word_at_position(path: Path, line: int, character: int) -> str | None:
    text = _read_text(path)
    if text is None:
        return None
    lines = text.splitlines()
    if line < 1 or line > len(lines):
        return None
    text_line = lines[line - 1]
    index = max(0, min(character - 1, len(text_line)))
    if index == len(text_line) and index > 0:
        index -= 1
    if (
        index > 0
        and not _is_identifier_char(text_line[index])
        and _is_identifier_char(text_line[index - 1])
    ):
        index -= 1
    if not text_line or not _is_identifier_char(text_line[index]):
        return None
    start = index
    while start > 0 and _is_identifier_char(text_line[start - 1]):
        start -= 1
    end = index + 1
    while end < len(text_line) and _is_identifier_char(text_line[end]):
        end += 1
    return text_line[start:end]


def _is_identifier_char(char: str) -> bool:
    return char == "_" or char == "$" or char.isalnum()


def _rank_definition_matches(
    symbols: list[CodeSymbol], query: str, source_file: Path | None
) -> list[CodeSymbol]:
    query_lower = query.lower()

    def sort_key(symbol: CodeSymbol) -> tuple[int, int, str, int]:
        exact = 0 if symbol.name.lower() == query_lower else 1
        same_file = 0 if source_file and symbol.path == source_file else 1
        return exact, same_file, _display_path(symbol.path), symbol.line

    return sorted(symbols, key=sort_key)


def _find_references(root: Path, query: str, limit: int) -> list[CodeReference]:
    pattern = re.compile(rf"\b{re.escape(query)}\b")
    references: list[CodeReference] = []
    for source_file in _iter_source_files(root):
        text = _read_text(source_file)
        if text is None:
            continue
        for line_number, text_line in enumerate(text.splitlines(), start=1):
            for match in pattern.finditer(text_line):
                references.append(
                    CodeReference(
                        path=source_file,
                        line=line_number,
                        column=match.start() + 1,
                        text=text_line.strip(),
                    )
                )
                if len(references) >= limit:
                    return references
    return references


def _diagnostics(target: Path) -> list[CodeDiagnostic]:
    files = [target] if target.is_file() else list(_iter_source_files(target))
    diagnostics: list[CodeDiagnostic] = []
    for source_file in files:
        if source_file.suffix.lower() != ".py":
            continue
        text = _read_text(source_file)
        if text is None:
            continue
        try:
            ast.parse(text, filename=str(source_file))
        except SyntaxError as exc:
            diagnostics.append(
                CodeDiagnostic(
                    path=source_file,
                    line=exc.lineno or 1,
                    column=exc.offset or 1,
                    message=exc.msg,
                )
            )
    return diagnostics


def _format_symbols(symbols: list[CodeSymbol], title: str, limit: int) -> str:
    if not symbols:
        return f"{title}\nNo symbols found."
    shown = symbols[:limit]
    lines = [
        f"{title} ({len(symbols)} match{'es' if len(symbols) != 1 else ''}; showing {len(shown)})"
    ]
    for symbol in shown:
        container = f" in {symbol.container}" if symbol.container else ""
        signature = f" - {symbol.signature}" if symbol.signature else ""
        lines.append(
            f"{_display_path(symbol.path)}:{symbol.line}:{symbol.column} "
            f"{symbol.kind} {symbol.name}{container}{signature}"
        )
    return "\n".join(lines)


def _format_references(references: list[CodeReference], query: str, limit: int) -> str:
    if not references:
        return f"References for {query}\nNo references found."
    shown = references[:limit]
    lines = [
        f"References for {query} ({len(references)} match{'es' if len(references) != 1 else ''}; showing {len(shown)})"
    ]
    for reference in shown:
        lines.append(
            f"{_display_path(reference.path)}:{reference.line}:{reference.column} {reference.text}"
        )
    return "\n".join(lines)


def _format_diagnostics(diagnostics: list[CodeDiagnostic], target: Path) -> str:
    if not diagnostics:
        if target.is_file() and target.suffix.lower() != ".py":
            return (
                f"Diagnostics for {_display_path(target)}\n"
                "No local diagnostics for this file type. Python syntax checks are supported."
            )
        return f"Diagnostics for {_display_path(target)}\nNo diagnostics found."
    lines = [f"Diagnostics ({len(diagnostics)} issue{'s' if len(diagnostics) != 1 else ''})"]
    for diagnostic in diagnostics:
        lines.append(
            f"{_display_path(diagnostic.path)}:{diagnostic.line}:{diagnostic.column} {diagnostic.message}"
        )
    return "\n".join(lines)
