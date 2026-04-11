from __future__ import annotations

import ast
from pathlib import Path


def python_ast_query(path: str, query_type: str = "symbols") -> str:
    """Inspect a Python file with the stdlib ast module."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        return f"Error: file not found: {path}"
    if file_path.suffix.lower() != ".py":
        return "Error: python_ast_query only supports .py files."

    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except Exception as exc:  # noqa: BLE001
        return f"Error parsing Python AST for {path}: {exc}"

    symbols: list[str] = []
    imports: list[str] = []
    calls: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(f"{type(node).__name__}:{node.name}:{getattr(node, 'lineno', '?')}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"from:{module}:{alias.name}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                calls.append(func.attr)

    if query_type == "symbols":
        payload = symbols
    elif query_type == "imports":
        payload = imports
    elif query_type == "calls":
        payload = calls
    else:
        return "Unsupported query_type. Use symbols, imports, or calls."

    if not payload:
        return f"No {query_type} found in {file_path}."
    return f"{query_type} for {file_path}:\n" + "\n".join(sorted(set(payload)))
