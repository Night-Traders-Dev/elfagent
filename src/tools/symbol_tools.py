from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_C_EXTENSIONS = {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh"}
_ASM_EXTENSIONS = {".s", ".S", ".asm"}
_TEXT_EXTENSIONS = _C_EXTENSIONS | _ASM_EXTENSIONS

_C_FUNCTION_DEF_RE = re.compile(
    r"^\s*(?:[A-Za-z_][\w\s\*\(\),]*?\s+)?([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{"
)
_C_FUNCTION_DECL_RE = re.compile(
    r"^\s*(?:extern\s+)?(?:[A-Za-z_][\w\s\*\(\),]*?\s+)?([A-Za-z_]\w*)\s*\([^;{}]*\)\s*;"
)
_C_STRUCT_RE = re.compile(r"^\s*(?:typedef\s+)?struct\s+([A-Za-z_]\w*)?")
_C_ENUM_RE = re.compile(r"^\s*(?:typedef\s+)?enum\s+([A-Za-z_]\w*)?")
_C_TYPEDEF_RE = re.compile(r"^\s*typedef\b.*?\b([A-Za-z_]\w*)\s*;")
_C_MACRO_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\b")

_ASM_LABEL_RE = re.compile(r"^\s*([A-Za-z_.$][\w.$@]*)\s*:\s*(?:[#;].*)?$")
_ASM_GLOBAL_RE = re.compile(r"^\s*\.globl?\s+([A-Za-z_.$][\w.$@]*)\b")
_ASM_TYPE_RE = re.compile(r"^\s*\.type\s+([A-Za-z_.$][\w.$@]*)\s*,\s*@?([A-Za-z_]\w*)")


@dataclass
class SymbolRecord:
    name: str
    kind: str
    path: str
    line: int
    signature: str = ""


def _resolve_path(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    return resolved


def _iter_source_files(root: Path, extensions: set[str]) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix in extensions else []
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"} for part in path.parts):
            continue
        if path.suffix in extensions:
            files.append(path)
    return files


def _parse_c_symbols(path: Path) -> list[SymbolRecord]:
    records: list[SymbolRecord] = []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        for regex, kind in (
            (_C_MACRO_RE, "macro"),
            (_C_FUNCTION_DEF_RE, "function"),
            (_C_FUNCTION_DECL_RE, "declaration"),
            (_C_STRUCT_RE, "struct"),
            (_C_ENUM_RE, "enum"),
            (_C_TYPEDEF_RE, "typedef"),
        ):
            match = regex.match(line)
            if not match:
                continue
            name = (match.group(1) or "").strip()
            if not name:
                continue
            records.append(
                SymbolRecord(
                    name=name,
                    kind=kind,
                    path=str(path),
                    line=lineno,
                    signature=stripped[:160],
                )
            )
            break
    return records


def _parse_asm_symbols(path: Path) -> list[SymbolRecord]:
    records: list[SymbolRecord] = []
    seen: set[tuple[str, int]] = set()
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        for regex, kind in (
            (_ASM_GLOBAL_RE, "global"),
            (_ASM_TYPE_RE, "type"),
            (_ASM_LABEL_RE, "label"),
        ):
            match = regex.match(line)
            if not match:
                continue
            name = (match.group(1) or "").strip()
            if not name or (name, lineno) in seen:
                continue
            seen.add((name, lineno))
            records.append(
                SymbolRecord(
                    name=name,
                    kind=kind,
                    path=str(path),
                    line=lineno,
                    signature=stripped[:160],
                )
            )
            break
    return records


def _format_records(records: list[SymbolRecord], root: Path, max_symbols: int) -> str:
    if not records:
        return "No symbols found."
    lines = []
    for record in records[:max_symbols]:
        try:
            display_path = Path(record.path).resolve().relative_to(root)
        except ValueError:
            display_path = record.path
        lines.append(
            f"{record.kind}: {record.name} [{display_path}:{record.line}] {record.signature}".rstrip()
        )
    if len(records) > max_symbols:
        lines.append(f"[... truncated at {max_symbols} symbols ...]")
    return "\n".join(lines)


def list_c_symbols(path: str, max_symbols: int = 200) -> str:
    """List C/C++ source symbols from a file or directory."""
    try:
        root = _resolve_path(path)
    except Exception as exc:  # noqa: BLE001
        return f"Error listing C symbols: {exc}"
    records: list[SymbolRecord] = []
    for file_path in _iter_source_files(root, _C_EXTENSIONS):
        try:
            records.extend(_parse_c_symbols(file_path))
        except Exception:
            continue
    records.sort(key=lambda item: (Path(item.path).name, item.line, item.name))
    header = f"C/C++ symbols for {root}:\n"
    return header + _format_records(records, root if root.is_dir() else root.parent, max_symbols)


def list_asm_symbols(path: str, max_symbols: int = 200) -> str:
    """List assembly symbols from a file or directory."""
    try:
        root = _resolve_path(path)
    except Exception as exc:  # noqa: BLE001
        return f"Error listing ASM symbols: {exc}"
    records: list[SymbolRecord] = []
    for file_path in _iter_source_files(root, _ASM_EXTENSIONS):
        try:
            records.extend(_parse_asm_symbols(file_path))
        except Exception:
            continue
    records.sort(key=lambda item: (Path(item.path).name, item.line, item.name))
    header = f"ASM symbols for {root}:\n"
    return header + _format_records(records, root if root.is_dir() else root.parent, max_symbols)


def find_symbol_references(
    symbol: str,
    path: str = ".",
    extensions: str | None = None,
    max_matches: int = 100,
) -> str:
    """Find likely references to a C or ASM symbol across source files."""
    try:
        root = _resolve_path(path)
    except Exception as exc:  # noqa: BLE001
        return f"Error finding symbol references: {exc}"

    extension_set = {
        item.strip() if item.strip().startswith(".") else "." + item.strip()
        for item in (extensions.split(",") if extensions else [])
        if item.strip()
    } or _TEXT_EXTENSIONS

    regex = re.compile(rf"\b{re.escape(symbol)}\b")
    matches: list[str] = []
    for file_path in _iter_source_files(root, extension_set):
        try:
            for lineno, line in enumerate(file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if regex.search(line):
                    try:
                        display_path = file_path.resolve().relative_to(root)
                    except ValueError:
                        display_path = file_path
                    matches.append(f"{display_path}:{lineno}: {line.strip()}")
                    if len(matches) >= max_matches:
                        matches.append(f"[... truncated at {max_matches} matches ...]")
                        return "\n".join(matches)
        except Exception:
            continue
    if not matches:
        return f"No references found for symbol '{symbol}' under {root}."
    return "\n".join(matches)


def read_binary_symbols(path: str, defined_only: bool = True, max_lines: int = 200) -> str:
    """Read a binary or object file symbol table via nm."""
    try:
        resolved = _resolve_path(path)
    except Exception as exc:  # noqa: BLE001
        return f"Error reading binary symbols: {exc}"

    argv = ["nm"]
    if defined_only:
        argv.append("--defined-only")
    argv.append(str(resolved))
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    except Exception as exc:  # noqa: BLE001
        return f"Error running nm on {resolved}: {exc}"

    output = ((result.stdout or "") + (result.stderr or "")).splitlines()
    if len(output) > max_lines:
        output = output[:max_lines] + [f"[... truncated at {max_lines} lines ...]"]
    return f"nm output for {resolved}:\n" + "\n".join(output)
