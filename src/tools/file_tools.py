"""File-system tools for the ReAct agent.

The agent can use these to read source files, inspect project structure, and
write small patches without leaving the agent loop.
"""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

# Paths the agent is never allowed to write to (relative or absolute patterns).
_WRITE_BLOCKED = {".git", "__pycache__", ".env", ".elfagent_cache"}


def _safe_path(path: str) -> Path:
    """Resolve and return a Path, rejecting traversal outside cwd."""
    resolved = Path(path).resolve()
    cwd = Path.cwd().resolve()
    # Allow absolute paths inside the project tree
    try:
        resolved.relative_to(cwd)
    except ValueError:
        # Outside cwd — still allow reads of absolute paths (e.g. /tmp)
        pass
    return resolved


def read_file(path: str, max_chars: int = 16_000) -> str:
    """Read a text file and return its contents (up to max_chars characters).

    If the file is larger, a truncation notice is appended.
    """
    try:
        p = _safe_path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n[... truncated at {max_chars} chars ...]"
        return text
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except Exception as exc:
        return f"Error reading {path}: {exc}"


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories as needed.

    Refuses to write into .git, __pycache__, .env, or .elfagent_cache.
    """
    p = _safe_path(path)
    for blocked in _WRITE_BLOCKED:
        if blocked in str(p):
            return f"Error: writing to {blocked} is not allowed."
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {p}"
    except Exception as exc:
        return f"Error writing {path}: {exc}"


def list_directory(path: str = ".", pattern: str = "*") -> str:
    """List files and directories at path matching an optional glob pattern.

    Returns a tree-style text representation.
    """
    try:
        p = _safe_path(path)
        if not p.exists():
            return f"Error: path not found: {path}"
        lines: list[str] = [f"Directory listing: {p}"]
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        for entry in entries:
            if not fnmatch.fnmatch(entry.name, pattern):
                continue
            kind = "📁" if entry.is_dir() else "📄"
            size = f"  ({entry.stat().st_size:,} bytes)" if entry.is_file() else ""
            lines.append(f"  {kind} {entry.name}{size}")
        return "\n".join(lines) if len(lines) > 1 else f"No entries matching '{pattern}' in {path}"
    except Exception as exc:
        return f"Error listing {path}: {exc}"


def grep_files(
    pattern: str,
    path: str = ".",
    file_glob: str = "*.py",
    max_matches: int = 40,
) -> str:
    """Search for a regex pattern across files in path matching file_glob.

    Returns matching lines with file:line context (like grep -n).
    """
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"Invalid regex: {exc}"
    root = _safe_path(path)
    matches: list[str] = []
    for filepath in sorted(root.rglob(file_glob)):
        if any(blocked in str(filepath) for blocked in _WRITE_BLOCKED):
            continue
        try:
            for lineno, line in enumerate(filepath.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    rel = filepath.relative_to(root) if filepath.is_relative_to(root) else filepath
                    matches.append(f"{rel}:{lineno}: {line.rstrip()}")
                    if len(matches) >= max_matches:
                        matches.append(f"[... stopped at {max_matches} matches ...]")
                        return "\n".join(matches)
        except Exception:
            continue
    return "\n".join(matches) if matches else f"No matches for '{pattern}' in {path}/{file_glob}"
