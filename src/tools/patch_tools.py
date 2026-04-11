from __future__ import annotations

import difflib
import os
import subprocess


def create_diff(
    before: str,
    after: str,
    before_label: str = "before",
    after_label: str = "after",
    inputs_are_paths: bool = False,
) -> str:
    """Create a unified diff from text blobs or from two file paths."""
    if inputs_are_paths:
        with open(before, "r", encoding="utf-8", errors="replace") as handle:
            before_text = handle.read()
        with open(after, "r", encoding="utf-8", errors="replace") as handle:
            after_text = handle.read()
        before_label = before
        after_label = after
    else:
        before_text = before
        after_text = after

    diff = difflib.unified_diff(
        before_text.splitlines(keepends=True),
        after_text.splitlines(keepends=True),
        fromfile=before_label,
        tofile=after_label,
    )
    return "".join(diff) or "No diff."


def apply_patch(diff: str, working_directory: str | None = None) -> str:
    """Apply a unified diff using the system patch tool."""
    if not diff or not diff.strip():
        return "Error: empty patch."

    cwd = working_directory or os.getcwd()
    try:
        result = subprocess.run(
            ["patch", "-p0", "--forward", "--reject-file=-"],
            input=diff,
            text=True,
            capture_output=True,
            cwd=cwd,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error applying patch: {exc}"

    output = (result.stdout or "") + (result.stderr or "")
    return f"[exit {result.returncode}]\n{output.strip()}".strip()
