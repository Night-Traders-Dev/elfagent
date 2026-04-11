"""Safe subprocess tools for the ReAct agent.

Only commands on the allowlist can be executed.  The agent can use these to
run builds, tests, linters, and inspect git state without arbitrary shell
access.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Allowlist: the base command (argv[0]) must be in this set.
# ---------------------------------------------------------------------------
_ALLOWED_COMMANDS: set[str] = {
    # version control
    "git",
    # build systems
    "make", "cmake", "ninja",
    # compilers / assemblers (cross-dev friendly)
    "gcc", "g++", "clang", "clang++",
    "arm-none-eabi-gcc", "arm-none-eabi-g++", "arm-none-eabi-as", "arm-none-eabi-ld",
    "riscv64-unknown-elf-gcc", "riscv64-unknown-elf-as", "riscv64-unknown-elf-ld",
    "as", "ld", "objdump", "objcopy", "readelf", "nm", "size",
    # python / nim / rust
    "python3", "python", "pip3", "pip",
    "nim", "nimble",
    "cargo", "rustc",
    # testing / linting
    "pytest", "ruff", "mypy", "flake8",
    # general utilities
    "ls", "cat", "head", "tail", "wc", "find", "grep", "diff", "patch",
    "echo", "pwd", "env",
}

_DEFAULT_TIMEOUT = int(os.getenv("SHELL_TOOL_TIMEOUT", "30"))
_MAX_OUTPUT_CHARS = int(os.getenv("SHELL_TOOL_MAX_OUTPUT", "8000"))


def _run(argv: list[str], cwd: str | None = None, timeout: int = _DEFAULT_TIMEOUT) -> str:
    """Internal runner — returns combined stdout+stderr as a string."""
    base = Path(argv[0]).name if argv else ""
    if base not in _ALLOWED_COMMANDS:
        return (
            f"Error: '{base}' is not on the shell tool allowlist.\n"
            f"Allowed commands: {', '.join(sorted(_ALLOWED_COMMANDS))}"
        )
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ},
        )
        out = (result.stdout or "") + (result.stderr or "")
        if len(out) > _MAX_OUTPUT_CHARS:
            out = out[:_MAX_OUTPUT_CHARS] + f"\n[... truncated at {_MAX_OUTPUT_CHARS} chars ...]"
        header = f"[exit {result.returncode}]\n"
        return header + out
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s: {argv}"
    except FileNotFoundError:
        return f"Error: command not found: {argv[0]}"
    except Exception as exc:
        return f"Error running command: {exc}"


def run_shell_command(command: str, working_directory: str | None = None) -> str:
    """Run an allowlisted shell command and return its output.

    The command is split with shlex so it handles quoted args correctly.
    Only commands on the allowlist (git, make, gcc, pytest, etc.) are permitted.

    Args:
        command: The shell command to run, e.g. 'pytest tests/ -v' or 'make all'.
        working_directory: Optional working directory (defaults to cwd).
    """
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return f"Error parsing command: {exc}"
    if not argv:
        return "Error: empty command."
    return _run(argv, cwd=working_directory)


def git_status(working_directory: str | None = None) -> str:
    """Run `git status --short` and return the output.

    Useful for the agent to see which files are modified before committing.
    """
    return _run(["git", "status", "--short"], cwd=working_directory)


def git_log(n: int = 10, working_directory: str | None = None) -> str:
    """Return the last n git log entries (oneline format)."""
    return _run(
        ["git", "log", f"--max-count={n}", "--oneline", "--decorate"],
        cwd=working_directory,
    )
