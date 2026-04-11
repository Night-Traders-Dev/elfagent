# ElfAgent — Patch Notes

## Fixes applied against live main branch (April 10, 2026)

### Feature 1 - AutoResearch workflow is now concrete (`reasoning/autoresearch.py`, `reasoning/web_reasoner.py`, `helpers/search_helper.py`)
Web research now runs a real iterative search/evaluate/refine loop instead of a single raw-search pass. The helper tracks coverage of salient query terms and launches follow-up searches when evidence is thin.

### Feature 2 - Engram selective memory layer (`core/engram.py`, `core/app.py`)
Added a persistent JSONL engram store that saves compact turn gists and retrieves only relevant prior memories for the current request.

### Feature 3 - TurboQuant runtime compression (`core/turboquant.py`, `orchestration/handoff.py`)
Added a budget-aware compression path that trims snippets, caps evidence lists, and compacts structured handoff packets before they reach the main model.

### Bug 7 - Weather queries were misclassified as simple code (`core/app.py`, `routing/policies.py`)
Removed `"weather"` from the simple-code fast path and added explicit weather/time-sensitive routing so live queries like current weather go to web research.

### Bug 1 — Double MCP init / panel on every turn (`core/app.py`)
`build_agent()` now accepts `preloaded_tools`. Per-turn agent rebuilds reuse the
already-loaded tool list, so `load_mcp_github_tools` only runs once at startup.
`run_turn_rich()` signature updated to accept `tools` and pass them through.

### Bug 2 — Agent tries to run ASM/C through `code_interpreter` (`core/app.py`)
Added `system_prompt` to `ReActAgent` that explicitly tells the model:
- `code_interpreter` is Python-only
- For assembly (ARM, RISC-V, x86, PowerPC, MIPS, etc.), C, C++, Nim, Rust, Zig,
  Ruby, or any other non-Python language, respond directly without running anything

### Bug 3 — Default 20-iteration cap causes `WorkflowRuntimeError` (`core/app.py`)
`.run()` now passes `max_iterations=50` and `early_stopping_method="generate"` so
the agent generates a graceful final response instead of throwing.

### Bug 4 — Unhandled `WorkflowRuntimeError` in background task (`core/app.py`)
`run_response()` is now wrapped in `try/except`. On failure, a `_Fallback` response
object is stored so the UI always has something to print instead of an empty panel
and an unretreived task exception.

### Bug 5 — `"in c"` substring false-positive in `is_simple_code_request` (`core/app.py`)
Changed `"in c"` → `" in c "` (space-padded) to avoid matching `"in powerpc"`,
`"in crystal"`, `"in clojure"`, etc.

### Bug 6 — `info_panel` fires on every turn for intentionally-disabled MCP (`integrations/github_mcp.py`)
The `ENABLE_GITHUB_MCP=0` early return now exits silently (comment only) instead of
rendering a full Rich panel on startup and every subsequent query.
