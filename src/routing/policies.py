# Coding keywords checked first so prompts like "find the bug" or
# "current implementation" don't misfire as web_research.
_CODING_KEYWORDS = {
    "code", "python", "nim", "assembly", "asm", "bug", "refactor",
    "stack trace", "stacktrace", "traceback", "debug", "debugger",
    "compile", "compiler", "linker", "function", "class", "method",
    "implement", "implementation", "algorithm", "snippet", "script",
    "syntax", "type error", "runtime error", "import", "module",
    "risc-v", "riscv", "arm", "x86", "aarch64", "mips", "powerpc",
    "fix this", "fix the", "write a function", "write a script",
    "write a class", "write a program",
}

_TIME_SENSITIVE_KEYWORDS = {
    "weather", "forecast", "temperature", "humidity", "wind", "rain", "snow",
    "today", "tonight", "tomorrow", "this week",
    "latest news", "breaking", "live score", "stock price",
}

_SEARCH_KEYWORDS = {
    "latest", "search", "look up", "news", "web", "docs", "documentation",
    "release notes", "changelog", "official site",
}


def _has(q: str, keywords) -> bool:
    return any(k in q for k in keywords)


def route_by_rules(query: str) -> dict:
    q = query.strip().lower()

    # --- Explicit slash-command routes ---
    if q.startswith("/summarize-medical"):
        return {"route": "local_summary_medical", "confidence": 0.99, "reason": "explicit command"}
    if q.startswith("/summarize-meeting"):
        return {"route": "local_summary_meeting", "confidence": 0.99, "reason": "explicit command"}

    # --- Coding intent takes precedence over ambiguous words like "find" / "current" ---
    if _has(q, _CODING_KEYWORDS):
        return {"route": "code_reasoning", "confidence": 0.82, "reason": "coding intent"}

    # --- Clearly time-sensitive / live-data queries ---
    if _has(q, _TIME_SENSITIVE_KEYWORDS):
        return {"route": "web_research", "confidence": 0.92, "reason": "time-sensitive intent"}

    # --- Remaining search-like signals ("find", "current" safe here; no coding keyword present) ---
    if _has(q, _SEARCH_KEYWORDS | {"find", "current", "look for"}):
        return {"route": "web_research", "confidence": 0.86, "reason": "search-like intent"}

    return {"route": "main_model", "confidence": 0.60, "reason": "fallback synthesis path"}
