def route_by_rules(query: str) -> dict:
    q = query.strip().lower()
    if q.startswith("/summarize-medical"):
        return {"route": "local_summary_medical", "confidence": 0.99, "reason": "explicit command"}
    if q.startswith("/summarize-meeting"):
        return {"route": "local_summary_meeting", "confidence": 0.99, "reason": "explicit command"}
    if any(k in q for k in ["latest", "search", "look up", "find", "news", "web", "docs"]):
        return {"route": "web_research", "confidence": 0.86, "reason": "search-like intent"}
    if any(k in q for k in ["code", "python", "nim", "assembly", "bug", "refactor", "stack trace"]):
        return {"route": "code_reasoning", "confidence": 0.82, "reason": "coding intent"}
    return {"route": "main_model", "confidence": 0.60, "reason": "fallback synthesis path"}
