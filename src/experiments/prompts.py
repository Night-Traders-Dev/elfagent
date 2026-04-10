ROUTER_PROMPTS = {
    "baseline": "Classify the task into one route.",
    "search_heavy": "Prefer web_research when freshness, docs, or lookup behavior is present.",
    "code_heavy": "Prefer code_reasoning when debugging, patching, or implementation details are present.",
}

HELPER_PROMPTS = {
    "baseline": "Compress search results into compact evidence.",
    "strict_json": "Return only strict JSON with title, url, snippet.",
    "high_signal": "Prefer relevance, dedupe aggressively, suppress fluff.",
}
