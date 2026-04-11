from routing.policies import route_by_rules


_SEARCH_HEAVY_BOOST = {"web_research"}
_CODE_HEAVY_BOOST = {"code_reasoning"}


class TaskRouter:
    """Rule-based task router.

    Parameters
    ----------
    system_prompt:
        Optional hint injected by the experiment harness.  Recognised phrases:
        - "prefer web_research" / "search_heavy"  → lower confidence threshold
          for web_research routes, raise it for code_reasoning.
        - "prefer code_reasoning" / "code_heavy"  → vice-versa.
        No-op in production (system_prompt is None by default).
    """

    model_name = "rule-based"

    def __init__(self, system_prompt: str | None = None):
        self._system_prompt = (system_prompt or "").lower()

    def _bias(self, route: dict) -> dict:
        """Adjust confidence based on the active system_prompt variant."""
        if not self._system_prompt:
            return route
        r = dict(route)
        if any(k in self._system_prompt for k in ("prefer web_research", "search_heavy", "freshness", "lookup")):
            if r["route"] == "web_research":
                r["confidence"] = min(r["confidence"] + 0.08, 0.99)
                r["reason"] += " [search_heavy bias]"
            elif r["route"] == "code_reasoning":
                r["confidence"] = max(r["confidence"] - 0.06, 0.10)
        elif any(k in self._system_prompt for k in ("prefer code_reasoning", "code_heavy", "debugging", "patching", "implementation")):
            if r["route"] == "code_reasoning":
                r["confidence"] = min(r["confidence"] + 0.08, 0.99)
                r["reason"] += " [code_heavy bias]"
            elif r["route"] == "web_research":
                r["confidence"] = max(r["confidence"] - 0.06, 0.10)
        return r

    def route(self, query: str) -> dict:
        base = route_by_rules(query)
        return self._bias(base)
