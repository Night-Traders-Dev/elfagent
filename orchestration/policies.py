def should_escalate(route_result: dict, threshold: float) -> bool:
    return route_result.get("confidence", 0.0) < threshold

def needs_main_model(route_name: str) -> bool:
    return route_name in {"web_research", "code_reasoning", "main_model"}
