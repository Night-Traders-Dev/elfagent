def is_simple_code_request(user_msg: str) -> bool:
    q = user_msg.lower()
    simple_markers = ["write a ", "script", "example", "hello world", "program that", "in ruby", "in python", " in c ", "in nim"]
    hard_markers = ["debug", "analyze this codebase", "refactor", "optimize", "architecture", "benchmark", "multi-file", "agent", "workflow"]
    return any(m in q for m in simple_markers) and not any(m in q for m in hard_markers)


def is_complex_refactor_request(user_msg: str) -> bool:
    q = user_msg.lower()
    markers = ["refactor", "rewrite this module", "restructure", "large codebase", "complex migration", "deep cleanup", "multi-file refactor"]
    return any(m in q for m in markers)
