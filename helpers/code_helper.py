from core.config import HELPER_MODEL

class CodeHelper:
    def __init__(self):
        self.model_name = HELPER_MODEL

    def normalize_request(self, query: str) -> dict:
        return {
            "task": "code_help",
            "query": query,
            "needs_patch": any(x in query.lower() for x in ["fix", "patch", "refactor"]),
            "needs_explanation": True,
        }
