from core.config import CODE_MODEL
from helpers.code_helper import CodeHelper

class CodeReasoner:
    def __init__(self):
        self.model_name = CODE_MODEL
        self.helper = CodeHelper()

    def run(self, query: str) -> dict:
        normalized = self.helper.normalize_request(query)
        return {
            "query": query,
            "normalized": normalized,
            "model": self.model_name,
            "next_step": "handoff_to_main_model",
        }
