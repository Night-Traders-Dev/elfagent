from core.config import CODE_MODEL
from helpers.code_helper import CodeHelper
from reasoning.task_planner import build_default_plan

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
            "plan": build_default_plan(query).to_dict(),
            "next_step": "handoff_to_main_model",
        }
