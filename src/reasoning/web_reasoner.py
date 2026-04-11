from core.config import SEARCH_MODEL
from reasoning.autoresearch import AutoResearchWorkflow

class WebReasoner:
    def __init__(self, search_tool):
        self.model_name = SEARCH_MODEL
        self.workflow = AutoResearchWorkflow(search_tool)

    def run(self, query: str) -> dict:
        return self.workflow.run(query)
