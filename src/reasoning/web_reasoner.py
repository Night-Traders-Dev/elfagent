from core.config import SEARCH_MODEL
from helpers.search_helper import SearchHelper

class WebReasoner:
    def __init__(self, search_tool):
        self.model_name = SEARCH_MODEL
        self.search_helper = SearchHelper()
        self.search_tool = search_tool

    def run(self, query: str) -> dict:
        queries = self.search_helper.build_queries(query)
        raw_results = []
        for q in queries:
            try:
                raw_results.extend(self.search_tool.duckduckgo_full_search(q, max_results=8))
            except Exception:
                continue
        return {
            "query": query,
            "search_queries": queries,
            "evidence": self.search_helper.compress_results(query, raw_results),
            "model": self.model_name,
            "raw_result_count": len(raw_results),
        }
