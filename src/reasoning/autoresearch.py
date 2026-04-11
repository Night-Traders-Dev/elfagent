from core.config import (
    AUTORESEARCH_MAX_QUERIES_PER_ROUND,
    AUTORESEARCH_MAX_ROUNDS,
    AUTORESEARCH_TARGET_COVERAGE,
    SEARCH_MODEL,
)
from helpers.search_helper import SearchHelper


class AutoResearchWorkflow:
    def __init__(
        self,
        search_tool=None,
        search_helper: SearchHelper | None = None,
        max_rounds: int = AUTORESEARCH_MAX_ROUNDS,
        max_queries_per_round: int = AUTORESEARCH_MAX_QUERIES_PER_ROUND,
        target_coverage: float = AUTORESEARCH_TARGET_COVERAGE,
    ):
        self.model_name = SEARCH_MODEL
        # Use the multi-engine search adapter; fall back to whatever was passed
        # (e.g. DuckDuckGoSearchToolSpec) so callers that pass a custom tool
        # still work.
        from tools.search_tools import MultiEngineSearch
        self._multi_engine = MultiEngineSearch()
        self._legacy_search_tool = search_tool  # kept for reference
        self.search_helper = search_helper or SearchHelper()
        self.max_rounds = max_rounds
        self.max_queries_per_round = max_queries_per_round
        self.target_coverage = target_coverage

    def _search_once(self, query: str) -> list[dict]:
        """Run one search query, using multi-engine with DDG fallback."""
        try:
            results = self._multi_engine.search(query, max_results=8)
            if results:
                return results
        except Exception:
            pass
        # Last-ditch: try the legacy tool if it was provided
        if self._legacy_search_tool is not None:
            try:
                return self._legacy_search_tool.duckduckgo_full_search(query, max_results=8)
            except Exception:
                pass
        return []

    def run(self, query: str) -> dict:
        initial_queries = self.search_helper.build_queries(
            query,
            limit=self.max_queries_per_round,
        )
        evidence = []
        rounds = []
        executed_queries = []
        total_raw_results = 0
        pending_queries = initial_queries

        for round_idx in range(1, self.max_rounds + 1):
            current_queries = [
                item for item in pending_queries
                if item not in executed_queries
            ][:self.max_queries_per_round]
            if not current_queries:
                break

            raw_results = []
            result_counts = {}
            for search_query in current_queries:
                results = self._search_once(search_query)
                raw_results.extend(results)
                result_counts[search_query] = len(results)
                executed_queries.append(search_query)

            total_raw_results += len(raw_results)
            new_evidence = self.search_helper.compress_results(query, raw_results, limit=3)
            evidence = self.search_helper.merge_evidence(query, evidence, new_evidence, limit=6)
            evaluation = self.search_helper.evaluate_coverage(query, evidence)
            rounds.append({
                "round": round_idx,
                "queries": current_queries,
                "result_counts": result_counts,
                "new_evidence": new_evidence[:2],
                "coverage_score": evaluation["coverage_score"],
                "missing_terms": evaluation["missing_terms"][:4],
                "notes": (
                    f"Round {round_idx} gathered {len(raw_results)} raw results. "
                    f"Coverage is now {evaluation['coverage_score']:.2f}."
                ),
            })

            if (
                evaluation["coverage_score"] >= self.target_coverage
                or not evaluation["missing_terms"]
            ):
                break

            pending_queries = self.search_helper.build_queries(
                query,
                missing_terms=evaluation["missing_terms"],
                evidence=evidence,
                limit=self.max_queries_per_round,
            )

        final_evaluation = self.search_helper.evaluate_coverage(query, evidence)
        return {
            "query": query,
            "workflow": "autoresearch",
            "plan": {
                "objective": query,
                "max_rounds": self.max_rounds,
                "target_coverage": self.target_coverage,
                "initial_queries": initial_queries,
            },
            "search_queries": executed_queries,
            "rounds": rounds,
            "evidence": evidence,
            "evaluation": final_evaluation,
            "model": self.model_name,
            "raw_result_count": total_raw_results,
        }
