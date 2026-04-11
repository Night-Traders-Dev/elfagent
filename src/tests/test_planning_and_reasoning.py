import unittest
from unittest.mock import patch

from reasoning.autoresearch import AutoResearchWorkflow
from reasoning.code_reasoner import CodeReasoner
from reasoning.critic_reasoner import CriticReasoner
from reasoning.task_planner import PlanExecutor, Step, build_default_plan


class FakeSearchTool:
    def duckduckgo_full_search(self, query, max_results=8):
        return []


class PassthroughEmbeddingReranker:
    def rerank(self, query, results):
        return results


class PlanningAndReasoningTests(unittest.TestCase):
    def test_code_reasoner_returns_structured_plan(self):
        payload = CodeReasoner().run("Refactor the bootloader startup sequence")
        self.assertIn("plan", payload)
        self.assertEqual(payload["plan"]["steps"][0]["status"], "pending")

    def test_plan_executor_updates_statuses(self):
        plan = build_default_plan("Inspect and change config")

        def runner(step, current_plan):
            return f"done:{step.title}"

        executed = PlanExecutor(step_runner=runner).run(plan)
        self.assertTrue(all(step.status == "done" for step in executed.steps))

    def test_autoresearch_uses_local_rag_before_web(self):
        workflow = AutoResearchWorkflow(FakeSearchTool(), max_rounds=1, max_queries_per_round=1, target_coverage=0.5)
        local_results = [
            {
                "title": "boot.md",
                "url": "/tmp/boot.md",
                "domain": "local",
                "snippet": "memory retrieval and bootloader stack setup",
            }
        ]
        workflow.search_helper._embedding_reranker = PassthroughEmbeddingReranker()
        with patch("tools.rag_tools.semantic_search_raw", return_value=local_results):
            payload = workflow.run("memory retrieval bootloader")

        self.assertTrue(payload["rounds"])
        self.assertEqual(payload["rounds"][0]["source"], "local_rag")
        self.assertTrue(payload["plan"]["local_rag_seeded"])

    def test_critic_reasoner_flags_invalid_python(self):
        review = CriticReasoner().review("```python\nif True print('x')\n```")
        self.assertFalse(review["ok"])
        self.assertTrue(review["findings"])


if __name__ == "__main__":
    unittest.main()
