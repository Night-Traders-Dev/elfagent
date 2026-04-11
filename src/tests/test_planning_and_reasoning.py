import asyncio
import unittest
from unittest.mock import patch

from reasoning.autoresearch import AutoResearchWorkflow
from reasoning.code_reasoner import CodeReasoner
from reasoning.critic_reasoner import CriticReasoner
from reasoning.task_planner import PlanExecutor, Step, StepResult, build_default_plan, should_use_task_planner


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

    def test_async_plan_executor_can_revise_and_retry(self):
        plan = build_default_plan("Fix the startup sequence")
        attempts = {"Implement change": 0}

        async def runner(step, current_plan):
            if step.title.startswith("Resolve blocker"):
                return StepResult(status="done", summary="Collected the missing linker detail.")
            if step.title == "Implement change":
                attempts["Implement change"] += 1
                if attempts["Implement change"] == 1:
                    return StepResult(status="revise", summary="Need one blocker-resolution step before implementation.")
            return StepResult(status="done", summary=f"done:{step.title}")

        def replanner(current_plan, failed_step, exc):
            failed_step.status = "pending"
            failed_step.error = ""
            current_plan.steps.insert(
                current_plan.steps.index(failed_step),
                Step(title="Resolve blocker for Implement change", description=str(exc)),
            )
            return current_plan

        executed = asyncio.run(PlanExecutor(step_runner=runner, replanner=replanner).run_async(plan))
        self.assertGreaterEqual(executed.revision_count, 1)
        self.assertTrue(any(step.title.startswith("Resolve blocker") and step.status == "done" for step in executed.steps))
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

    def test_should_use_task_planner_for_complex_code_request(self):
        route = {"route": "code_reasoning"}
        normalized = {"needs_patch": True}
        self.assertTrue(should_use_task_planner("Refactor and verify this multi-file patch", route, normalized))


if __name__ == "__main__":
    unittest.main()
