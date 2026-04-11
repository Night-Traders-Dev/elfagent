import unittest

from reasoning.autoresearch import AutoResearchWorkflow


class FakeSearchTool:
    def __init__(self, mapping):
        self.mapping = mapping

    def duckduckgo_full_search(self, query, max_results=8):
        return self.mapping.get(query, [])


class AutoResearchWorkflowTests(unittest.TestCase):
    def test_autoresearch_runs_rounds_and_collects_evidence(self):
        query = "python memory retrieval docs"
        search_tool = FakeSearchTool({
            "python memory retrieval docs": [
                {
                    "title": "Memory retrieval guide",
                    "href": "https://example.com/memory",
                    "body": "Python memory retrieval techniques and retrieval patterns.",
                }
            ],
            "python memory retrieval docs official docs": [
                {
                    "title": "Official docs",
                    "href": "https://docs.example.com/python-memory",
                    "body": "Official docs for Python memory retrieval APIs.",
                }
            ],
            "python memory retrieval docs reference": [
                {
                    "title": "Reference",
                    "href": "https://reference.example.com/python-memory",
                    "body": "Reference material for memory retrieval docs.",
                }
            ],
        })
        workflow = AutoResearchWorkflow(
            search_tool,
            max_rounds=2,
            max_queries_per_round=2,
            target_coverage=0.5,
        )

        payload = workflow.run(query)
        self.assertEqual(payload["workflow"], "autoresearch")
        self.assertTrue(payload["rounds"])
        self.assertTrue(payload["evidence"])
        self.assertGreaterEqual(payload["evaluation"]["coverage_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
