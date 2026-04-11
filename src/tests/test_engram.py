import os
import tempfile
import unittest

from core.engram import EngramStore


class EngramStoreTests(unittest.TestCase):
    def test_retrieve_prefers_matching_engram(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "engrams.jsonl")
            store = EngramStore(
                path=path,
                enabled=True,
                max_records=10,
                retrieval_limit=2,
                max_gist_chars=180,
            )
            store.remember(
                "Implement selective memory retrieval for coding tasks",
                "Added a memory store that ranks prior turns by overlap.",
                route="code_reasoning",
            )
            store.remember(
                "Summarize this meeting transcript",
                "Created a concise summary with action items.",
                route="local_summary_meeting",
            )

            matches = store.retrieve("How does selective memory retrieval work?")
            self.assertTrue(matches)
            self.assertEqual(matches[0]["route"], "code_reasoning")
            self.assertIn("memory", matches[0]["gist"].lower())


if __name__ == "__main__":
    unittest.main()
