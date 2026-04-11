import unittest

from core.turboquant import TurboQuantCompressor


class TurboQuantTests(unittest.TestCase):
    def test_compress_text_respects_budget(self):
        compressor = TurboQuantCompressor(
            enabled=True,
            max_snippet_chars=80,
            max_packet_chars=200,
            max_evidence_items=3,
        )
        text = (
            "TurboQuant keeps the important sentence about compression. "
            "This sentence is filler. "
            "Compression should preserve query terms and fit a budget."
        )
        compacted = compressor.compress_text(text, budget=80, query="compression budget")
        self.assertLessEqual(len(compacted), 80)
        self.assertIn("compression", compacted.lower())

    def test_compact_packet_trims_large_payload(self):
        compressor = TurboQuantCompressor(
            enabled=True,
            max_snippet_chars=60,
            max_packet_chars=450,
            max_evidence_items=2,
        )
        packet = {
            "user_query": "research compact memory systems",
            "route": {"route": "web_research", "confidence": 0.9, "reason": "test"},
            "payload": {
                "evidence": [
                    {
                        "title": "Doc",
                        "url": "https://example.com/doc",
                        "snippet": " ".join(["memory"] * 80),
                    }
                    for _ in range(4)
                ],
                "rounds": [
                    {
                        "round": 1,
                        "queries": ["research compact memory systems", "memory systems overview"],
                        "new_evidence": [
                            {
                                "title": "Doc",
                                "url": "https://example.com/doc",
                                "snippet": " ".join(["memory"] * 50),
                            }
                        ],
                        "notes": " ".join(["coverage"] * 40),
                    },
                    {
                        "round": 2,
                        "queries": ["research compact memory systems docs"],
                        "new_evidence": [],
                        "notes": " ".join(["coverage"] * 40),
                    },
                ],
            },
            "instructions": ["a", "b", "c"],
        }
        compacted, stats = compressor.compact_packet(packet)
        self.assertTrue(stats["applied"])
        self.assertLess(stats["final_chars"], stats["original_chars"])
        self.assertLessEqual(compressor.measure(compacted), compressor.max_packet_chars)


if __name__ == "__main__":
    unittest.main()
