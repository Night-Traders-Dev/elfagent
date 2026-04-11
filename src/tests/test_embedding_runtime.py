import sys
import types
import unittest
from unittest.mock import patch

from helpers.embedding_reranker import EmbeddingReranker
from tools import rag_tools


class FakeSentenceTransformer:
    instances = []

    def __init__(self, model_name, cache_folder=None, device=None):
        self.model_name = model_name
        self.cache_folder = cache_folder
        self.device = device
        self.fail_once = device != "cpu"
        FakeSentenceTransformer.instances.append(self)

    def encode(self, texts, show_progress_bar=False):
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("cuda kernel image is not available")
        return [FakeVector([1.0, 0.0]) for _ in texts]


class FakeVector(list):
    def __matmul__(self, other):
        return sum(a * b for a, b in zip(self, other))


class EmbeddingRuntimeTests(unittest.TestCase):
    def setUp(self):
        FakeSentenceTransformer.instances = []
        rag_tools._EMBEDDING_MODEL = None
        rag_tools._EMBEDDING_DEVICE = "cpu"
        self.fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)

    def test_embedding_reranker_uses_cpu_by_default(self):
        reranker = EmbeddingReranker()
        with patch.dict(sys.modules, {"sentence_transformers": self.fake_module}):
            reranker._get_model()

        self.assertTrue(FakeSentenceTransformer.instances)
        self.assertEqual(FakeSentenceTransformer.instances[0].device, "cpu")

    def test_embedding_reranker_falls_back_to_cpu_after_runtime_error(self):
        reranker = EmbeddingReranker()
        reranker._device = "cuda"
        results = [{"title": "Boot", "snippet": "memory retrieval docs", "url": "https://example.com"}]

        with patch.dict(sys.modules, {"sentence_transformers": self.fake_module}):
            ranked = reranker.rerank("memory retrieval", results)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(reranker._device, "cpu")
        self.assertGreaterEqual(len(FakeSentenceTransformer.instances), 2)
        self.assertEqual(FakeSentenceTransformer.instances[-1].device, "cpu")

    def test_rag_encoder_retries_on_cpu(self):
        rag_tools._EMBEDDING_DEVICE = "cuda"
        with patch.dict(sys.modules, {"sentence_transformers": self.fake_module}):
            vectors = rag_tools._encode_texts(["boot stack setup"])

        self.assertEqual(vectors, [[1.0, 0.0]])
        self.assertEqual(rag_tools._EMBEDDING_DEVICE, "cpu")


if __name__ == "__main__":
    unittest.main()
