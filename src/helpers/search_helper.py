from core.config import HELPER_MODEL
from helpers.reranker import ResultReranker
from helpers.embedding_reranker import EmbeddingReranker

class SearchHelper:
    def __init__(self):
        self.model_name = HELPER_MODEL
        self.reranker = ResultReranker()
        self.embedding_reranker = EmbeddingReranker()

    def build_queries(self, query: str):
        return [query.strip()][:3]

    def compress_results(self, query: str, results):
        ranked = self.reranker.rerank(query, results)
        ranked = self.embedding_reranker.rerank(query, ranked)
        compressed = []
        for item in ranked[:5]:
            compressed.append({
                "title": item.get("title") or item.get("name") or "",
                "url": item.get("href") or item.get("url") or "",
                "snippet": (item.get("body") or item.get("content") or item.get("snippet") or "")[:500],
                "score": item.get("score", 0.0),
                "embedding_score": item.get("embedding_score", 0.0),
            })
        return compressed
