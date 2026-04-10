try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    SentenceTransformer = None
    HAS_ST = False


class EmbeddingReranker:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name) if HAS_ST else None

    def rerank(self, query: str, results: list):
        if not HAS_ST or not results:
            return results
        docs = [" ".join([
            r.get("title") or r.get("name") or "",
            r.get("snippet") or r.get("body") or r.get("content") or "",
            r.get("url") or r.get("href") or "",
        ]) for r in results]
        q_emb = self.model.encode([query])[0]
        d_embs = self.model.encode(docs)
        scored = []
        for item, emb in zip(results, d_embs):
            score = float((q_emb @ emb) / (((q_emb @ q_emb) ** 0.5) * ((emb @ emb) ** 0.5) + 1e-8))
            enriched = dict(item)
            enriched["embedding_score"] = score
            scored.append(enriched)
        return sorted(scored, key=lambda x: x.get("embedding_score", 0.0), reverse=True)
