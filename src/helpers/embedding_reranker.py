from core.config import EMBEDDING_MODEL_ID, HF_CACHE_DIR

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    SentenceTransformer = None
    HAS_ST = False


class EmbeddingReranker:
    """Lazy-loading embedding reranker.

    SentenceTransformer is expensive to initialise (loads weights into memory).
    We defer that work until the first actual rerank() call so that constructing
    an EmbeddingReranker at import time (e.g. inside build_runtime_profile) does
    not trigger a full model load.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_ID):
        self.model_name = model_name
        self._model = None  # loaded on first use

    def _get_model(self):
        if self._model is None and HAS_ST:
            self._model = SentenceTransformer(self.model_name, cache_folder=HF_CACHE_DIR)
        return self._model

    def rerank(self, query: str, results: list):
        model = self._get_model()
        if not model or not results:
            return results
        docs = [
            " ".join([
                r.get("title") or r.get("name") or "",
                r.get("snippet") or r.get("body") or r.get("content") or "",
                r.get("url") or r.get("href") or "",
            ])
            for r in results
        ]
        q_emb = model.encode([query])[0]
        d_embs = model.encode(docs)
        scored = []
        for item, emb in zip(results, d_embs):
            score = float(
                (q_emb @ emb) / (((q_emb @ q_emb) ** 0.5) * ((emb @ emb) ** 0.5) + 1e-8)
            )
            enriched = dict(item)
            enriched["embedding_score"] = score
            scored.append(enriched)
        return sorted(scored, key=lambda x: x.get("embedding_score", 0.0), reverse=True)
