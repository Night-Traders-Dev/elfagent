from core.config import EMBEDDING_DEVICE, EMBEDDING_MODEL_ID, HF_CACHE_DIR

# NOTE: `sentence_transformers` is intentionally NOT imported at module level.
# Importing it triggers a CUDA device scan and torch initialisation even if no
# model is ever loaded.  We defer the import (and the model load) to the first
# actual rerank() call via _get_model().


class EmbeddingReranker:
    """Lazy-loading semantic reranker.

    Neither the `sentence_transformers` package nor the model weights are
    touched until rerank() is called for the first time.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_ID):
        self.model_name = model_name
        self._model = None
        self._device = EMBEDDING_DEVICE

    def _load_model(self, device: str):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        return SentenceTransformer(
            self.model_name,
            cache_folder=HF_CACHE_DIR,
            device=device,
        )

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            self._model = self._load_model(self._device)
        except Exception:  # noqa: BLE001
            self._model = None
        return self._model

    def _reload_on_cpu(self):
        self._device = "cpu"
        try:
            self._model = self._load_model("cpu")
        except Exception:  # noqa: BLE001
            self._model = None
        return self._model

    def rerank(self, query: str, results: list) -> list:
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
        try:
            q_emb = model.encode([query])[0]
            d_embs = model.encode(docs)
        except RuntimeError:
            model = self._reload_on_cpu()
            if not model:
                return results
            try:
                q_emb = model.encode([query])[0]
                d_embs = model.encode(docs)
            except Exception:  # noqa: BLE001
                return results
        except Exception:  # noqa: BLE001
            return results
        scored = []
        for item, emb in zip(results, d_embs):
            score = float(
                (q_emb @ emb)
                / (((q_emb @ q_emb) ** 0.5) * ((emb @ emb) ** 0.5) + 1e-8)
            )
            enriched = dict(item)
            enriched["embedding_score"] = score
            scored.append(enriched)
        return sorted(scored, key=lambda x: x.get("embedding_score", 0.0), reverse=True)
