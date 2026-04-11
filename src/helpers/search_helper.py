import re

from core.config import AUTORESEARCH_MAX_QUERIES_PER_ROUND, HELPER_MODEL
from core.turboquant import TurboQuantCompressor
from helpers.reranker import ResultReranker

# EmbeddingReranker is imported lazily so that merely constructing a
# SearchHelper does not trigger a SentenceTransformer / CUDA initialisation.

_WORD_RE = re.compile(r"[a-z0-9_]{3,}")
_STOPWORDS = {
    "about", "after", "again", "also", "because", "could", "does", "from",
    "have", "into", "just", "only", "other", "over", "some", "than", "that",
    "their", "there", "these", "they", "this", "what", "when", "where",
    "which", "with", "would",
}


class SearchHelper:
    def __init__(self):
        self.model_name = HELPER_MODEL
        self.reranker = ResultReranker()
        self._embedding_reranker = None  # lazy — loaded on first rerank call
        self.turboquant = TurboQuantCompressor()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embedding_reranker(self):
        """Return the EmbeddingReranker, constructing it on first access."""
        if self._embedding_reranker is None:
            from helpers.embedding_reranker import EmbeddingReranker  # noqa: PLC0415
            self._embedding_reranker = EmbeddingReranker()
        return self._embedding_reranker

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def salient_terms(self, text: str, limit: int = 6) -> list[str]:
        seen: list[str] = []
        for token in _WORD_RE.findall((text or "").lower()):
            if token in _STOPWORDS or token in seen:
                continue
            seen.append(token)
            if len(seen) >= limit:
                break
        return seen

    def build_queries(
        self,
        query: str,
        missing_terms: list[str] | None = None,
        evidence: list[dict] | None = None,
        limit: int = AUTORESEARCH_MAX_QUERIES_PER_ROUND,
    ) -> list[str]:
        base = query.strip()
        if not base:
            return []

        lowered = base.lower()
        queries = [base]

        if any(t in lowered for t in ["latest", "news", "today", "recent", "current"]):
            queries.extend([f"{base} official update", f"{base} timeline"])
        elif any(t in lowered for t in ["docs", "documentation", "api", "sdk", "reference"]):
            queries.extend([f"{base} official docs", f"{base} reference"])
        elif any(t in lowered for t in ["compare", "comparison", "versus", " vs "]):
            queries.extend([f"{base} benchmark", f"{base} differences"])
        else:
            queries.extend([f"{base} overview", f"{base} examples"])

        if missing_terms:
            gap_phrase = " ".join(missing_terms[:2]).strip()
            if gap_phrase:
                queries.insert(1, f"{base} {gap_phrase}")
                queries.append(f"{gap_phrase} {base}")

        if evidence:
            domains = [item.get("domain", "") for item in evidence if item.get("domain")]
            if domains:
                queries.append(f"{base} site:{domains[0]}")

        deduped: list[str] = []
        seen_set: set[str] = set()
        for item in queries:
            normalized = " ".join(item.split())
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen_set:
                continue
            seen_set.add(key)
            deduped.append(normalized)
        return deduped[:limit]

    def compress_results(self, query: str, results, limit: int = 5) -> list[dict]:
        ranked = self.reranker.rerank(query, results)
        ranked = self._get_embedding_reranker().rerank(query, ranked)
        compressed = []
        for item in ranked[:limit]:
            compressed.append({
                "title": item.get("title") or item.get("name") or "",
                "url": item.get("href") or item.get("url") or "",
                "domain": item.get("domain") or "",
                "snippet": self.turboquant.compress_text(
                    (item.get("body") or item.get("content") or item.get("snippet") or "")[:1000],
                    query=query,
                ),
                "score": item.get("score", 0.0),
                "embedding_score": item.get("embedding_score", 0.0),
            })
        return compressed

    def merge_evidence(
        self, query: str, existing: list[dict], new_items: list[dict], limit: int = 6
    ) -> list[dict]:
        merged = list(existing) + list(new_items)
        reranked = self.reranker.rerank(query, merged)
        reranked = self._get_embedding_reranker().rerank(query, reranked)
        compacted = self.turboquant.compress_evidence(query, reranked, max_items=limit)
        return compacted[:limit]

    def evaluate_coverage(self, query: str, evidence: list[dict]) -> dict:
        query_terms = self.salient_terms(query)
        if not query_terms:
            return {
                "coverage_score": 1.0,
                "covered_terms": [],
                "missing_terms": [],
                "notes": "No salient query terms required explicit coverage.",
            }

        evidence_text = " ".join([
            " ".join([
                item.get("title") or "",
                item.get("snippet") or "",
                item.get("url") or "",
            ])
            for item in evidence
        ]).lower()

        covered_terms = [t for t in query_terms if t in evidence_text]
        missing_terms = [t for t in query_terms if t not in evidence_text]
        coverage_score = round(len(covered_terms) / len(query_terms), 2)
        return {
            "coverage_score": coverage_score,
            "covered_terms": covered_terms,
            "missing_terms": missing_terms,
            "notes": (
                f"Covered {len(covered_terms)} of {len(query_terms)} salient query terms "
                f"across {len(evidence)} evidence items."
            ),
        }
