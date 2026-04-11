import json
import os
import re
from datetime import datetime, timezone

from core.config import (
    ENABLE_ENGRAMS,
    ENGRAM_MAX_GIST_CHARS,
    ENGRAM_MAX_RECORDS,
    ENGRAM_PATH,
    ENGRAM_RETRIEVAL_LIMIT,
)
from core.turboquant import TurboQuantCompressor

_WORD_RE = re.compile(r"[a-z0-9_]{3,}")
_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "could",
    "from",
    "have",
    "just",
    "only",
    "other",
    "than",
    "that",
    "their",
    "there",
    "these",
    "they",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}


def _terms(text: str, limit: int = 8) -> list[str]:
    seen = []
    for token in _WORD_RE.findall((text or "").lower()):
        if token in _STOPWORDS or token in seen:
            continue
        seen.append(token)
        if len(seen) >= limit:
            break
    return seen


class EngramStore:
    def __init__(
        self,
        path: str = ENGRAM_PATH,
        enabled: bool = ENABLE_ENGRAMS,
        max_records: int = ENGRAM_MAX_RECORDS,
        retrieval_limit: int = ENGRAM_RETRIEVAL_LIMIT,
        max_gist_chars: int = ENGRAM_MAX_GIST_CHARS,
    ):
        self.path = path
        self.enabled = enabled
        self.max_records = max_records
        self.retrieval_limit = retrieval_limit
        self.max_gist_chars = max_gist_chars
        self.compressor = TurboQuantCompressor()

    def _ensure_parent_dir(self):
        parent = os.path.dirname(self.path) or "."
        os.makedirs(parent, exist_ok=True)

    def _load_records(self) -> list[dict]:
        if not self.enabled or not os.path.exists(self.path):
            return []

        records = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records[-self.max_records:]

    def _rewrite_records(self, records: list[dict]):
        self._ensure_parent_dir()
        with open(self.path, "w", encoding="utf-8") as handle:
            for record in records[-self.max_records:]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _should_store(self, user_msg: str, assistant_msg: str) -> bool:
        if not self.enabled:
            return False
        if not user_msg or user_msg.strip().startswith("/"):
            return False
        if len(user_msg.strip()) < 8:
            return False
        if not assistant_msg or assistant_msg.strip().startswith("(Agent stopped early:"):
            return False
        return True

    def _build_gist(self, user_msg: str, assistant_msg: str) -> str:
        user_part = self.compressor.compress_text(user_msg, budget=140, query=user_msg)
        answer_budget = max(self.max_gist_chars - len(user_part) - 18, 80)
        answer_part = self.compressor.compress_text(assistant_msg, budget=answer_budget, query=user_msg)
        return f"User: {user_part} | Outcome: {answer_part}"

    def remember(
        self,
        user_msg: str,
        assistant_msg: str,
        route: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        if not self._should_store(user_msg, assistant_msg):
            return None

        record = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "route": route or "unknown",
            "query": self.compressor.compress_text(user_msg, budget=160, query=user_msg),
            "gist": self._build_gist(user_msg, assistant_msg),
            "terms": _terms(f"{user_msg} {assistant_msg}"),
            "metadata": metadata or {},
        }

        records = self._load_records()
        records.append(record)
        self._rewrite_records(records)
        return record

    def retrieve(self, query: str, limit: int | None = None) -> list[dict]:
        if not self.enabled or not query:
            return []

        records = self._load_records()
        if not records:
            return []

        query_terms = set(_terms(query))
        scored = []
        total_records = max(len(records), 1)
        for idx, record in enumerate(records):
            record_terms = set(record.get("terms") or _terms(record.get("gist", "")))
            overlap = query_terms & record_terms
            base_score = len(overlap) / max(len(query_terms), 1)
            text = " ".join([record.get("query", ""), record.get("gist", "")]).lower()
            phrase_bonus = 0.15 if query.lower() in text else 0.0
            recency_bonus = ((idx + 1) / total_records) * 0.25
            score = round(base_score + phrase_bonus + recency_bonus, 4)
            if not overlap and score < 0.2:
                continue

            enriched = dict(record)
            enriched["score"] = score
            enriched["matched_terms"] = sorted(overlap)
            scored.append(enriched)

        scored.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return scored[:(limit or self.retrieval_limit)]

    def format_for_prompt(self, matches: list[dict]) -> str:
        if not matches:
            return ""

        lines = []
        for match in matches:
            route = match.get("route", "unknown")
            score = match.get("score", 0.0)
            gist = match.get("gist", "")
            lines.append(f"- [{route} score={score:.2f}] {gist}")
        return (
            "Relevant engram memory is below. Use it only when it helps the current request.\n"
            + "\n".join(lines)
        )
