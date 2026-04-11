import copy
import json
import re

from core.config import (
    ENABLE_TURBOQUANT,
    TURBOQUANT_MAX_EVIDENCE_ITEMS,
    TURBOQUANT_MAX_PACKET_CHARS,
    TURBOQUANT_MAX_SNIPPET_CHARS,
)

_WORD_RE = re.compile(r"[a-z0-9_]{3,}")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "between",
    "could",
    "does",
    "from",
    "have",
    "into",
    "just",
    "more",
    "most",
    "only",
    "other",
    "over",
    "some",
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


def _query_terms(text: str) -> set[str]:
    return {
        token for token in _WORD_RE.findall((text or "").lower())
        if token not in _STOPWORDS
    }


class TurboQuantCompressor:
    def __init__(
        self,
        enabled: bool = ENABLE_TURBOQUANT,
        max_snippet_chars: int = TURBOQUANT_MAX_SNIPPET_CHARS,
        max_packet_chars: int = TURBOQUANT_MAX_PACKET_CHARS,
        max_evidence_items: int = TURBOQUANT_MAX_EVIDENCE_ITEMS,
    ):
        self.enabled = enabled
        self.max_snippet_chars = max_snippet_chars
        self.max_packet_chars = max_packet_chars
        self.max_evidence_items = max_evidence_items

    def measure(self, value) -> int:
        if isinstance(value, str):
            return len(value)
        try:
            return len(json.dumps(value, ensure_ascii=False, sort_keys=True))
        except TypeError:
            return len(str(value))

    def compress_text(self, text: str, budget: int | None = None, query: str | None = None) -> str:
        normalized = " ".join((text or "").split())
        if not normalized:
            return ""

        target = budget or self.max_snippet_chars
        if not self.enabled or len(normalized) <= target:
            return normalized

        pieces = [piece.strip() for piece in _SENTENCE_RE.split(normalized) if piece.strip()]
        if not pieces:
            return normalized[:max(target - 3, 0)].rstrip() + "..."

        terms = _query_terms(query or "")
        scored = []
        for idx, piece in enumerate(pieces):
            lowered = piece.lower()
            hits = sum(1 for term in terms if term in lowered)
            scored.append((hits, -idx, piece))

        selected = []
        selected_set = set()
        used_chars = 0
        for _, _, piece in sorted(scored, reverse=True):
            extra = len(piece) + (1 if selected else 0)
            if selected and used_chars + extra > target:
                continue
            selected.append(piece)
            selected_set.add(piece)
            used_chars += extra
            if used_chars >= int(target * 0.8):
                break

        if not selected:
            return normalized[:max(target - 3, 0)].rstrip() + "..."

        compacted = " ".join(piece for piece in pieces if piece in selected_set).strip()
        if len(compacted) <= target:
            return compacted
        return compacted[:max(target - 3, 0)].rstrip() + "..."

    def compress_evidence(self, query: str, evidence: list[dict], max_items: int | None = None) -> list[dict]:
        if not evidence:
            return []

        limit = max_items or self.max_evidence_items
        compacted = []
        for item in evidence[:limit]:
            entry = dict(item)
            entry["snippet"] = self.compress_text(entry.get("snippet", ""), query=query)
            compacted.append(entry)
        return compacted

    def compact_packet(self, packet: dict) -> tuple[dict, dict]:
        original_chars = self.measure(packet)
        if not self.enabled or original_chars <= self.max_packet_chars:
            return packet, {
                "enabled": self.enabled,
                "applied": False,
                "original_chars": original_chars,
                "final_chars": original_chars,
                "saved_chars": 0,
            }

        compacted = copy.deepcopy(packet)
        user_query = compacted.get("user_query", "")
        payload = compacted.setdefault("payload", {})

        if isinstance(payload.get("evidence"), list):
            payload["evidence"] = self.compress_evidence(user_query, payload["evidence"])

        if isinstance(payload.get("rounds"), list):
            trimmed_rounds = []
            for round_item in payload["rounds"]:
                entry = dict(round_item)
                if isinstance(entry.get("queries"), list):
                    entry["queries"] = entry["queries"][:2]
                if isinstance(entry.get("new_evidence"), list):
                    entry["new_evidence"] = self.compress_evidence(
                        user_query,
                        entry["new_evidence"],
                        max_items=2,
                    )
                if isinstance(entry.get("notes"), str):
                    entry["notes"] = self.compress_text(
                        entry["notes"],
                        budget=min(220, self.max_snippet_chars),
                        query=user_query,
                    )
                trimmed_rounds.append(entry)
            payload["rounds"] = trimmed_rounds

        if isinstance(payload.get("engram_matches"), list):
            trimmed_engrams = []
            for match in payload["engram_matches"][:3]:
                entry = {
                    "route": match.get("route", "unknown"),
                    "score": match.get("score", 0.0),
                    "created_at": match.get("created_at"),
                    "gist": self.compress_text(
                        match.get("gist", ""),
                        budget=min(180, self.max_snippet_chars),
                        query=user_query,
                    ),
                }
                trimmed_engrams.append(entry)
            payload["engram_matches"] = trimmed_engrams

        evaluation = payload.get("evaluation")
        if isinstance(evaluation, dict):
            if isinstance(evaluation.get("missing_terms"), list):
                evaluation["missing_terms"] = evaluation["missing_terms"][:4]
            if isinstance(evaluation.get("covered_terms"), list):
                evaluation["covered_terms"] = evaluation["covered_terms"][:6]
            if isinstance(evaluation.get("notes"), str):
                evaluation["notes"] = self.compress_text(
                    evaluation["notes"],
                    budget=min(220, self.max_snippet_chars),
                    query=user_query,
                )

        last_size = None
        while self.measure(compacted) > self.max_packet_chars:
            current_size = self.measure(compacted)
            if current_size == last_size:
                break
            last_size = current_size

            if isinstance(payload.get("rounds"), list) and len(payload["rounds"]) > 1:
                payload["rounds"] = payload["rounds"][:-1]
                continue
            if isinstance(payload.get("rounds"), list) and payload["rounds"]:
                active_round = payload["rounds"][-1]
                if isinstance(active_round.get("new_evidence"), list) and active_round["new_evidence"]:
                    active_round["new_evidence"] = active_round["new_evidence"][:-1]
                    continue
                if isinstance(active_round.get("result_counts"), dict) and len(active_round["result_counts"]) > 1:
                    first_query = next(iter(active_round["result_counts"]))
                    active_round["result_counts"] = {first_query: active_round["result_counts"][first_query]}
                    continue
                if isinstance(active_round.get("notes"), str) and len(active_round["notes"]) > 80:
                    active_round["notes"] = self.compress_text(
                        active_round["notes"],
                        budget=80,
                        query=user_query,
                    )
                    continue
                payload["rounds"] = payload["rounds"][:-1]
                continue
            if isinstance(payload.get("evidence"), list) and len(payload["evidence"]) > 3:
                payload["evidence"] = payload["evidence"][:-1]
                continue
            if isinstance(payload.get("evidence"), list) and len(payload["evidence"]) > 1:
                payload["evidence"] = payload["evidence"][:-1]
                continue
            if isinstance(payload.get("evidence"), list) and payload["evidence"]:
                snippet = payload["evidence"][0].get("snippet", "")
                if len(snippet) > 48:
                    payload["evidence"][0]["snippet"] = self.compress_text(
                        snippet,
                        budget=48,
                        query=user_query,
                    )
                    continue
            if isinstance(compacted.get("instructions"), list) and len(compacted["instructions"]) > 2:
                compacted["instructions"] = compacted["instructions"][:-1]
                continue
            if isinstance(compacted.get("user_query"), str) and len(compacted["user_query"]) > 120:
                compacted["user_query"] = self.compress_text(
                    compacted["user_query"],
                    budget=120,
                    query=user_query,
                )
                continue

        final_chars = self.measure(compacted)
        return compacted, {
            "enabled": self.enabled,
            "applied": True,
            "original_chars": original_chars,
            "final_chars": final_chars,
            "saved_chars": max(original_chars - final_chars, 0),
        }
