from urllib.parse import urlparse


class ResultReranker:
    def dedupe(self, results):
        seen = set()
        deduped = []
        for item in results:
            url = item.get("url") or item.get("href") or ""
            title = item.get("title") or item.get("name") or ""
            key = (url.strip().lower(), title.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def score(self, query: str, item: dict) -> float:
        q_terms = {x for x in query.lower().split() if x}
        text = " ".join([
            item.get("title") or item.get("name") or "",
            item.get("snippet") or item.get("body") or item.get("content") or "",
            item.get("url") or item.get("href") or "",
        ]).lower()
        overlap = sum(1 for t in q_terms if t in text)
        domain_bonus = 0.2 if urlparse(item.get("url") or item.get("href") or "").netloc else 0.0
        return overlap + domain_bonus

    def rerank(self, query: str, results):
        deduped = self.dedupe(results)
        ranked = sorted(deduped, key=lambda r: self.score(query, r), reverse=True)
        return ranked
