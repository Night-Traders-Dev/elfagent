"""Multi-engine web search with automatic rate-limit failover.

Engine priority order (highest to lowest):
  1. Brave Search  (free-tier API key via BRAVE_API_KEY env var; skipped if absent)
  2. SearXNG       (self-hosted; base URL via SEARXNG_URL env var; skipped if absent)
  3. DuckDuckGo    (no key needed; rate-limited)
  4. Wikipedia     (summary + search; no key needed)
  5. HTML scrape   (last-resort: fetches a Google search result page)

For each engine, an empty result list or an HTTP 429 / connection error causes
the next engine to be tried.  The first engine that returns at least one result
wins and the rest are skipped for that query.
"""
from __future__ import annotations

import os
import time
import logging
from typing import Any

import requests
from bs4 import BeautifulSoup

from core.config import USER_AGENT

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (all optional via env)
# ---------------------------------------------------------------------------
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
SEARXNG_URL = os.getenv("SEARXNG_URL", "")          # e.g. http://localhost:8080
SEARXNG_TIMEOUT = int(os.getenv("SEARXNG_TIMEOUT", "8"))
DDG_PAUSE = float(os.getenv("DDG_PAUSE", "1.5"))    # seconds between DDG calls
_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}

# ---------------------------------------------------------------------------
# Individual engine helpers
# ---------------------------------------------------------------------------

def _normalize(items: list[dict], engine: str) -> list[dict]:
    """Ensure every result dict has title / url / snippet / engine keys."""
    out = []
    for r in items:
        out.append({
            "title":   r.get("title") or r.get("name") or "",
            "url":     r.get("url") or r.get("href") or r.get("link") or "",
            "snippet": r.get("body") or r.get("snippet") or r.get("content") or r.get("extract") or "",
            "domain":  r.get("domain") or "",
            "engine":  engine,
        })
    return [r for r in out if r["url"]]


def _brave(query: str, max_results: int = 8) -> list[dict]:
    if not BRAVE_API_KEY:
        return []
    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                **_HEADERS,
                "Accept": "application/json",
                "X-Subscription-Token": BRAVE_API_KEY,
            },
            params={"q": query, "count": max_results, "text_decorations": False},
            timeout=10,
        )
        if resp.status_code == 429:
            log.warning("Brave: rate limited")
            return []
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("web", {}).get("results", [])
        return _normalize(raw, "brave")
    except Exception as exc:
        log.debug("Brave search error: %s", exc)
        return []


def _searxng(query: str, max_results: int = 8) -> list[dict]:
    if not SEARXNG_URL:
        return []
    try:
        resp = requests.get(
            SEARXNG_URL.rstrip("/") + "/search",
            params={"q": query, "format": "json", "engines": "google,bing,wikipedia"},
            headers=_HEADERS,
            timeout=SEARXNG_TIMEOUT,
        )
        if resp.status_code == 429:
            log.warning("SearXNG: rate limited")
            return []
        resp.raise_for_status()
        raw = resp.json().get("results", [])
        return _normalize(raw[:max_results], "searxng")
    except Exception as exc:
        log.debug("SearXNG error: %s", exc)
        return []


def _ddg(query: str, max_results: int = 8) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
        time.sleep(DDG_PAUSE)
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))
        return _normalize(raw, "ddg")
    except Exception as exc:
        msg = str(exc).lower()
        if "ratelimit" in msg or "202" in msg or "blocked" in msg:
            log.warning("DDG: rate limited")
        else:
            log.debug("DDG error: %s", exc)
        return []


def _wikipedia(query: str, max_results: int = 5) -> list[dict]:
    """Wikipedia search + page summary via the public REST API (no key needed)."""
    results: list[dict] = []
    try:
        # Search endpoint
        sresp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
                "utf8": 1,
            },
            headers=_HEADERS,
            timeout=8,
        )
        sresp.raise_for_status()
        hits = sresp.json().get("query", {}).get("search", [])
        for hit in hits[:max_results]:
            title = hit.get("title", "")
            snippet = BeautifulSoup(hit.get("snippet", ""), "lxml").get_text()
            url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
            results.append({"title": title, "url": url, "snippet": snippet,
                             "domain": "en.wikipedia.org", "engine": "wikipedia"})
    except Exception as exc:
        log.debug("Wikipedia search error: %s", exc)
    return results


def _scrape_fallback(query: str, max_results: int = 5) -> list[dict]:
    """Last resort: parse a DuckDuckGo HTML results page."""
    results: list[dict] = []
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={**_HEADERS, "Accept": "text/html"},
            timeout=12,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for div in soup.select("div.g")[:max_results]:
            a = div.select_one("a")
            snip = div.select_one("div.VwiC3b, div[data-sncf]")
            if not a:
                continue
            href = a.get("href", "")
            results.append({
                "title": a.select_one("h3").get_text(strip=True) if a.select_one("h3") else a.get_text(strip=True),
                "url": href,
                "snippet": snip.get_text(strip=True) if snip else "",
                "domain": "",
                "engine": "scrape_fallback",
            })
    except Exception as exc:
        log.debug("Scrape fallback error: %s", exc)
    return results


# ---------------------------------------------------------------------------
# Multi-engine facade
# ---------------------------------------------------------------------------


def _playwright(query: str, max_results: int = 8) -> list[dict]:
    """Headless browser fallback to bypass anti-bot protection."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return []

    results = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()
            page.goto(f"https://www.google.com/search?q={query}&hl=en", timeout=15000)
            page.wait_for_timeout(1500)

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(page.content(), "lxml")

            for div in soup.select("div.g")[:max_results]:
                a = div.select_one("a")
                snip = div.select_one("div.VwiC3b, div[data-sncf]")
                if a and a.get("href"):
                    results.append({
                        "title": a.select_one("h3").get_text(strip=True) if a.select_one("h3") else a.get_text(strip=True),
                        "url": a.get("href"),
                        "snippet": snip.get_text(strip=True) if snip else "",
                        "domain": "",
                        "engine": "playwright",
                    })
            browser.close()
    except Exception:
        pass
    return results

class MultiEngineSearch:
    """Try engines in priority order; return on first non-empty batch.

    Used by AutoResearchWorkflow._search_once and exposed as a FunctionTool
    for the ReAct agent.
    """

    _ENGINE_ORDER = ["playwright"]

    def search(self, query: str, max_results: int = 8) -> list[dict]:
        for engine in self._ENGINE_ORDER:
            fn = {
                "brave":          lambda q, n: _brave(q, n),
                "searxng":        lambda q, n: _searxng(q, n),
                "ddg":            lambda q, n: _ddg(q, n),
                "wikipedia":      lambda q, n: _wikipedia(q, n),
                "playwright":     lambda q, n: _playwright(q, n),
                "scrape_fallback": lambda q, n: _scrape_fallback(q, n),
            }[engine]
            results = fn(query, max_results)
            if results:
                log.debug("MultiEngineSearch: %s returned %d results for %r", engine, len(results), query)
                return results
        return []

    # Convenience alias used by AutoResearchWorkflow
    def duckduckgo_full_search(self, query: str, max_results: int = 8) -> list[dict]:
        """Drop-in replacement for the old DuckDuckGoSearchToolSpec method."""
        return self.search(query, max_results)


# ---------------------------------------------------------------------------
# Standalone FunctionTool-friendly wrappers
# ---------------------------------------------------------------------------

_mes = MultiEngineSearch()


def web_search(query: str, max_results: int = 8) -> str:
    """Search the web using multiple engines with automatic rate-limit failover.

    Tries Brave Search (if BRAVE_API_KEY is set), SearXNG (if SEARXNG_URL is
    set), DuckDuckGo, Wikipedia, and an HTML scrape fallback in that order.
    Returns a formatted string of results suitable for an LLM to read.
    """
    results = _mes.search(query, max_results)
    if not results:
        return f"No results found for: {query}"
    lines = [f"Web search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. [{r['engine']}] {r['title']}")
        lines.append(f"   URL: {r['url']}")
        if r["snippet"]:
            lines.append(f"   {r['snippet'][:300]}")
        lines.append("")
    return "\n".join(lines)


def wikipedia_search(query: str, max_results: int = 5) -> str:
    """Search Wikipedia and return article summaries."""
    results = _wikipedia(query, max_results)
    if not results:
        return f"No Wikipedia results found for: {query}"
    lines = [f"Wikipedia results for: {query}\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   URL: {r['url']}")
        if r["snippet"]:
            lines.append(f"   {r['snippet'][:400]}")
        lines.append("")
    return "\n".join(lines)


def brave_search(query: str, max_results: int = 8) -> str:
    """Search using Brave Search API (requires BRAVE_API_KEY env var).

    Returns an error string if the API key is not configured.
    """
    if not BRAVE_API_KEY:
        return "Brave Search is not configured. Set the BRAVE_API_KEY environment variable."
    results = _brave(query, max_results)
    if not results:
        return f"No Brave Search results for: {query}"
    lines = [f"Brave Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        lines.append(f"   URL: {r['url']}")
        if r["snippet"]:
            lines.append(f"   {r['snippet'][:300]}")
        lines.append("")
    return "\n".join(lines)
