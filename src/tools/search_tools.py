import urllib.parse
import logging

logger = logging.getLogger(__name__)


class SearchError(Exception):
    pass


class MultiEngineSearch:
    def _playwright(self, query: str, max_results: int = 8) -> list[dict]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise SearchError("Playwright is not installed. Run: pip install playwright && playwright install chromium")

        results = []
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]
            )
            try:
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    viewport={"width": 1920, "height": 1080}
                )
                page = context.new_page()

                url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
                page.goto(url, timeout=15000)

                try:
                    page.wait_for_selector(".result__body", timeout=10000)
                except Exception:
                    logger.warning("Timed out waiting for .result__body — page may have been blocked or is empty.")
                    return results

                from bs4 import BeautifulSoup
                soup = BeautifulSoup(page.content(), "lxml")

                for div in soup.select(".result__body")[:max_results]:
                    a = div.select_one(".result__a")
                    snip = div.select_one(".result__snippet")
                    if not (a and a.get("href")):
                        continue

                    href = a.get("href")

                    if href.startswith("//duckduckgo.com/l/"):
                        params = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                        uddg = params.get("uddg", [""])
                        href = urllib.parse.unquote(uddg[0]) if uddg[0] else ""

                    if not href.startswith("http"):
                        continue

                    results.append({
                        "title": a.get_text(strip=True),
                        "url": href,
                        "snippet": snip.get_text(strip=True) if snip else "",
                        "domain": urllib.parse.urlparse(href).netloc,
                        "engine": "ddg_playwright",
                    })
            finally:
                browser.close()

        return results

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        return self._playwright(query, max_results)


_searcher: MultiEngineSearch | None = None


def _get_searcher() -> MultiEngineSearch:
    global _searcher
    if _searcher is None:
        _searcher = MultiEngineSearch()
    return _searcher


def web_search(query: str, max_results: int = 5) -> str:
    try:
        results = _get_searcher().search(query, max_results)
    except SearchError as e:
        return f"Search error: {e}"
    except Exception as e:
        logger.exception("Unexpected error during web_search")
        return f"Search failed unexpectedly: {e}"

    if not results:
        return "Search returned no results. The headless browser may have been blocked."

    lines = []
    for r in results:
        lines.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}\n---")
    return "\n".join(lines)
