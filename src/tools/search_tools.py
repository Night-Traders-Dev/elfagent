import re
import urllib.parse

class MultiEngineSearch:
    def __init__(self):
        self._ENGINE_ORDER = ["playwright"]

    def _playwright(self, query: str, max_results: int = 8) -> list[dict]:
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
                        href = a.get("href")
                        if href.startswith("/url?q="):
                            href = href.split("/url?q=")[1].split("&")[0]

                        title_el = a.select_one("h3")
                        title = title_el.get_text(strip=True) if title_el else a.get_text(strip=True)

                        if href and href.startswith("http"):
                            results.append({
                                "title": title,
                                "url": urllib.parse.unquote(href),
                                "snippet": snip.get_text(strip=True) if snip else "",
                                "domain": urllib.parse.urlparse(href).netlloc,
                                "engine": "google_playwright"
                            })
                browser.close()
        except Exception as e:
            print(f"Playwright error: {e}")
        return results


    def search(self, query: str, max_results: int = 5) -> list[dict]:
        return self._playwright(query, max_results)

# Global singleton
_searcher = MultiEngineSearch()

def web_search(query: str, max_results: int = 5) -> str:
    """Search the web. ONLY USES GOOGLE IN HEADLESS BROWSER NOW."""
    results = _searcher.search(query, max_results)
    
    if not results:
        return "Search returned no results."
        
    out = []
    for r in results:
        out.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}\n---")
    return "\n".join(out)

def wikipedia_search(query: str, max_results: int = 3) -> str:
    return "Disabled."
    
def brave_search(query: str, max_results: int = 5) -> str:
    return web_search(query, max_results)
