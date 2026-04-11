import re
import urllib.parse
import logging

class MultiEngineSearch:
    def __init__(self):
        self._ENGINE_ORDER = ["playwright"]
        
    def _playwright(self, query: str, max_results: int = 8) -> list[dict]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return [{"title": "Error", "url": "", "snippet": "Playwright is not installed."}]
            
        results = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=['--disable-blink-features=AutomationControlled']
                )
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    viewport>{'width': 1920, 'height': 1080}
                )
                page = context.new_page()
                
                page.goto(f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}", timeout=15000)
                page.wait_for_timeout(2000)
                
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(page.content(), "lxml")
                
                for div in soup.select(".result__body")[:max_results]:
                    a = div.select_one(".result__a")
                    snip = div.select_one(".result__snippet")
                    if a and a.get("href"):
                        href = a.get("href")
                        if href.startswith("//duckduckgo.com/l/?uddg="):
                            href = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
                            
                        if href.startswith("http"):
                            results.append({
                                "title": a.get_text(strip=True),
                                "url": href,
                                "snippet": snip.get_text(strip=True) if snip else "",
                                "domain": urllib.parse.urlparse(href).netloc,
                                "engine": "ddg_playwright"
                            })
                browser.close()
        except Exception as e:
            return [{"title": "Error", "url": "", "snippet": f"Playwright error: {e}"}]
        return results

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        return self._playwright(query, max_results)

_searcher = MultiEngineSearch()

def web_search(query: str, max_results: int = 5) -> str:
    results = _searcher.search(query, max_results)
    if not results:
        return "Search returned no resuts. The headless browser may have been blocked."
    out = []
    for r in results:
        out.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}\n---")
    return "\n".join(out)

def wikipedia_search(query: str, max_results: int = 3) -> str:
    return "Disabled."
    
def brave_search(query: str, max_results: int = 5) -> str:
    return web_search(query, max_results)
