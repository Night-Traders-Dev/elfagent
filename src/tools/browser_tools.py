from __future__ import annotations

def browser_extract(url: str) -> str:
    """Fetch the fully rendered text content of a webpage using a headless browser."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "Error: playwright is not installed."

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=20000, wait_until="domcontentloaded")
            page.wait_for_timeout(2000) 
            text = page.evaluate("document.body.innerText")
            browser.close()

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            cleaned = "
".join(lines)
            if len(cleaned) > 15000:
                cleaned = cleaned[:15000] + "

[... truncated ...]"
            return cleaned
    except Exception as exc:
        return f"Browser extraction failed: {exc}"
