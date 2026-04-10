import os, sys, subprocess, requests, yaml
from datetime import datetime, timezone
from urllib.parse import quote, urlparse
from bs4 import BeautifulSoup
from readability import Document
from markdownify import markdownify as md
from core.config import USER_AGENT
from utils.formatting import sanitize_filename
from utils.web_extract import meta_content, split_keywords

def slurp_url(url: str, output_dir: str = "slurps") -> str:
    os.makedirs(output_dir, exist_ok=True)
    resp = requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})
    resp.raise_for_status()
    html = resp.text
    soup = BeautifulSoup(html, "lxml")
    doc = Document(html)
    title = doc.short_title() or meta_content(soup, "og:title", "twitter:title") or (soup.title.string.strip() if soup.title and soup.title.string else "") or "Untitled"
    readable_html = doc.summary(html_partial=True)
    markdown_body = md(readable_html, heading_style="ATX", bullets="-", strip=["script", "style"]).strip()
    parsed = urlparse(url)
    metadata = {
        "title": title, "source_url": url, "domain": parsed.netloc,
        "site_name": meta_content(soup, "og:site_name", "application-name"),
        "author": meta_content(soup, "author", "article:author", "twitter:creator"),
        "description": meta_content(soup, "description", "og:description", "twitter:description"),
        "published": meta_content(soup, "article:published_time", "pubdate", "date"),
        "modified": meta_content(soup, "article:modified_time", "lastmod"),
        "canonical": ((soup.find("link", rel="canonical").get("href").strip()) if soup.find("link", rel="canonical") and soup.find("link", rel="canonical").get("href") else ""),
        "tags": split_keywords(meta_content(soup, "keywords")),
        "slurped_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata = {k: v for k, v in metadata.items() if v not in ("", [], None)}
    if not markdown_body:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        markdown_body = "\n\n".join(p for p in paragraphs if p).strip() or f"# {title}\n\nSource: {url}\n"
    basename = sanitize_filename(title)
    filename = os.path.join(output_dir, f"{basename}.md")
    if os.path.exists(filename):
        filename = os.path.join(output_dir, f"{basename}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("---\n" + yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False, default_flow_style=False).strip() + "\n---\n\n" + markdown_body + "\n")
    return f"Saved cleaned Markdown to {filename}"

def slurp_to_obsidian(url: str) -> str:
    uri = f"obsidian://slurp?url={quote(url, safe='')}"
    if sys.platform.startswith("linux"):
        subprocess.Popen(["xdg-open", uri])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", uri])
    elif os.name == "nt":
        os.startfile(uri)
    else:
        return f"Constructed Obsidian Slurp URI: {uri} (open it manually on this platform)"
    return f"Sent URL to Obsidian Slurp via {uri}"
