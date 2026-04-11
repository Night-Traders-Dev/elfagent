import os
import re

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name)

def slurp_url(url: str, output_dir: str = "slurps") -> str:
    os.makedirs(output_dir, exist_ok=True)
    
    from tools.browser_tools import browser_extract
    text_content = browser_extract(url)
    
    if text_content.startswith("Error") or text_content.startswith("Browser extraction failed"):
        return f"Failed to slurp {url}: {text_content}"
        
    safe_name = sanitize_filename(url) + ".txt"
    out_path = os.path.join(output_dir, safe_name)
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n\n")
        f.write(text_content)
        
    preview = text_content[:400] + "..." if len(text_content) > 400 else text_content
    return f"Successfully slurped via Browser to {out_path}.\nPreview:\n{preview}"

def slurp_to_obsidian(url: str) -> str:
    return "Skipped for headless."
