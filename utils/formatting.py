import re

def s(obj):
    try:
        return str(obj)
    except Exception:
        return repr(obj)

def truncate(text, max_len=4000):
    text = s(text)
    return text if len(text) <= max_len else text[:max_len] + "\n...[TRUNCATED]..."

def format_elapsed(seconds: float) -> str:
    if seconds is None:
        return "-"
    if seconds < 1:
        return f"{seconds:.2f}s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds % 60
    return f"{minutes}m {rem:.1f}s"

def sanitize_filename(name: str, max_len: int = 120) -> str:
    name = (name or "slurped-page").strip().lower()
    name = re.sub(r"[\\/:*?\"<>|]+", "-", name)
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-.")
    return (name or "slurped-page")[:max_len].rstrip("-.")
