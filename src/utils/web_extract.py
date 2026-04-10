from bs4 import BeautifulSoup

def meta_content(soup: BeautifulSoup, *keys):
    for key in keys:
        tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return ""

def split_keywords(value: str):
    return [x.strip() for x in (value or "").split(",") if x.strip()]
