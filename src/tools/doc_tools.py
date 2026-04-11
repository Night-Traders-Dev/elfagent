from __future__ import annotations

from pathlib import Path


def _open_pdf(path: str):
    try:
        import fitz  # noqa: PLC0415
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "PyMuPDF is not installed. Add it to the environment or install requirements.txt."
        ) from exc

    pdf_path = Path(path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    return fitz.open(pdf_path), pdf_path


def parse_pdf(path: str, max_pages: int | None = None, max_chars_per_page: int = 4000) -> str:
    """Extract text from a PDF and annotate each section with page numbers."""
    try:
        document, pdf_path = _open_pdf(path)
    except Exception as exc:  # noqa: BLE001
        return f"Error parsing PDF {path}: {exc}"

    pages: list[str] = []
    try:
        total_pages = len(document)
        limit = min(total_pages, max_pages) if max_pages else total_pages
        for page_index in range(limit):
            page = document.load_page(page_index)
            text = (page.get_text("text") or "").strip()
            if len(text) > max_chars_per_page:
                text = text[:max_chars_per_page].rstrip() + "\n[... page text truncated ...]"
            if not text:
                text = "[No extractable text on this page.]"
            pages.append(f"--- Page {page_index + 1} ---\n{text}")
    finally:
        document.close()

    return (
        f"PDF text for {pdf_path} ({len(pages)} page(s) extracted):\n\n"
        + "\n\n".join(pages)
    )


def extract_pdf_tables(path: str, max_pages: int | None = None, max_tables_per_page: int = 5) -> str:
    """Extract tables from a PDF via PyMuPDF's table detector."""
    try:
        document, pdf_path = _open_pdf(path)
    except Exception as exc:  # noqa: BLE001
        return f"Error extracting tables from PDF {path}: {exc}"

    blocks: list[str] = []
    try:
        total_pages = len(document)
        limit = min(total_pages, max_pages) if max_pages else total_pages
        for page_index in range(limit):
            page = document.load_page(page_index)
            if not hasattr(page, "find_tables"):
                return (
                    f"Table extraction is not available for {pdf_path}. "
                    "Upgrade PyMuPDF to a version that supports page.find_tables()."
                )
            finder = page.find_tables()
            tables = list(getattr(finder, "tables", [])[:max_tables_per_page])
            for table_index, table in enumerate(tables, start=1):
                rows = table.extract()
                if not rows:
                    continue
                rendered_rows = []
                for row in rows:
                    rendered_rows.append(" | ".join("" if cell is None else str(cell).strip() for cell in row))
                blocks.append(
                    f"--- Page {page_index + 1}, Table {table_index} ---\n"
                    + "\n".join(rendered_rows)
                )
    finally:
        document.close()

    if not blocks:
        return f"No tables detected in PDF {pdf_path}."
    return f"Extracted tables for {pdf_path}:\n\n" + "\n\n".join(blocks)
