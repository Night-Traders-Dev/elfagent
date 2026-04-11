from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.config import EMBEDDING_MODEL_ID, HF_CACHE_DIR, RAG_INDEX_PATH

_BLOCKED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "model_cache",
    "telemetry_logs",
    "benchmarks_out",
}
_DEFAULT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".s",
    ".asm",
    ".nim",
    ".rs",
}
_INDEX_CACHE: dict[str, dict] = {}
_EMBEDDING_MODEL = None


@dataclass
class IndexedChunk:
    path: str
    title: str
    text: str
    chunk_index: int
    start_line: int
    end_line: int

    def to_record(self) -> dict:
        return {
            "path": self.path,
            "title": self.title,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


def _safe_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _iter_text_files(root: Path, extensions: set[str]) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in _BLOCKED_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in extensions:
            continue
        files.append(path)
    return files


def _chunk_text(path: Path, text: str, chunk_size: int, overlap: int) -> list[IndexedChunk]:
    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[IndexedChunk] = []
    current_lines: list[str] = []
    current_chars = 0
    chunk_index = 0
    line_start = 1
    overlap_lines = max(overlap // 80, 1)

    for lineno, line in enumerate(lines, start=1):
        next_len = current_chars + len(line) + 1
        if current_lines and next_len > chunk_size:
            chunks.append(
                IndexedChunk(
                    path=str(path),
                    title=path.name,
                    text="\n".join(current_lines).strip(),
                    chunk_index=chunk_index,
                    start_line=line_start,
                    end_line=lineno - 1,
                )
            )
            chunk_index += 1
            current_lines = current_lines[-overlap_lines:] if overlap_lines else []
            line_start = max(lineno - len(current_lines), 1)
            current_chars = sum(len(item) + 1 for item in current_lines)
        current_lines.append(line)
        current_chars += len(line) + 1

    if current_lines:
        chunks.append(
            IndexedChunk(
                path=str(path),
                title=path.name,
                text="\n".join(current_lines).strip(),
                chunk_index=chunk_index,
                start_line=line_start,
                end_line=len(lines),
            )
        )
    return [chunk for chunk in chunks if chunk.text]


def _get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_ID, cache_folder=HF_CACHE_DIR)
    return _EMBEDDING_MODEL


def _encode_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _get_embedding_model()
    vectors = model.encode(texts, show_progress_bar=False)
    return [[float(value) for value in vector] for vector in vectors]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    numerator = sum(x * y for x, y in zip(a, b))
    a_norm = math.sqrt(sum(x * x for x in a))
    b_norm = math.sqrt(sum(y * y for y in b))
    if not a_norm or not b_norm:
        return 0.0
    return numerator / (a_norm * b_norm)


def _load_index(index_path: str | None = None) -> dict | None:
    resolved = str(Path(index_path or RAG_INDEX_PATH).expanduser().resolve())
    if resolved in _INDEX_CACHE:
        return _INDEX_CACHE[resolved]
    if not os.path.exists(resolved):
        return None
    with open(resolved, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _INDEX_CACHE[resolved] = payload
    return payload


def _save_index(index_payload: dict, index_path: str | None = None) -> str:
    resolved = Path(index_path or RAG_INDEX_PATH).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved, "w", encoding="utf-8") as handle:
        json.dump(index_payload, handle, ensure_ascii=False)
    _INDEX_CACHE[str(resolved)] = index_payload
    return str(resolved)


def index_directory(
    path: str,
    index_path: str | None = None,
    extensions: str | None = None,
    chunk_size: int = 1200,
    overlap: int = 160,
    max_files: int = 500,
) -> str:
    """Build a local semantic index for source code and docs under a directory."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return f"Error: directory not found: {path}"
    if not root.is_dir():
        return f"Error: path is not a directory: {path}"

    extension_set = {
        item.strip().lower()
        for item in (extensions.split(",") if extensions else [])
        if item.strip()
    } or _DEFAULT_EXTENSIONS
    files = _iter_text_files(root, extension_set)[:max_files]
    chunks: list[IndexedChunk] = []
    for file_path in files:
        try:
            chunks.extend(_chunk_text(file_path, _safe_text(file_path), chunk_size, overlap))
        except Exception:
            continue

    if not chunks:
        return f"No indexable text files found under {root}."

    try:
        embeddings = _encode_texts([chunk.text for chunk in chunks])
    except Exception as exc:  # noqa: BLE001
        return f"Error building semantic index for {root}: {exc}"
    payload = {
        "version": 1,
        "root_path": str(root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": EMBEDDING_MODEL_ID,
        "documents": [chunk.to_record() for chunk in chunks],
        "embeddings": embeddings,
    }
    resolved = _save_index(payload, index_path=index_path)
    return (
        f"Indexed {len(files)} files into {len(chunks)} semantic chunks.\n"
        f"Root: {root}\n"
        f"Index: {resolved}\n"
        f"Embedding model: {EMBEDDING_MODEL_ID}"
    )


def semantic_search_raw(query: str, top_k: int = 5, index_path: str | None = None) -> list[dict]:
    """Return structured local semantic matches for a query."""
    payload = _load_index(index_path=index_path)
    if not payload or not payload.get("documents"):
        return []

    query_vector = _encode_texts([query])[0]
    results: list[dict] = []
    for document, embedding in zip(payload["documents"], payload["embeddings"]):
        score = _cosine_similarity(query_vector, embedding)
        results.append(
            {
                "title": document.get("title") or Path(document["path"]).name,
                "url": document["path"],
                "path": document["path"],
                "domain": "local",
                "snippet": document.get("text", ""),
                "score": round(score, 4),
                "chunk_index": document.get("chunk_index", 0),
                "start_line": document.get("start_line", 1),
                "end_line": document.get("end_line", 1),
            }
        )
    results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return results[:top_k]


def semantic_search(query: str, top_k: int = 5, index_path: str | None = None) -> str:
    """Search the local semantic index and return formatted matches."""
    try:
        results = semantic_search_raw(query, top_k=top_k, index_path=index_path)
    except Exception as exc:  # noqa: BLE001
        return f"Error performing semantic search: {exc}"
    if not results:
        return (
            "No local semantic index results were found. "
            "Build one first with index_directory(path)."
        )

    lines = [f"Local semantic search results for: {query}\n"]
    for idx, item in enumerate(results, start=1):
        lines.append(
            f"{idx}. {item['title']} "
            f"(score={item['score']:.3f}, lines {item['start_line']}-{item['end_line']})"
        )
        lines.append(f"   Path: {item['path']}")
        lines.append(f"   {item['snippet'][:320].replace(chr(10), ' ')}")
        lines.append("")
    return "\n".join(lines)
