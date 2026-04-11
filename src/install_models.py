from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

from core.config import (
    EMBEDDING_MODEL_ID,
    HF_CACHE_DIR,
    MEDICAL_MODEL_ID,
    MEETING_MODEL_ID,
    USER_AGENT,
)

# Files required per loader kind – avoids pulling entire repo snapshots.
_ALLOW_PATTERNS: dict[str, list[str]] = {
    "transformers_seq2seq": [
        "config.json",
        "tokenizer*.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "spiece.model",
        "vocab.json",
        "merges.txt",
        "*.safetensors",
        "*.bin",
        "generation_config.json",
    ],
    "sentence_transformer": [
        "config.json",
        "tokenizer*.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "sentence_bert_config.json",
        "modules.json",
        "*.safetensors",
        "*.bin",
        "1_Pooling/config.json",
    ],
}

_CACHE_POINTER_FILE = ".elfagent_cache"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    loader_kind: str
    description: str


def build_model_specs() -> tuple[ModelSpec, ...]:
    return (
        ModelSpec(
            name="medical_summary",
            repo_id=MEDICAL_MODEL_ID,
            loader_kind="transformers_seq2seq",
            description="medical summarization bundle",
        ),
        ModelSpec(
            name="meeting_summary",
            repo_id=MEETING_MODEL_ID,
            loader_kind="transformers_seq2seq",
            description="meeting summarization bundle",
        ),
        ModelSpec(
            name="embedding_reranker",
            repo_id=EMBEDDING_MODEL_ID,
            loader_kind="sentence_transformer",
            description="embedding reranker bundle",
        ),
    )


def configure_cache_env(cache_dir: str) -> str:
    """Set env vars *and* write a pointer file so runtime config picks up the dir."""
    resolved = os.path.abspath(cache_dir)
    os.makedirs(resolved, exist_ok=True)
    for var in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME"):
        os.environ[var] = resolved
    # Write a pointer that core/config.py can read on next launch
    try:
        pointer = os.path.join(os.path.dirname(os.path.abspath(__file__)), _CACHE_POINTER_FILE)
        with open(pointer, "w", encoding="utf-8") as fh:
            fh.write(resolved)
    except OSError:
        pass
    return resolved


def snapshot_model(
    spec: ModelSpec,
    cache_dir: str,
    force_download: bool,
    local_files_only: bool,
) -> str:
    from huggingface_hub import snapshot_download

    allow = _ALLOW_PATTERNS.get(spec.loader_kind)
    return snapshot_download(
        repo_id=spec.repo_id,
        repo_type="model",
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        user_agent=USER_AGENT,
        allow_patterns=allow,
    )


def verify_model(spec: ModelSpec, cache_dir: str) -> None:
    if spec.loader_kind == "transformers_seq2seq":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        AutoTokenizer.from_pretrained(spec.repo_id, cache_dir=cache_dir, local_files_only=True)
        AutoModelForSeq2SeqLM.from_pretrained(spec.repo_id, cache_dir=cache_dir, local_files_only=True)
        return

    if spec.loader_kind == "sentence_transformer":
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(spec.repo_id, cache_folder=cache_dir, local_files_only=True)
        return

    raise ValueError(f"Unknown loader kind: {spec.loader_kind}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch the Hugging Face models used by ElfAgent.",
    )
    parser.add_argument(
        "--cache-dir",
        default=HF_CACHE_DIR,
        help=(
            "Directory for Hugging Face and Sentence Transformers caches. "
            "Also writes a .elfagent_cache file so the runtime uses the same path "
            "without needing exported env vars."
        ),
    )
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download even if cached.")
    parser.add_argument("--local-files-only", action="store_true",
                        help="No network requests; verify local files only.")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip loader verification after download.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = configure_cache_env(args.cache_dir)
    specs = build_model_specs()

    print(f"Using Hugging Face cache: {cache_dir}")
    print(f"Preparing {len(specs)} model bundles...")

    failures: list[str] = []
    for spec in specs:
        print(f"- Fetching {spec.repo_id} ({spec.description})")
        try:
            snapshot_path = snapshot_model(
                spec,
                cache_dir=cache_dir,
                force_download=args.force_download,
                local_files_only=args.local_files_only,
            )
            print(f"  snapshot: {snapshot_path}")
            if not args.skip_verify:
                verify_model(spec, cache_dir)
                print("  verified with runtime loader")
        except Exception as exc:
            failures.append(f"{spec.repo_id}: {exc}")
            print(f"  failed: {exc}", file=sys.stderr)

    if failures:
        print("\nModel install failed for:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("All required model bundles are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
