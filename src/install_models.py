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
    resolved = os.path.abspath(cache_dir)
    os.makedirs(resolved, exist_ok=True)
    os.environ["HF_HOME"] = resolved
    os.environ["HF_HUB_CACHE"] = resolved
    os.environ["TRANSFORMERS_CACHE"] = resolved
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = resolved
    return resolved


def snapshot_model(spec: ModelSpec, cache_dir: str, force_download: bool, local_files_only: bool) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=spec.repo_id,
        repo_type="model",
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        user_agent=USER_AGENT,
    )


def verify_model(spec: ModelSpec, cache_dir: str) -> None:
    if spec.loader_kind == "transformers_seq2seq":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        AutoTokenizer.from_pretrained(
            spec.repo_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        AutoModelForSeq2SeqLM.from_pretrained(
            spec.repo_id,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        return

    if spec.loader_kind == "sentence_transformer":
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(
            spec.repo_id,
            cache_folder=cache_dir,
            local_files_only=True,
        )
        return

    raise ValueError(f"Unknown loader kind: {spec.loader_kind}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch the Hugging Face models used by ElfAgent.",
    )
    parser.add_argument(
        "--cache-dir",
        default=HF_CACHE_DIR,
        help="Directory to use for Hugging Face and Sentence Transformers caches.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download model snapshots even if they already exist in cache.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not make network requests; verify that all required files already exist locally.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip loader verification after downloading model snapshots.",
    )
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
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("All required model bundles are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
