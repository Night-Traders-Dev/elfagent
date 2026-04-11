from __future__ import annotations

import json
from pathlib import Path

import yaml


def parse_structured_file(path: str, max_chars: int = 16000) -> str:
    """Parse JSON, TOML, YAML, or CBOR into a normalized JSON view."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        return f"Error: structured file not found: {path}"

    suffix = file_path.suffix.lower()
    raw = file_path.read_bytes()
    try:
        if suffix == ".json":
            data = json.loads(raw.decode("utf-8"))
        elif suffix == ".toml":
            import tomllib  # noqa: PLC0415

            data = tomllib.loads(raw.decode("utf-8"))
        elif suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(raw.decode("utf-8"))
        elif suffix == ".cbor":
            try:
                import cbor2  # noqa: PLC0415
            except ImportError as exc:  # noqa: BLE001
                return (
                    "Error: cbor2 is required to parse CBOR files. "
                    "Install it if you want CBOR support."
                )
            data = cbor2.loads(raw)
        else:
            return (
                "Unsupported structured file type. "
                "Use JSON, TOML, YAML, or CBOR."
            )
    except Exception as exc:  # noqa: BLE001
        return f"Error parsing structured file {path}: {exc}"

    rendered = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars] + "\n[... structured output truncated ...]"
    return f"Parsed structured file {file_path}:\n{rendered}"
