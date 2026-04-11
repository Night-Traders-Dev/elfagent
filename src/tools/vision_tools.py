from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import requests

from core.config import OLLAMA_BASE_URL, VISION_MODEL, VISION_OCR_MODEL


def _resolve_image(path: str) -> Path:
    image_path = Path(path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if not image_path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    mime, _ = mimetypes.guess_type(str(image_path))
    if mime and not mime.startswith("image/"):
        raise ValueError(f"Path does not look like an image file: {path}")
    return image_path


def _candidate_models(primary: str) -> list[str]:
    models = [primary, "moondream2", "moondream", "llava", "bakllava"]
    deduped: list[str] = []
    for model in models:
        if model and model not in deduped:
            deduped.append(model)
    return deduped


def _invoke_ollama_vision(path: str, prompt: str, primary_model: str) -> tuple[str, str]:
    image_path = _resolve_image(path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    last_error = "Unknown Ollama vision error."

    for model_name in _candidate_models(primary_model):
        try:
            response = requests.post(
                OLLAMA_BASE_URL.rstrip("/") + "/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "images": [encoded],
                    "stream": False,
                },
                timeout=180,
            )
            if response.status_code >= 400:
                last_error = f"{response.status_code}: {response.text[:400]}"
                continue
            payload = response.json()
            text = (payload.get("response") or "").strip()
            if text:
                return model_name, text
            last_error = "Ollama returned an empty vision response."
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

    raise RuntimeError(
        "Unable to complete the vision request via Ollama. "
        f"Tried models: {', '.join(_candidate_models(primary_model))}. Last error: {last_error}"
    )


def describe_image(path: str, prompt: str | None = None) -> str:
    """Describe the important technical details in a local image file."""
    vision_prompt = prompt or (
        "Describe this image with high technical fidelity. "
        "Call out visible labels, diagrams, connectors, pins, UI messages, "
        "measurements, and anything that would help with debugging or reverse engineering."
    )
    try:
        model_name, text = _invoke_ollama_vision(path, vision_prompt, VISION_MODEL)
        return f"Vision description ({model_name}) for {Path(path).name}:\n{text}"
    except Exception as exc:  # noqa: BLE001
        return f"Error describing image {path}: {exc}"


def ocr_image(path: str, prompt: str | None = None) -> str:
    """Extract visible text from a local image file using a vision-capable Ollama model."""
    ocr_prompt = prompt or (
        "Read all visible text in this image exactly when possible. "
        "Preserve line breaks, labels, table structure, headings, error codes, "
        "and any values that look like identifiers or measurements."
    )
    try:
        model_name, text = _invoke_ollama_vision(path, ocr_prompt, VISION_OCR_MODEL)
        return f"OCR result ({model_name}) for {Path(path).name}:\n{text}"
    except Exception as exc:  # noqa: BLE001
        return f"Error extracting text from image {path}: {exc}"
