from __future__ import annotations

import json

import requests


def _parse_headers(headers) -> dict[str, str]:
    if headers is None:
        return {}
    if isinstance(headers, dict):
        return {str(key): str(value) for key, value in headers.items()}
    if isinstance(headers, str):
        stripped = headers.strip()
        if not stripped:
            return {}
        if stripped.startswith("{"):
            payload = json.loads(stripped)
            return {str(key): str(value) for key, value in payload.items()}
        parsed: dict[str, str] = {}
        for line in stripped.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip()
        return parsed
    raise ValueError("headers must be a dict, JSON string, or newline-delimited header block")


def http_request(
    method: str,
    url: str,
    headers=None,
    body: str | None = None,
    timeout: int = 30,
) -> str:
    """Make an HTTP request, including POST/PUT/PATCH calls to local or remote services."""
    try:
        parsed_headers = _parse_headers(headers)
        kwargs = {
            "method": method.upper(),
            "url": url,
            "headers": parsed_headers,
            "timeout": timeout,
        }
        if body is not None:
            if "application/json" in parsed_headers.get("Content-Type", "").lower():
                kwargs["json"] = json.loads(body)
            else:
                kwargs["data"] = body
        response = requests.request(**kwargs)
        response_text = response.text
        if len(response_text) > 4000:
            response_text = response_text[:4000] + "\n[... body truncated ...]"
        response_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() in {"content-type", "content-length", "server", "date"}
        }
        return (
            f"HTTP {response.request.method} {url}\n"
            f"Status: {response.status_code}\n"
            f"Headers: {json.dumps(response_headers, ensure_ascii=False)}\n\n"
            f"{response_text}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error performing HTTP request to {url}: {exc}"
