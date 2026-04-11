from __future__ import annotations

import ast
import json
import re

_CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


class CriticReasoner:
    """Lightweight verifier for final answers."""

    def review(self, answer_text: str) -> dict:
        findings: list[str] = []
        for language, body in _CODE_BLOCK_RE.findall(answer_text or ""):
            lang = language.strip().lower()
            snippet = body.strip()
            if not snippet:
                continue
            if lang in {"python", "py"}:
                try:
                    ast.parse(snippet)
                except SyntaxError as exc:
                    findings.append(f"Python block may not parse: line {exc.lineno}: {exc.msg}")
            elif lang == "json":
                try:
                    json.loads(snippet)
                except json.JSONDecodeError as exc:
                    findings.append(f"JSON block may be invalid: line {exc.lineno}: {exc.msg}")

        return {
            "ok": not findings,
            "findings": findings,
        }
