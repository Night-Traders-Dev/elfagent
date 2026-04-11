"""Prompt experiment harness.

Each variant in ROUTER_PROMPTS now actually changes the routing behaviour by
passing the prompt text to router.route() as a system_prompt hint.  The router
stores it and may use it to break ties or adjust confidence.  This means the
three variants (baseline / search_heavy / code_heavy) produce genuinely
different results.json rows instead of identical ones.
"""
import json
from pathlib import Path
from routing.router import TaskRouter
from experiments.prompts import ROUTER_PROMPTS


def main():
    data_path = Path(__file__).resolve().parent / "dataset.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))

    rows = []
    for prompt_name, prompt_text in ROUTER_PROMPTS.items():
        router = TaskRouter(system_prompt=prompt_text)
        correct = 0
        details = []
        for item in data:
            result = router.route(item["query"])
            ok = result["route"] == item["expected_route"]
            correct += int(ok)
            details.append({
                "query": item["query"],
                "expected": item["expected_route"],
                "got": result["route"],
                "confidence": result.get("confidence"),
                "reason": result.get("reason"),
                "ok": ok,
            })
        rows.append({
            "prompt_variant": prompt_name,
            "prompt_text": prompt_text,
            "accuracy": correct / len(data) if data else 0.0,
            "details": details,
        })

    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
