import json
from pathlib import Path
from routing.router import TaskRouter
from experiments.prompts import ROUTER_PROMPTS


def main():
    data_path = Path(__file__).resolve().parent / "dataset.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    router = TaskRouter()

    rows = []
    for prompt_name in ROUTER_PROMPTS:
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
                "ok": ok,
            })
        rows.append({
            "prompt_variant": prompt_name,
            "accuracy": correct / len(data),
            "details": details,
        })

    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
