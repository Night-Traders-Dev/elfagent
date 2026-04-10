import json
from pathlib import Path
from routing.router import TaskRouter


def main():
    dataset_path = Path(__file__).resolve().parent.parent / "experiments" / "dataset.json"
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    router = TaskRouter()
    total = len(data)
    correct = 0
    details = []
    for item in data:
        got = router.route(item["query"])
        ok = got["route"] == item["expected_route"]
        correct += int(ok)
        details.append({
            "query": item["query"],
            "expected": item["expected_route"],
            "got": got["route"],
            "confidence": got.get("confidence", 0.0),
            "ok": ok,
        })
    out = {
        "accuracy": correct / total if total else 0.0,
        "total": total,
        "details": details,
    }
    out_path = Path(__file__).resolve().parent / "route_eval_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
