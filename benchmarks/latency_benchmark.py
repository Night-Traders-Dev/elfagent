import json
import time
from pathlib import Path
from routing.router import TaskRouter
from core.config import BENCHMARK_DIR

QUERIES = [
    "Find the latest RP2350 SDK docs",
    "Refactor this Python stack trace handler",
    "Explain the pros and cons of staged model routing",
]


def main():
    Path(BENCHMARK_DIR).mkdir(parents=True, exist_ok=True)
    router = TaskRouter()
    rows = []
    for q in QUERIES:
        t0 = time.perf_counter()
        route = router.route(q)
        dt = time.perf_counter() - t0
        rows.append({"query": q, "route": route, "latency_s": dt})
    out = Path(BENCHMARK_DIR) / "router_latency.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
