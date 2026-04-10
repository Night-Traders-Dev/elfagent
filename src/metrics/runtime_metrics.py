import json
import os
from datetime import datetime, timezone
from core.config import BENCHMARK_DIR


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


class RuntimeMetrics:
    def __init__(self, filename="runtime_metrics.jsonl"):
        ensure_dir(BENCHMARK_DIR)
        self.path = os.path.join(BENCHMARK_DIR, filename)

    def emit(self, metric_name: str, value, labels=None):
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "metric": metric_name,
            "value": value,
            "labels": labels or {},
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
