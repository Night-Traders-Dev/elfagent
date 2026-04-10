import json
import os
from datetime import datetime, timezone
from core.config import TELEMETRY_DIR


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class JsonlTelemetryLogger:
    def __init__(self, filename: str = "agent_runs.jsonl"):
        ensure_dir(TELEMETRY_DIR)
        self.path = os.path.join(TELEMETRY_DIR, filename)

    def write(self, event_type: str, payload: dict):
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
