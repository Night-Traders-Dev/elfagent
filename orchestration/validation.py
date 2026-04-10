import json
from pathlib import Path

try:
    import jsonschema
except ImportError:
    jsonschema = None


def validate_handoff_packet(packet: dict) -> None:
    if jsonschema is None:
        required = {"user_query", "route", "payload", "instructions"}
        missing = required - set(packet.keys())
        if missing:
            raise ValueError(f"Missing required packet keys: {sorted(missing)}")
        return

    schema_path = Path(__file__).resolve().parent.parent / "schemas" / "handoff_packet.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(packet, schema)
