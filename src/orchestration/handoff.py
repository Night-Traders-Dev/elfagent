from core.turboquant import TurboQuantCompressor
from orchestration.validation import validate_handoff_packet

turboquant = TurboQuantCompressor()


def build_handoff_packet(user_query: str, route: dict, payload: dict) -> dict:
    packet = {
        "user_query": user_query,
        "route": route,
        "payload": payload,
        "instructions": [
            "Use the payload as high-signal context.",
            "Do not restate all evidence verbatim.",
            "Prefer concise synthesis over raw dump.",
        ],
    }
    compacted, compression_stats = turboquant.compact_packet(packet)
    compacted.setdefault("payload", {})["turboquant"] = compression_stats
    validate_handoff_packet(compacted)
    return compacted
