from orchestration.validation import validate_handoff_packet

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
    validate_handoff_packet(packet)
    return packet
