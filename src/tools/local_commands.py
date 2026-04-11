import os, asyncio
from summary.summary_models import summarize_medical_text, summarize_meeting_text


def list_tool_names(tools):
    names = []
    for t in tools:
        name = getattr(t, "metadata", None)
        if name and hasattr(name, "name"):
            names.append(name.name)
        elif hasattr(t, "name"):
            names.append(t.name)
        else:
            names.append(type(t).__name__)
    return sorted(set(names))


async def handle_local_command(user_msg, tools):
    """Return a (sentinel, ...) tuple or None.

    Sentinels:
        ("print", text)  – print text locally, do not send to agent
        ("quit",)        – save and exit
        ("clear",)       – clear the terminal
    Returns None for non-command input (pass to agent).
    """
    raw = user_msg.strip()
    cmd = raw.lower()

    if cmd == "/help":
        text = (
            "Commands:\n"
            "  /help                          – show this message\n"
            "  /tools                         – list loaded tools\n"
            "  /clear                         – clear the terminal\n"
            "  /pwd                           – print working directory\n"
            "  /exit  /quit                   – save memory and exit\n"
            "  /compact                       – force memory compaction\n"
            "  /summarize-medical <text|file> – medical summarisation\n"
            "  /summarize-meeting <text|file> – meeting summarisation"
        )
        return ("print", text)

    if cmd == "/tools":
        names = list_tool_names(tools)
        text = "Loaded tools:\n- " + "\n- ".join(names) if names else "No tools loaded."
        return ("print", text)

    if cmd == "/pwd":
        return ("print", os.getcwd())

    if cmd in ("/clear",):
        return ("clear",)

    if cmd in ("/exit", "/quit"):
        return ("quit",)

    if raw.startswith("/summarize-medical "):
        payload = raw[len("/summarize-medical "):].strip()
        if os.path.isfile(payload):
            with open(payload, "r", encoding="utf-8") as f:
                payload = f.read()
        result = await asyncio.to_thread(summarize_medical_text, payload)
        return ("print", result)

    if raw.startswith("/summarize-meeting "):
        payload = raw[len("/summarize-meeting "):].strip()
        if os.path.isfile(payload):
            with open(payload, "r", encoding="utf-8") as f:
                payload = f.read()
        result = await asyncio.to_thread(summarize_meeting_text, payload)
        return ("print", result)

    return None
