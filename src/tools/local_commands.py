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
    raw = user_msg.strip(); cmd = raw.lower()
    if cmd == "/help":
        return "Commands:\n/help\n/tools\n/clear\n/pwd\n/exit
/compact\n/summarize-medical <text or file>\n/summarize-meeting <text or file>"
    if cmd == "/tools":
        return "Loaded tools:\n- " + "\n- ".join(list_tool_names(tools))
    if cmd == "/pwd":
        return os.getcwd()
    if raw.startswith("/summarize-medical "):
        payload = raw[len("/summarize-medical "):].strip()
        if os.path.isfile(payload):
            with open(payload, "r", encoding="utf-8") as f: payload = f.read()
        return await asyncio.to_thread(summarize_medical_text, payload)
    if raw.startswith("/summarize-meeting "):
        payload = raw[len("/summarize-meeting "):].strip()
        if os.path.isfile(payload):
            with open(payload, "r", encoding="utf-8") as f: payload = f.read()
        return await asyncio.to_thread(summarize_meeting_text, payload)
    return None
