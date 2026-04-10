from utils.formatting import truncate, s

def extract_blocks_text(blocks):
    parts = []
    for block in blocks or []:
        if hasattr(block, "content") and getattr(block, "content", None):
            parts.append(s(block.content))
        elif hasattr(block, "text") and getattr(block, "text", None):
            parts.append(s(block.text))
        else:
            parts.append(s(block))
    return "\n\n".join(p for p in parts if p).strip()

def extract_tool_output(event):
    tool_output = getattr(event, "tool_output", None)
    if tool_output is None:
        return "No output returned."
    if hasattr(tool_output, "content") and getattr(tool_output, "content", None):
        return truncate(tool_output.content)
    if hasattr(tool_output, "blocks") and getattr(tool_output, "blocks", None):
        text = extract_blocks_text(tool_output.blocks)
        if text:
            return truncate(text)
    if hasattr(tool_output, "raw_output") and getattr(tool_output, "raw_output", None):
        return truncate(tool_output.raw_output)
    return truncate(tool_output)

def extract_response_text(response):
    if response is None: return ""
    if isinstance(response, str): return response
    if hasattr(response, "response"):
        nested = extract_response_text(response.response)
        if nested: return nested
    if hasattr(response, "blocks") and getattr(response, "blocks", None):
        text = extract_blocks_text(response.blocks)
        if text: return text
    if hasattr(response, "content") and getattr(response, "content", None):
        return s(response.content)
    return s(response)

def extract_thinking_text(event):
    response = getattr(event, "response", None)
    if response is None or not hasattr(response, "blocks"):
        return ""
    thoughts = []
    for block in response.blocks:
        if getattr(block, "block_type", "") == "thinking":
            content = getattr(block, "content", "")
            if content: thoughts.append(s(content))
    return "\n\n".join(thoughts).strip()
