import re

def split_text_into_token_chunks(text: str, tokenizer, max_input_tokens: int = 900):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()] or [text.strip()]
    chunks, current_parts, current_tokens = [], [], 0
    for para in paragraphs:
        token_count = len(tokenizer.encode(para, add_special_tokens=False))
        if token_count > max_input_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
                if sent_tokens > max_input_tokens:
                    words, temp = sent.split(), []
                    for word in words:
                        candidate = (" ".join(temp + [word])).strip()
                        if len(tokenizer.encode(candidate, add_special_tokens=False)) <= max_input_tokens:
                            temp.append(word)
                        else:
                            if temp:
                                chunks.append(" ".join(temp))
                            temp = [word]
                    if temp:
                        chunks.append(" ".join(temp))
                    continue
                if current_tokens + sent_tokens <= max_input_tokens:
                    current_parts.append(sent)
                    current_tokens += sent_tokens
                else:
                    if current_parts:
                        chunks.append("\n\n".join(current_parts))
                    current_parts, current_tokens = [sent], sent_tokens
            continue
        if current_tokens + token_count <= max_input_tokens:
            current_parts.append(para)
            current_tokens += token_count
        else:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
            current_parts, current_tokens = [para], token_count
    if current_parts:
        chunks.append("\n\n".join(current_parts))
    return [c.strip() for c in chunks if c.strip()]
