import os
import threading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from core.config import HF_CACHE_DIR, MEDICAL_MODEL_ID, MEETING_MODEL_ID
from summary.chunking import split_text_into_token_chunks

_model_lock = threading.Lock()
_medical_bundle = None
_meeting_bundle = None

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_hf_bundle(model_id: str):
    ensure_dir(HF_CACHE_DIR)
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, cache_dir=HF_CACHE_DIR)
    return {"tokenizer": tokenizer, "model": model}

def get_medical_bundle():
    global _medical_bundle
    with _model_lock:
        if _medical_bundle is None:
            _medical_bundle = load_hf_bundle(MEDICAL_MODEL_ID)
    return _medical_bundle

def get_meeting_bundle():
    global _meeting_bundle
    with _model_lock:
        if _meeting_bundle is None:
            _meeting_bundle = load_hf_bundle(MEETING_MODEL_ID)
    return _meeting_bundle

def summarize_with_bundle(bundle, text: str, max_length: int = 180, min_length: int = 40) -> str:
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = next(model.parameters()).device
    inputs = tokenizer(text.strip(), return_tensors="pt", truncation=True, max_length=min(getattr(model.config, "max_position_embeddings", 1024), 1024)).to(device)
    with __import__("torch").inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_length, min_new_tokens=min_length)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

def summarize_medical_text(text: str, max_length: int = 180, min_length: int = 40) -> str:
    if not text or not text.strip():
        return "No text provided."
    return summarize_with_bundle(get_medical_bundle(), text, max_length=max_length, min_length=min_length)

def summarize_meeting_text(text: str, max_length: int = 180, min_length: int = 40, max_input_tokens: int = 900, second_pass: bool = True) -> str:
    if not text or not text.strip():
        return "No text provided."
    bundle = get_meeting_bundle()
    tokenizer = bundle["tokenizer"]
    chunks = split_text_into_token_chunks(text.strip(), tokenizer=tokenizer, max_input_tokens=max_input_tokens)
    if len(chunks) == 1:
        return summarize_with_bundle(bundle, chunks[0], max_length=max_length, min_length=min_length)
    partials = [f"Chunk {idx}: {summarize_with_bundle(bundle, chunk, max_length=max_length, min_length=min_length)}" for idx, chunk in enumerate(chunks, start=1)]
    combined = "\n".join(partials)
    if second_pass and len(tokenizer.encode(combined, add_special_tokens=False)) <= max_input_tokens:
        return summarize_with_bundle(bundle, combined, max_length=max_length, min_length=min_length)
    return "\n".join(partials)

def prefetch_models():
    get_medical_bundle(); get_meeting_bundle()
