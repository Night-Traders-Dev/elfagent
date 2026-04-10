import os
from summary.summary_models import prefetch_models
from core.config import HF_CACHE_DIR
os.makedirs(HF_CACHE_DIR, exist_ok=True)
print(f"Using Hugging Face cache: {HF_CACHE_DIR}")
prefetch_models()
print("All models downloaded.")
