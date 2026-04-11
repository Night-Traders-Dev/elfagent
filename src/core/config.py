import os
from dotenv import load_dotenv

load_dotenv()

FIM_MODEL = os.getenv("FIM_MODEL", "qwen2.5-coder:7b")
MAIN_MODEL = os.getenv("MAIN_MODEL", "qwen3.5:9b")
SEARCH_MODEL = os.getenv("SEARCH_MODEL", "phi3.5")
CODE_MODEL = os.getenv("CODE_MODEL", "qwen2.5-coder:7b")
REFACTOR_MODEL = os.getenv("REFACTOR_MODEL", "qwen2.5-coder:14b")
HELPER_MODEL = os.getenv("HELPER_MODEL", "phi3.5")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "phi3.5")
MEMORY_PATH = os.getenv("MEMORY_PATH", "agent_memory.json")
CHAT_STORE_KEY = os.getenv("CHAT_STORE_KEY", "coding_agent_01")
GITHUB_PAT = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
ENABLE_GITHUB_MCP = os.getenv("ENABLE_GITHUB_MCP", "0").lower() in {"1", "true", "yes", "on"}
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", os.path.join("model_cache", "huggingface"))
HF_HOME = os.getenv("HF_HOME", HF_CACHE_DIR)
HF_HUB_CACHE = os.getenv("HF_HUB_CACHE", HF_CACHE_DIR)
SENTENCE_TRANSFORMERS_HOME = os.getenv("SENTENCE_TRANSFORMERS_HOME", HF_CACHE_DIR)
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", HF_HUB_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", SENTENCE_TRANSFORMERS_HOME)
USER_AGENT = "ElfAgent/3.0 (+https://github.com/)"
MEDICAL_MODEL_ID = "Falconsai/medical_summarization"
MEETING_MODEL_ID = "knkarthick/MEETING_SUMMARY"
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "moondream2")
VISION_OCR_MODEL = os.getenv("VISION_OCR_MODEL", VISION_MODEL)
RAG_INDEX_PATH = os.getenv("RAG_INDEX_PATH", ".elfagent_cache/local_rag_index.json")
RICH_REFRESH_PER_SECOND = float(os.getenv("RICH_REFRESH_PER_SECOND", "12"))
ROUTER_ESCALATION_THRESHOLD = float(os.getenv("ROUTER_ESCALATION_THRESHOLD", "0.75"))
TELEMETRY_DIR = os.getenv("TELEMETRY_DIR", "telemetry_logs")
BENCHMARK_DIR = os.getenv("BENCHMARK_DIR", "benchmarks_out")
ENABLE_CONSOLE_OTEL = os.getenv("ENABLE_CONSOLE_OTEL", "0").lower() in {"1", "true", "yes", "on"}
FAST_PATH_SIMPLE_CODE = os.getenv("FAST_PATH_SIMPLE_CODE", "1").lower() in {"1", "true", "yes", "on"}
MAX_FINAL_ANSWER_CHARS = int(os.getenv("MAX_FINAL_ANSWER_CHARS", "4000"))
ENABLE_TURBOQUANT = os.getenv("ENABLE_TURBOQUANT", "1").lower() in {"1", "true", "yes", "on"}
TURBOQUANT_MAX_SNIPPET_CHARS = int(os.getenv("TURBOQUANT_MAX_SNIPPET_CHARS", "280"))
TURBOQUANT_MAX_PACKET_CHARS = int(os.getenv("TURBOQUANT_MAX_PACKET_CHARS", "4500"))
TURBOQUANT_MAX_EVIDENCE_ITEMS = int(os.getenv("TURBOQUANT_MAX_EVIDENCE_ITEMS", "5"))
ENABLE_ENGRAMS = os.getenv("ENABLE_ENGRAMS", "1").lower() in {"1", "true", "yes", "on"}
ENGRAM_PATH = os.getenv("ENGRAM_PATH", "agent_engrams.jsonl")
ENGRAM_MAX_RECORDS = int(os.getenv("ENGRAM_MAX_RECORDS", "200"))
ENGRAM_RETRIEVAL_LIMIT = int(os.getenv("ENGRAM_RETRIEVAL_LIMIT", "3"))
ENGRAM_MAX_GIST_CHARS = int(os.getenv("ENGRAM_MAX_GIST_CHARS", "260"))
AUTORESEARCH_MAX_ROUNDS = int(os.getenv("AUTORESEARCH_MAX_ROUNDS", "2"))
AUTORESEARCH_MAX_QUERIES_PER_ROUND = int(os.getenv("AUTORESEARCH_MAX_QUERIES_PER_ROUND", "3"))
AUTORESEARCH_TARGET_COVERAGE = float(os.getenv("AUTORESEARCH_TARGET_COVERAGE", "0.67"))
