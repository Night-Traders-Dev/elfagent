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
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)
USER_AGENT = "ElfAgent/3.0 (+https://github.com/)"
MEDICAL_MODEL_ID = "Falconsai/medical_summarization"
MEETING_MODEL_ID = "knkarthick/MEETING_SUMMARY"
RICH_REFRESH_PER_SECOND = float(os.getenv("RICH_REFRESH_PER_SECOND", "12"))
ROUTER_ESCALATION_THRESHOLD = float(os.getenv("ROUTER_ESCALATION_THRESHOLD", "0.75"))
TELEMETRY_DIR = os.getenv("TELEMETRY_DIR", "telemetry_logs")
BENCHMARK_DIR = os.getenv("BENCHMARK_DIR", "benchmarks_out")
ENABLE_CONSOLE_OTEL = os.getenv("ENABLE_CONSOLE_OTEL", "0").lower() in {"1", "true", "yes", "on"}
FAST_PATH_SIMPLE_CODE = os.getenv("FAST_PATH_SIMPLE_CODE", "1").lower() in {"1", "true", "yes", "on"}
MAX_FINAL_ANSWER_CHARS = int(os.getenv("MAX_FINAL_ANSWER_CHARS", "4000"))
