from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from core.config import MEMORY_PATH, CHAT_STORE_KEY

def init_chat_store(path=MEMORY_PATH):
    try:
        return SimpleChatStore.from_persist_path(persist_path=path)
    except Exception:
        return SimpleChatStore()

def build_memory(chat_store):
    return ChatMemoryBuffer.from_defaults(token_limit=10000, chat_store=chat_store, chat_store_key=CHAT_STORE_KEY)
