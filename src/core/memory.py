import os
import time
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama
from core.config import CHAT_STORE_KEY, MAIN_MODEL

def init_chat_store(filepath: str) -> SimpleChatStore:
    """Loads existing chat store from disk, or creates a new one."""
    if os.path.exists(filepath):
        try:
            return SimpleChatStore.from_persist_path(filepath)
        except Exception as e:
            print(f"Failed to load memory from {filepath}: {e}")
    return SimpleChatStore()


def _compact_history(messages: list[ChatMessage]) -> list[ChatMessage]:
    """
    Summarizes the first half of the message history to keep the context window
    lean, preserving the most recent messages exactly.
    """
    # Keep the system prompt if present
    start_idx = 1 if messages and messages[0].role == MessageRole.SYSTEM else 0

    # Need at least 10 messages to bother compacting
    if len(messages) - start_idx < 10:
        return messages

    # Split: compact the oldest 50% of the conversational messages
    # Keep the newest 50% exactly as they are
    usable_msgs = messages[start_idx:]
    split_point = len(usable_msgs) // 2

    to_summarize = usable_msgs[:split_point]
    to_keep = usable_msgs[split_point:]

    # Build the prompt for the local LLM to summarize
    # Format the old history
    history_text = "\n".join([
        f"{m.role.value.upper()}: {m.content}" for m in to_summarize
    ])

    # We use the MAIN_MODEL to do the summarization. We use a short timeout.
    llm = Ollama(model=MAIN_MODEL, request_timeout=60.0)

    prompt = (
        "Summarize the following conversation history concisely.\n"
        "Focus only on facts, user preferences, and key technical context established.\n"
        "Do NOT include greetings or conversational filler.\n\n"
        f"HISTORY:\n{history_text}\n\n"
        "SUMMARY:"
    )

    try:
        response = llm.complete(prompt)
        summary_content = f"[Prior context summarized: {response.text.strip()}]"

        # Build the new compacted message list
        new_messages = messages[:start_idx]
        new_messages.append(ChatMessage(role=MessageRole.USER, content="Here is our past context."))
        new_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=summary_content))
        new_messages.extend(to_keep)

        return new_messages
    except Exception as e:
        # If summarization fails, just return the original messages to avoid breaking
        print(f"[Memory] Compaction failed: {e}")
        return messages


class CompactingChatMemoryBuffer(ChatMemoryBuffer):
    """
    A ChatMemoryBuffer that automatically compacts (summarizes) older
    messages when the token count gets too large.
    """

    def get(self, initial_token_count: int = 0, **kwargs) -> list[ChatMessage]:
        messages = super().get(initial_token_count=initial_token_count, **kwargs)

        # We roughly estimate tokens by chars / 4
        # If the history is longer than ~6000 tokens (24,000 chars), compact it.
        total_chars = sum(len(m.content or "") for m in messages)

        if total_chars > 24000:
            compacted = _compact_history(messages)
            # Update the underlying chat store so we don't have to re-summarize
            # next time.
            self.chat_store.set_messages(self.chat_store_key, compacted)
            # Re-fetch from the store to ensure state is synced
            return super().get(initial_token_count=initial_token_count, **kwargs)

        return messages



    def force_compact(self):
        messages = self.chat_store.get_messages(self.chat_store_key)
        if len(messages) < 10:
            return False, "Not enough messages to compact (need at least 10)."

        compacted = _compact_history(messages)
        self.chat_store.set_messages(self.chat_store_key, compacted)
        return True, f"Compacted {len(messages)} messages down to {len(compacted)}."

def build_memory(chat_store: SimpleChatStore, token_limit: int = 8192) -> CompactingChatMemoryBuffer:
    """Builds the memory buffer from the chat store."""
    return CompactingChatMemoryBuffer.from_defaults(
        chat_store=chat_store,
        chat_store_key=CHAT_STORE_KEY,
        token_limit=token_limit
    )
