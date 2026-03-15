from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.memory.agent_memory import AgentMemory


def create_agent_memory(
    agent_id: str,
    *,
    short_term_max_messages: int = 10,
    long_term_persist_path: str | None = None,
    long_term_retrieve_n: int = 5,
    forum=None,
    whisper_store=None,
) -> AgentMemory:
    """
    Create an AgentMemory with short-term buffer (last 5-10 messages) and optional
    long-term Chroma store (full conversation history). Optionally pass forum and
    whisper_store for council-style shared context.
    """
    short_term = ShortTermMemory(max_messages=short_term_max_messages)
    long_term = None
    if long_term_persist_path is not None:
        long_term = LongTermMemory(
            agent_id=agent_id,
            persist_directory=long_term_persist_path,
        )
    return AgentMemory(
        short_term=short_term,
        long_term=long_term,
        long_term_retrieve_n=long_term_retrieve_n,
        forum=forum,
        whisper_store=whisper_store,
        agent_id=agent_id,
    )


__all__ = [
    "ShortTermMemory",
    "LongTermMemory",
    "AgentMemory",
    "create_agent_memory",
]
