from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.memory.agent_memory import AgentMemory


def create_agent_memory(
    agent_id: str,
    *,
    short_term_max_messages: int = 20,
    long_term_persist_path: str | None = None,
    long_term_retrieve_n: int = 5,
) -> AgentMemory:
    """
    Create an AgentMemory with short-term buffer and optional long-term Chroma store.
    Each agent_id gets its own short-term buffer; if long_term_persist_path is set,
    long-term uses a dedicated Chroma collection for this agent.
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
    )


__all__ = [
    "ShortTermMemory",
    "LongTermMemory",
    "AgentMemory",
    "create_agent_memory",
]
