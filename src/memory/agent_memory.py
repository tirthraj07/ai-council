"""
Combined agent memory: short-term (in-context) and long-term (vector DB).
Use this as the single memory object for an Agent.
"""

from typing import Any

from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory


class AgentMemory:
    """
    Single memory facade for an agent: short-term buffer (injected into prompt)
    and long-term vector store (retrieved by similarity). HistoryTool and the
    agent use this to read recent messages and to build full context.
    """

    def __init__(
        self,
        short_term: ShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
        short_term_max_messages: int = 10,
        long_term_retrieve_n: int = 5,
    ):
        """
        Args:
            short_term: Short-term memory. If None, a new one is created.
            long_term: Long-term memory. If None, agent has no long-term memory.
            short_term_max_messages: Used only when short_term is None. Default 10 (last 5-10 messages).
            long_term_retrieve_n: Number of long-term memories to retrieve per query.
        """
        self._short_term = short_term or ShortTermMemory(max_messages=short_term_max_messages)
        self._long_term = long_term
        self._long_term_retrieve_n = long_term_retrieve_n

    @property
    def short_term(self) -> ShortTermMemory:
        return self._short_term

    @property
    def long_term(self) -> LongTermMemory | None:
        return self._long_term

    def get_last_messages(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent messages (for tools like HistoryTool)."""
        return self._short_term.get_last_messages(limit=limit)

    def add_to_short_term(self, role: str, content: str | dict[str, Any]) -> None:
        """Append one message to short-term memory."""
        self._short_term.add(role=role, content=content)

    def add_messages_to_short_term(self, messages: list[dict[str, Any]]) -> None:
        """Append multiple messages to short-term memory."""
        self._short_term.add_messages(messages)

    def get_recent_messages(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return recent messages for prompt injection."""
        return self._short_term.get_recent(limit=limit)

    def retrieve_long_term(self, query: str, n_results: int | None = None) -> list[dict[str, Any]]:
        """
        Retrieve relevant long-term memories for the given query.
        Returns empty list if no long-term memory is configured.
        """
        if self._long_term is None:
            return []
        n = n_results if n_results is not None else self._long_term_retrieve_n
        return self._long_term.retrieve(query=query, n_results=n)

    def store_long_term(self, content: str, metadata: dict[str, Any] | None = None) -> str | None:
        """
        Store a memory in long-term. Returns document id or None if long-term not configured.
        """
        if self._long_term is None:
            return None
        return self._long_term.add(content=content, metadata=metadata)

    def append_turn_to_long_term(
        self,
        user_message: str,
        assistant_message: str,
        tool_calls_summary: str | None = None,
    ) -> str | None:
        """
        Append one full conversation turn to long-term history (entire conversation stored).
        Returns document id or None if long-term not configured.
        """
        if self._long_term is None:
            return None
        return self._long_term.append_turn(
            user_message=user_message,
            assistant_message=assistant_message,
            tool_calls_summary=tool_calls_summary,
        )

    def get_long_term_context(
        self,
        query: str,
        n_results: int | None = None,
    ) -> str:
        """
        Return formatted long-term memories for the query, to be appended to the
        system prompt. Returns empty string if no long-term memory or no results.
        """
        retrieved = self.retrieve_long_term(query, n_results=n_results or self._long_term_retrieve_n)
        if not retrieved:
            return ""
        lines = [f"- {r['content']}" for r in retrieved]
        return "\n[Relevant long-term memory]\n" + "\n".join(lines)

    def build_context_messages(
        self,
        current_query: str,
        include_short_term: bool = True,
        short_term_limit: int | None = None,
        long_term_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the list of messages to inject into the prompt before the current turn:
        recent short-term messages only. Use get_long_term_context() for system prompt.
        Does not include the current user message.
        """
        if not include_short_term:
            return []
        return self.get_recent_messages(limit=short_term_limit)
