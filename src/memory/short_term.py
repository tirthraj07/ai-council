"""
Short-term memory: recent conversation turns kept in context and injected into the LLM prompt.
Uses a sliding window so the prompt does not grow unbounded.
"""

from collections import deque
from typing import Any


class ShortTermMemory:
    """
    In-memory buffer of recent messages. Content is injected directly into
    the prompt sent to the LLM (no separate retrieval step).
    """

    def __init__(self, max_messages: int = 20):
        """
        Args:
            max_messages: Maximum number of messages to retain. Oldest are dropped when exceeded.
        """
        self._max_messages = max(1, max_messages)
        self._messages: deque[dict[str, Any]] = deque(maxlen=self._max_messages)

    def add(self, role: str, content: str | dict[str, Any]) -> None:
        """Append a single message (e.g. 'user', 'assistant', 'system')."""
        self._messages.append({"role": role, "content": content})

    def add_messages(self, messages: list[dict[str, Any]]) -> None:
        """Append multiple messages. Each must have 'role' and 'content'."""
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role is not None and content is not None:
                self._messages.append({"role": role, "content": content})

    def get_recent(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Return the most recent messages for injection into the prompt.
        If limit is None, returns all buffered messages.
        """
        recent = list(self._messages)
        if limit is not None and limit > 0:
            recent = recent[-limit:]
        return recent

    def get_last_messages(self, limit: int = 10) -> list[dict[str, Any]]:
        """Alias for get_recent for compatibility with tools like HistoryTool."""
        return self.get_recent(limit=limit)

    def clear(self) -> None:
        """Clear all buffered messages."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
