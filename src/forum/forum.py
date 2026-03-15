"""
Common forum: shared channel where agents broadcast messages. All agents can see
recent broadcasts when their context is built.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class Broadcast:
    sender: str
    message: str

    def format(self) -> str:
        return f"{self.sender}: {self.message}"


class Forum:
    """
    Shared message board for agent broadcasts. Holds recent broadcasts in order;
    oldest are dropped when max size is reached.
    """

    def __init__(self, max_broadcasts: int = 50):
        self._max_broadcasts = max(1, max_broadcasts)
        self._broadcasts: deque[Broadcast] = deque(maxlen=self._max_broadcasts)

    def broadcast(self, sender: str, message: str) -> None:
        """Post a message to the common forum from the given agent."""
        self._broadcasts.append(Broadcast(sender=sender, message=message))

    def get_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent broadcasts (newest last)."""
        recent = list(self._broadcasts)[-limit:]
        return [{"sender": b.sender, "message": b.message} for b in recent]

    def format_recent(self, limit: int = 20) -> str:
        """Return recent broadcasts as a single string for prompt injection."""
        recent = self.get_recent(limit=limit)
        if not recent:
            return ""
        lines = [f"{r['sender']}: {r['message']}" for r in recent]
        return "[Common forum]\n" + "\n".join(lines)
