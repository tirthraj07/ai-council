"""
Whisper store: private messages between agents. Each agent has an inbox of
messages sent to them via whisper.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


@dataclass
class Whisper:
    from_agent: str
    message: str

    def format(self) -> str:
        return f"From {self.from_agent}: {self.message}"


class WhisperStore:
    """
    Per-agent inbox for whispered messages. Agents send via whisper(to_agent, message);
    the recipient sees recent whispers when their context is built.
    """

    def __init__(self, max_per_inbox: int = 50):
        self._max_per_inbox = max(1, max_per_inbox)
        self._inboxes: dict[str, deque[Whisper]] = defaultdict(
            lambda: deque(maxlen=self._max_per_inbox)
        )

    def send(self, from_agent: str, to_agent: str, message: str) -> None:
        """Deliver a whisper from one agent to another."""
        self._inboxes[to_agent].append(Whisper(from_agent=from_agent, message=message))

    def get_for_agent(self, agent_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent whispers addressed to the given agent (newest last)."""
        inbox = self._inboxes.get(agent_id, deque())
        recent = list(inbox)[-limit:]
        return [{"from_agent": w.from_agent, "message": w.message} for w in recent]

    def format_for_agent(self, agent_id: str, limit: int = 20) -> str:
        """Return whispers for the agent as a single string for prompt injection."""
        recent = self.get_for_agent(agent_id, limit=limit)
        if not recent:
            return ""
        lines = [f"From {r['from_agent']}: {r['message']}" for r in recent]
        return "[Whispers to you]\n" + "\n".join(lines)
