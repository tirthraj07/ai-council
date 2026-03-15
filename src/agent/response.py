"""Unified response type for LLM.generate()."""

from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Returned by LLM.generate(). Either a message or a tool call."""

    type: str  # "message" | "tool_call"
    content: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None

    @classmethod
    def message(cls, content: str) -> "LLMResponse":
        return cls(type="message", content=content or "")

    @classmethod
    def tool_call(cls, tool_name: str, arguments: dict[str, Any]) -> "LLMResponse":
        return cls(
            type="tool_call",
            content="",
            tool_name=tool_name,
            arguments=arguments or {},
        )
