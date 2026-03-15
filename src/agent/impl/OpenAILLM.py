"""OpenAI LLM implementation using the openai package."""

import json
from typing import Any

from src.agent import LLM
from src.agent.response import LLMResponse

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def _messages_to_openai(messages: list[dict]) -> list[dict]:
    """Convert to OpenAI chat format: role (system/user/assistant) and content or tool_calls."""
    out = []
    for msg in messages:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content")
        if isinstance(content, dict):
            content = str(content)
        if role == "tool":
            out.append({"role": "tool", "content": content or "", "tool_call_id": "call_0"})
            continue
        if role == "assistant":
            role = "assistant"
        elif role != "system":
            role = "user"
        out.append({"role": role, "content": content or ""})
    return out


def _tools_to_openai(tools) -> list[dict]:
    """Build OpenAI tools list."""
    if not tools:
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": getattr(t, "name", None) or "tool",
                "description": getattr(t, "description", "") or "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message content"},
                        "to_agent": {"type": "string", "description": "Agent name"},
                        "limit": {"type": "integer", "description": "Limit"},
                        "query": {"type": "string", "description": "Search query for web search"},
                        "max_results": {"type": "integer", "description": "Max search results"},
                        "scrape_top": {"type": "integer", "description": "Top results to scrape (0-2)"},
                    },
                },
            },
        }
        for t in tools
    ]


class OpenAILLM(LLM):
    """LLM backed by OpenAI API. Set OPENAI_API_KEY or pass api_key in config."""

    def __init__(self, provider_name: str, model_name: str, **kwargs):
        super().__init__(provider_name, model_name, **kwargs)
        if OpenAI is None:
            raise ImportError("Install openai: pip install openai")
        self._client = OpenAI(api_key=kwargs.get("api_key"))

    def generate(self, messages, tools=None):
        if not messages:
            return LLMResponse.message("")
        openai_messages = _messages_to_openai(messages)
        tool_list = _tools_to_openai(tools) if tools else []
        kwargs = {
            "model": self.model_name,
            "messages": openai_messages,
        }
        if self.config.get("temperature") is not None:
            kwargs["temperature"] = self.config["temperature"]
        if tool_list:
            kwargs["tools"] = tool_list
            kwargs["tool_choice"] = "auto"
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            return LLMResponse.message(f"Error: {e}")
        choice = response.choices[0] if response.choices else None
        if not choice or not choice.message:
            return LLMResponse.message("")
        msg = choice.message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            name = tc.function.name if hasattr(tc.function, "name") else ""
            args_str = tc.function.arguments if hasattr(tc.function, "arguments") else "{}"
            try:
                args = json.loads(args_str)
            except Exception:
                args = {}
            return LLMResponse.tool_call(name, args)
        return LLMResponse.message((msg.content or "").strip())
