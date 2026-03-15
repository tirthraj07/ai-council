"""Ollama LLM implementation for local models."""

import json
from typing import Any

from src.agent import LLM
from src.agent.response import LLMResponse

try:
    import ollama
except ImportError:
    ollama = None



def _messages_to_ollama(messages: list[dict]) -> list[dict]:
    """Convert to Ollama /api/chat format: role (system/user/assistant) and content."""
    out = []
    for msg in messages:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content")
        if isinstance(content, dict):
            content = str(content)
        content = (content or "").strip()
        if role == "tool":
            out.append({"role": "user", "content": f"[Tool result]\n{content}"})
            continue
        if role == "assistant":
            role = "assistant"
        elif role != "system":
            role = "user"
        out.append({"role": role, "content": content})
    return out


class OllamaLLM(LLM):
    """LLM backed by local Ollama. No API key; ensure Ollama is running and model is pulled."""

    def __init__(self, provider_name: str, model_name: str, **kwargs):
        super().__init__(provider_name, model_name, **kwargs)
        if ollama is None:
            raise ImportError("Install ollama: pip install ollama")

    def generate(self, messages, tools=None, on_stream=None):
        if not messages:
            return LLMResponse.message("")
        ollama_messages = _messages_to_ollama(messages)
        opts = {}
        if self.config.get("temperature") is not None:
            opts["temperature"] = self.config["temperature"]
        try:
            if on_stream is not None and not tools:
                return self._generate_stream(ollama_messages, opts, on_stream)
            response = ollama.chat(
                model=self.model_name,
                messages=ollama_messages,
                options=opts or None,
            )
        except Exception as e:
            return LLMResponse.message(f"Error: {e}")
        if not response or "message" not in response:
            return LLMResponse.message("")
        msg = response["message"]
        content = (msg.get("content") or "").strip()
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            tc = tool_calls[0] if isinstance(tool_calls, list) else tool_calls
            name = tc.get("function", {}).get("name", "") if isinstance(tc, dict) else ""
            args = tc.get("function", {}).get("arguments", {}) if isinstance(tc, dict) else {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            return LLMResponse.tool_call(name, args or {})
        return LLMResponse.message(content)

    def _generate_stream(self, ollama_messages: list, opts: dict, on_stream):
        """Stream response and call on_stream for each content delta. Returns full response."""
        try:
            content_parts = []
            for chunk in ollama.chat(
                model=self.model_name,
                messages=ollama_messages,
                options=opts or None,
                stream=True,
            ):
                msg = chunk.get("message", {}) if isinstance(chunk, dict) else {}
                delta = msg.get("content", "") or ""
                if delta:
                    content_parts.append(delta)
                    on_stream(delta)
            return LLMResponse.message("".join(content_parts).strip())
        except Exception as e:
            return LLMResponse.message(f"Error: {e}")
