"""Gemini LLM implementation using google-generativeai."""

from typing import Any

from src.agent import LLM
from src.agent.response import LLMResponse

try:
    import google.generativeai as genai
except ImportError:
    genai = None


def _messages_to_gemini_contents(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert messages to (system_instruction, history). History is list of {"role": "user"|"model", "parts": [str]}."""
    if not genai:
        raise ImportError("Install google-generativeai: pip install google-generativeai")
    system_parts = []
    history = []
    for msg in messages:
        role = (msg.get("role") or "user").lower()
        content = msg.get("content")
        if isinstance(content, dict):
            content = str(content)
        content = (content or "").strip()
        if role == "system":
            system_parts.append(content)
            continue
        if role == "tool":
            history.append({"role": "user", "parts": [f"[Tool result]\n{content}"]})
            continue
        if role == "user":
            history.append({"role": "user", "parts": [content]})
        elif role == "assistant":
            history.append({"role": "model", "parts": [content]})
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, history


def _tools_to_gemini(tools) -> list[dict]:
    """Build Gemini function declarations from tool list."""
    if not tools:
        return []
    decls = []
    for t in tools:
        name = getattr(t, "name", None) or "tool"
        desc = getattr(t, "description", "") or ""
        decls.append({
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message or content"},
                    "to_agent": {"type": "string", "description": "Agent name to send to"},
                    "limit": {"type": "integer", "description": "Limit for history"},
                    "query": {"type": "string", "description": "Search query for web search"},
                    "max_results": {"type": "integer", "description": "Max search results to return"},
                    "scrape_top": {"type": "integer", "description": "Number of top results to scrape (0-2)"},
                },
            },
        })
    return decls


class GeminiLLM(LLM):
    """LLM backed by Google Gemini (google-generativeai). Set GOOGLE_API_KEY."""

    def __init__(self, provider_name: str, model_name: str, **kwargs):
        super().__init__(provider_name, model_name, **kwargs)
        if genai is None:
            raise ImportError("Install google-generativeai: pip install google-generativeai")
        api_key = kwargs.get("api_key") or __import__("os").environ.get("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
        self._model_name = model_name

    def generate(self, messages, tools=None):
        system_instruction, history = _messages_to_gemini_contents(messages)
        if not history:
            return LLMResponse.message("")
        tool_list = _tools_to_gemini(tools) if tools else []
        gen_config = {}
        if self.config.get("temperature") is not None:
            gen_config["temperature"] = self.config["temperature"]
        model_kwargs = {}
        if system_instruction:
            model_kwargs["system_instruction"] = system_instruction
        if gen_config:
            model_kwargs["generation_config"] = gen_config
        if tool_list:
            model_kwargs["tools"] = [{"function_declarations": tool_list}]
        model = genai.GenerativeModel(self._model_name, **model_kwargs)
        try:
            chat = model.start_chat(history=history[:-1] if len(history) > 1 else [])
            last = history[-1]
            if last.get("role") != "user" or not last.get("parts"):
                return LLMResponse.message("")
            response = chat.send_message(last["parts"][0])
        except Exception as e:
            return LLMResponse.message(f"Error: {e}")
        if not response or not response.candidates:
            return LLMResponse.message("")
        parts = response.candidates[0].content.parts
        if not parts:
            return LLMResponse.message("")
        part = parts[0]
        if hasattr(part, "function_call") and part.function_call:
            fc = part.function_call
            name = getattr(fc, "name", None) or ""
            args = dict(getattr(fc, "args", None) or {})
            return LLMResponse.tool_call(name, args)
        text = getattr(part, "text", None) or str(part)
        return LLMResponse.message(text)
