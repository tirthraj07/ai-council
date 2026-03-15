"""Gemini LLM implementation using the google-genai package (google.genai)."""

from typing import Any

from src.agent import LLM
from src.agent.response import LLMResponse

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


def _normalize_content(content: Any) -> str:
    """Normalize message content to a single string. Handles str, dict, or list of parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        return str(content).strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
            else:
                parts.append(str(part))
        return "\n".join(parts).strip()
    return str(content).strip()


def _messages_to_contents(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Convert messages to (system_instruction, contents). Contents are Content-like dicts with role and parts."""
    system_parts = []
    contents = []
    for msg in messages:
        role = (msg.get("role") or "user").lower()
        content = _normalize_content(msg.get("content"))
        if role == "system":
            system_parts.append(content)
            continue
        if role == "tool":
            contents.append({
                "role": "user",
                "parts": [{"text": f"[Tool result]\n{content}"}],
            })
            continue
        if role == "user":
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _tools_to_genai(tools) -> list[dict]:
    """Build google.genai Tool function_declarations from tool list."""
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
                "type": "OBJECT",
                "properties": {
                    "message": {"type": "STRING", "description": "Message or content"},
                    "to_agent": {"type": "STRING", "description": "Agent name to send to"},
                    "limit": {"type": "INTEGER", "description": "Limit for history"},
                    "query": {"type": "STRING", "description": "Search query for web search"},
                    "max_results": {"type": "INTEGER", "description": "Max search results to return"},
                    "scrape_top": {"type": "INTEGER", "description": "Number of top results to scrape (0-2)"},
                },
            },
        })
    return decls


class GeminiLLM(LLM):
    """LLM backed by Google Gemini (google.genai). Set GOOGLE_API_KEY."""

    def __init__(self, provider_name: str, model_name: str, **kwargs):
        super().__init__(provider_name, model_name, **kwargs)
        if genai is None or types is None:
            raise ImportError("Install google-genai: pip install google-genai")
        api_key = kwargs.get("api_key") or __import__("os").environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY must be set for GeminiLLM")
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name

    def generate(self, messages, tools=None, on_stream=None):
        system_instruction, contents = _messages_to_contents(messages)
        if not contents:
            return LLMResponse.message("")
        config_dict = {}
        if system_instruction:
            config_dict["system_instruction"] = system_instruction
        if self.config.get("temperature") is not None:
            config_dict["temperature"] = self.config["temperature"]
        tool_list = _tools_to_genai(tools) if tools else []
        if tool_list:
            config_dict["tools"] = [{"function_declarations": tool_list}]
        config = types.GenerateContentConfig(**config_dict) if config_dict else None
        try:
            if on_stream is not None and not tool_list:
                return self._generate_stream(contents, config, on_stream)
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=config,
            )
        except Exception as e:
            return LLMResponse.message(f"Error: {e}")
        if not response or not getattr(response, "candidates", None):
            return LLMResponse.message("")
        function_calls = getattr(response, "function_calls", None)
        if function_calls and len(function_calls) > 0:
            fc = function_calls[0]
            name = getattr(fc, "name", None) or ""
            args = dict(getattr(fc, "args", None) or {})
            return LLMResponse.tool_call(name, args)
        text = getattr(response, "text", None) or ""
        return LLMResponse.message(text or "")

    def _generate_stream(self, contents, config, on_stream):
        """Stream content and call on_stream for each text chunk. Returns full response."""
        try:
            full_text = []
            for chunk in self._client.models.generate_content_stream(
                model=self._model_name,
                contents=contents,
                config=config,
            ):
                if not chunk or not getattr(chunk, "candidates", None):
                    continue
                for c in chunk.candidates:
                    content = getattr(c, "content", None)
                    if not content:
                        continue
                    for part in getattr(content, "parts", []) or []:
                        if getattr(part, "text", None):
                            t = part.text or ""
                            if t:
                                full_text.append(t)
                                on_stream(t)
            return LLMResponse.message("".join(full_text).strip())
        except Exception as e:
            return LLMResponse.message(f"Error: {e}")
