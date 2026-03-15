from abc import ABC, abstractmethod
from typing import Callable

class LLM(ABC):
    def __init__(self, provider_name: str, model_name: str, **kwargs):
        self.provider_name = provider_name
        self.model_name    = model_name
        self.config        = kwargs

    @abstractmethod
    def generate(self, messages, tools=None, on_stream: Callable[[str], None] | None = None):
        """Generate a response. If on_stream is provided, call it with each content chunk (text only)."""
        pass