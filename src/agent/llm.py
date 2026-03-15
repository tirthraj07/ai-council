from abc import ABC, abstractmethod

class LLM(ABC):
    def __init__(self, provider_name: str, model_name: str, **kwargs):
        self.provider_name = provider_name
        self.model_name    = model_name
        self.config        = kwargs

    @abstractmethod
    def generate(self, messages, tools=None):
        pass