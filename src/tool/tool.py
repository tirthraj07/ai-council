from abc import ABC, abstractmethod


class Tool(ABC):
    """Base for tools the agent can call. Expose name, description, and parameters for the LLM."""

    name: str = ""
    description: str = ""
    parameters: str = ""  # How to call, e.g. "message: str" or "to_agent: str, message: str"

    @abstractmethod
    def run(self, **kwargs):
        pass