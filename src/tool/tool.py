from abc import ABC, abstractmethod

class Tool(ABC):

    name: str
    description: str

    @abstractmethod
    def run(self, **kwargs):
        pass