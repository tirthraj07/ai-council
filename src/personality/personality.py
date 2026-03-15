from dataclasses import dataclass

@dataclass
class Personality:
    name: str
    system_prompt: str
    temperature: float = 0.7