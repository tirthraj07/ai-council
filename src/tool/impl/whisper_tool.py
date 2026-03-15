from src.tool import Tool


class WhisperTool(Tool):
    """Send a private message to another agent."""

    name = "whisper"
    description = "Send a private message to a specific agent. Only that agent sees it."
    parameters = "to_agent (str): the agent's name to send to; message (str): the private message content"

    def __init__(self, whisper_store, sender_name: str, agent_names: list[str]):
        self._whisper_store = whisper_store
        self._sender_name = sender_name
        self._agent_names = list(agent_names)

    def run(self, to_agent: str, message: str) -> str:
        to_agent = str(to_agent).strip()
        if not to_agent:
            return "Error: to_agent cannot be empty."
        if to_agent == self._sender_name:
            return "You cannot whisper to yourself."
        if self._agent_names and to_agent not in self._agent_names:
            return f"Unknown agent: {to_agent}. Known agents: {', '.join(self._agent_names)}."
        if not message or not str(message).strip():
            return "Error: message cannot be empty."
        self._whisper_store.send(self._sender_name, to_agent, str(message).strip())
        return f"Whisper sent to {to_agent}."
