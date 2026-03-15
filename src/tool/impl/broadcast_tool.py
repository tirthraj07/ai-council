from src.tool import Tool


class BroadcastTool(Tool):
    """Post a message to the common forum so all agents can see it."""

    name = "broadcast"
    description = "Send a message to the common forum. All agents can see broadcasts."
    parameters = "message (str): the text to post to the forum"

    def __init__(self, forum, sender_name: str):
        self._forum = forum
        self._sender_name = sender_name

    def run(self, message: str) -> str:
        if not message or not str(message).strip():
            return "Error: message cannot be empty."
        self._forum.broadcast(self._sender_name, str(message).strip())
        return f"Broadcast sent to the forum as {self._sender_name}."
