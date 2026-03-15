from src.tool import Tool

class HistoryTool(Tool):

    name = "get_history"
    description = "Fetch previous conversation messages"

    def __init__(self, memory):
        self.memory = memory

    def run(self, limit: int = 10):
        return self.memory.get_last_messages(limit)