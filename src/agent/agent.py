
class Agent:
    def __init__(self, llm, personality, tools, memory):
        self.llm = llm
        self.personality = personality
        self.tools = tools
        self.memory = memory

    def run(self, messages):
        while True:
            response = self.llm.generate(
                messages,
                tools=self.tools.list()
            )

            if response.type == "tool_call":

                tool = self.tools.get(response.tool_name)

                result = tool.run(**response.arguments)

                messages.append({
                    "role": "tool",
                    "content": result
                })

            else:
                return response.content