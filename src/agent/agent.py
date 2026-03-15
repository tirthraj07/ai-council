from typing import Any


class Agent:
    def __init__(self, llm, personality, tools, memory):
        self.llm = llm
        self.personality = personality
        self.tools = tools
        self.memory = memory

    def build_messages_for_turn(self, user_message: str) -> list[dict[str, Any]]:
        """
        Build the full message list for one turn: system (personality + long-term
        context), short-term messages, and the current user message.
        """
        system_content = self.personality.system_prompt
        if hasattr(self.memory, "get_long_term_context"):
            long_term = self.memory.get_long_term_context(user_message)
            if long_term:
                system_content = system_content + "\n" + long_term
        messages = [{"role": "system", "content": system_content}]
        if hasattr(self.memory, "build_context_messages"):
            context = self.memory.build_context_messages(user_message)
            messages.extend(context)
        messages.append({"role": "user", "content": user_message})
        return messages

    def run_turn(self, user_message: str) -> str:
        """
        Run one conversation turn: inject short-term and long-term memory into
        the prompt, generate a response, then append the exchange to short-term.
        Returns the assistant reply content.
        """
        messages = self.build_messages_for_turn(user_message)
        response_content = self.run(messages)
        if hasattr(self.memory, "add_to_short_term"):
            self.memory.add_to_short_term("user", user_message)
            self.memory.add_to_short_term("assistant", response_content)
        return response_content

    def run(self, messages: list[dict[str, Any]]) -> str:
        """Run the agent on a pre-built message list; returns final assistant content."""
        messages = list(messages)
        while True:
            response = self.llm.generate(
                messages,
                tools=self.tools.list(),
            )
            if response.type == "tool_call":
                tool = self.tools.get(response.tool_name)
                result = tool.run(**response.arguments)
                messages.append({"role": "tool", "content": result})
            else:
                return response.content