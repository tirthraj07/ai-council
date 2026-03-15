from typing import Any


def _format_tool_info(tools) -> str:
    """Format tool names and descriptions for injection into the system prompt."""
    if not tools:
        return ""
    lines = ["Available tools you can call:"]
    for t in tools:
        name = getattr(t, "name", str(t))
        desc = getattr(t, "description", "")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


class Agent:
    def __init__(self, llm, personality, tools, memory):
        self.llm = llm
        self.personality = personality
        self.tools = tools
        self.memory = memory

    def build_messages_for_turn(
        self,
        user_message: str,
        short_term_limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Build the full message list for one turn: system (personality + long-term
        context + tool info), last 5-10 short-term messages, and the current user message.
        """
        system_content = self.personality.system_prompt
        if hasattr(self.memory, "get_long_term_context"):
            long_term = self.memory.get_long_term_context(user_message)
            if long_term:
                system_content = system_content + "\n" + long_term
        tool_info = _format_tool_info(self.tools.list())
        if tool_info:
            system_content = system_content + "\n\n" + tool_info
        messages = [{"role": "system", "content": system_content}]
        if hasattr(self.memory, "build_context_messages"):
            context = self.memory.build_context_messages(
                user_message,
                short_term_limit=short_term_limit,
            )
            messages.extend(context)
        messages.append({"role": "user", "content": user_message})
        return messages

    def run_turn(
        self,
        user_message: str,
        short_term_limit: int = 10,
    ) -> str:
        """
        Run one conversation turn: inject system/personality, tool info, long-term
        context, and last 5-10 messages into the prompt; then append the exchange
        to short-term and to long-term (full conversation history).
        Returns the assistant reply content.
        """
        messages = self.build_messages_for_turn(user_message, short_term_limit=short_term_limit)
        response_content = self.run(messages)
        if hasattr(self.memory, "add_to_short_term"):
            self.memory.add_to_short_term("user", user_message)
            self.memory.add_to_short_term("assistant", response_content)
        if hasattr(self.memory, "append_turn_to_long_term"):
            self.memory.append_turn_to_long_term(user_message, response_content)
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