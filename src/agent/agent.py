import json
from typing import Any


def _format_tool_info(tools) -> str:
    """Format tool names, descriptions, and parameters for injection into the system prompt."""
    if not tools:
        return ""
    lines = [
        "Available tools you can call. When you want to use a tool, request it; the runner will execute it and return the result, then you can continue or give your final response.",
        "",
    ]
    for t in tools:
        name = getattr(t, "name", str(t))
        desc = getattr(t, "description", "")
        params = getattr(t, "parameters", "")
        if params:
            lines.append(f"- {name}({params}): {desc}")
        else:
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
        on_tool_call=None,
        on_stream=None,
    ) -> str:
        """
        Run one conversation turn: inject system/personality, tool info, long-term
        context, and last 5-10 messages into the prompt; then append the exchange
        to short-term and to long-term (full conversation history).
        Optional on_tool_call(tool_name, arguments, result) for transcript logging.
        Optional on_stream(chunk) to stream each text chunk to the terminal.
        Returns the assistant reply content.
        """
        messages = self.build_messages_for_turn(user_message, short_term_limit=short_term_limit)
        response_content = self.run(messages, on_tool_call=on_tool_call, on_stream=on_stream)
        if hasattr(self.memory, "add_to_short_term"):
            self.memory.add_to_short_term("user", user_message)
            self.memory.add_to_short_term("assistant", response_content)
        if hasattr(self.memory, "append_turn_to_long_term"):
            self.memory.append_turn_to_long_term(user_message, response_content)
        return response_content

    def run(
        self,
        messages: list[dict[str, Any]],
        on_tool_call=None,
        on_stream=None,
    ) -> str:
        """
        Run the agent on a pre-built message list. The runner intercepts tool calls:
        when the LLM requests a tool, we execute it, append the result to the
        conversation, and hand control back to the LLM. This repeats until the
        LLM returns a final text response (no tool call). Returns that final content.
        Optional on_tool_call(tool_name, arguments, result) invoked after each tool use.
        Optional on_stream(chunk) to stream each text chunk (only for final text response).
        """
        messages = list(messages)
        tool_call_id = 0
        while True:
            response = self.llm.generate(
                messages,
                tools=self.tools.list(),
                on_stream=on_stream,
            )
            if response.type == "tool_call":
                tool_name = response.tool_name
                call_id = f"call_{tool_call_id}"
                tool_call_id += 1
                assistant_tool_calls = [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(response.arguments),
                        },
                    }
                ]
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": assistant_tool_calls,
                })
                try:
                    tool = self.tools.get(tool_name)
                except KeyError:
                    available = [t.name for t in self.tools.list() if getattr(t, "name", None)]
                    result = (
                        f"Error: the tool '{tool_name}' does not exist. "
                        f"Available tools: {', '.join(available) or 'none'}."
                    )
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": call_id,
                        "tool_name": tool_name,
                    })
                    if on_tool_call is not None:
                        on_tool_call(tool_name, response.arguments, result)
                else:
                    result = tool.run(**response.arguments)
                    messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": call_id,
                        "tool_name": tool_name,
                    })
                    if on_tool_call is not None:
                        on_tool_call(tool_name, response.arguments, result)
            else:
                return response.content