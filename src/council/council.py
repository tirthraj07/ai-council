"""
Council: runs multiple named LLM agents with a shared forum (broadcast) and
per-agent whisper inboxes. Each agent has broadcast and whisper tools.
"""

from src.forum import Forum, WhisperStore
from src.tool.impl.broadcast_tool import BroadcastTool
from src.tool.impl.whisper_tool import WhisperTool


class Council:
    """
    Spawns and runs multiple agents, each with a unique name. All agents share
    a common forum (broadcast) and can send private messages (whisper) to each other.
    """

    def __init__(
        self,
        forum: Forum | None = None,
        whisper_store: WhisperStore | None = None,
    ):
        self._forum = forum or Forum()
        self._whisper_store = whisper_store or WhisperStore()
        self._agents: dict[str, object] = {}
        self._agent_names: list[str] = []
        self._summary_agent_name: str | None = None

    @property
    def forum(self) -> Forum:
        return self._forum

    @property
    def whisper_store(self) -> WhisperStore:
        return self._whisper_store

    @property
    def agent_names(self) -> list[str]:
        return list(self._agent_names)

    @property
    def debate_agent_names(self) -> list[str]:
        """Agent names that participate in rounds, excluding the summary agent."""
        if self._summary_agent_name is None:
            return list(self._agent_names)
        return [n for n in self._agent_names if n != self._summary_agent_name]

    def set_summary_agent(self, name: str) -> None:
        """Mark the named agent as the one used for per-round and final summaries."""
        name = name.strip()
        if not name:
            raise ValueError("Summary agent name cannot be empty.")
        if name not in self._agents:
            raise KeyError(f"Agent '{name}' must be added with add_agent before set_summary_agent.")
        self._summary_agent_name = name

    def add_agent(self, name: str, agent: object, role: str = "debate") -> None:
        """
        Register an agent with a unique name. Injects shared forum and whisper
        context into the agent's memory. Debate agents get broadcast and whisper
        tools; the summary agent (role='summary') does not.
        """
        name = name.strip()
        if not name:
            raise ValueError("Agent name cannot be empty.")
        if name in self._agents:
            raise ValueError(f"Agent named '{name}' is already registered.")
        self._agents[name] = agent
        self._agent_names.append(name)
        if hasattr(agent, "memory") and agent.memory is not None:
            if hasattr(agent.memory, "set_shared_context"):
                agent.memory.set_shared_context(
                    forum=self._forum,
                    whisper_store=self._whisper_store,
                    agent_id=name,
                )
        if role == "debate" and hasattr(agent, "tools") and agent.tools is not None:
            agent.tools.register(BroadcastTool(self._forum, name))
            agent.tools.register(WhisperTool(self._whisper_store, name, self._agent_names))
        return None

    def get_agent(self, name: str) -> object:
        """Return the agent with the given name."""
        if name not in self._agents:
            raise KeyError(f"Unknown agent: {name}. Known agents: {', '.join(self._agent_names)}.")
        return self._agents[name]

    def run_turn(
        self,
        agent_name: str,
        user_message: str,
        short_term_limit: int = 10,
        on_tool_call=None,
        on_stream=None,
    ) -> str:
        """
        Run one turn for the named agent. The agent sees recent forum broadcasts
        and whispers to them, then processes the user message.
        Optional on_tool_call(tool_name, arguments, result) for transcript logging.
        Optional on_stream(chunk) to stream each text chunk to the terminal.
        """
        agent = self.get_agent(agent_name)
        if not hasattr(agent, "run_turn"):
            raise TypeError(f"Agent '{agent_name}' has no run_turn method.")
        return agent.run_turn(
            user_message,
            short_term_limit=short_term_limit,
            on_tool_call=on_tool_call,
            on_stream=on_stream,
        )
