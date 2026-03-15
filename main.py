from pathlib import Path

from src.agent import Agent
from src.agent.impl.GeminiLLM import GeminiLLM
from src.council import Council, RoundRunner
from src.memory import create_agent_memory
from src.personality import Personality
from src.tool import ToolRegistry
from src.tool.impl.history_tool import HistoryTool


def main():
    persist_dir = Path(__file__).parent / "data" / "chroma"
    transcript_path = Path(__file__).parent / "debate_transcript.txt"
    council = Council()

    def make_agent(name: str, system_prompt: str, role: str = "debate"):
        memory = create_agent_memory(
            agent_id=name,
            short_term_max_messages=10,
            long_term_persist_path=str(persist_dir),
            long_term_retrieve_n=5,
            forum=council.forum,
            whisper_store=council.whisper_store,
        )
        tools = ToolRegistry()
        tools.register(HistoryTool(memory))
        llm = GeminiLLM(provider_name="gemini", model_name="gemini-1.5-flash")
        personality = Personality(name=name, system_prompt=system_prompt)
        agent = Agent(llm=llm, personality=personality, tools=tools, memory=memory)
        council.add_agent(name, agent, role=role)
        return agent

    make_agent(
        "critic",
        "You are a critical analyst that challenges assumptions. You can broadcast to the forum or whisper to other agents.",
    )
    make_agent(
        "assistant",
        "You are a helpful assistant. You can broadcast to the forum or whisper to other agents.",
    )
    make_agent(
        "moderator",
        "You are a moderator who summarizes discussions. Summarize clearly and neutrally. Return only the summary text.",
        role="summary",
    )
    council.set_summary_agent("moderator")

    print("Debate agents:", council.debate_agent_names)
    print("Summary agent: moderator")

    runner = RoundRunner(council, transcript_path=transcript_path)
    runner.run_debate()


if __name__ == "__main__":
    main()
