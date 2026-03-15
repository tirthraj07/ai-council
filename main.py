from pathlib import Path

from src.personality import Personality
from src.memory import create_agent_memory

def main():
    persist_dir = Path(__file__).parent / "data" / "chroma"

    critic_memory = create_agent_memory(
        agent_id="critic",
        short_term_max_messages=20,
        long_term_persist_path=str(persist_dir),
        long_term_retrieve_n=5,
    )
    critic = Personality(
        name="critic",
        system_prompt="You are a critical analyst that challenges assumptions.",
    )
    # critic_agent = Agent(llm=..., personality=critic, tools=..., memory=critic_memory)
    # critic_agent.run_turn("What do you think of X?")

    assistant_memory = create_agent_memory(
        agent_id="assistant",
        short_term_max_messages=20,
        long_term_persist_path=str(persist_dir),
        long_term_retrieve_n=5,
    )
    assistant = Personality(
        name="assistant",
        system_prompt="You are a helpful assistant.",
    )
    # assistant_agent = Agent(llm=..., personality=assistant, tools=..., memory=assistant_memory)


if __name__ == "__main__":
    main()
