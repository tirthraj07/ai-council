# Council: multiple agents with broadcast and whisper

The Council runs multiple named LLM agents. Each agent has a unique name and two communication tools:

- **Broadcast**: send a message to the common forum (visible to all agents).
- **Whisper**: send a private message to a specific agent.

## Usage

1. Create a `Council` (optionally with your own `Forum` and `WhisperStore`).
2. Create each agent (LLM, personality, tools, memory) and add it with `council.add_agent(name, agent)`.
3. Run turns with `council.run_turn(agent_name, user_message)`.

When an agent runs a turn, their context includes:

- System prompt and personality
- Recent **forum broadcasts** (so they see what others have said)
- **Whispers to them** (private messages they received)
- Last 5–10 short-term messages
- Tool list (including `broadcast` and `whisper`)

So an agent can choose to reply in text, or call `broadcast(message)` to post to the forum, or `whisper(to_agent, message)` to message another agent by name.

## Example

```python
from src.council import Council
from src.agent import Agent
from src.memory import create_agent_memory
from src.personality import Personality
from src.tool import ToolRegistry
from src.tool.impl.history_tool import HistoryTool

council = Council()

# Create and register agents (each with unique name)
for name, prompt in [
    ("critic", "You are a critical analyst..."),
    ("assistant", "You are a helpful assistant..."),
]:
    memory = create_agent_memory(
        name,
        forum=council.forum,
        whisper_store=council.whisper_store,
    )
    tools = ToolRegistry()
    tools.register(HistoryTool(memory))
    agent = Agent(llm=..., personality=Personality(name=name, system_prompt=prompt), tools=tools, memory=memory)
    council.add_agent(name, agent)

# Run a turn; the agent sees the forum and whispers and can broadcast or whisper
reply = council.run_turn("critic", "What do you think?")
```

## Forum and WhisperStore

- **Forum**: `forum.broadcast(sender, message)` posts to the common channel. `forum.get_recent(limit)` / `forum.format_recent(limit)` for prompt injection.
- **WhisperStore**: `whisper_store.send(from_agent, to_agent, message)` delivers a whisper. `whisper_store.get_for_agent(agent_id, limit)` / `whisper_store.format_for_agent(agent_id, limit)` for prompt injection.

Agents do not call these directly; they use the **broadcast** and **whisper** tools, which the Council registers on each agent when you call `add_agent`.
