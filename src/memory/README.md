# Agent memory

Each agent has two kinds of memory:

- **Short-term (in-context)**: Injected into the LLM prompt each turn. It includes:
  - System prompt and personality
  - Last 5–10 messages (configurable)
  - Tool information (names and descriptions of tools the agent can call)
- **Long-term**: Full conversation history stored in a vector database (Chroma). Each agent has its own collection. Every turn is appended; retrieval by similarity injects relevant past context into the system prompt.

## Usage

### Create memory for an agent

```python
from src.memory import create_agent_memory

memory = create_agent_memory(
    agent_id="my_agent",
    short_term_max_messages=10,
    long_term_persist_path="./data/chroma",
    long_term_retrieve_n=5,
)
```

- `agent_id`: Unique id for the agent; used as the Chroma collection name.
- `short_term_max_messages`: Number of recent messages to keep and inject (default 10; use 5–10 for “last 5–10 messages”).
- `long_term_persist_path`: Directory for Chroma persistence. Omit for in-memory only (no long-term).
- `long_term_retrieve_n`: How many long-term (conversation) turns to retrieve by similarity per turn.

### Use with Agent

Pass the same `AgentMemory` instance to `Agent` and to any tools that need it (e.g. `HistoryTool`):

```python
from src.agent import Agent
from src.tool.impl.history_tool import HistoryTool

memory = create_agent_memory("critic", long_term_persist_path="./data/chroma")
tools = ToolRegistry()
tools.register(HistoryTool(memory))

agent = Agent(llm=llm, personality=personality, tools=tools, memory=memory)
reply = agent.run_turn("What do you think of X?")
```

- `run_turn(user_message)` builds the prompt as: **system** (personality + retrieved long-term context + tool list) + **last 5–10 messages** + **current user message**. After the turn it appends the exchange to short-term and to long-term (full history).
- `run(messages)` uses a pre-built message list and does not update memory.

### What is stored where

| Content                    | Short-term (in prompt)     | Long-term (Chroma)        |
|---------------------------|----------------------------|----------------------------|
| System prompt & personality | Injected each turn         | Not stored                 |
| Tool names & descriptions | Injected each turn         | Not stored                 |
| Last 5–10 messages        | Injected each turn         | —                          |
| Full conversation history| —                          | One document per turn      |

Each turn is stored in long-term as a single document (user + assistant text, optional tool summary) so you can retrieve similar past turns. You can also store one-off facts with `store_long_term(...)`.

### Optional: store extra long-term facts

```python
agent.memory.store_long_term("User prefers concise answers.", metadata={"source": "feedback"})
```

## Dependencies

Long-term memory requires Chroma:

```bash
pip install chromadb
```

Chroma uses its default embedding model unless you pass a custom `embedding_function` when creating `LongTermMemory` directly.
