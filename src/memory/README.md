# Agent memory

Each agent has two kinds of memory:

- **Short-term**: Recent conversation turns kept in memory and injected into the LLM prompt (in-context). Implemented as a sliding window of messages.
- **Long-term**: Stored in a vector database (Chroma). Each agent has its own collection. Memories are retrieved by similarity and injected into the system prompt when relevant.

## Usage

### Create memory for an agent

```python
from src.memory import create_agent_memory

memory = create_agent_memory(
    agent_id="my_agent",
    short_term_max_messages=20,
    long_term_persist_path="./data/chroma",
    long_term_retrieve_n=5,
)
```

- `agent_id`: Unique id for the agent; used as the Chroma collection name.
- `short_term_max_messages`: Max number of recent messages to keep and inject.
- `long_term_persist_path`: Directory for Chroma persistence. Omit for in-memory only (no long-term).
- `long_term_retrieve_n`: How many long-term memories to retrieve per turn.

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

- `run_turn(user_message)` builds the prompt from personality + long-term retrieval + short-term messages + current message, runs the LLM, then appends the exchange to short-term.
- `run(messages)` uses a pre-built message list and does not update memory.

### Storing long-term memories

To add something to long-term memory (e.g. after a turn or from a tool):

```python
agent.memory.store_long_term("User prefers concise answers.", metadata={"source": "feedback"})
```

Retrieval is automatic: each turn, the current user message is used as the query to fetch relevant long-term memories, which are appended to the system prompt.

## Dependencies

Long-term memory requires Chroma:

```bash
pip install chromadb
```

Chroma uses its default embedding model unless you pass a custom `embedding_function` when creating `LongTermMemory` directly.
