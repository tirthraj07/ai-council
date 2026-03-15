# AI Council

A multi-agent debate system where several LLM agents with distinct personalities discuss a topic in rounds. Each agent can broadcast to a shared forum, whisper to another agent, search the web, and use conversation history. A summary agent wraps each round and the full debate. Useful for idea stress-testing, market-style feedback, and structured deliberation without a single model’s bias.

## What This Project Is

- **Council**: A set of named agents that share a common **forum** (broadcast channel) and can send **whispers** (private messages) to each other.
- **Rounds**: You give an initial topic. Each round, agents speak in random order. On their turn, each agent sees the forum, any whispers to them, round summaries so far, and can call tools (broadcast, whisper, search, history) before posting their view. When everyone has spoken, the summary agent summarizes the round. Summaries are carried into the next round.
- **Personalities**: Agents are defined by system prompts (e.g. Visionary, Skeptic, Engineer, Investor). Each keeps its own short-term and long-term memory (Chroma). The default config ships 11 debate personalities plus a moderator for summaries.
- **Tools**: Every debate agent can **broadcast** (post to the forum), **whisper** (message one agent), **search** (web search and optional scraping for market research or facts), and **get_history** (recent conversation). The runner intercepts tool calls, runs them, and returns results to the LLM until it gives a final reply.
- **Output**: A transcript file records each round (user message, agent turns with tool calls and whispers, round summary) and a final consolidated summary.

## Example: How the Council Can Help

You have an idea and want to stress-test it. The council simulates a structured debate.

**You:** “We’re thinking of an AI study planner that suggests schedules and resources per course.”

**Round 1**

- **Visionary** might say: “If this gains traction, it could reshape how students plan. The long-term upside is large.” and broadcast that.
- **Skeptic** might say: “This depends on students trusting AI with their grades. Most won’t.” and broadcast.
- **Engineer** might call `search(query="study planner app API integration")`, then broadcast: “The algorithm is straightforward; the hard part is data and integrations.”
- **Investor** might ask about TAM and who pays; **User Advocate** might raise switching cost from existing planners; **Market Analyst** might cite competitors.
- **Risk Analyst** might mention liability if advice is bad; **Pragmatist** might suggest: “Start with a Chrome extension for one platform.”
- **Contrarian**, **Historian**, and **Psychologist** add alternative angles, history of similar products, and adoption psychology.

**Round summary (moderator):** “Main points: potential scale vs. trust and data; need clear MVP and positioning; regulatory and liability concerns.”

You can add context for the next round or type `/quit`. After the debate, the moderator gives a **final summary**: what was discussed, for/against, risks, and a consolidated view. The transcript is saved to a file.

So the council helps by: surfacing objections (Skeptic, Risk), feasibility (Engineer, Pragmatist), market and users (Investor, User Advocate, Market Analyst), and upside (Visionary, Contrarian), with web search and history for grounding.

## Setup

### Requirements

- Python 3.13+
- At least one LLM provider (Gemini, OpenAI, or Ollama)

### Install

```bash
cd ai-council
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### API keys (for cloud providers)

- **Gemini**: Set `GOOGLE_API_KEY`.
- **OpenAI**: Set `OPENAI_API_KEY`.
- **Ollama**: No key; run Ollama locally and pull a model (e.g. `ollama pull llama3.2`).

Optional: **Web search** uses DuckDuckGo (no key). Scraping uses `requests` (already in dependencies).

## Configuring Agents

Agents are defined in a YAML file. By default the app uses `config/council.yaml`.

### Config file location

- Default: `config/council.yaml` (relative to the project root).
- Override: set `COUNCIL_CONFIG` to the config path, or run with `--config path/to/council.yaml`.

### Top-level options

| Key | Description |
|-----|-------------|
| `memory` | Optional. `persist_directory` (Chroma), `short_term_max_messages`, `long_term_retrieve_n`. |
| `transcript_path` | Where to write the debate transcript. Default `debate_transcript.txt`. |
| `agents` | List of agent definitions (see below). |
| `summary_agent` | Name of the agent that does per-round and final summaries (e.g. `moderator`). |

### Agent definition

Each entry under `agents` can have:

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique id (e.g. `visionary`, `skeptic`). |
| `system_prompt` | Yes | Personality and instructions (multi-line YAML supported). |
| `provider` | No | `gemini`, `openai`, or `ollama`. Default `gemini`. |
| `model` | No | Model name (e.g. `gemini-1.5-flash`, `gpt-4o`, `llama3.2`). Default `gemini-1.5-flash`. |
| `role` | No | `debate` (speaks in rounds, has broadcast/whisper/search/history) or `summary` (summarizes only). Default `debate`. |
| `temperature` | No | Float for the LLM. Default `0.7`. |

Example:

```yaml
agents:
  - name: visionary
    system_prompt: |
      You are The Visionary. Your core mindset is "What could this become?"
      You are optimistic and long-term. Argue for potential and scale.
      You can broadcast to the forum or whisper to other agents.
    provider: gemini
    model: gemini-1.5-flash
    role: debate
    temperature: 0.75

  - name: moderator
    system_prompt: |
      You summarize discussions clearly and neutrally. Return only the summary.
    provider: gemini
    model: gemini-1.5-flash
    role: summary
    temperature: 0.5

summary_agent: moderator
```

Paths in `memory.persist_directory` and `transcript_path` are resolved relative to the config file (or project root when using `--config`). See `config/README.md` and `config/council.yaml` for the full schema and the default 11 debate personas plus moderator.

## Running the Council

From the project root (with the venv activated):

```bash
python main.py
```

Or with a custom config:

```bash
python main.py --config path/to/council.yaml
```

### Flow

1. **Initial topic**  
   You’re prompted: “Enter your thought, idea, or topic to discuss (start of round 1):”  
   Your reply is broadcast to the forum as the user message for round 1.

2. **Round mode**  
   You’re prompted: “Enter number of rounds to run (or press Enter to prompt after each round; type /quit to end):”
   - **Enter a number** (e.g. `3`): the council runs exactly that many rounds. There is no prompt between rounds; the debate continues using the same topic and round summaries.
   - **Press Enter**: after each round you’re asked to add context for the next round or type `/quit` to end. If you add context, it’s broadcast as the next round’s user message.

3. **During each round**  
   Debate agents are ordered at random. Each agent gets a turn: they see the forum, whispers to them, previous round summaries, and can use tools (including `search`) before broadcasting. Tool calls and whispers are logged in the transcript. Then the summary agent summarizes the round; that summary is stored and shown in the next round.

4. **After the debate**  
   The summary agent produces a final consolidated summary (all rounds, main points, for/against). It’s printed and appended to the transcript file.

### Transcript

The transcript file (see `transcript_path` in config) contains:

- For each round: `-- Round N`, `[User]` + your message, then for each agent `[agent_name]`, any `<tool-call>...`, whispers, and the agent’s reply, then `[summary]` and the round summary.
- At the end: `-- Final summary` and the consolidated summary.

## Project layout

- `main.py` – Entry point; loads config, builds the council, runs the debate.
- `config/council.yaml` – Default agent and memory config.
- `config/README.md` – Short config schema reference.
- `src/config/` – YAML loading and council construction from config.
- `src/council/` – Council (forum, whisper store, agent registry), round runner, transcript writing.
- `src/agent/` – Agent loop, tool-call handling; LLM implementations (Gemini, OpenAI, Ollama).
- `src/forum/` – Forum (broadcasts) and WhisperStore (private messages).
- `src/memory/` – Short-term buffer and long-term Chroma store per agent.
- `src/tool/` – Tool interface and implementations (broadcast, whisper, history, search).
- `src/personality/` – Personality (name, system_prompt, temperature).

## License and contribution

See the repository for license and contribution guidelines.
