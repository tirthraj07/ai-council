"""
Load council configuration from YAML and build Council with agents.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

from src.agent import Agent, LLM
from src.agent.impl.GeminiLLM import GeminiLLM
from src.agent.impl.OpenAILLM import OpenAILLM
from src.agent.impl.OllamaLLM import OllamaLLM
from src.council import Council
from src.memory import create_agent_memory
from src.personality import Personality
from src.tool import ToolRegistry
from src.tool.impl.history_tool import HistoryTool
from src.tool.impl.search_tool import SearchTool

_PROVIDER_LLM: dict[str, type] = {
    "gemini": GeminiLLM,
    "openai": OpenAILLM,
    "ollama": OllamaLLM,
}


def _get_llm_class(provider: str) -> type:
    provider = (provider or "").strip().lower()
    if provider not in _PROVIDER_LLM:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: {', '.join(_PROVIDER_LLM)}"
        )
    return _PROVIDER_LLM[provider]


_DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-flash-latest",
    "openai": "gpt-5-mini",
    "ollama": "llama3:8b",
}

# Appended to every agent's system prompt so debate stays direct and evidence-based.
_SHARED_SYSTEM_INSTRUCTION = (
    "\n\n"
    "Do not sugar-coat your responses. Explain things as they are; be direct and clear. "
    "When possible and relevant, support your statements with numbers, data, or facts."
)


def _create_llm(provider: str, model: str, **kwargs: Any) -> LLM:
    if not model or not str(model).strip():
        model = _DEFAULT_MODELS.get(provider, "gemini-flash-latest")
    cls = _get_llm_class(provider)
    return cls(provider_name=provider, model_name=model, **kwargs)


def load_council_config(path: str | Path) -> dict[str, Any]:
    """Load and parse the council YAML file. Returns the raw config dict."""
    if yaml is None:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping (dict).")
    return data


def _create_agent_memory_for_entry(
    entry: dict[str, Any],
    forum: Any,
    whisper_store: Any,
    short_term_max: int,
    long_term_n: int,
    chroma_client: Any = None,
) -> tuple[str, Any]:
    """Create AgentMemory for one config entry. Used for parallel setup. Returns (agent_name, memory)."""
    name = str(entry.get("name", "")).strip()
    if not name:
        raise ValueError("Each agent must have 'name'.")
    memory = create_agent_memory(
        agent_id=name,
        short_term_max_messages=short_term_max,
        long_term_retrieve_n=long_term_n,
        long_term_client=chroma_client,
        forum=forum,
        whisper_store=whisper_store,
    )
    return (name, memory)


def build_council_from_config(
    config_path: str | Path,
    *,
    base_dir: Path | None = None,
) -> tuple[Council, dict[str, Any]]:
    """
    Load config from YAML and build a Council with all agents.
    Returns (council, options) where options includes transcript_path, etc.
    Paths in the config are resolved relative to the config file's directory
    unless base_dir is set.
    """
    config = load_council_config(config_path)
    config_dir = Path(config_path).resolve().parent
    base = Path(base_dir) if base_dir is not None else config_dir

    memory_cfg = config.get("memory") or {}
    persist_dir = memory_cfg.get("persist_directory")
    if persist_dir:
        persist_dir = str((base / persist_dir).resolve())
    short_term_max = memory_cfg.get("short_term_max_messages", 10)
    long_term_n = memory_cfg.get("long_term_retrieve_n", 5)

    transcript_path = config.get("transcript_path", "debate_transcript.txt")
    transcript_path = base / transcript_path

    council = Council()
    agents_cfg = config.get("agents")
    if not agents_cfg:
        raise ValueError("Config must contain 'agents' list with at least one agent.")

    if persist_dir:
        if chromadb is None:
            raise ImportError("Chroma is required for persistent memory. Install with: pip install chromadb")
        print("Setting up agent memories (Chroma)...", flush=True)
        chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        memories_by_name: dict[str, Any] = {}
        total = len(agents_cfg)
        with ThreadPoolExecutor(max_workers=min(total, 16)) as executor:
            futures = {
                executor.submit(
                    _create_agent_memory_for_entry,
                    entry,
                    council.forum,
                    council.whisper_store,
                    short_term_max,
                    long_term_n,
                    chroma_client,
                ): entry
                for entry in agents_cfg
            }
            done = 0
            for future in as_completed(futures):
                name, memory = future.result()
                memories_by_name[name] = memory
                done += 1
                print(f"  Memory ready: {name} ({done}/{total})", flush=True)
        print("Agent memories ready.", flush=True)
    else:
        memories_by_name = {}

    for entry in agents_cfg:
        name = entry.get("name")
        if not name:
            raise ValueError("Each agent must have 'name'.")
        name = str(name).strip()
        system_prompt = entry.get("system_prompt", "")
        if isinstance(system_prompt, list):
            system_prompt = "\n".join(system_prompt)
        system_prompt = (system_prompt.strip() + _SHARED_SYSTEM_INSTRUCTION).strip()
        provider = entry.get("provider", "gemini")
        model = entry.get("model") or _DEFAULT_MODELS.get(provider, "gemini-flash-latest")
        role = entry.get("role", "debate")
        temperature = float(entry.get("temperature", 0.7))

        if name in memories_by_name:
            memory = memories_by_name[name]
        else:
            memory = create_agent_memory(
                agent_id=name,
                short_term_max_messages=short_term_max,
                long_term_persist_path=persist_dir,
                long_term_retrieve_n=long_term_n,
                forum=council.forum,
                whisper_store=council.whisper_store,
            )
        tools = ToolRegistry()
        tools.register(HistoryTool(memory))
        try:
            tools.register(SearchTool(max_results=5, scrape_top=1))
        except ImportError:
            pass
        llm = _create_llm(provider, model, temperature=temperature)
        personality = Personality(
            name=name,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        agent = Agent(llm=llm, personality=personality, tools=tools, memory=memory)
        council.add_agent(name, agent, role=role)

    summary_name = config.get("summary_agent")
    if summary_name:
        summary_name = str(summary_name).strip()
        council.set_summary_agent(summary_name)

    options = {
        "transcript_path": transcript_path,
        "config_path": Path(config_path).resolve(),
    }
    return council, options
