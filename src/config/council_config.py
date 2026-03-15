"""
Load council configuration from YAML and build Council with agents.
"""

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

from src.agent import Agent, LLM
from src.agent.impl.GeminiLLM import GeminiLLM
from src.council import Council
from src.memory import create_agent_memory
from src.personality import Personality
from src.tool import ToolRegistry
from src.tool.impl.history_tool import HistoryTool

_PROVIDER_LLM: dict[str, type] = {
    "gemini": GeminiLLM,
}


def _get_llm_class(provider: str) -> type:
    provider = (provider or "").strip().lower()
    if provider not in _PROVIDER_LLM:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported: {', '.join(_PROVIDER_LLM)}"
        )
    return _PROVIDER_LLM[provider]


def _create_llm(provider: str, model: str, **kwargs: Any) -> LLM:
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

    for entry in agents_cfg:
        name = entry.get("name")
        if not name:
            raise ValueError("Each agent must have 'name'.")
        name = str(name).strip()
        system_prompt = entry.get("system_prompt", "")
        if isinstance(system_prompt, list):
            system_prompt = "\n".join(system_prompt)
        provider = entry.get("provider", "gemini")
        model = entry.get("model", "gemini-1.5-flash")
        role = entry.get("role", "debate")
        temperature = float(entry.get("temperature", 0.7))

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
        llm = _create_llm(provider, model)
        personality = Personality(
            name=name,
            system_prompt=system_prompt.strip(),
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
