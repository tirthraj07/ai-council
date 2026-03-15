# Council configuration

The council is defined in YAML. By default `main.py` loads `config/council.yaml`. Override with the `COUNCIL_CONFIG` environment variable or `--config` argument.

## Schema

- **memory** (optional): `persist_directory`, `short_term_max_messages`, `long_term_retrieve_n`. Paths are relative to the config file directory (or project root when using `--config` with a path).
- **transcript_path** (optional): Where to write the debate transcript. Default `debate_transcript.txt`.
- **agents**: List of agent definitions. Each entry:
  - **name** (required): Unique agent name.
  - **system_prompt** (required): Personality and instructions (string or multi-line).
  - **provider** (optional): LLM provider, e.g. `gemini`. Default `gemini`.
  - **model** (optional): Model name, e.g. `gemini-1.5-flash`. Default `gemini-1.5-flash`.
  - **role** (optional): `debate` (speaks in rounds, has broadcast/whisper) or `summary` (summarizes only). Default `debate`.
  - **temperature** (optional): Float for the LLM. Default `0.7`.
- **summary_agent** (optional): Name of the agent that performs per-round and final summaries. Must match one of the agent names and should have `role: summary`.

## Example

See `council.yaml` in this directory.
