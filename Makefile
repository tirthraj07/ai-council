# ai-council Makefile
# Run from project root: make clean-memory

# Paths matching config/council.yaml (memory.persist_directory, transcript_path)
CHROMA_DIR := data/chroma
TRANSCRIPT_FILE := debate_transcript.txt

.PHONY: clean-memory clean-memory-help

clean-memory:
	@echo Cleaning persistent memory and transcript...
	@uv run python scripts/clean_memory.py

clean-memory-help:
	@echo "clean-memory: Remove ChromaDB data (data/chroma) and debate transcript (debate_transcript.txt)."
	@echo "             Use this to reset long-term memory and start a fresh debate."
