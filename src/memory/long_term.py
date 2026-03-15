"""
Long-term memory: persisted in a vector database (Chroma). Each agent has its own
collection so memories are isolated per agent. Retrieved by similarity and injected
into the prompt when relevant.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None  # type: ignore[assignment]


class LongTermMemory:
    """
    Vector-backed long-term memory using Chroma. One collection per agent.
    Call add() to store facts or conversation summaries; call retrieve() to get
    relevant memories for the current context.
    """

    def __init__(
        self,
        agent_id: str,
        persist_directory: str | Path | None = None,
        embedding_function: Any = None,
    ):
        """
        Args:
            agent_id: Unique identifier for this agent. Used as the Chroma collection name.
            persist_directory: Directory for Chroma persistence. If None, uses in-memory only.
            embedding_function: Chroma embedding function. If None, uses Chroma default.
        """
        if chromadb is None:
            raise ImportError(
                "Chroma is required for long-term memory. Install with: pip install chromadb"
            )
        self._agent_id = self._sanitize_collection_name(agent_id)
        self._persist_directory = str(persist_directory) if persist_directory else None
        self._embedding_function = embedding_function
        self._client = self._create_client()
        kwargs = {"name": self._agent_id}
        if self._embedding_function is not None:
            kwargs["embedding_function"] = self._embedding_function
        self._collection = self._client.get_or_create_collection(**kwargs)

    @staticmethod
    def _sanitize_collection_name(name: str) -> str:
        """Chroma collection names must match [a-zA-Z0-9_.-]+."""
        return "".join(c if c.isalnum() or c in "_.-" else "_" for c in name)

    def _create_client(self):
        if self._persist_directory:
            return chromadb.PersistentClient(
                path=self._persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        return chromadb.Client(settings=Settings(anonymized_telemetry=False))

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        id: str | None = None,
    ) -> str:
        """
        Store a memory (e.g. fact, summary, or important exchange).
        Returns the id of the added document.
        """
        doc_id = id or str(uuid.uuid4())
        meta = metadata or {}
        self._collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[meta],
        )
        return doc_id

    def add_many(
        self,
        contents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Store multiple memories. Returns list of assigned ids."""
        count = len(contents)
        if metadatas is None:
            metadatas = [{}] * count
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(count)]
        if len(metadatas) != count or len(ids) != count:
            raise ValueError("contents, metadatas, and ids must have the same length")
        self._collection.add(ids=ids, documents=contents, metadatas=metadatas)
        return ids

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve memories relevant to the query. Returns list of dicts with
        'content' and 'metadata' for each result.
        """
        result = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas"],
        )
        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        return [
            {"content": doc, "metadata": meta or {}}
            for doc, meta in zip(documents, metadatas)
        ]

    def delete(self, ids: list[str]) -> None:
        """Remove memories by id."""
        if ids:
            self._collection.delete(ids=ids)
