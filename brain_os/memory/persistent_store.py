"""
BrAIn OS — L3 Persistent Store (Cold Storage)

The deepest tier of the memory hierarchy. Uses ChromaDB for permanent,
append-only semantic storage with full vector search.

OS Analogy: This is the hard disk / SSD. ~15ms access, unlimited capacity,
data persists forever (append-only).

Key properties:
    - ChromaDB-backed with per-agent collections
    - Append-only — data is never deleted (full history)
    - Full semantic search via embeddings
    - Metadata filtering (timestamps, tags, sources)
    - Used for long-term knowledge base and cross-session recall
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import chromadb

logger = logging.getLogger(__name__)


class PersistentStore:
    """
    L3 Memory — ChromaDB-backed persistent vector store.

    Each agent gets its own ChromaDB collection for complete isolation.
    Data is append-only and persists across sessions and restarts.

    Usage:
        store = PersistentStore(persist_dir="./data/chromadb")
        await store.store(
            agent_id="research",
            content="Paris is the capital of France",
            embedding=[0.1, 0.2, ...],
            metadata={"source": "web_search", "confidence": 0.95}
        )
        results = await store.search("capital of France", agent_id="research", top_k=5)
    """

    COLLECTION_PREFIX = "brain_os"

    def __init__(self, persist_dir: str = "./data/chromadb"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collections: dict[str, chromadb.Collection] = {}
        logger.info("Persistent store initialized at %s", persist_dir)

    def _get_collection(self, agent_id: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection for an agent."""
        if agent_id not in self._collections:
            collection_name = f"{self.COLLECTION_PREFIX}_{agent_id}"
            self._collections[agent_id] = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )
        return self._collections[agent_id]

    # ------------------------------------------------------------------
    # Write Operations
    # ------------------------------------------------------------------

    def store(
        self,
        agent_id: str,
        content: str,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
    ) -> str:
        """
        Store content in the persistent store.

        Args:
            agent_id: Agent identifier (collection isolation)
            content: Text content to store
            embedding: Pre-computed embedding vector (if None, ChromaDB computes one)
            metadata: Additional metadata (source, timestamp, tags, etc.)
            doc_id: Optional document ID (auto-generated if not provided)

        Returns:
            The document ID of the stored item
        """
        collection = self._get_collection(agent_id)
        doc_id = doc_id or str(uuid.uuid4())

        # Build metadata with timestamp
        full_metadata = {
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
        }
        if metadata:
            full_metadata.update(metadata)

        # Store with or without pre-computed embeddings
        kwargs: dict[str, Any] = {
            "ids": [doc_id],
            "documents": [content],
            "metadatas": [full_metadata],
        }
        if embedding is not None:
            kwargs["embeddings"] = [embedding]

        collection.upsert(**kwargs)

        logger.debug(
            "Persistent store: stored doc_id=%s for agent=%s (len=%d)",
            doc_id,
            agent_id,
            len(content),
        )
        return doc_id

    def store_batch(
        self,
        agent_id: str,
        contents: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[str]:
        """Store multiple items in a single batch operation."""
        collection = self._get_collection(agent_id)

        if doc_ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in contents]

        now = datetime.now(timezone.utc).isoformat()
        full_metadatas = []
        for i, content in enumerate(contents):
            meta = {"stored_at": now, "agent_id": agent_id}
            if metadatas and i < len(metadatas):
                meta.update(metadatas[i])
            full_metadatas.append(meta)

        kwargs: dict[str, Any] = {
            "ids": doc_ids,
            "documents": contents,
            "metadatas": full_metadatas,
        }
        if embeddings is not None:
            kwargs["embeddings"] = embeddings

        collection.upsert(**kwargs)
        logger.info("Persistent store: batch stored %d docs for agent=%s", len(contents), agent_id)
        return doc_ids

    # ------------------------------------------------------------------
    # Read / Search Operations
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        agent_id: str,
        query_embedding: list[float] | None = None,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search over the persistent store.

        Args:
            query: Text query for search
            agent_id: Agent whose collection to search
            query_embedding: Pre-computed query embedding (faster if provided)
            top_k: Number of results to return
            where: Optional metadata filter (ChromaDB where clause)

        Returns:
            List of results, each containing:
                {"id": str, "content": str, "metadata": dict, "distance": float}
        """
        collection = self._get_collection(agent_id)

        kwargs: dict[str, Any] = {"n_results": top_k}

        if query_embedding is not None:
            kwargs["query_embeddings"] = [query_embedding]
        else:
            kwargs["query_texts"] = [query]

        if where:
            kwargs["where"] = where

        try:
            raw_results = collection.query(**kwargs)
        except Exception as e:
            logger.error("Persistent store search failed: %s", e)
            return []

        # Format results
        results = []
        if raw_results and raw_results["ids"]:
            for i, doc_id in enumerate(raw_results["ids"][0]):
                results.append({
                    "id": doc_id,
                    "content": raw_results["documents"][0][i] if raw_results["documents"] else "",
                    "metadata": raw_results["metadatas"][0][i] if raw_results["metadatas"] else {},
                    "distance": raw_results["distances"][0][i] if raw_results["distances"] else 0.0,
                })

        return results

    def get_by_id(self, agent_id: str, doc_id: str) -> dict[str, Any] | None:
        """Retrieve a specific document by ID."""
        collection = self._get_collection(agent_id)
        try:
            result = collection.get(ids=[doc_id], include=["documents", "metadatas", "embeddings"])
            if result and result["ids"]:
                return {
                    "id": result["ids"][0],
                    "content": result["documents"][0] if result["documents"] else "",
                    "metadata": result["metadatas"][0] if result["metadatas"] else {},
                }
        except Exception as e:
            logger.error("Persistent store get_by_id failed: %s", e)
        return None

    # ------------------------------------------------------------------
    # Management Operations
    # ------------------------------------------------------------------

    def count(self, agent_id: str) -> int:
        """Count total documents for an agent."""
        collection = self._get_collection(agent_id)
        return collection.count()

    def list_agents(self) -> list[str]:
        """List all agent IDs that have collections."""
        collections = self._client.list_collections()
        return [
            c.name.replace(f"{self.COLLECTION_PREFIX}_", "")
            for c in collections
            if c.name.startswith(self.COLLECTION_PREFIX)
        ]

    def delete_collection(self, agent_id: str) -> None:
        """Delete an entire agent collection (use with caution)."""
        collection_name = f"{self.COLLECTION_PREFIX}_{agent_id}"
        try:
            self._client.delete_collection(collection_name)
            self._collections.pop(agent_id, None)
            logger.warning("Persistent store: deleted collection for agent=%s", agent_id)
        except Exception as e:
            logger.error("Failed to delete collection: %s", e)
