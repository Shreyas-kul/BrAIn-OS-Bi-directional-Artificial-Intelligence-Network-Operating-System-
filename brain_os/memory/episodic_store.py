"""
BrAIn OS — L2 Episodic Store (Warm Cache)

The middle tier of the memory hierarchy. Combines Redis hash maps for
structured data with vector embeddings for semantic search.

OS Analogy: This is the CPU L2/L3 cache + swap. ~2ms access, moderate capacity,
session-scoped with TTL expiry.

Key properties:
    - Redis-backed for persistence across process restarts
    - Per-agent namespace isolation via key prefix (mem:{pid}:*)
    - Vector embeddings stored alongside data for semantic recall
    - Session-scoped with configurable TTL
    - Supports both exact-match and semantic (cosine similarity) retrieval
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import numpy as np
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class EpisodicStore:
    """
    L2 Memory — Redis-backed episodic store with vector search.

    Stores data as Redis hashes with per-agent namespace isolation.
    Embeddings are stored in a parallel key space for semantic retrieval.

    Usage:
        store = EpisodicStore(redis_client, session_ttl=3600)
        await store.store(agent_pid="research_001", key="finding_1", value=data, embedding=vec)
        results = await store.search_semantic(agent_pid="research_001", query_embedding=q, top_k=5)
    """

    # Redis key prefixes for namespace isolation
    DATA_PREFIX = "brain:mem"
    EMBED_PREFIX = "brain:emb"
    INDEX_PREFIX = "brain:idx"

    def __init__(self, redis_client: redis.Redis, session_ttl: int = 3600):
        self._redis = redis_client
        self._session_ttl = session_ttl

    # ------------------------------------------------------------------
    # Write Operations
    # ------------------------------------------------------------------

    async def store(
        self,
        agent_pid: str,
        key: str,
        value: Any,
        embedding: list[float] | np.ndarray | None = None,
    ) -> None:
        """
        Store a key-value pair with optional embedding in the episodic store.

        Args:
            agent_pid: Agent process ID (namespace isolation)
            key: Storage key
            value: Any JSON-serializable value
            embedding: Optional vector embedding for semantic search
        """
        data_key = f"{self.DATA_PREFIX}:{agent_pid}"
        serialized = json.dumps(value, default=str)

        # Store the data in a hash map
        await self._redis.hset(data_key, key, serialized)
        await self._redis.expire(data_key, self._session_ttl)

        # Store the embedding if provided
        if embedding is not None:
            emb_key = f"{self.EMBED_PREFIX}:{agent_pid}:{key}"
            emb_bytes = np.array(embedding, dtype=np.float32).tobytes()
            await self._redis.set(emb_key, emb_bytes, ex=self._session_ttl)

            # Add to the embedding index for this agent
            idx_key = f"{self.INDEX_PREFIX}:{agent_pid}"
            await self._redis.sadd(idx_key, key)
            await self._redis.expire(idx_key, self._session_ttl)

        logger.debug(
            "Episodic store: wrote key=%s for agent=%s (has_embedding=%s)",
            key,
            agent_pid,
            embedding is not None,
        )

    # ------------------------------------------------------------------
    # Read Operations
    # ------------------------------------------------------------------

    async def retrieve(self, agent_pid: str, key: str) -> Any | None:
        """Exact-match retrieval by key."""
        data_key = f"{self.DATA_PREFIX}:{agent_pid}"
        raw = await self._redis.hget(data_key, key)
        if raw is None:
            return None
        return json.loads(raw)

    async def retrieve_all(self, agent_pid: str) -> dict[str, Any]:
        """Retrieve all stored items for an agent."""
        data_key = f"{self.DATA_PREFIX}:{agent_pid}"
        raw_items = await self._redis.hgetall(data_key)
        return {k.decode() if isinstance(k, bytes) else k: json.loads(v) for k, v in raw_items.items()}

    async def search_semantic(
        self,
        agent_pid: str,
        query_embedding: list[float] | np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Semantic search via cosine similarity over stored embeddings.

        Returns top-k results sorted by similarity score, each containing:
            {"key": str, "value": Any, "score": float}
        """
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm

        # Get all embedding keys for this agent
        idx_key = f"{self.INDEX_PREFIX}:{agent_pid}"
        member_keys = await self._redis.smembers(idx_key)
        if not member_keys:
            return []

        # Compute cosine similarity for each stored embedding
        scored_results: list[tuple[str, float]] = []
        for member in member_keys:
            member_str = member.decode() if isinstance(member, bytes) else member
            emb_key = f"{self.EMBED_PREFIX}:{agent_pid}:{member_str}"
            emb_bytes = await self._redis.get(emb_key)
            if emb_bytes is None:
                continue

            stored_vec = np.frombuffer(emb_bytes, dtype=np.float32)
            stored_norm = np.linalg.norm(stored_vec)
            if stored_norm == 0:
                continue

            cosine_sim = float(np.dot(query_vec, stored_vec / stored_norm))
            scored_results.append((member_str, cosine_sim))

        # Sort by score descending, take top-k
        scored_results.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_results[:top_k]

        # Fetch the actual values for top results
        results = []
        data_key = f"{self.DATA_PREFIX}:{agent_pid}"
        for key, score in top_results:
            raw = await self._redis.hget(data_key, key)
            value = json.loads(raw) if raw else None
            results.append({"key": key, "value": value, "score": score})

        return results

    # ------------------------------------------------------------------
    # Management Operations
    # ------------------------------------------------------------------

    async def delete(self, agent_pid: str, key: str) -> bool:
        """Delete a specific key and its embedding."""
        data_key = f"{self.DATA_PREFIX}:{agent_pid}"
        emb_key = f"{self.EMBED_PREFIX}:{agent_pid}:{key}"
        idx_key = f"{self.INDEX_PREFIX}:{agent_pid}"

        deleted = await self._redis.hdel(data_key, key)
        await self._redis.delete(emb_key)
        await self._redis.srem(idx_key, key)
        return deleted > 0

    async def clear_agent(self, agent_pid: str) -> None:
        """Clear all data for a specific agent (session end)."""
        # Get all embedding keys for this agent
        idx_key = f"{self.INDEX_PREFIX}:{agent_pid}"
        members = await self._redis.smembers(idx_key)

        # Delete embeddings
        for member in members:
            member_str = member.decode() if isinstance(member, bytes) else member
            emb_key = f"{self.EMBED_PREFIX}:{agent_pid}:{member_str}"
            await self._redis.delete(emb_key)

        # Delete data hash and index
        data_key = f"{self.DATA_PREFIX}:{agent_pid}"
        await self._redis.delete(data_key, idx_key)

        logger.info("Episodic store: cleared all data for agent=%s", agent_pid)

    async def count(self, agent_pid: str) -> int:
        """Count stored items for an agent."""
        data_key = f"{self.DATA_PREFIX}:{agent_pid}"
        return await self._redis.hlen(data_key)

    async def generate_key(self, content: str) -> str:
        """Generate a deterministic key from content (for deduplication)."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
