"""
BrAIn OS — Hierarchical Memory Controller

The unified memory manager that orchestrates L1 → L2 → L3 like an OS
manages the CPU cache hierarchy. Provides a single API for all memory
operations: remember(), recall(), and forget().

Write path:  remember() → L1 (immediate) → L2 (async write-through) → L3 (async persist)
Read path:   recall()   → L1 (check first) → L2 (semantic search) → L3 (deep recall)
                           with L1 cache promotion on hits from lower tiers.

The controller also handles:
    - Embedding generation (via sentence-transformers)
    - Reranking (via cross-encoder)
    - Eviction coordination across tiers
    - Memory statistics for monitoring
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import redis.asyncio as redis

from brain_os.config.settings import get_settings
from brain_os.memory.episodic_store import EpisodicStore
from brain_os.memory.persistent_store import PersistentStore
from brain_os.memory.reranker import ReRanker
from brain_os.memory.working_context import WorkingContext

logger = logging.getLogger(__name__)

# Lazy-loaded embedding model
_embedding_model = None


def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load the sentence-transformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", model_name)
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


class MemoryController:
    """
    OS-style hierarchical memory manager.

    Orchestrates the three memory tiers and provides a unified API:
        - remember(): Write data through the hierarchy
        - recall():   Read data with cascading search
        - forget():   Remove data from specific tiers

    Each agent gets its own isolated memory space through the controller.

    Usage:
        controller = MemoryController(redis_client=redis, persist_dir="./data/chromadb")
        await controller.remember("agent_001", "finding", "Paris is the capital of France")
        results = await controller.recall("agent_001", "What is the capital of France?", top_k=5)
    """

    def __init__(
        self,
        redis_client: redis.Redis | None = None,
        persist_dir: str = "./data/chromadb",
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        settings = get_settings()

        if redis_client is None:
            try:
                redis_client = redis.from_url(settings.redis.url)
            except Exception:
                try:
                    import fakeredis.aioredis as fake_aio
                    redis_client = fake_aio.FakeRedis()
                    logger.info("Using FakeRedis fallback for MemoryController L2 store")
                except Exception:
                    redis_client = None

        # Initialize memory tiers
        self._working_contexts: dict[str, WorkingContext] = {}
        self._episodic = EpisodicStore(
            redis_client=redis_client,
            session_ttl=settings.memory.l2_session_ttl_seconds,
        )
        self._persistent = PersistentStore(persist_dir=persist_dir)
        self._reranker = ReRanker(model_name=reranker_model)

        # Model names (lazy-loaded)
        self._embedding_model_name = embedding_model

        # Config
        self._l1_ttl = settings.memory.l1_cache_ttl_seconds
        self._l1_max_items = settings.memory.l1_cache_max_items

        logger.info("MemoryController initialized (L1 → L2 → L3)")

    # ------------------------------------------------------------------
    # Working Context Management
    # ------------------------------------------------------------------

    def _get_context(self, agent_pid: str) -> WorkingContext:
        """Get or create the L1 working context for an agent."""
        if agent_pid not in self._working_contexts:
            self._working_contexts[agent_pid] = WorkingContext(
                ttl_seconds=self._l1_ttl,
                max_items=self._l1_max_items,
            )
        return self._working_contexts[agent_pid]

    # ------------------------------------------------------------------
    # Embedding Generation
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the given text."""
        model = _get_embedding_model(self._embedding_model_name)
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (batched for efficiency)."""
        model = _get_embedding_model(self._embedding_model_name)
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    # ------------------------------------------------------------------
    # Write Path: remember()
    # ------------------------------------------------------------------

    async def remember(
        self,
        agent_pid: str,
        key: str,
        value: Any,
        persist: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Store data through the memory hierarchy.

        Write path:
            1. L1 — Immediate write to working context (~0ms)
            2. L2 — Async write-through to Redis with embedding (~2ms)
            3. L3 — Async persist to ChromaDB (if persist=True, ~15ms)

        Args:
            agent_pid: Agent process ID (namespace isolation)
            key: Storage key
            value: The data to store (any JSON-serializable value)
            persist: Whether to persist to L3 (ChromaDB)
            metadata: Optional metadata for L3 storage
        """
        # L1: Immediate write
        context = self._get_context(agent_pid)
        context.set(key, value)

        # Generate embedding for the value
        text_value = value if isinstance(value, str) else str(value)
        embedding = self._embed(text_value)

        # L2: Async write-through to Redis
        await self._episodic.store(
            agent_pid=agent_pid,
            key=key,
            value=value,
            embedding=embedding,
        )

        # L3: Async persist to ChromaDB
        if persist:
            self._persistent.store(
                agent_id=agent_pid,
                content=text_value,
                embedding=embedding,
                metadata=metadata,
                doc_id=key,
            )

        logger.debug(
            "Memory: remembered key=%s for agent=%s (L1+L2%s)",
            key,
            agent_pid,
            "+L3" if persist else "",
        )

    # ------------------------------------------------------------------
    # Read Path: recall()
    # ------------------------------------------------------------------

    async def recall(
        self,
        agent_pid: str,
        query: str,
        top_k: int = 5,
        use_reranker: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Recall data from the memory hierarchy with cascading search.

        Read path:
            1. L1 — Check working context for exact match (~0ms)
            2. L2 — Semantic search in Redis + vectors (~2ms)
            3. L3 — Deep recall from ChromaDB (~15ms)
            4. Rerank — Cross-encoder refinement of combined results

        Results from lower tiers are promoted to L1 (cache warming).

        Args:
            agent_pid: Agent process ID
            query: Natural language query for semantic search
            top_k: Number of results to return
            use_reranker: Whether to apply cross-encoder reranking

        Returns:
            List of results sorted by relevance, each containing:
                {"key": str, "value": Any, "score": float, "source": str}
        """
        results: list[dict[str, Any]] = []

        # L1: Check working context for exact match
        context = self._get_context(agent_pid)
        l1_hit = context.get(query)
        if l1_hit is not None:
            results.append({
                "key": query,
                "value": l1_hit,
                "score": 1.0,
                "source": "L1_working_context",
            })

        # Generate query embedding for semantic search
        query_embedding = self._embed(query)

        # L2: Semantic search in episodic store
        l2_results = await self._episodic.search_semantic(
            agent_pid=agent_pid,
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Fetch more for reranking
        )
        for r in l2_results:
            results.append({
                "key": r["key"],
                "value": r["value"],
                "score": r["score"],
                "source": "L2_episodic",
            })

        # L3: Deep recall from persistent store
        l3_results = self._persistent.search(
            query=query,
            agent_id=agent_pid,
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Fetch more for reranking
        )
        for r in l3_results:
            results.append({
                "key": r["id"],
                "value": r["content"],
                "score": 1.0 - r.get("distance", 0.0),  # Convert distance to similarity
                "source": "L3_persistent",
                "metadata": r.get("metadata", {}),
            })

        if not results:
            return []

        # Deduplicate by key (prefer higher-tier results)
        seen_keys: set[str] = set()
        unique_results = []
        for r in results:
            if r["key"] not in seen_keys:
                seen_keys.add(r["key"])
                unique_results.append(r)
        results = unique_results

        # Apply cross-encoder reranking for precision
        if use_reranker and len(results) > 1:
            documents = [
                str(r["value"]) if not isinstance(r["value"], str) else r["value"]
                for r in results
            ]
            reranked = await self._reranker.rerank(
                query=query,
                documents=documents,
                top_k=top_k,
            )
            # Map reranked indices back to results
            final_results = []
            for rr in reranked:
                orig_idx = rr["original_index"]
                result = results[orig_idx].copy()
                result["rerank_score"] = rr["score"]
                final_results.append(result)
            results = final_results
        else:
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

        # Promote top results to L1 (cache warming)
        for r in results[:3]:
            context.set(r["key"], r["value"])

        logger.debug(
            "Memory: recalled %d results for agent=%s query='%s...'",
            len(results),
            agent_pid,
            query[:50],
        )

        return results

    # ------------------------------------------------------------------
    # Direct L1 Access (for current task state)
    # ------------------------------------------------------------------

    def get_working(self, agent_pid: str, key: str) -> Any | None:
        """Direct L1 read — for current task variables, not semantic search."""
        return self._get_context(agent_pid).get(key)

    def set_working(self, agent_pid: str, key: str, value: Any) -> None:
        """Direct L1 write — for current task variables."""
        self._get_context(agent_pid).set(key, value)

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    async def clear_agent_memory(self, agent_pid: str) -> None:
        """Clear all memory for an agent across all tiers."""
        # L1
        if agent_pid in self._working_contexts:
            self._working_contexts[agent_pid].clear()
            del self._working_contexts[agent_pid]

        # L2
        await self._episodic.clear_agent(agent_pid)

        # L3 is append-only — we don't delete persistent data
        logger.info("Memory: cleared L1+L2 for agent=%s (L3 preserved)", agent_pid)

    def get_stats(self, agent_pid: str) -> dict[str, Any]:
        """Get memory statistics for an agent."""
        context = self._get_context(agent_pid)
        return {
            "l1": context.stats,
            "l3_count": self._persistent.count(agent_pid),
        }
