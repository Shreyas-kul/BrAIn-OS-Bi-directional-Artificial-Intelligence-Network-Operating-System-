"""
BrAIn OS — Two-Stage Reranking Pipeline

After initial retrieval from memory (L2/L3), results pass through a
cross-encoder reranker for precision refinement.

Pipeline:
    Stage 1 — Bi-Encoder (fast, approximate): Cosine similarity on pre-computed
              embeddings. Retrieves top-50 candidates. Already done in L2/L3.

    Stage 2 — Cross-Encoder (accurate, slow): Jointly scores each (query, document)
              pair through a cross-encoder model. Re-orders and returns top-5.

Why two stages? Bi-encoders are fast but miss nuance (they encode query and doc
independently). Cross-encoders are accurate but slow on large sets (they process
query+doc together). Two-stage gives both speed AND precision — 40%+ relevance
improvement over single-stage retrieval.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-load the cross-encoder to avoid slow startup
_cross_encoder = None


def _get_cross_encoder(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Lazy-load the cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder model: %s", model_name)
        _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


class ReRanker:
    """
    Cross-encoder reranking pipeline for memory recall precision.

    Takes candidate documents from bi-encoder retrieval (Stage 1)
    and re-scores them with a cross-encoder (Stage 2) for higher accuracy.

    Usage:
        reranker = ReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        results = await reranker.rerank(
            query="What is the capital of France?",
            documents=["Paris is the capital...", "France is in Europe...", ...],
            top_k=5
        )
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Re-rank documents using the cross-encoder model.

        Args:
            query: The search query
            documents: List of candidate documents from Stage 1 retrieval
            top_k: Number of top results to return
            score_threshold: Minimum score to include in results

        Returns:
            List of re-ranked results, each containing:
                {"content": str, "score": float, "original_index": int}
        """
        if not documents:
            return []

        # If we have fewer documents than top_k, just score them all
        if len(documents) <= top_k:
            top_k = len(documents)

        # Create (query, document) pairs for cross-encoder
        pairs = [(query, doc) for doc in documents]

        # Score all pairs with the cross-encoder
        cross_encoder = _get_cross_encoder(self._model_name)
        scores = cross_encoder.predict(pairs)

        # Sort by score descending
        scored_docs = [
            {"content": doc, "score": float(score), "original_index": i}
            for i, (doc, score) in enumerate(zip(documents, scores))
        ]
        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        # Filter by threshold and take top-k
        results = [
            doc for doc in scored_docs[:top_k] if doc["score"] >= score_threshold
        ]

        logger.debug(
            "Reranker: %d candidates → %d results (top score=%.4f)",
            len(documents),
            len(results),
            results[0]["score"] if results else 0.0,
        )

        return results

    async def rerank_with_metadata(
        self,
        query: str,
        items: list[dict[str, Any]],
        content_key: str = "content",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Re-rank items that have metadata attached.

        Preserves all original metadata while adding reranker scores.

        Args:
            query: The search query
            items: List of dicts, each with at least a content_key field
            content_key: Key in each dict that contains the text to score
            top_k: Number of results to return

        Returns:
            Re-ranked items with added "rerank_score" field
        """
        if not items:
            return []

        documents = [item.get(content_key, "") for item in items]
        pairs = [(query, doc) for doc in documents]

        cross_encoder = _get_cross_encoder(self._model_name)
        scores = cross_encoder.predict(pairs)

        # Attach scores to original items
        for item, score in zip(items, scores):
            item["rerank_score"] = float(score)

        # Sort by rerank score
        items.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        return items[:top_k]
