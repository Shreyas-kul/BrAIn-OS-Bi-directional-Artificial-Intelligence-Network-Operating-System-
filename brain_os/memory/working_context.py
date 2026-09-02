"""
BrAIn OS — L1 Working Context (Hot Cache)

The fastest tier of the memory hierarchy. Each agent gets its own isolated
working context — a TTL-evicting, capacity-limited in-process dictionary.

OS Analogy: This is the CPU L1 cache. ~0ms access, small capacity, auto-evicts.

Key properties:
    - Per-agent namespace isolation (no cross-agent reads/writes)
    - TTL-based eviction (default 5 min)
    - LRU eviction when capacity exceeded (default 100 items)
    - Thread-safe via asyncio (single event loop)
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    """A single entry in the working context with TTL tracking."""

    value: Any
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if this entry has exceeded its TTL."""
        return (time.monotonic() - self.created_at) > ttl_seconds

    def touch(self) -> None:
        """Update last access time (for LRU tracking)."""
        self.last_accessed = time.monotonic()


class WorkingContext:
    """
    L1 Memory — Per-agent isolated working context.

    This is a TTL + LRU evicting dictionary that provides ~0ms reads
    for the agent's current task state. Each agent process gets its own
    WorkingContext instance — no agent can see another agent's context.

    Usage:
        ctx = WorkingContext(ttl_seconds=300, max_items=100)
        ctx.set("current_task", task_data)
        result = ctx.get("current_task")  # ~0ms
    """

    def __init__(self, ttl_seconds: float = 300.0, max_items: int = 100):
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._ttl_seconds = ttl_seconds
        self._max_items = max_items
        self._hits = 0
        self._misses = 0

    def set(self, key: str, value: Any) -> None:
        """
        Store a value in the working context.

        If capacity is exceeded, the least-recently-used item is evicted.
        """
        # Evict expired entries first
        self._evict_expired()

        # If key exists, remove it so we can re-insert at the end (most recent)
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = CacheEntry(value=value)
        else:
            # Check capacity — evict LRU if full
            while len(self._store) >= self._max_items:
                self._store.popitem(last=False)  # Remove oldest (least recent)
            self._store[key] = CacheEntry(value=value)

    def get(self, key: str) -> Any | None:
        """
        Retrieve a value from the working context.

        Returns None if the key doesn't exist or has expired.
        Expired entries are cleaned up on access.
        """
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None

        # Check TTL
        if entry.is_expired(self._ttl_seconds):
            del self._store[key]
            self._misses += 1
            return None

        # Cache hit — update LRU position
        entry.touch()
        self._store.move_to_end(key)
        self._hits += 1
        return entry.value

    def delete(self, key: str) -> bool:
        """Remove a specific key. Returns True if it existed."""
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        """Wipe the entire working context."""
        self._store.clear()

    def has(self, key: str) -> bool:
        """Check if a non-expired key exists."""
        entry = self._store.get(key)
        if entry is None:
            return False
        if entry.is_expired(self._ttl_seconds):
            del self._store[key]
            return False
        return True

    def keys(self) -> list[str]:
        """Return all non-expired keys."""
        self._evict_expired()
        return list(self._store.keys())

    def size(self) -> int:
        """Return the count of non-expired items."""
        self._evict_expired()
        return len(self._store)

    def __len__(self) -> int:
        """Support len() operator on WorkingContext."""
        return self.size()

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache statistics for monitoring."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "size": len(self._store),
            "max_items": self._max_items,
            "ttl_seconds": self._ttl_seconds,
        }

    def _evict_expired(self) -> int:
        """Remove all expired entries. Returns count of evicted items."""
        expired_keys = [
            k for k, v in self._store.items() if v.is_expired(self._ttl_seconds)
        ]
        for key in expired_keys:
            del self._store[key]
        return len(expired_keys)

    def get_all(self) -> dict[str, Any]:
        """Return all non-expired items as a plain dict (for checkpointing)."""
        self._evict_expired()
        return {k: v.value for k, v in self._store.items()}
