"""
Unit tests for BrAIn OS Memory Hierarchy (L1 Working Context & Controller).
"""

import time
import pytest

from brain_os.memory.working_context import WorkingContext


def test_working_context_basic():
    ctx = WorkingContext(ttl_seconds=10.0, max_items=5)
    ctx.set("key1", "val1")
    ctx.set("key2", "val2")

    assert ctx.get("key1") == "val1"
    assert ctx.get("key2") == "val2"
    assert ctx.get("nonexistent") is None
    assert len(ctx) == 2


def test_working_context_lru_eviction():
    ctx = WorkingContext(ttl_seconds=10.0, max_items=2)
    ctx.set("k1", "v1")
    ctx.set("k2", "v2")
    # Adding a 3rd item should evict oldest (k1)
    ctx.set("k3", "v3")

    assert ctx.get("k1") is None
    assert ctx.get("k2") == "v2"
    assert ctx.get("k3") == "v3"


def test_working_context_ttl_expiration():
    ctx = WorkingContext(ttl_seconds=0.05, max_items=10)
    ctx.set("fast_expire", "hello")
    assert ctx.get("fast_expire") == "hello"

    time.sleep(0.06)
    assert ctx.get("fast_expire") is None
