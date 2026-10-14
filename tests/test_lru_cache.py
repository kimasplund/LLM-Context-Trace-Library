"""Tests for LRUCache implementation in lctl/core/events.py."""

from datetime import datetime, timedelta

import pytest

from lctl.core.events import Chain, Event, EventType, LRUCache, ReplayEngine, State


class TestLRUCache:
    """Tests for LRUCache class."""

    def test_cache_creation_default_size(self):
        """Test creating cache with default max size."""
        cache = LRUCache()
        assert cache.max_size == 100
        assert len(cache) == 0

    def test_cache_creation_custom_size(self):
        """Test creating cache with custom max size."""
        cache = LRUCache(max_size=50)
        assert cache.max_size == 50

    def test_cache_creation_invalid_size(self):
        """Test that max_size < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_size must be at least 1"):
            LRUCache(max_size=0)
        with pytest.raises(ValueError, match="max_size must be at least 1"):
            LRUCache(max_size=-5)

    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = LRUCache(max_size=10)
        state = State()
        state.facts["F1"] = {"text": "test", "confidence": 0.9}

        cache.put(1, state)
        retrieved = cache.get(1)

        assert retrieved is state
        assert len(cache) == 1

    def test_get_nonexistent_key(self):
        """Test get returns None for nonexistent key."""
        cache = LRUCache(max_size=10)
        assert cache.get(999) is None

    def test_put_updates_existing_key(self):
        """Test put updates value for existing key."""
        cache = LRUCache(max_size=10)
        state1 = State()
        state2 = State()
        state2.facts["F2"] = {"text": "updated", "confidence": 0.8}

        cache.put(1, state1)
        cache.put(1, state2)

        assert cache.get(1) is state2
        assert len(cache) == 1

    def test_lru_eviction_at_capacity(self):
        """Test that oldest item is evicted when at capacity."""
        cache = LRUCache(max_size=3)

        cache.put(1, State())
        cache.put(2, State())
        cache.put(3, State())
        assert len(cache) == 3
        assert 1 in cache
        assert 2 in cache
        assert 3 in cache

        # Adding 4th item should evict key=1 (oldest)
        cache.put(4, State())
        assert len(cache) == 3
        assert 1 not in cache
        assert 2 in cache
        assert 3 in cache
        assert 4 in cache

    def test_get_moves_to_most_recent(self):
        """Test that get moves item to most recently used position."""
        cache = LRUCache(max_size=3)

        cache.put(1, State())
        cache.put(2, State())
        cache.put(3, State())

        # Access key=1, making it most recently used
        cache.get(1)

        # Adding 4th item should evict key=2 (now oldest)
        cache.put(4, State())
        assert 1 in cache  # Should still be present (was accessed)
        assert 2 not in cache  # Should be evicted
        assert 3 in cache
        assert 4 in cache

    def test_put_existing_moves_to_most_recent(self):
        """Test that putting existing key moves it to most recently used."""
        cache = LRUCache(max_size=3)

        cache.put(1, State())
        cache.put(2, State())
        cache.put(3, State())

        # Update key=1, making it most recently used
        cache.put(1, State())

        # Adding 4th item should evict key=2 (now oldest)
        cache.put(4, State())
        assert 1 in cache
        assert 2 not in cache
        assert 3 in cache
        assert 4 in cache

    def test_clear(self):
        """Test clear removes all items."""
        cache = LRUCache(max_size=10)
        cache.put(1, State())
        cache.put(2, State())
        cache.put(3, State())

        cache.clear()

        assert len(cache) == 0
        assert 1 not in cache
        assert 2 not in cache
        assert 3 not in cache

    def test_contains(self):
        """Test __contains__ method."""
        cache = LRUCache(max_size=10)
        cache.put(1, State())

        assert 1 in cache
        assert 2 not in cache

    def test_contains_does_not_affect_lru_order(self):
        """Test that checking containment doesn't affect LRU order."""
        cache = LRUCache(max_size=3)
        cache.put(1, State())
        cache.put(2, State())
        cache.put(3, State())

        # Check containment of key=1 (should not move it)
        _ = 1 in cache

        # Add new item - should still evict key=1 (oldest)
        cache.put(4, State())
        assert 1 not in cache  # Was evicted despite containment check

    def test_len(self):
        """Test __len__ method."""
        cache = LRUCache(max_size=10)
        assert len(cache) == 0

        cache.put(1, State())
        assert len(cache) == 1

        cache.put(2, State())
        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0

    def test_keys(self):
        """Test keys method returns keys in LRU order."""
        cache = LRUCache(max_size=5)
        cache.put(1, State())
        cache.put(2, State())
        cache.put(3, State())

        # Keys should be in insertion order (oldest first)
        assert cache.keys() == [1, 2, 3]

        # Access key=1, moving it to end
        cache.get(1)
        assert cache.keys() == [2, 3, 1]

    def test_single_item_cache(self):
        """Test cache with max_size=1."""
        cache = LRUCache(max_size=1)

        cache.put(1, State())
        assert len(cache) == 1
        assert 1 in cache

        cache.put(2, State())
        assert len(cache) == 1
        assert 1 not in cache
        assert 2 in cache

    def test_cache_with_various_key_values(self):
        """Test cache with various integer keys."""
        cache = LRUCache(max_size=10)

        cache.put(0, State())
        cache.put(100, State())
        cache.put(-5, State())  # Negative keys should work

        assert 0 in cache
        assert 100 in cache
        assert -5 in cache


class TestReplayEngineWithLRUCache:
    """Tests for ReplayEngine integration with LRUCache."""

    @pytest.fixture
    def base_timestamp(self) -> datetime:
        """Fixed timestamp for reproducible tests."""
        return datetime(2025, 1, 15, 10, 0, 0)

    @pytest.fixture
    def large_chain(self, base_timestamp: datetime) -> Chain:
        """Create a chain with many events for cache testing."""
        chain = Chain(id="large-chain")
        for i in range(200):
            chain.add_event(
                Event(
                    seq=i + 1,
                    type=EventType.FACT_ADDED,
                    timestamp=base_timestamp + timedelta(milliseconds=i),
                    agent=f"agent{i % 5}",
                    data={"id": f"F{i}", "text": f"Fact {i}", "confidence": 0.9},
                )
            )
        return chain

    def test_replay_engine_default_cache_size(self, large_chain: Chain):
        """Test ReplayEngine uses default cache size of 100."""
        engine = ReplayEngine(large_chain)
        assert engine._state_cache.max_size == 100

    def test_replay_engine_custom_cache_size(self, large_chain: Chain):
        """Test ReplayEngine with custom cache size."""
        engine = ReplayEngine(large_chain, cache_size=50)
        assert engine._state_cache.max_size == 50

    def test_cache_populated_during_replay(self, large_chain: Chain):
        """Test that cache is populated during replay_to calls."""
        engine = ReplayEngine(large_chain, cache_size=10)

        engine.replay_to(50)
        assert len(engine._state_cache) == 1
        assert 50 in engine._state_cache

        engine.replay_to(100)
        assert len(engine._state_cache) == 2
        assert 100 in engine._state_cache

    def test_cache_eviction_during_replay(self, large_chain: Chain):
        """Test that cache evicts old entries when full."""
        engine = ReplayEngine(large_chain, cache_size=3)

        # Fill cache
        engine.replay_to(10)
        engine.replay_to(20)
        engine.replay_to(30)
        assert len(engine._state_cache) == 3

        # Add more entries, should evict oldest
        engine.replay_to(40)
        assert len(engine._state_cache) == 3
        assert 10 not in engine._state_cache
        assert 40 in engine._state_cache

    def test_cache_hit_uses_cached_state(self, large_chain: Chain):
        """Test that replay uses cached state when available."""
        engine = ReplayEngine(large_chain, cache_size=10)

        # First replay populates cache
        state1 = engine.replay_to(50)

        # Inject a marker into cached state to verify it's used
        engine._state_cache.get(50).facts["MARKER"] = {"text": "marker"}

        # Replay to same point should use cache and include marker
        state2 = engine.replay_to(50)
        # The marker should propagate since we copy from cache
        assert "MARKER" in state2.facts

    def test_cache_enables_partial_replay(self, large_chain: Chain):
        """Test that cache enables efficient partial replay."""
        engine = ReplayEngine(large_chain, cache_size=10)

        # Build up cache at seq=50
        engine.replay_to(50)

        # Replay to seq=60 should start from cached seq=50
        state = engine.replay_to(60)
        # State should have 60 events worth of facts
        assert state.metrics["event_count"] == 60

    def test_replay_correctness_with_small_cache(self, large_chain: Chain):
        """Test replay produces correct results with small cache."""
        engine_small = ReplayEngine(large_chain, cache_size=3)
        engine_large = ReplayEngine(large_chain, cache_size=100)

        # Replay same sequences with different cache sizes
        for seq in [10, 50, 100, 150, 200]:
            state_small = engine_small.replay_to(seq)
            state_large = engine_large.replay_to(seq)

            assert state_small.metrics["event_count"] == state_large.metrics["event_count"]
            assert len(state_small.facts) == len(state_large.facts)

    def test_replay_all_with_cache(self, large_chain: Chain):
        """Test replay_all works correctly with cache."""
        engine = ReplayEngine(large_chain, cache_size=10)

        state = engine.replay_all()

        assert state.metrics["event_count"] == 200
        assert len(state.facts) == 200

    def test_multiple_replays_same_sequence(self, large_chain: Chain):
        """Test multiple replays to same sequence use cache."""
        engine = ReplayEngine(large_chain, cache_size=10)

        state1 = engine.replay_to(100)
        state2 = engine.replay_to(100)
        state3 = engine.replay_to(100)

        # All should produce equivalent results
        assert state1.metrics["event_count"] == state2.metrics["event_count"]
        assert state2.metrics["event_count"] == state3.metrics["event_count"]

    def test_backward_replay_uses_nearest_cache(self, large_chain: Chain):
        """Test that replaying backward uses nearest earlier cached state."""
        engine = ReplayEngine(large_chain, cache_size=10)

        # Build cache at multiple points
        engine.replay_to(100)
        engine.replay_to(150)
        engine.replay_to(200)

        # Replay to point between cached states
        state = engine.replay_to(120)

        # Should work correctly (uses seq=100 as base)
        assert state.metrics["event_count"] == 120

    def test_cache_keys_order_maintained(self, large_chain: Chain):
        """Test that cache maintains proper LRU order."""
        engine = ReplayEngine(large_chain, cache_size=3)

        engine.replay_to(10)
        engine.replay_to(20)
        engine.replay_to(30)

        # Access seq=10, making it most recently used
        engine.replay_to(10)

        # Add new entry, should evict seq=20 (now oldest)
        engine.replay_to(40)

        assert 10 in engine._state_cache
        assert 20 not in engine._state_cache
        assert 30 in engine._state_cache
        assert 40 in engine._state_cache
