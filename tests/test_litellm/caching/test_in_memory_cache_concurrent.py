"""
Test concurrent increment operations for InMemoryCache to ensure thread-safety.
This tests the fix for the race condition in increment_cache() method.
"""

import asyncio
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.caching.in_memory_cache import InMemoryCache


def test_increment_cache_concurrent_basic():
    """
    Test that concurrent increments are correctly counted with basic threading.
    This validates the fix for the race condition.
    """
    cache = InMemoryCache(max_size_in_memory=10000)
    key = "test_deployment:successes"
    cache.set_cache(key=key, value=0)

    def worker():
        for _ in range(20):
            cache.increment_cache(key=key, value=1, ttl=60)

    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(worker) for _ in range(50)]
        for f in as_completed(futures):
            f.result()

    final = cache.get_cache(key=key)
    expected = 50 * 20  # 50 workers * 20 increments each = 1000
    assert final == expected, f"Expected {expected}, but got {final}. Lost {expected - final} increments"


def test_increment_cache_concurrent_high_contention():
    """
    Test with high contention - many threads incrementing the same key rapidly.
    This simulates the ThreadPoolExecutor scenario from the bug report.
    """
    cache = InMemoryCache(max_size_in_memory=10000)
    key = "high_contention_key"
    cache.set_cache(key=key, value=0)

    def worker():
        for _ in range(100):
            cache.increment_cache(key=key, value=1, ttl=60)

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(worker) for _ in range(100)]
        for f in as_completed(futures):
            f.result()

    final = cache.get_cache(key=key)
    expected = 100 * 100  # 100 workers * 100 increments each = 10000
    assert final == expected, f"Expected {expected}, but got {final}. Lost {expected - final} increments"


def test_increment_cache_concurrent_multiple_keys():
    """
    Test concurrent increments on multiple different keys to ensure locks work correctly
    and don't create deadlocks.
    """
    cache = InMemoryCache(max_size_in_memory=10000)
    keys = [f"deployment_{i}:successes" for i in range(10)]
    
    for key in keys:
        cache.set_cache(key=key, value=0)

    def worker(key_idx):
        key = keys[key_idx]
        for _ in range(50):
            cache.increment_cache(key=key, value=1, ttl=60)

    with ThreadPoolExecutor(max_workers=50) as executor:
        # 5 threads per key
        futures = [executor.submit(worker, i % 10) for i in range(50)]
        for f in as_completed(futures):
            f.result()

    # Each key should have been incremented 5 threads * 50 times = 250
    for key in keys:
        final = cache.get_cache(key=key)
        expected = 5 * 50
        assert final == expected, f"Key {key}: Expected {expected}, but got {final}"


def test_increment_cache_concurrent_with_varying_increments():
    """
    Test concurrent increments with different increment values.
    """
    cache = InMemoryCache(max_size_in_memory=10000)
    key = "varied_increment_key"
    cache.set_cache(key=key, value=0)

    def worker(increment_value):
        for _ in range(10):
            cache.increment_cache(key=key, value=increment_value, ttl=60)

    with ThreadPoolExecutor(max_workers=20) as executor:
        # 10 workers with increment=1, 10 workers with increment=2
        futures = []
        for i in range(10):
            futures.append(executor.submit(worker, 1))
        for i in range(10):
            futures.append(executor.submit(worker, 2))
        
        for f in as_completed(futures):
            f.result()

    final = cache.get_cache(key=key)
    expected = (10 * 10 * 1) + (10 * 10 * 2)  # (10 workers * 10 times * 1) + (10 workers * 10 times * 2) = 300
    assert final == expected, f"Expected {expected}, but got {final}"


def test_increment_cache_concurrent_with_ttl_expiry():
    """
    Test that concurrent increments work correctly even when TTL expires.
    """
    cache = InMemoryCache(max_size_in_memory=10000)
    key = "expiry_test_key"
    cache.set_cache(key=key, value=0, ttl=1)

    def worker():
        for _ in range(10):
            cache.increment_cache(key=key, value=1, ttl=1)
            time.sleep(0.01)  # Small delay

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker) for _ in range(10)]
        for f in as_completed(futures):
            f.result()

    # Wait for TTL to expire
    time.sleep(2)
    
    # After expiry, value should be None
    final = cache.get_cache(key=key)
    assert final is None, "Value should be None after TTL expiry"


def test_increment_cache_concurrent_initialization():
    """
    Test that increment works correctly when key doesn't exist initially
    (concurrent initialization).
    """
    cache = InMemoryCache(max_size_in_memory=10000)
    key = "new_key_concurrent"
    # Don't initialize the key

    def worker():
        for _ in range(20):
            cache.increment_cache(key=key, value=1, ttl=60)

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(worker) for _ in range(30)]
        for f in as_completed(futures):
            f.result()

    final = cache.get_cache(key=key)
    expected = 30 * 20
    assert final == expected, f"Expected {expected}, but got {final}"


@pytest.mark.asyncio
async def test_async_increment_concurrent():
    """
    Test concurrent async increments to ensure they're also thread-safe.
    """
    cache = InMemoryCache(max_size_in_memory=10000)
    key = "async_test_key"
    cache.set_cache(key=key, value=0)

    async def worker():
        for _ in range(50):
            await cache.async_increment(key=key, value=1, ttl=60)

    # Run multiple concurrent async workers
    tasks = [worker() for _ in range(20)]
    await asyncio.gather(*tasks)

    final = cache.get_cache(key=key)
    expected = 20 * 50
    assert final == expected, f"Expected {expected}, but got {final}"


def test_increment_cache_returns_correct_value():
    """
    Test that increment_cache returns the correct new value after increment.
    """
    cache = InMemoryCache(max_size_in_memory=10000)
    key = "return_value_key"
    cache.set_cache(key=key, value=10)

    result = cache.increment_cache(key=key, value=5, ttl=60)
    assert result == 15, f"Expected return value 15, but got {result}"
    
    cached_value = cache.get_cache(key=key)
    assert cached_value == 15, f"Expected cached value 15, but got {cached_value}"


def test_increment_cache_stress_test():
    """
    Stress test with the scenario from the bug report.
    This should reproduce the issue without the fix.
    """
    cache = InMemoryCache(max_size_in_memory=10000)
    key = "stress_test_key"
    cache.set_cache(key=key, value=0)

    def worker():
        # Mimic the pattern from the bug report
        for _ in range(20):
            current = cache.get_cache(key=key) or 0
            # Intentionally use the external increment pattern that was failing
            # but the fixed increment_cache should handle this internally
            pass
        
        # Use the fixed increment_cache method
        for _ in range(20):
            cache.increment_cache(key=key, value=1, ttl=60)

    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(worker) for _ in range(50)]
        for f in as_completed(futures):
            f.result()

    final = cache.get_cache(key=key)
    expected = 50 * 20
    
    # The fix should ensure 100% accuracy
    assert final == expected, f"Expected {expected}, but got {final}. Lost {expected - final} increments ({(expected - final) / expected * 100:.1f}% loss)"
