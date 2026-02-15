"""
Tests for router previous_models race condition fix.

This test verifies that the Router's log_retry() method properly handles
concurrent access to the previous_models list without data loss.
"""

import threading
from typing import List
from unittest.mock import MagicMock

import pytest

from litellm import Router


def test_log_retry_concurrent_access():
    """
    Test that concurrent calls to log_retry don't lose data due to race conditions.
    
    This test simulates the TOCTOU race condition that could occur when multiple
    threads simultaneously call log_retry() and modify the shared previous_models list.
    
    Without proper synchronization, the pattern:
    - if len(self.previous_models) > 3: self.previous_models.pop(0)
    - self.previous_models.append(previous_model)
    
    Would cause data loss as threads read stale values and overwrite each other's changes.
    
    Test configuration: 20 threads × 100 iterations = 2000 concurrent operations
    """
    # Create a router instance
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "sk-test",
                },
            }
        ],
        num_retries=2,
    )
    
    # Track all unique models that were added
    added_models: List[str] = []
    num_threads = 20
    iterations_per_thread = 100  # 100 iterations per thread for thorough testing
    
    def worker(thread_id: int):
        """Simulate concurrent retry logging from different threads."""
        for i in range(iterations_per_thread):
            # Create a unique exception for this thread/iteration
            exception = Exception(f"Error from thread {thread_id}, iteration {i}")
            
            # Create kwargs with metadata
            kwargs = {
                "metadata": {
                    "model_group": "gpt-3.5-turbo",
                    "thread_id": thread_id,
                    "iteration": i,
                }
            }
            
            # Call log_retry which modifies previous_models
            try:
                result_kwargs = router.log_retry(kwargs=kwargs, e=exception)
                
                # Track what was added to verify later
                with threading.Lock():  # Lock for our tracking list
                    added_models.append(f"thread_{thread_id}_iter_{i}")
                    
            except Exception as e:
                # Should not raise any exceptions
                pytest.fail(f"log_retry raised exception: {e}")
    
    # Create and start threads
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    # Verify results
    expected_total = num_threads * iterations_per_thread
    actual_added = len(added_models)
    
    # All calls should have completed successfully
    assert actual_added == expected_total, (
        f"Expected {expected_total} log_retry calls, but only {actual_added} completed"
    )
    
    # The previous_models list should have at most 4 entries (limited by the pop logic)
    # but could have fewer if the test completes with fewer in the list
    assert len(router.previous_models) <= 4, (
        f"previous_models should have at most 4 entries, but has {len(router.previous_models)}"
    )
    
    # All entries should be unique (no duplicates from race conditions)
    # Check that the list doesn't have duplicate entries based on the exception string
    exception_strings = [model.get("exception_string", "") for model in router.previous_models]
    unique_exceptions = set(exception_strings)
    
    assert len(exception_strings) == len(unique_exceptions), (
        f"Found duplicate entries in previous_models: {exception_strings}"
    )


def test_log_retry_thread_safety_stress():
    """
    Stress test for thread safety of log_retry method.
    
    This test runs many more threads to increase the probability of catching
    any race conditions that might still exist.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "sk-test",
                },
            }
        ],
        num_retries=2,
    )
    
    success_count = 0
    success_lock = threading.Lock()
    
    def worker(thread_id: int):
        """Rapidly call log_retry to stress test the lock."""
        nonlocal success_count
        for i in range(100):
            exception = Exception(f"T{thread_id}I{i}")
            kwargs = {"metadata": {"id": f"{thread_id}_{i}"}}
            
            router.log_retry(kwargs=kwargs, e=exception)
            
            with success_lock:
                success_count += 1
    
    # Use even more threads for stress testing
    num_threads = 50
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    # Verify all calls completed
    expected = num_threads * 100
    assert success_count == expected, (
        f"Expected {expected} successful calls, got {success_count}"
    )
    
    # List should not exceed maximum size
    assert len(router.previous_models) <= 4


def test_log_retry_previous_models_copy():
    """
    Test that log_retry returns a copy of previous_models, not a reference.
    
    This prevents external code from modifying the internal list without synchronization.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "sk-test",
                },
            }
        ],
    )
    
    # Add some models
    for i in range(3):
        exception = Exception(f"Error {i}")
        kwargs = {"metadata": {}}
        result_kwargs = router.log_retry(kwargs=kwargs, e=exception)
        previous_models_from_kwargs = result_kwargs["metadata"]["previous_models"]
        
        # Modify the returned list
        previous_models_from_kwargs.append({"external": "modification"})
    
    # The router's internal list should not have the external modifications
    for model in router.previous_models:
        assert "external" not in model, (
            "External modification affected internal previous_models list"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
