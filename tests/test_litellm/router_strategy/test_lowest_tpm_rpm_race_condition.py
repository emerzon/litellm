"""
Test for the race condition fix in lowest_tpm_rpm.py

This test validates that concurrent TPM/RPM updates are handled correctly
and no data is lost due to race conditions.
"""

import asyncio
import sys
import threading
import time
from datetime import datetime
from unittest.mock import MagicMock

# Try to import pytest, but don't fail if it's not available
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Mock pytest.mark.asyncio for direct execution
    class MockPytest:
        class mark:
            @staticmethod
            def asyncio(func):
                return func
    pytest = MockPytest()

# Add path to litellm
sys.path.insert(0, "/home/runner/work/litellm/litellm")

from litellm.caching.dual_cache import DualCache
from litellm.router_strategy.lowest_tpm_rpm import LowestTPMLoggingHandler


def test_concurrent_tpm_updates():
    """
    Test that concurrent TPM updates don't lose data due to race conditions.
    
    This simulates multiple threads updating the same deployment's TPM counter
    simultaneously, which was causing >60% data loss before the fix.
    """
    cache = DualCache(default_in_memory_ttl=60)
    handler = LowestTPMLoggingHandler(router_cache=cache)
    
    model_group = "test-model-group"
    deployment_id = "deployment-1"
    current_minute = datetime.now().strftime("%H-%M")
    tpm_key = f"{model_group}:tpm:{current_minute}"
    
    # Initialize with some usage
    cache.set_cache(key=tpm_key, value={deployment_id: 50}, ttl=60)
    
    num_threads = 20
    tokens_per_thread = 10
    expected_total = 50 + (num_threads * tokens_per_thread)
    
    def simulate_success_event():
        """Simulate a successful request completing"""
        kwargs = {
            "litellm_params": {
                "metadata": {"model_group": model_group},
                "model_info": {"id": deployment_id},
            }
        }
        response_obj = {"usage": {"total_tokens": tokens_per_thread}}
        
        handler.log_success_event(kwargs, response_obj, time.time(), time.time())
    
    # Run concurrent updates
    threads = [threading.Thread(target=simulate_success_event) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Verify all updates were recorded
    final_value = cache.get_cache(key=tpm_key) or {}
    actual_tpm = final_value.get(deployment_id, 0)
    
    print(f"Expected TPM: {expected_total}")
    print(f"Actual TPM: {actual_tpm}")
    
    # With the fix, we should have 100% accuracy
    assert actual_tpm == expected_total, (
        f"Lost TPM data! Expected {expected_total}, got {actual_tpm}. "
        f"Loss: {expected_total - actual_tpm} tokens "
        f"({(expected_total - actual_tpm) / expected_total * 100:.1f}%)"
    )


def test_concurrent_rpm_updates():
    """
    Test that concurrent RPM updates don't lose data due to race conditions.
    
    This simulates multiple threads updating the same deployment's RPM counter
    simultaneously, which was causing >60% data loss before the fix.
    """
    cache = DualCache(default_in_memory_ttl=60)
    handler = LowestTPMLoggingHandler(router_cache=cache)
    
    model_group = "test-model-group"
    deployment_id = "deployment-1"
    current_minute = datetime.now().strftime("%H-%M")
    rpm_key = f"{model_group}:rpm:{current_minute}"
    
    # Initialize with some usage
    cache.set_cache(key=rpm_key, value={deployment_id: 5}, ttl=60)
    
    num_threads = 20
    expected_total = 5 + num_threads
    
    def simulate_success_event():
        """Simulate a successful request completing"""
        kwargs = {
            "litellm_params": {
                "metadata": {"model_group": model_group},
                "model_info": {"id": deployment_id},
            }
        }
        response_obj = {"usage": {"total_tokens": 10}}
        
        handler.log_success_event(kwargs, response_obj, time.time(), time.time())
    
    # Run concurrent updates
    threads = [threading.Thread(target=simulate_success_event) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Verify all updates were recorded
    final_value = cache.get_cache(key=rpm_key) or {}
    actual_rpm = final_value.get(deployment_id, 0)
    
    print(f"Expected RPM: {expected_total}")
    print(f"Actual RPM: {actual_rpm}")
    
    # With the fix, we should have 100% accuracy
    assert actual_rpm == expected_total, (
        f"Lost RPM data! Expected {expected_total}, got {actual_rpm}. "
        f"Loss: {expected_total - actual_rpm} requests "
        f"({(expected_total - actual_rpm) / expected_total * 100:.1f}%)"
    )


@pytest.mark.asyncio
async def test_async_concurrent_tpm_updates():
    """
    Test that async concurrent TPM updates don't lose data due to race conditions.
    
    This tests the async version of the fix.
    """
    cache = DualCache(default_in_memory_ttl=60)
    handler = LowestTPMLoggingHandler(router_cache=cache)
    
    model_group = "test-model-group"
    deployment_id = "deployment-1"
    current_minute = datetime.now().strftime("%H-%M")
    tpm_key = f"{model_group}:tpm:{current_minute}"
    
    # Initialize with some usage
    await cache.async_set_cache(key=tpm_key, value={deployment_id: 50}, ttl=60)
    
    num_tasks = 20
    tokens_per_task = 10
    expected_total = 50 + (num_tasks * tokens_per_task)
    
    async def simulate_success_event():
        """Simulate a successful async request completing"""
        kwargs = {
            "litellm_params": {
                "metadata": {"model_group": model_group},
                "model_info": {"id": deployment_id},
            }
        }
        response_obj = {"usage": {"total_tokens": tokens_per_task}}
        
        await handler.async_log_success_event(kwargs, response_obj, time.time(), time.time())
    
    # Run concurrent updates
    tasks = [simulate_success_event() for _ in range(num_tasks)]
    await asyncio.gather(*tasks)
    
    # Verify all updates were recorded
    final_value = await cache.async_get_cache(key=tpm_key) or {}
    actual_tpm = final_value.get(deployment_id, 0)
    
    print(f"Expected TPM (async): {expected_total}")
    print(f"Actual TPM (async): {actual_tpm}")
    
    # With the fix, we should have 100% accuracy
    assert actual_tpm == expected_total, (
        f"Lost async TPM data! Expected {expected_total}, got {actual_tpm}. "
        f"Loss: {expected_total - actual_tpm} tokens "
        f"({(expected_total - actual_tpm) / expected_total * 100:.1f}%)"
    )


@pytest.mark.asyncio
async def test_async_concurrent_rpm_updates():
    """
    Test that async concurrent RPM updates don't lose data due to race conditions.
    
    This tests the async version of the fix.
    """
    cache = DualCache(default_in_memory_ttl=60)
    handler = LowestTPMLoggingHandler(router_cache=cache)
    
    model_group = "test-model-group"
    deployment_id = "deployment-1"
    current_minute = datetime.now().strftime("%H-%M")
    rpm_key = f"{model_group}:rpm:{current_minute}"
    
    # Initialize with some usage
    await cache.async_set_cache(key=rpm_key, value={deployment_id: 5}, ttl=60)
    
    num_tasks = 20
    expected_total = 5 + num_tasks
    
    async def simulate_success_event():
        """Simulate a successful async request completing"""
        kwargs = {
            "litellm_params": {
                "metadata": {"model_group": model_group},
                "model_info": {"id": deployment_id},
            }
        }
        response_obj = {"usage": {"total_tokens": 10}}
        
        await handler.async_log_success_event(kwargs, response_obj, time.time(), time.time())
    
    # Run concurrent updates
    tasks = [simulate_success_event() for _ in range(num_tasks)]
    await asyncio.gather(*tasks)
    
    # Verify all updates were recorded
    final_value = await cache.async_get_cache(key=rpm_key) or {}
    actual_rpm = final_value.get(deployment_id, 0)
    
    print(f"Expected RPM (async): {expected_total}")
    print(f"Actual RPM (async): {actual_rpm}")
    
    # With the fix, we should have 100% accuracy
    assert actual_rpm == expected_total, (
        f"Lost async RPM data! Expected {expected_total}, got {actual_rpm}. "
        f"Loss: {expected_total - actual_rpm} requests "
        f"({(expected_total - actual_rpm) / expected_total * 100:.1f}%)"
    )


def test_multiple_deployments_concurrent():
    """
    Test that concurrent updates to different deployments work correctly.
    
    This ensures the locking doesn't cause issues when updating different deployments.
    """
    cache = DualCache(default_in_memory_ttl=60)
    handler = LowestTPMLoggingHandler(router_cache=cache)
    
    model_group = "test-model-group"
    current_minute = datetime.now().strftime("%H-%M")
    tpm_key = f"{model_group}:tpm:{current_minute}"
    
    deployments = ["deployment-1", "deployment-2", "deployment-3"]
    num_threads_per_deployment = 10
    tokens_per_thread = 10
    
    def simulate_success_event(deployment_id):
        """Simulate a successful request completing for a specific deployment"""
        kwargs = {
            "litellm_params": {
                "metadata": {"model_group": model_group},
                "model_info": {"id": deployment_id},
            }
        }
        response_obj = {"usage": {"total_tokens": tokens_per_thread}}
        
        handler.log_success_event(kwargs, response_obj, time.time(), time.time())
    
    # Run concurrent updates for all deployments
    threads = []
    for deployment in deployments:
        for _ in range(num_threads_per_deployment):
            t = threading.Thread(target=simulate_success_event, args=(deployment,))
            threads.append(t)
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Verify all updates were recorded for each deployment
    final_value = cache.get_cache(key=tpm_key) or {}
    expected_per_deployment = num_threads_per_deployment * tokens_per_thread
    
    for deployment in deployments:
        actual_tpm = final_value.get(deployment, 0)
        print(f"{deployment}: Expected {expected_per_deployment}, Actual {actual_tpm}")
        
        assert actual_tpm == expected_per_deployment, (
            f"Lost data for {deployment}! Expected {expected_per_deployment}, got {actual_tpm}."
        )


if __name__ == "__main__":
    # Run sync tests
    print("\n" + "="*70)
    print("TEST: Concurrent TPM Updates (Sync)")
    print("="*70)
    test_concurrent_tpm_updates()
    print("✓ PASSED")
    
    print("\n" + "="*70)
    print("TEST: Concurrent RPM Updates (Sync)")
    print("="*70)
    test_concurrent_rpm_updates()
    print("✓ PASSED")
    
    print("\n" + "="*70)
    print("TEST: Multiple Deployments Concurrent Updates")
    print("="*70)
    test_multiple_deployments_concurrent()
    print("✓ PASSED")
    
    # Run async tests
    print("\n" + "="*70)
    print("TEST: Concurrent TPM Updates (Async)")
    print("="*70)
    asyncio.run(test_async_concurrent_tpm_updates())
    print("✓ PASSED")
    
    print("\n" + "="*70)
    print("TEST: Concurrent RPM Updates (Async)")
    print("="*70)
    asyncio.run(test_async_concurrent_rpm_updates())
    print("✓ PASSED")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
