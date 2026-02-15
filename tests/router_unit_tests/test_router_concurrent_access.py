"""
Test suite for Router concurrent access thread safety.

Tests the fix for race conditions when:
- Background scheduler jobs modify the model list
- Concurrent API requests read from the model list
- Multiple upsert/delete operations happen simultaneously
"""

import sys
import os
import pytest
import threading
import time
from typing import List

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm import Router
from litellm.types.router import Deployment


def create_model(idx: int) -> dict:
    """Helper to create a test model configuration."""
    return {
        "model_name": "gpt-4",
        "litellm_params": {
            "model": f"openai/gpt-4-{idx}",
            "api_key": f"test-key-{idx}",
        },
        "model_info": {"id": f"deployment-id-{idx}"},
    }


class TestRouterConcurrentAccess:
    """Test cases for router thread safety under concurrent access"""

    def test_concurrent_upsert_and_read(self):
        """
        Test that concurrent upsert operations and reads don't cause race conditions.
        
        Simulates background scheduler jobs upserting deployments while
        API requests read from the model list.
        """
        router = Router(model_list=[create_model(0)])
        
        errors = []
        requests_completed = [0]
        
        def api_request_handler():
            """Simulate API requests calling get_available_deployment."""
            for _ in range(50):
                try:
                    deployment = router.get_available_deployment(
                        model="gpt-4",
                        messages=[{"role": "user", "content": "test"}],
                    )
                    if deployment:
                        requests_completed[0] += 1
                except Exception as e:
                    error_msg = str(e).lower()
                    if "index" in error_msg or "range" in error_msg or "list" in error_msg:
                        errors.append(f"IndexError in API request: {e}")
                
                time.sleep(0.0001)  # Small delay to increase interleaving
        
        def scheduler_job():
            """Simulate background scheduler upserting deployments."""
            for i in range(1, 20):
                try:
                    router.upsert_deployment(deployment=Deployment(**create_model(i)))
                except Exception as e:
                    errors.append(f"Error in scheduler upsert: {e}")
                
                time.sleep(0.0001)
        
        # Launch concurrent threads
        api_threads = [threading.Thread(target=api_request_handler) for _ in range(3)]
        scheduler_thread = threading.Thread(target=scheduler_job)
        
        for t in api_threads:
            t.start()
        scheduler_thread.start()
        
        for t in api_threads:
            t.join()
        scheduler_thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Race condition detected! Errors: {errors}"
        assert requests_completed[0] > 0, "No requests completed successfully"
        
        # Verify final state is consistent
        assert len(router.model_list) > 0
        assert len(router.model_id_to_deployment_index_map) == len(router.model_list)

    def test_concurrent_delete_and_read(self):
        """
        Test that concurrent delete operations and reads don't cause race conditions.
        """
        # Initialize router with multiple deployments
        initial_models = [create_model(i) for i in range(20)]
        router = Router(model_list=initial_models)
        
        errors = []
        reads_completed = [0]
        
        def reader_thread():
            """Simulate reading from the router."""
            for _ in range(30):
                try:
                    deployment = router.get_deployment(model_id=f"deployment-id-5")
                    reads_completed[0] += 1
                except Exception as e:
                    error_msg = str(e).lower()
                    if "index" in error_msg or "range" in error_msg:
                        errors.append(f"IndexError in reader: {e}")
                
                time.sleep(0.0001)
        
        def deleter_thread():
            """Simulate deleting deployments."""
            for i in range(10, 15):
                try:
                    router.delete_deployment(id=f"deployment-id-{i}")
                except Exception as e:
                    if "not found" not in str(e).lower():
                        errors.append(f"Error in deleter: {e}")
                
                time.sleep(0.0002)
        
        # Launch concurrent threads
        readers = [threading.Thread(target=reader_thread) for _ in range(3)]
        deleter = threading.Thread(target=deleter_thread)
        
        for t in readers:
            t.start()
        deleter.start()
        
        for t in readers:
            t.join()
        deleter.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Race condition detected! Errors: {errors}"
        assert reads_completed[0] > 0, "No reads completed successfully"
        
        # Verify final state is consistent
        assert len(router.model_id_to_deployment_index_map) == len(router.model_list)

    def test_concurrent_add_and_get_all_deployments(self):
        """
        Test that concurrent add operations and _get_all_deployments calls
        don't cause race conditions.
        """
        router = Router(model_list=[])
        
        errors = []
        reads_completed = [0]
        
        def reader_thread():
            """Simulate reading all deployments."""
            for _ in range(40):
                try:
                    deployments = router._get_all_deployments(model_name="gpt-4")
                    reads_completed[0] += 1
                except Exception as e:
                    error_msg = str(e).lower()
                    if "index" in error_msg or "range" in error_msg:
                        errors.append(f"IndexError in _get_all_deployments: {e}")
                
                time.sleep(0.0001)
        
        def adder_thread():
            """Simulate adding deployments."""
            for i in range(15):
                try:
                    router.add_deployment(deployment=Deployment(**create_model(i)))
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        errors.append(f"Error in adder: {e}")
                
                time.sleep(0.0002)
        
        # Launch concurrent threads
        readers = [threading.Thread(target=reader_thread) for _ in range(3)]
        adder = threading.Thread(target=adder_thread)
        
        for t in readers:
            t.start()
        adder.start()
        
        for t in readers:
            t.join()
        adder.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Race condition detected! Errors: {errors}"
        assert reads_completed[0] > 0, "No reads completed successfully"
        
        # Verify final state is consistent
        assert len(router.model_list) > 0
        assert len(router.model_id_to_deployment_index_map) == len(router.model_list)

    def test_concurrent_has_model_id_and_upsert(self):
        """
        Test that concurrent has_model_id checks and upsert operations
        don't cause race conditions.
        """
        router = Router(model_list=[create_model(0)])
        
        errors = []
        checks_completed = [0]
        
        def checker_thread():
            """Simulate checking if model_id exists."""
            for i in range(50):
                try:
                    exists = router.has_model_id(f"deployment-id-{i % 10}")
                    checks_completed[0] += 1
                except Exception as e:
                    errors.append(f"Error in checker: {e}")
                
                time.sleep(0.0001)
        
        def upserter_thread():
            """Simulate upserting deployments."""
            for i in range(1, 10):
                try:
                    router.upsert_deployment(deployment=Deployment(**create_model(i)))
                except Exception as e:
                    errors.append(f"Error in upserter: {e}")
                
                time.sleep(0.0002)
        
        # Launch concurrent threads
        checkers = [threading.Thread(target=checker_thread) for _ in range(3)]
        upserter = threading.Thread(target=upserter_thread)
        
        for t in checkers:
            t.start()
        upserter.start()
        
        for t in checkers:
            t.join()
        upserter.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Race condition detected! Errors: {errors}"
        assert checks_completed[0] > 0, "No checks completed successfully"
        
        # Verify final state is consistent
        assert len(router.model_id_to_deployment_index_map) == len(router.model_list)

    def test_concurrent_get_deployment_and_delete(self):
        """
        Test that concurrent get_deployment and delete operations
        don't cause race conditions.
        """
        # Initialize router with multiple deployments
        initial_models = [create_model(i) for i in range(15)]
        router = Router(model_list=initial_models)
        
        errors = []
        gets_completed = [0]
        
        def getter_thread():
            """Simulate getting deployments."""
            for i in range(40):
                try:
                    deployment = router.get_deployment(model_id=f"deployment-id-{i % 15}")
                    gets_completed[0] += 1
                except Exception as e:
                    error_msg = str(e).lower()
                    if "index" in error_msg or "range" in error_msg:
                        errors.append(f"IndexError in getter: {e}")
                
                time.sleep(0.0001)
        
        def deleter_thread():
            """Simulate deleting deployments."""
            for i in range(10, 15):
                try:
                    router.delete_deployment(id=f"deployment-id-{i}")
                except Exception as e:
                    if "not found" not in str(e).lower():
                        errors.append(f"Error in deleter: {e}")
                
                time.sleep(0.0003)
        
        # Launch concurrent threads
        getters = [threading.Thread(target=getter_thread) for _ in range(3)]
        deleter = threading.Thread(target=deleter_thread)
        
        for t in getters:
            t.start()
        deleter.start()
        
        for t in getters:
            t.join()
        deleter.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Race condition detected! Errors: {errors}"
        assert gets_completed[0] > 0, "No gets completed successfully"
        
        # Verify final state is consistent
        assert len(router.model_id_to_deployment_index_map) == len(router.model_list)

    def test_stress_concurrent_mixed_operations(self):
        """
        Stress test with mixed operations (add, upsert, delete, read) happening concurrently.
        
        This is the most comprehensive test simulating real-world usage patterns.
        """
        # Initialize with some deployments
        initial_models = [create_model(i) for i in range(5)]
        router = Router(model_list=initial_models)
        
        errors = []
        operation_counts = {"reads": [0], "writes": [0]}
        
        def reader_thread():
            """Simulate various read operations."""
            for i in range(30):
                try:
                    # Mix of different read operations
                    if i % 3 == 0:
                        router.get_deployment(model_id=f"deployment-id-{i % 10}")
                    elif i % 3 == 1:
                        router.has_model_id(f"deployment-id-{i % 10}")
                    else:
                        router._get_all_deployments(model_name="gpt-4")
                    
                    operation_counts["reads"][0] += 1
                except Exception as e:
                    error_msg = str(e).lower()
                    if "index" in error_msg or "range" in error_msg:
                        errors.append(f"IndexError in reader: {e}")
                
                time.sleep(0.0001)
        
        def writer_thread():
            """Simulate various write operations."""
            for i in range(5, 15):
                try:
                    # Mix of different write operations
                    if i % 2 == 0:
                        router.upsert_deployment(deployment=Deployment(**create_model(i)))
                    else:
                        router.add_deployment(deployment=Deployment(**create_model(i)))
                    
                    operation_counts["writes"][0] += 1
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        errors.append(f"Error in writer: {e}")
                
                time.sleep(0.0002)
        
        # Launch many concurrent threads
        readers = [threading.Thread(target=reader_thread) for _ in range(5)]
        writers = [threading.Thread(target=writer_thread) for _ in range(2)]
        
        all_threads = readers + writers
        
        for t in all_threads:
            t.start()
        
        for t in all_threads:
            t.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Race condition detected! Errors: {errors}"
        assert operation_counts["reads"][0] > 0, "No reads completed successfully"
        assert operation_counts["writes"][0] > 0, "No writes completed successfully"
        
        # Verify final state is consistent
        assert len(router.model_list) > 0
        assert len(router.model_id_to_deployment_index_map) == len(router.model_list)
        
        # Verify index consistency
        for model_id, idx in router.model_id_to_deployment_index_map.items():
            assert idx < len(router.model_list), f"Index {idx} out of bounds for list of length {len(router.model_list)}"
            model = router.model_list[idx]
            assert model.get("model_info", {}).get("id") == model_id, f"Inconsistent index mapping for {model_id}"
