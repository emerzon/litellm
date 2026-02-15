"""
Test for Cache class mutable default argument fix.

This test ensures that each Cache instance has an independent
supported_call_types list, preventing cross-instance state leakage.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.caching.caching import Cache


def test_cache_independent_supported_call_types():
    """
    Test that each Cache instance has its own independent supported_call_types list.
    
    This test verifies the fix for the mutable default argument bug where
    multiple Cache instances were sharing the same list object.
    """
    # Create two cache instances
    cache_a = Cache(type='local')
    cache_b = Cache(type='local')
    
    # Verify they have different list objects (not the same reference)
    assert cache_a.supported_call_types is not cache_b.supported_call_types, \
        "Cache instances should have independent supported_call_types lists"
    
    # Verify they have the same initial values
    assert cache_a.supported_call_types == cache_b.supported_call_types, \
        "Cache instances should start with the same default values"
    
    # Verify initial length
    initial_length = len(cache_a.supported_call_types)
    assert len(cache_a.supported_call_types) == initial_length
    assert len(cache_b.supported_call_types) == initial_length
    
    # Modify one instance's list
    cache_a.supported_call_types.append('custom_call_type')
    
    # Verify that only cache_a was modified
    assert len(cache_a.supported_call_types) == initial_length + 1, \
        "cache_a should have one additional element"
    assert len(cache_b.supported_call_types) == initial_length, \
        "cache_b should remain unchanged"
    assert 'custom_call_type' in cache_a.supported_call_types, \
        "cache_a should contain the custom call type"
    assert 'custom_call_type' not in cache_b.supported_call_types, \
        "cache_b should not contain the custom call type"


def test_cache_custom_supported_call_types():
    """
    Test that Cache instances with custom supported_call_types
    also get independent copies.
    """
    custom_types = ["completion", "embedding"]
    
    cache_a = Cache(type='local', supported_call_types=custom_types)
    cache_b = Cache(type='local', supported_call_types=custom_types)
    
    # Verify they have different list objects
    assert cache_a.supported_call_types is not cache_b.supported_call_types, \
        "Cache instances should have independent supported_call_types lists even with custom values"
    
    # Verify they don't share the original custom_types list
    assert cache_a.supported_call_types is not custom_types, \
        "Cache instance should have a copy, not the original list"
    
    # Verify they have the same values
    assert cache_a.supported_call_types == custom_types
    assert cache_b.supported_call_types == custom_types
    
    # Modify one instance
    cache_a.supported_call_types.append('new_type')
    
    # Verify isolation
    assert 'new_type' in cache_a.supported_call_types
    assert 'new_type' not in cache_b.supported_call_types
    assert 'new_type' not in custom_types, \
        "Original custom_types list should remain unchanged"


def test_cache_default_supported_call_types_values():
    """
    Test that the default supported_call_types contains expected values.
    """
    cache = Cache(type='local')
    
    # Verify the expected default call types are present
    expected_types = [
        "completion",
        "acompletion",
        "embedding",
        "aembedding",
        "atranscription",
        "transcription",
        "atext_completion",
        "text_completion",
        "arerank",
        "rerank",
        "responses",
        "aresponses",
    ]
    
    assert set(cache.supported_call_types) == set(expected_types), \
        "Cache should have all expected default supported call types"
