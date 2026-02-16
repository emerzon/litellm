"""
Test suite for BaseTokenUsageProcessor.combine_usage_objects() optimization.

This test file validates the correctness and performance of the optimized
combine_usage_objects function, which uses model_fields instead of dir()
for better performance.
"""
import pytest
from litellm.cost_calculator import BaseTokenUsageProcessor
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
    Usage,
)


def test_combine_usage_objects_single_object():
    """Test combining a single usage object."""
    usage = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
    )
    result = BaseTokenUsageProcessor.combine_usage_objects([usage])
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 50
    assert result.total_tokens == 150


def test_combine_usage_objects_multiple_objects():
    """Test combining multiple usage objects with basic fields."""
    usage1 = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    usage2 = Usage(prompt_tokens=200, completion_tokens=100, total_tokens=300)
    usage3 = Usage(prompt_tokens=150, completion_tokens=75, total_tokens=225)

    result = BaseTokenUsageProcessor.combine_usage_objects([usage1, usage2, usage3])
    assert result.prompt_tokens == 450
    assert result.completion_tokens == 225
    assert result.total_tokens == 675


def test_combine_usage_objects_with_prompt_tokens_details():
    """Test combining usage objects with nested prompt_tokens_details."""
    usage1 = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=10, cached_tokens=20, text_tokens=70, image_tokens=0
        ),
    )
    usage2 = Usage(
        prompt_tokens=200,
        completion_tokens=100,
        total_tokens=300,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=15, cached_tokens=30, text_tokens=155, image_tokens=0
        ),
    )

    result = BaseTokenUsageProcessor.combine_usage_objects([usage1, usage2])

    assert result.prompt_tokens == 300
    assert result.completion_tokens == 150
    assert result.total_tokens == 450
    assert result.prompt_tokens_details is not None
    assert result.prompt_tokens_details.audio_tokens == 25
    assert result.prompt_tokens_details.cached_tokens == 50
    assert result.prompt_tokens_details.text_tokens == 225
    assert result.prompt_tokens_details.image_tokens == 0


def test_combine_usage_objects_with_completion_tokens_details():
    """Test combining usage objects with nested completion_tokens_details."""
    usage1 = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            audio_tokens=5, reasoning_tokens=15, accepted_prediction_tokens=10
        ),
    )
    usage2 = Usage(
        prompt_tokens=200,
        completion_tokens=100,
        total_tokens=300,
        completion_tokens_details=CompletionTokensDetailsWrapper(
            audio_tokens=10, reasoning_tokens=30, accepted_prediction_tokens=20
        ),
    )

    result = BaseTokenUsageProcessor.combine_usage_objects([usage1, usage2])

    assert result.prompt_tokens == 300
    assert result.completion_tokens == 150
    assert result.total_tokens == 450
    assert result.completion_tokens_details is not None
    assert result.completion_tokens_details.audio_tokens == 15
    assert result.completion_tokens_details.reasoning_tokens == 45
    assert result.completion_tokens_details.accepted_prediction_tokens == 30


def test_combine_usage_objects_with_cost():
    """Test combining usage objects with cost field."""
    usage1 = Usage(
        prompt_tokens=100, completion_tokens=50, total_tokens=150, cost=0.001
    )
    usage2 = Usage(
        prompt_tokens=200, completion_tokens=100, total_tokens=300, cost=0.002
    )
    usage3 = Usage(
        prompt_tokens=150, completion_tokens=75, total_tokens=225, cost=0.0015
    )

    result = BaseTokenUsageProcessor.combine_usage_objects([usage1, usage2, usage3])

    assert result.prompt_tokens == 450
    assert result.completion_tokens == 225
    assert result.total_tokens == 675
    assert result.cost is not None
    assert abs(result.cost - 0.0045) < 0.0001


def test_combine_usage_objects_empty_list():
    """Test combining an empty list of usage objects."""
    result = BaseTokenUsageProcessor.combine_usage_objects([])
    # Should return an empty Usage object with None or 0 for all fields
    assert result.prompt_tokens is None or result.prompt_tokens == 0
    assert result.completion_tokens is None or result.completion_tokens == 0


def test_combine_usage_objects_with_none_values():
    """Test combining usage objects where some fields are None."""
    usage1 = Usage(prompt_tokens=100, completion_tokens=None, total_tokens=100)
    usage2 = Usage(prompt_tokens=200, completion_tokens=50, total_tokens=250)

    result = BaseTokenUsageProcessor.combine_usage_objects([usage1, usage2])

    assert result.prompt_tokens == 300
    assert result.completion_tokens == 50
    assert result.total_tokens == 350


def test_combine_usage_objects_all_fields():
    """Test combining usage objects with all possible fields populated."""
    usage1 = Usage(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cost=0.001,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=10,
            cached_tokens=20,
            text_tokens=70,
            image_tokens=0,
            web_search_requests=2,
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            audio_tokens=5,
            reasoning_tokens=15,
            accepted_prediction_tokens=10,
            rejected_prediction_tokens=5,
        ),
    )
    usage2 = Usage(
        prompt_tokens=200,
        completion_tokens=100,
        total_tokens=300,
        cost=0.002,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            audio_tokens=15,
            cached_tokens=30,
            text_tokens=155,
            image_tokens=0,
            web_search_requests=3,
        ),
        completion_tokens_details=CompletionTokensDetailsWrapper(
            audio_tokens=10,
            reasoning_tokens=30,
            accepted_prediction_tokens=20,
            rejected_prediction_tokens=10,
        ),
    )

    result = BaseTokenUsageProcessor.combine_usage_objects([usage1, usage2])

    # Verify basic fields
    assert result.prompt_tokens == 300
    assert result.completion_tokens == 150
    assert result.total_tokens == 450
    assert abs(result.cost - 0.003) < 0.0001

    # Verify prompt_tokens_details
    assert result.prompt_tokens_details is not None
    assert result.prompt_tokens_details.audio_tokens == 25
    assert result.prompt_tokens_details.cached_tokens == 50
    assert result.prompt_tokens_details.text_tokens == 225
    assert result.prompt_tokens_details.image_tokens == 0
    assert result.prompt_tokens_details.web_search_requests == 5

    # Verify completion_tokens_details
    assert result.completion_tokens_details is not None
    assert result.completion_tokens_details.audio_tokens == 15
    assert result.completion_tokens_details.reasoning_tokens == 45
    assert result.completion_tokens_details.accepted_prediction_tokens == 30
    assert result.completion_tokens_details.rejected_prediction_tokens == 15


def test_combine_usage_objects_caching_behavior():
    """Test that the field caching mechanism works correctly."""
    # First call should compute and cache fields
    usage1 = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    result1 = BaseTokenUsageProcessor.combine_usage_objects([usage1])

    # Second call should use cached fields
    usage2 = Usage(prompt_tokens=200, completion_tokens=100, total_tokens=300)
    result2 = BaseTokenUsageProcessor.combine_usage_objects([usage2])

    # Both should produce correct results
    assert result1.prompt_tokens == 100
    assert result1.completion_tokens == 50
    assert result2.prompt_tokens == 200
    assert result2.completion_tokens == 100

    # Verify the cache was populated
    assert BaseTokenUsageProcessor._usage_numeric_fields_cache is not None
    assert len(BaseTokenUsageProcessor._usage_numeric_fields_cache) > 0


def test_combine_usage_objects_large_batch():
    """Test combining a large number of usage objects (performance regression test)."""
    # Create 100 usage objects
    usage_objects = [
        Usage(
            prompt_tokens=100 + i,
            completion_tokens=50 + i,
            total_tokens=150 + 2 * i,
            cost=0.001 * i,
            prompt_tokens_details=PromptTokensDetailsWrapper(
                audio_tokens=i, cached_tokens=i * 2, text_tokens=100, image_tokens=0
            ),
            completion_tokens_details=CompletionTokensDetailsWrapper(
                audio_tokens=i,
                reasoning_tokens=i * 3,
                accepted_prediction_tokens=i,
                rejected_prediction_tokens=0,
            ),
        )
        for i in range(100)
    ]

    result = BaseTokenUsageProcessor.combine_usage_objects(usage_objects)

    # Verify the sum is correct
    expected_prompt_tokens = sum(100 + i for i in range(100))
    expected_completion_tokens = sum(50 + i for i in range(100))
    expected_total_tokens = sum(150 + 2 * i for i in range(100))

    assert result.prompt_tokens == expected_prompt_tokens
    assert result.completion_tokens == expected_completion_tokens
    assert result.total_tokens == expected_total_tokens

    # Verify nested details are correctly summed
    assert result.prompt_tokens_details is not None
    expected_audio = sum(i for i in range(100))
    expected_cached = sum(i * 2 for i in range(100))
    assert result.prompt_tokens_details.audio_tokens == expected_audio
    assert result.prompt_tokens_details.cached_tokens == expected_cached
