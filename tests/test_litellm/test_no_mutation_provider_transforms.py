"""
Test that provider transformation helpers do not mutate caller-owned message lists.

Multiple provider transform helpers mutate caller-owned messages lists in place,
which can surprise direct/internal callers that reuse the same list object across calls.
"""

import sys
import os
from copy import deepcopy

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.llms.vertex_ai.gemini.transformation import (
    _transform_system_message as gemini_transform_system,
)
from litellm.llms.replicate.chat.transformation import ReplicateConfig


def test_anthropic_translate_system_message_does_not_mutate_input():
    """Test that Anthropic's translate_system_message does not mutate the input messages list."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]
    original_messages = deepcopy(messages)
    original_length = len(messages)

    config = AnthropicConfig()
    filtered_messages, system_message_list = config.translate_system_message(
        messages=messages
    )

    # Verify input list was NOT mutated
    assert messages == original_messages, "Input messages list should not be mutated"
    assert len(messages) == original_length, "Input messages list length should not change"

    # Verify the function still works correctly
    assert len(system_message_list) == 1, "Should extract one system message"
    assert len(filtered_messages) == 1, "Should return one non-system message"
    assert filtered_messages[0]["role"] == "user", "Remaining message should be user message"


def test_anthropic_translate_system_message_multiple_system_messages():
    """Test handling of multiple system messages."""
    messages = [
        {"role": "system", "content": "System message 1"},
        {"role": "user", "content": "User message"},
        {"role": "system", "content": "System message 2"},
        {"role": "assistant", "content": "Assistant message"},
    ]
    original_messages = deepcopy(messages)

    config = AnthropicConfig()
    filtered_messages, system_message_list = config.translate_system_message(
        messages=messages
    )

    # Verify input list was NOT mutated
    assert messages == original_messages, "Input messages list should not be mutated"

    # Verify correct extraction
    assert len(system_message_list) == 2, "Should extract two system messages"
    assert len(filtered_messages) == 2, "Should return two non-system messages"

    # Verify content and order of extracted system messages
    assert len(system_message_list) >= 2, "Should have at least 2 system messages"
    assert "text" in system_message_list[0], "System message should have 'text' key"
    assert "text" in system_message_list[1], "System message should have 'text' key"
    assert system_message_list[0]["text"] == "System message 1", "First system message content should match"
    assert system_message_list[1]["text"] == "System message 2", "Second system message content should match"

    # Verify remaining messages
    assert filtered_messages[0]["role"] == "user", "First remaining message should be user"
    assert filtered_messages[1]["role"] == "assistant", "Second remaining message should be assistant"


def test_bedrock_transform_system_message_does_not_mutate_input():
    """Test that Bedrock's _transform_system_message does not mutate the input messages list."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]
    original_messages = deepcopy(messages)
    original_length = len(messages)

    config = AmazonConverseConfig()
    filtered_messages, system_content_blocks = config._transform_system_message(
        messages=messages, model="anthropic.claude-3-sonnet-20240229-v1:0"
    )

    # Verify input list was NOT mutated
    assert messages == original_messages, "Input messages list should not be mutated"
    assert len(messages) == original_length, "Input messages list length should not change"

    # Verify the function still works correctly
    assert len(system_content_blocks) == 1, "Should extract one system message"
    assert len(filtered_messages) == 1, "Should return one non-system message"
    assert filtered_messages[0]["role"] == "user", "Remaining message should be user message"


def test_gemini_transform_system_message_does_not_mutate_input():
    """Test that Gemini's _transform_system_message does not mutate the input messages list."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]
    original_messages = deepcopy(messages)
    original_length = len(messages)

    system_instructions, filtered_messages = gemini_transform_system(
        supports_system_message=True, messages=messages
    )

    # Verify input list was NOT mutated
    assert messages == original_messages, "Input messages list should not be mutated"
    assert len(messages) == original_length, "Input messages list length should not change"

    # Verify the function still works correctly
    assert system_instructions is not None, "Should extract system instructions"
    assert len(filtered_messages) == 1, "Should return one non-system message"
    assert filtered_messages[0]["role"] == "user", "Remaining message should be user message"


def test_gemini_transform_system_message_when_not_supported():
    """Test that Gemini's _transform_system_message handles unsupported case correctly."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]
    original_messages = deepcopy(messages)

    system_instructions, filtered_messages = gemini_transform_system(
        supports_system_message=False, messages=messages
    )

    # Verify input list was NOT mutated
    assert messages == original_messages, "Input messages list should not be mutated"

    # Verify the function still works correctly
    assert system_instructions is None, "Should not extract system instructions when not supported"
    assert len(filtered_messages) == len(messages), "Should return all messages"


def test_replicate_transform_request_does_not_mutate_input():
    """Test that Replicate's transform_request does not mutate the input messages list."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]
    original_messages = deepcopy(messages)
    original_length = len(messages)

    config = ReplicateConfig()
    result = config.transform_request(
        model="replicate/meta/llama-2-7b-chat",
        messages=messages,
        optional_params={"supports_system_prompt": True},
        litellm_params={},
        headers={},
    )

    # Verify input list was NOT mutated
    assert messages == original_messages, "Input messages list should not be mutated"
    assert len(messages) == original_length, "Input messages list length should not change"

    # Verify the function still works correctly
    assert isinstance(result, dict), "Should return a dict"
    assert "input" in result, "Result should contain input"


def test_replicate_transform_request_without_system_prompt_support():
    """Test Replicate's transform_request when system prompt is not supported."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]
    original_messages = deepcopy(messages)

    config = ReplicateConfig()
    result = config.transform_request(
        model="replicate/meta/llama-2-7b-chat",
        messages=messages,
        optional_params={"supports_system_prompt": False},
        litellm_params={},
        headers={},
    )

    # Verify input list was NOT mutated
    assert messages == original_messages, "Input messages list should not be mutated"

    # Verify the function still works correctly
    assert isinstance(result, dict), "Should return a dict"
