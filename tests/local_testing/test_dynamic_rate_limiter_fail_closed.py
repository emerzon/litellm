# What is this?
## Unit tests for dynamic_rate_limiter.py fail-closed behavior
## Tests that rate limiter properly fails closed (rejects requests) when cache errors occur

import asyncio
import sys
import os
import pytest

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm import DualCache, Router
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.dynamic_rate_limiter import (
    _PROXY_DynamicRateLimitHandler as DynamicRateLimitHandler,
)
from fastapi import HTTPException


class FailingUsageCache:
    """Mock cache that fails on read operations"""
    async def async_get_cache(self, model):
        raise RuntimeError('cache backend down')
    
    async def async_set_cache_sadd(self, model, value):
        # Allow writes to succeed (though they won't be used in our test)
        return None


@pytest.fixture
def dynamic_rate_limit_handler() -> DynamicRateLimitHandler:
    internal_cache = DualCache()
    return DynamicRateLimitHandler(internal_usage_cache=internal_cache)


@pytest.fixture
def user_api_key_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key='sk-test', token='test-token', metadata={})


@pytest.mark.asyncio
async def test_cache_error_fails_closed(dynamic_rate_limit_handler, user_api_key_auth):
    """
    Test that when cache lookup fails, the rate limiter fails closed (rejects request).
    
    This is a security test - we should never allow requests through when we can't 
    properly track rate limits due to cache errors.
    """
    model = "test-model"
    
    # Set up router with rate limits
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "tpm": 100,
                    "rpm": 10,
                },
            }
        ]
    )
    dynamic_rate_limit_handler.update_variables(llm_router=llm_router)
    
    # Replace the internal cache with a failing one
    dynamic_rate_limit_handler.internal_usage_cache = FailingUsageCache()
    
    # Attempt to make a request - should raise an exception (fail closed)
    with pytest.raises(Exception) as exc_info:
        await dynamic_rate_limit_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_auth,
            cache=DualCache(),
            data={"model": model},
            call_type="completion",
        )
    
    # Verify that an appropriate error was raised (not a 429, but an internal error)
    # The specific exception type may vary, but it should NOT allow the request through
    assert exc_info.value is not None, "Expected an exception to be raised on cache failure"


@pytest.mark.asyncio
async def test_check_available_usage_raises_on_cache_error(dynamic_rate_limit_handler):
    """
    Test that check_available_usage raises an exception instead of returning None values
    when cache operations fail.
    """
    model = "test-model"
    
    # Set up router
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "tpm": 100,
                },
            }
        ]
    )
    dynamic_rate_limit_handler.update_variables(llm_router=llm_router)
    
    # Replace the internal cache with a failing one
    dynamic_rate_limit_handler.internal_usage_cache = FailingUsageCache()
    
    # check_available_usage should raise an exception, not return (None, None, None, None, None)
    with pytest.raises(Exception) as exc_info:
        result = await dynamic_rate_limit_handler.check_available_usage(
            model=model, priority=None
        )
    
    # Verify exception was raised
    assert exc_info.value is not None


@pytest.mark.asyncio  
async def test_normal_operation_still_works(dynamic_rate_limit_handler, user_api_key_auth):
    """
    Test that normal operation (with working cache) still functions correctly after fix.
    This ensures our fix doesn't break existing functionality.
    """
    model = "test-model"
    
    # Set up router with sufficient limits
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "tpm": 1000,
                    "rpm": 100,
                },
            }
        ]
    )
    dynamic_rate_limit_handler.update_variables(llm_router=llm_router)
    
    # With a working cache, this should not raise an exception
    result = await dynamic_rate_limit_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_auth,
        cache=DualCache(),
        data={"model": model},
        call_type="completion",
    )
    
    # Result should be None (no exception raised)
    assert result is None
