"""
Test for Prisma token refresh deadlock fix.

This test validates that PrismaWrapper.__getattr__ does not deadlock when called
from within a running asyncio event loop with an expired IAM token.
"""
import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.db.prisma_client import PrismaWrapper


@pytest.mark.asyncio
async def test_no_deadlock_when_called_from_running_loop():
    """
    Test that __getattr__ does not deadlock when called from within a running event loop.
    
    This validates the fix for the issue where asyncio.run_coroutine_threadsafe()
    was called from the same thread as the running event loop, causing a 30-second timeout.
    """
    # Create a mock Prisma client
    mock_prisma = MagicMock()
    mock_prisma.some_attribute = "test_value"
    
    # Create PrismaWrapper with IAM token auth enabled
    wrapper = PrismaWrapper(mock_prisma, iam_token_db_auth=True)
    
    # Mock the token expiration check to always return True
    def mock_is_token_expired(db_url):
        return True
    
    # Mock the token generation
    def mock_get_rds_iam_token():
        return "postgresql://user:newtoken@host:5432/db"
    
    # Track if recreate was called
    recreate_called = asyncio.Event()
    
    # Mock the recreate method
    async def mock_recreate_prisma_client(new_db_url):
        await asyncio.sleep(0.1)  # Simulate some async work
        recreate_called.set()
    
    wrapper.is_token_expired = mock_is_token_expired
    wrapper.get_rds_iam_token = mock_get_rds_iam_token
    wrapper.recreate_prisma_client = mock_recreate_prisma_client
    
    # Set DATABASE_URL environment variable
    os.environ["DATABASE_URL"] = "postgresql://user:oldtoken@host:5432/db"
    
    # Call __getattr__ from within a running event loop
    start_time = time.time()
    result = wrapper.__getattr__("some_attribute")
    elapsed = time.time() - start_time
    
    # Verify the call completed quickly (< 1 second)
    assert elapsed < 1.0, f"Call took {elapsed:.2f}s, expected < 1s (no deadlock)"
    
    # Verify we got the attribute
    assert result == "test_value"
    
    # Give the background task time to complete
    await asyncio.sleep(0.2)
    
    # Verify the background refresh was scheduled
    assert recreate_called.is_set(), "Background token refresh should have been called"
