"""
Reproduction script for Prisma token refresh deadlock.

This script demonstrates the deadlock that occurs when PrismaWrapper.__getattr__
is called from within a running asyncio event loop and the IAM token is expired.
"""
import asyncio
import os
import sys
import time
from unittest.mock import MagicMock, patch

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from litellm.proxy.db.prisma_client import PrismaWrapper


def mock_is_token_expired(db_url):
    """Mock that always returns True to simulate an expired token."""
    return True


def mock_get_rds_iam_token():
    """Mock that returns a dummy new database URL."""
    return "postgresql://user:newtoken@host:5432/db"


async def mock_recreate_prisma_client(new_db_url):
    """Mock that simulates async reconnection."""
    await asyncio.sleep(0.1)  # Simulate some async work
    print("Mock recreate_prisma_client called")


async def main():
    """
    Main test function that triggers the deadlock scenario.
    
    This creates a PrismaWrapper with IAM token auth enabled,
    patches the token expiration check to always return True,
    and then calls __getattr__ from within a running event loop.
    """
    # Create a mock Prisma client
    mock_prisma = MagicMock()
    mock_prisma.some_attribute = "test_value"
    
    # Create PrismaWrapper with IAM token auth enabled
    wrapper = PrismaWrapper(mock_prisma, iam_token_db_auth=True)
    
    # Patch methods to simulate expired token scenario
    wrapper.is_token_expired = mock_is_token_expired
    wrapper.get_rds_iam_token = mock_get_rds_iam_token
    wrapper.recreate_prisma_client = mock_recreate_prisma_client
    
    # Set DATABASE_URL environment variable
    os.environ["DATABASE_URL"] = "postgresql://user:oldtoken@host:5432/db"
    
    print("Starting getattr() call inside running event loop - this will block")
    start_time = time.time()
    
    try:
        # This is the problematic call - it will deadlock for 30 seconds
        _ = wrapper.__getattr__("some_attribute")
        print("getattr succeeded (unexpected!)")
    except Exception as e:
        print(f"getattr raised (as expected) - traceback:")
        import traceback
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    print(f"Elapsed time for getattr call: {elapsed:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
