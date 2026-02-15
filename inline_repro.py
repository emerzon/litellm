"""
Reproduction script for Prisma token refresh deadlock.

This script demonstrates that the deadlock has been fixed when PrismaWrapper.__getattr__
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
    Main test function that verifies the deadlock is fixed.
    
    This creates a PrismaWrapper with IAM token auth enabled,
    patches the token expiration check to always return True,
    and then calls __getattr__ from within a running event loop.
    
    Expected behavior after fix: Should complete quickly without blocking.
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
    
    print("Starting getattr() call inside running event loop")
    start_time = time.time()
    
    try:
        # This should now complete quickly without blocking
        result = wrapper.__getattr__("some_attribute")
        print(f"getattr succeeded! Returned: {result}")
    except Exception as e:
        print(f"getattr raised an exception:")
        import traceback
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    print(f"Elapsed time for getattr call: {elapsed:.2f} seconds")
    
    if elapsed < 1.0:
        print("✓ SUCCESS: Call completed quickly without deadlock!")
    else:
        print("✗ FAILURE: Call took too long, possible deadlock")
    
    # Give background task time to complete
    await asyncio.sleep(0.2)
    print("Test completed")


if __name__ == "__main__":
    asyncio.run(main())
