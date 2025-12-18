"""Test configuration and shared fixtures.

Author: Samaya
Created: 2025
Description: Provides the asynchronous test backend for anyio-based tests.
"""

import pytest


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
