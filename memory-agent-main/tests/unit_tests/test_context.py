"""Unit tests for the AgentConfig configuration class.

Author: Samaya
Created: 2025
Description: Verifies that configuration values are initialized and overridden correctly.
"""

import os

from memory_agent.context import AgentConfig


def test_context_init() -> None:
    config = AgentConfig(user_id="test-user")
    assert config.user_id == "test-user"


def test_context_init_with_env_vars() -> None:
    os.environ["USER_ID"] = "test-user"
    config = AgentConfig()
    assert config.user_id == "test-user"


def test_context_init_with_env_vars_and_passed_values() -> None:
    os.environ["USER_ID"] = "test-user"
    config = AgentConfig(user_id="actual-user")
    assert config.user_id == "actual-user"
