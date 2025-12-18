"""Utility helpers for working with chat models.

Author: Samaya
Created: 2025
Description: Provides a thin wrapper for initializing chat models by provider and name.
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel


def create_chat_model(model_specification: str) -> BaseChatModel:
    """Create a chat model instance from a provider/model specification string.

    Args:
        model_specification: String in the format 'provider/model-name'.
    """
    provider, model_name = model_specification.split("/", maxsplit=1)
    return init_chat_model(model_name, model_provider=provider)
