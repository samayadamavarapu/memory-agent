"""Conversation state model for the memory graph.

Author: Samaya
Created: 2025
Description: Defines the structured state tracked as the agent processes messages.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


@dataclass(kw_only=True)
class ConversationState:
    """Graph state containing the conversation message history."""

    messages: Annotated[list[AnyMessage], add_messages]
    """Ordered list of messages exchanged in the conversation."""


__all__ = [
    "ConversationState",
]
