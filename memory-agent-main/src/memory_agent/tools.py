"""Tool definitions for the conversational memory agent.

Author: Samaya
Created: 2025
Description: Implements the memory upsert operation used by the chat model.
"""

import uuid
from typing import Annotated

from langchain_core.tools import InjectedToolArg
from langgraph.store.base import BaseStore


async def save_memory_record(
    content: str,
    context: str,
    *,
    record_id: uuid.UUID | None = None,
    # The following arguments are injected by the runtime and hidden from the model.
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
) -> str:
    """Create or update a stored memory record.

    Args:
        content: Main content of the memory, such as a user preference.
        context: Additional context for when or how the memory was collected.
        record_id: Identifier of an existing memory to update, if applicable.
    """
    memory_key = record_id or uuid.uuid4()
    await store.aput(
        ("memories", user_id),
        key=str(memory_key),
        value={"content": content, "context": context},
    )
    return f"Stored memory {memory_key}"
