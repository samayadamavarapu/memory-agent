"""Integration tests for the conversational memory graph.

Author: Samaya
Created: 2025
Description: Ensures that user messages result in stored memories for the correct user.
"""

from typing import List

import langsmith as ls
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from memory_agent.context import AgentConfig
from memory_agent.graph import builder

pytestmark = pytest.mark.anyio


@ls.unit
@pytest.mark.parametrize(
    "conversation",
    [
        ["The user likes pizza and wants this preference remembered."],
        [
            "The user enjoys playing tennis. Remember this.",
            "The user also has a pet dog.",
            "The dog is a golden retriever and is 5 years old. Remember this as well.",
        ],
        [
            "The user works as a software engineer and is interested in AI.",
            "The user is focused on machine learning algorithms and natural language processing.",
            "The user is trying to improve sentiment analysis accuracy in multilingual text.",
            "Progress has been made using transformer models, but context and idioms remain challenging.",
            "Some language pairs are more difficult due to structural and cultural differences.",
        ],
    ],
    ids=["short", "medium", "long"],
)
async def test_memory_storage(conversation: List[str]) -> None:
    storage = InMemoryStore()

    graph = builder.compile(store=storage, checkpointer=InMemorySaver())
    user_id = "test-user"

    for content in conversation:
        await graph.ainvoke(
            {"messages": [("user", content)]},
            {"thread_id": "thread"},
            context=AgentConfig(user_id=user_id),
        )

    namespace = ("memories", user_id)
    user_records = storage.search(namespace)

    ls.expect(len(user_records)).to_be_greater_than(0)

    bad_namespace = ("memories", "wrong-user")
    bad_records = storage.search(bad_namespace)
    ls.expect(len(bad_records)).to_equal(0)
