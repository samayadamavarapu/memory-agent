"""Graph definition for the conversational memory agent.

Author: Samaya
Created: 2025
Description: Builds a LangGraph state machine that reads and writes user memories.
"""

import asyncio
import logging
from datetime import datetime
from typing import cast, Dict, Any

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from memory_agent import tools, utils
from memory_agent.context import AgentConfig
from memory_agent.state import ConversationState

logger = logging.getLogger(__name__)


async def generate_response(
    state: ConversationState, runtime: Runtime[AgentConfig]
) -> Dict[str, Any]:
    """Generate a response from the chat model and optionally request memory updates."""
    user_id = runtime.context.user_id
    model = runtime.context.model
    system_prompt_template = runtime.context.system_prompt

    # Retrieve the most recent user records related to the conversation.
    user_records = await cast(BaseStore, runtime.store).asearch(
        ("memories", user_id),
        query=str([m.content for m in state.messages[-3:]]),
        limit=10,
    )

    # Format user records for inclusion in the prompt.
    memory_context = "\n".join(
        f"[{record.key}]: {record.value} (similarity: {record.score})"
        for record in user_records
    )
    if memory_context:
        memory_context = f"""
<memories>
{memory_context}
</memories>"""

    # Prepare the system message with user memories and current time.
    system_message = system_prompt_template.format(
        user_info=memory_context, time=datetime.now().isoformat()
    )

    # Create the chat model and invoke it with memory tools.
    llm = utils.create_chat_model(model)
    response = await llm.bind_tools([tools.save_memory_record]).ainvoke(
        [{"role": "system", "content": system_message}, *state.messages]
    )
    return {"messages": [response]}


async def persist_records(
    state: ConversationState, runtime: Runtime[AgentConfig]
) -> Dict[str, Any]:
    """Persist any memory tool calls emitted by the last model invocation."""
    tool_calls = getattr(state.messages[-1], "tool_calls", [])

    saved_records = await asyncio.gather(
        *(
            tools.save_memory_record(
                **tc["args"],
                user_id=runtime.context.user_id,
                store=cast(BaseStore, runtime.store),
            )
            for tc in tool_calls
        )
    )

    tool_responses = [
        {
            "role": "tool",
            "content": record,
            "tool_call_id": tc["id"],
        }
        for tc, record in zip(tool_calls, saved_records)
    ]
    return {"messages": tool_responses}


def select_next_step(state: ConversationState) -> str:
    """Determine the next step based on whether tool calls are present."""
    last_message = state.messages[-1]
    if getattr(last_message, "tool_calls", None):
        return "persist_records"
    return END


# Create the graph and all nodes.
builder = StateGraph(ConversationState, context_schema=AgentConfig)

# Define the flow of the memory extraction process.
builder.add_node(generate_response)
builder.add_edge("__start__", "generate_response")
builder.add_node(persist_records)
builder.add_conditional_edges(
    "generate_response", select_next_step, ["persist_records", END]
)
# After storing records, the graph currently returns control to the user.
builder.add_edge("persist_records", "generate_response")
graph = builder.compile()
graph.name = "MemoryAgent"


__all__ = ["graph"]
