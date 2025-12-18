"""Runtime configuration for the conversational memory agent.

Author: Samaya
Created: 2025
Description: Defines the context object used to parameterize the memory graph at runtime.
"""

import os
from dataclasses import dataclass, field, fields

from typing_extensions import Annotated

from memory_agent import prompts


@dataclass(kw_only=True)
class AgentConfig:
    """Configuration and environment for a memory-enabled conversation."""

    user_id: str = "default"
    """Identifier for the user whose memories are being stored and retrieved."""

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-sonnet-4-5-20250929",
        metadata={
            "description": "The language model to use for the agent "
            "in the form provider/model-name."
        },
    )

    system_prompt: str = prompts.DEFAULT_SYSTEM_MESSAGE

    def __post_init__(self) -> None:
        """Populate unset fields from matching environment variables."""
        for f in fields(self):
            if not f.init:
                continue

            current_value = getattr(self, f.name)
            if current_value == f.default:
                env_value = os.environ.get(f.name.upper(), f.default)
                setattr(self, f.name, env_value)
