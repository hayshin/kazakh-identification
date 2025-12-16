from agno.models.openai import OpenAIChat
from agno.db.sqlite import SqliteDb
from agno.agent import Agent as AgnoAgent
from agno.db.in_memory import InMemoryDb
from pathlib import Path
from typing import Optional, Union, List

from pydantic import BaseModel


def create_agent_from_config(
    short_name: str,
    id: str,
    name: str,
    role: str,
    tools: Optional[Union[List, object]] = None,
    output_schema: Optional[BaseModel] = None,
    add_datetime_to_context: bool = False,
    has_memory: bool = True,
    db: Optional[SqliteDb] = None,
    num_history_runs: int = 20,
    prompt_path: Optional[Path] = None,
    model: Optional[OpenAIChat] = None,
) -> AgnoAgent:
    """
    Factory function to create agents with automatic prompt loading.

    Args:
        agent_name: Name of the agent (used to find prompt in data/{agent_name}/prompt.md)
        id: Agent ID
        name: Agent display name
        role: Agent role description
        tools: Regular tools to add (can be a single tool or list of tools)
        mcp_tools: MCP tools to add (can be a single tool or list of tools)
        add_datetime_to_context: Whether to add datetime to context
        has_memory: Whether agent has memory/database
        db: Database instance (defaults to InMemoryDb if not provided)
        num_history_runs: Number of history runs to include in context (default: 20)
        prompt_path: Custom path to prompt file (overrides default location)
        model: Custom model instance (defaults to OpenAIChat(id="gpt-4.1"))

    Returns:
        Configured AgnoAgent instance
    """
    # Load prompt file - try multiple locations
    if prompt_path is None:
        # Try data/{short_name}/prompt.md first
        data_prompt = Path("data/") / short_name.lower() / "prompt.md"

        if data_prompt.exists():
            prompt_path = data_prompt
        else:
            raise FileNotFoundError(f"Prompt file not found for {short_name}")

    with open(prompt_path, "r", encoding="utf-8") as file:
        system_message = file.read()

    # Create model if not provided
    if model is None:
        model = OpenAIChat(id="gpt-5-mini")

    # Create agent
    agent = AgnoAgent(
        model=model,
        id=id,
        name=name,
        role=role,
        system_message=system_message,
        output_schema=output_schema,
        markdown=True,
        tools=tools,
        add_datetime_to_context=add_datetime_to_context,
        debug_mode=True,
    )

    # Configure memory if needed
    if has_memory:
        agent.db = db or InMemoryDb()
        agent.add_history_to_context = True
        agent.num_history_runs = num_history_runs

    return agent
