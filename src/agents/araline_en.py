from agno.models.openai import OpenAIChat
from src.agents.agent import create_agent_from_config

async def create_agent_araline_en(model=OpenAIChat(id="gpt-5-mini")):
    """Create the Araline support agent."""
    return create_agent_from_config(
        short_name="araline-en",
        id="araline-en-booking-agent",
        name="Araline Tech Support Agent in English Language",
        role="You are an expert technical support agent for Araline, a telecom company.",
        add_datetime_to_context=True,
        has_memory=True,
        model=model,
    )
