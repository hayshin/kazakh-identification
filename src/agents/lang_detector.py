from src.agents.agent import create_agent_from_config
from src.models.base import LangDetectorChoices

def create_agent_lang_detector():
    """Create the lang detector agent."""
    return create_agent_from_config(
        short_name="lang-detector",
        id="lang-detector-agent",
        name="Lang Detector Agent",
        role="You are a language detection agent that analyzes user queries and determines the language they are written in.",
        has_memory=False,
        output_schema=LangDetectorChoices,
    )
