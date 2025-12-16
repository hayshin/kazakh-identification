from src.agents.agent import create_agent_from_config

async def create_agent_araline_ru():
    """Create the Araline support agent."""
    return create_agent_from_config(
        short_name="araline-ru",
        id="araline-ru-booking-agent",
        name="Araline Tech Support Agent in Russian Language",
        role="Ты эксперт технической поддержки для Araline, телекоммуникационной компании.",
        add_datetime_to_context=True,
        has_memory=True,
    )
