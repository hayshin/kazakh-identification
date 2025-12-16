from src.agents.agent import create_agent_from_config

async def create_agent_araline_kk():
    """Create the Araline support agent."""
    return create_agent_from_config(
        short_name="araline-kk",
        id="araline-kk-booking-agent",
        name="Araline Tech Support Agent in Kazakh Language",
        role="Сіз Araline телекоммуникациялық компаниясының техникалық қолдау сарапшысысыз.",
        add_datetime_to_context=True,
        has_memory=True,
    )
