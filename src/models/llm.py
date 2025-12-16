from agno import agent
from src.models.base import Model, Language, LangDetectorChoices
from src.agents.lang_detector import create_agent_lang_detector

class LLM(Model):
    def __init__(self):
        self.agent = create_agent_lang_detector()
        pass
    
    def ask_agent(self, txt: str) -> LangDetectorChoices:
        response = self.agent.run(input=txt)
        return LangDetectorChoices.parse_raw(response)

    def detect_lang_single(self, txt: str) -> Language:
        return self.ask_agent(txt).primary_lang

    def detect_lang_probabilities(self, txt: str) -> LangDetectorChoices:
        return self.ask_agent(txt)