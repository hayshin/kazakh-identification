from abc import ABC, abstractmethod
from typing import Literal, TypeAlias

from pydantic import BaseModel, Field

Language: TypeAlias = Literal["kazakh", "russian", "other"]

class LangDetectorChoices(BaseModel):
    """Router decision with probabilities for each agent."""

    kazakh: float = Field(
        description="Probability (0.0-1.0), that this query is written in Kazakh language",
        ge=0.0,
        le=1.0,
    )
    russian: float = Field(
        description="Probability (0.0-1.0), that this query is written in Russian language",
        ge=0.0,
        le=1.0,
    )
    other: float = Field(
        description="Probability (0.0-1.0), that this query is not written in Kazakh or Russian language",
        ge=0.0,
        le=1.0,
    )
    primary_lang: Literal["kazakh", "russian", "other"] = Field(
        description="Primary language with the highest probability"
    )

class Model(ABC):
    """Abstract base class for language detection models."""

    @abstractmethod
    def detect_lang_single(self, txt: str) -> Language:
        """
        Detect the language of a single text input.
        
        Args:
            txt: Input text to detect language for.
            
        Returns:
            Language: The detected language.
        """
        pass

    @abstractmethod
    def detect_lang_probabilities(self, txt: str) -> LangDetectorChoices:
        """
        Detect language probabilities for a text input.
        
        Args:
            txt: Input text to analyze.
            
        Returns:
            LanguageDetectionMap: Dictionary mapping language names to probabilities.
        """
        pass
