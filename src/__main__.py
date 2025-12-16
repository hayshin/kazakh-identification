from typing import Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.base import LangDetectorChoices, Language
from src.models.kaznlp import KazNLP
from src.models.llm import LLM

app = FastAPI(title="Kazakh Language Identification API")

# Initialize models
kaznlp_model = KazNLP()
llm_model = LLM()


class TextInput(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {
        "message": "Kazakh Language Identification API",
        "endpoints": {
            "kaznlp": {
                "/kaznlp/detect": "Detect primary language using KazNLP model",
                "/kaznlp/probabilities": "Get language probabilities using KazNLP model",
            },
            "llm": {
                "/llm/detect": "Detect primary language using LLM model",
                "/llm/probabilities": "Get language probabilities using LLM model",
            }
        }
    }


# KazNLP routes
@app.post("/kaznlp/detect", response_model=dict)
def kaznlp_detect_language(input_data: TextInput) -> dict:
    """Detect the primary language of the input text using KazNLP model."""
    try:
        language = kaznlp_model.detect_lang_single(input_data.text)
        return {"text": input_data.text, "language": language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting language: {str(e)}")


@app.post("/kaznlp/probabilities", response_model=LangDetectorChoices)
def kaznlp_language_probabilities(input_data: TextInput) -> LangDetectorChoices:
    """Get language detection probabilities for the input text using KazNLP model."""
    try:
        probabilities = kaznlp_model.detect_lang_probabilities(input_data.text)
        return probabilities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating probabilities: {str(e)}")


# LLM routes
@app.post("/llm/detect", response_model=dict)
def llm_detect_language(input_data: TextInput) -> dict:
    """Detect the primary language of the input text using LLM model."""
    try:
        language = llm_model.detect_lang_single(input_data.text)
        return {"text": input_data.text, "language": language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting language: {str(e)}")


@app.post("/llm/probabilities", response_model=LangDetectorChoices)
def llm_language_probabilities(input_data: TextInput) -> LangDetectorChoices:
    """Get language detection probabilities for the input text using LLM model."""
    try:
        probabilities = llm_model.detect_lang_probabilities(input_data.text)
        return probabilities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating probabilities: {str(e)}")