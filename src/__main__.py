from typing import Union, Optional, Literal
import asyncio

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama

from src.models.base import LangDetectorChoices, Language
from src.models.kaznlp import KazNLP
from src.models.llm import LLM
from src.models.fasttext import FastText
from src.models.knn import KNN
from src.models.bilstm import BiLSTM
from src.agents.araline_en import create_agent_araline_en
from src.agents.araline_kk import create_agent_araline_kk
from src.agents.araline_ru import create_agent_araline_ru

app = FastAPI(title="Kazakh Language Identification API")

# Initialize models
kaznlp_model = KazNLP()
llm_model = LLM()
fasttext_model = FastText()
knn_model = KNN()
bilstm_model = BiLSTM()

# Model registry with weights for ensembling
models_config = [
    {"name": "kaznlp", "model": kaznlp_model, "weight": 1.0},
    # {"name": "llm", "model": llm_model, "weight": 1.0},
    {"name": "fasttext", "model": fasttext_model, "weight": 1.0},
    {"name": "knn", "model": knn_model, "weight": 1.0},
    {"name": "bilstm", "model": bilstm_model, "weight": 1.0},
]


class TextInput(BaseModel):
    text: str


class ChatInput(BaseModel):
    message: str
    model: Literal["gpt-5-mini", "gpt-5-nano", "local"] = "gpt-5-mini"


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
            },
            "fasttext": {
                "/fasttext/detect": "Detect primary language using FastText model",
                "/fasttext/probabilities": "Get language probabilities using FastText model",
            },
            "knn": {
                "/knn/detect": "Detect primary language using KNN model",
                "/knn/probabilities": "Get language probabilities using KNN model",
            },
            "bilstm": {
                "/bilstm/detect": "Detect primary language using BiLSTM model",
                "/bilstm/probabilities": "Get language probabilities using BiLSTM model",
            },
            "all": {
                "/all/probabilities": "Get language probabilities from all models at once",
            },
            "chat": {
                "/chat": "Send message to Araline chatbot with language detection and dual-agent response",
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


# FastText routes
@app.post("/fasttext/detect", response_model=dict)
def fasttext_detect_language(input_data: TextInput) -> dict:
    """Detect the primary language of the input text using FastText model."""
    try:
        language = fasttext_model.detect_lang_single(input_data.text)
        return {"text": input_data.text, "language": language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting language: {str(e)}")


@app.post("/fasttext/probabilities", response_model=LangDetectorChoices)
def fasttext_language_probabilities(input_data: TextInput) -> LangDetectorChoices:
    """Get language detection probabilities for the input text using FastText model."""
    try:
        probabilities = fasttext_model.detect_lang_probabilities(input_data.text)
        return probabilities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating probabilities: {str(e)}")


# KNN routes
@app.post("/knn/detect", response_model=dict)
def knn_detect_language(input_data: TextInput) -> dict:
    """Detect the primary language of the input text using KNN model."""
    try:
        language = knn_model.detect_lang_single(input_data.text)
        return {"text": input_data.text, "language": language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting language: {str(e)}")


@app.post("/knn/probabilities", response_model=LangDetectorChoices)
def knn_language_probabilities(input_data: TextInput) -> LangDetectorChoices:
    """Get language detection probabilities for the input text using KNN model."""
    try:
        probabilities = knn_model.detect_lang_probabilities(input_data.text)
        return probabilities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating probabilities: {str(e)}")


# BiLSTM routes
@app.post("/bilstm/detect", response_model=dict)
def bilstm_detect_language(input_data: TextInput) -> dict:
    """Detect the primary language of the input text using BiLSTM model."""
    try:
        language = bilstm_model.detect_lang_single(input_data.text)
        return {"text": input_data.text, "language": language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting language: {str(e)}")


@app.post("/bilstm/probabilities", response_model=LangDetectorChoices)
def bilstm_language_probabilities(input_data: TextInput) -> LangDetectorChoices:
    """Get language detection probabilities for the input text using BiLSTM model."""
    try:
        probabilities = bilstm_model.detect_lang_probabilities(input_data.text)
        return probabilities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating probabilities: {str(e)}")


# All models routes
class ModelResult(BaseModel):
    """Result from a single model."""
    model: str
    result: Union[LangDetectorChoices, dict]


@app.post("/all/probabilities", response_model=dict)
def all_models_probabilities(input_data: TextInput) -> dict:
    """Get language detection probabilities from all models and weighted ensemble."""
    results = []
    ensemble_scores = {"kazakh": 0.0, "russian": 0.0, "other": 0.0}
    total_weight = 0.0

    def _extract_probabilities(raw_result):
        if isinstance(raw_result, LangDetectorChoices):
            return {
                "kazakh": float(raw_result.kazakh),
                "russian": float(raw_result.russian),
                "other": float(raw_result.other),
                "primary_lang": raw_result.primary_lang,
            }
        return {
            "kazakh": float(raw_result.get("kazakh", 0.0)),
            "russian": float(raw_result.get("russian", 0.0)),
            "other": float(raw_result.get("other", 0.0)),
            "primary_lang": raw_result.get("primary_lang", "other"),
        }

    for entry in models_config:
        name = entry["name"]
        model = entry["model"]
        weight = float(entry.get("weight", 1.0))

        try:
            probs = model.detect_lang_probabilities(input_data.text)
            results.append({"model": name, "result": probs})

            parsed = _extract_probabilities(probs)
            ensemble_scores["kazakh"] += weight * parsed["kazakh"]
            ensemble_scores["russian"] += weight * parsed["russian"]
            ensemble_scores["other"] += weight * parsed["other"]
            total_weight += weight
        except Exception as e:
            results.append({"model": name, "error": f"Error calculating probabilities: {str(e)}"})

    if total_weight > 0:
        final_probs = {k: v / total_weight for k, v in ensemble_scores.items()}
    else:
        final_probs = {"kazakh": 0.0, "russian": 0.0, "other": 0.0}

    final_language = max(final_probs, key=final_probs.get)

    return {
        "models": results,
        "ensemble": {
            "probabilities": final_probs,
            "primary_lang": final_language,
            "weight_sum": total_weight,
        },
    }


# Chat route with language detection and dual-agent response
@app.post("/chat")
async def chat_with_araline(
    input_data: ChatInput,
    session_id: Optional[str] = Query(default=None, description="Session ID for conversation tracking")
) -> dict:
    """
    Send message to Araline chatbot with automatic language detection.
    Returns responses from both English agent and user-language agent.
    """
    try:
        # Step 1: Detect language using ensemble
        ensemble_result = all_models_probabilities(TextInput(text=input_data.message))
        detected_lang = ensemble_result["ensemble"]["primary_lang"]
        
        # Step 2: Select model based on input
        if input_data.model == "gpt-5-mini":
            model = OpenAIChat(id="gpt-5-mini")
        elif input_data.model == "gpt-5-nano":
            model = OpenAIChat(id="gpt-5-nano")
        elif input_data.model == "local":
            model = Ollama(id="gemma3")
        else:
            model = OpenAIChat(id="gpt-5-mini")
        
        # Step 3: Create agents with selected model
        agent_en = await create_agent_araline_en(model=model)
        
        # Select language-specific agent based on detection
        if detected_lang == "kazakh":
            agent_lang = await create_agent_araline_kk(model=model)
        elif detected_lang == "russian":
            agent_lang = await create_agent_araline_ru(model=model)
        else:
            # Default to English for "other"
            agent_lang = await create_agent_araline_en(model=model)
        
        # Step 4: Get responses from both agents in parallel
        # Use static user_id and provided session_id for conversation context
        response_en_task = asyncio.create_task(
            asyncio.to_thread(agent_en.run, input=input_data.message, user_id="araline-en-user", session_id=session_id)
        )
        response_lang_task = asyncio.create_task(
            asyncio.to_thread(agent_lang.run, input=input_data.message, user_id="araline-lang-user", session_id=session_id)
        )
        
        response_en, response_lang = await asyncio.gather(response_en_task, response_lang_task)
        
        return {
            "detected_language": detected_lang,
            "language_probabilities": ensemble_result["ensemble"]["probabilities"],
            "model_used": input_data.model,
            "session_id": session_id,
            "responses": {
                "english": response_en.content if hasattr(response_en, 'content') else str(response_en),
                "user_language": response_lang.content if hasattr(response_lang, 'content') else str(response_lang),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")