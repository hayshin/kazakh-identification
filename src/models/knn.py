import os
import sys

from src.models.base import Model, LangDetectorChoices, Language
import joblib

def inverse_distance_weights(distances):
    """Custom weight function for KNN model."""
    return 1 / (distances + 1e-3)

# Ensure pickle can resolve the custom weight function when loading the model.
sys.modules.setdefault("__main__", sys.modules[__name__]).inverse_distance_weights = inverse_distance_weights

model = joblib.load(os.path.join("data", "models", "knn.pkl"))

def predict_language_with_words(text) -> LangDetectorChoices:
    probs = model.predict_proba([text])[0]
    labels = model.classes_

    model_probs = dict(zip(labels, probs))
    best_label = max(model_probs, key=model_probs.get)
    if best_label == "KK":
        best_label = "kazakh"
    elif best_label == "RU":
        best_label = "russian"
    else:
        best_label = "other"

    return {
        "primary_lang": best_label,
        "kazakh": model_probs["KK"],
        "russian": model_probs["RU"],
        "other": 1 - model_probs["KK"] - model_probs["RU"],
    }

class KNN(Model):
    def detect_lang_single(self, txt: str) -> Language:
        return predict_language_with_words(txt)["primary_lang"]

    def detect_lang_probabilities(self, txt: str) -> LangDetectorChoices:
        return predict_language_with_words(txt)

        
