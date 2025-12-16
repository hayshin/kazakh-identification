import os
import fasttext

from src.models.base import Model, LangDetectorChoices, Language

model_bin = os.path.join("data", "models", "fasttext.bin")

model = fasttext.load_model(model_bin)

label_map = {"__label__ru": "russian", "__label__kk": "kazakh"}

def predict_language(text: str):
    labels, probs = model.predict([text], k=3)
    result = dict(zip(labels[0], probs[0]))

    return {
        "russian": float(result.get("__label__ru", 0.0)),
        "kazakh": float(result.get("__label__kk", 0.0)),
        "other": float(1.0 - sum(result.values())),
        "primary_lang": label_map.get(labels[0][0], "other"),
    }

class FastText(Model):
    def detect_lang_single(self, txt: str) -> Language:
        return predict_language(txt)["primary_lang"]

    def detect_lang_probabilities(self, txt: str) -> LangDetectorChoices:
        return predict_language(txt)

        
