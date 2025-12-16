import os
import numpy as np
import tensorflow as tf

from src.models.base import Model, LangDetectorChoices, Language


_MODEL_PATH = os.path.join("data", "models", "bilstm.keras")
_bilstm_model = tf.keras.models.load_model(_MODEL_PATH)


def predict_language(text: str) -> LangDetectorChoices:
    x = tf.constant([[text]], dtype=tf.string)
    p = _bilstm_model.predict(x, verbose=0).ravel()
    prob_kk = float(p[0])
    prob_ru = float(1.0 - prob_kk)
    primary = "kazakh" if prob_kk >= 0.5 else "russian"
    return {
        "russian": prob_ru,
        "kazakh": prob_kk,
        "other": float(0.0),
        "primary_lang": primary,
    }


class BiLSTM(Model):
    def detect_lang_single(self, txt: str) -> Language:
        return predict_language(txt)["primary_lang"]

    def detect_lang_probabilities(self, txt: str) -> LangDetectorChoices:
        return predict_language(txt)
