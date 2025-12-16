import os
import joblib

def inverse_distance_weights(distances):
    """Custom weight function for KNN model."""
    return 1 / (distances + 1e-3)

def predict_language_with_words(text, model):
    probs = model.predict_proba([text])[0]
    labels = model.classes_

    model_probs = dict(zip(labels, probs))
    best_label = max(model_probs, key=model_probs.get)


    return {
        "text": text,
        "prediction": best_label,
        "confidence": f"{model_probs[best_label] * 100:.2f}%",
        "model_details": {
            k: f"{v * 100:.2f}%" for k, v in model_probs.items()
        },
    }

model = joblib.load(os.path.join("data", "knn", "lang_knn_model1.pkl"))

text = "Сегодня погода очень хорошая, сондықтан серуендеуге шығайық"

res = predict_language_with_words(text, model)

print("--- РЕЗУЛЬТАТЫ ТЕСТА ---")
print("Text:", res["text"])
print(f"Pred: {res['prediction']} (Conf: {res['confidence']})")
print("Model Detail:", res["model_details"])