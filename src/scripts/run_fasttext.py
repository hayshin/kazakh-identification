import os
import fasttext

model_bin = os.path.join("data", "fasttext", "ru_kk_model.bin")

model = fasttext.load_model(model_bin)

label_map = {"__label__ru": "russian", "__label__kk": "kazakh"}

def predict_language(text: str):
    labels, probs = model.predict([text], k=3)
    print(labels[0], probs[0])
    result = dict(zip(labels[0], probs[0]))

    return {
        "russian": float(result.get("__label__ru", 0.0)),
        "kazakh": float(result.get("__label__kk", 0.0)),
        "other": float(1.0 - sum(result.values())),
        "primary_lang": label_map.get(labels[0][0], "other"),
    }

sentences = ["Net ya tebya ne ponimaiu"]
print(predict_language(sentences[0]))