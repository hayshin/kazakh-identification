import os
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(os.path.join("data", "models", "bilstm.keras"))

label_map = {"KK": "kazakh", "RU": "russian"}

def predict_language(text: str):
	x = tf.constant([[text]], dtype=tf.string)
	p = model.predict(x, verbose=0).ravel()
	prob_kk = float(p[0])
	prob_ru = float(1.0 - prob_kk)
	primary = label_map["KK"] if prob_kk >= 0.5 else label_map["RU"]
	return {
		"russian": prob_ru,
		"kazakh": prob_kk,
		"other": float(0.0),
		"primary_lang": primary,
	}

if __name__ == "__main__":
	sentence = "Net ya tebya ne ponimaiu"
	print(predict_language(sentence))