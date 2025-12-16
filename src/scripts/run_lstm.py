import os
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model(os.path.join("data", "bidirectional-lstm", "best_bilstm.keras"))

x = tf.constant([["Net ya tebya ne ponimaiu"], ["Красивые глаза можешь айтуга кайдан алдн"]], dtype=tf.string)
p = model.predict(x, verbose=0).ravel()
p = np.array(p)
labels = np.where(p >= 0.7, "KK", "RU")
print(labels)  # ['RU' 'KK']