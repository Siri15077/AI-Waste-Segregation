import tensorflow as tf
from PIL import Image
import numpy as np

MODEL_PATH = "model/best_waste_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


def predict_waste(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((160, 160))

    image_array = np.array(image) / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = predictions[0][predicted_index] * 100

    return predicted_class, confidence