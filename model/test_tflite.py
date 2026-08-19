import tensorflow as tf
from PIL import Image
import numpy as np

MODEL_PATH = "model/waste_model.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input:", input_details)
print("Output:", output_details)

image_path = input("Enter image path: ")

image = Image.open(image_path).convert("RGB")
image = image.resize((160, 160))

image_array = np.array(image, dtype=np.float32) / 255.0
image_array = np.expand_dims(image_array, axis=0)

interpreter.set_tensor(
    input_details[0]["index"],
    image_array
)

interpreter.invoke()

predictions = interpreter.get_tensor(
    output_details[0]["index"]
)

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

predicted_index = np.argmax(predictions[0])
predicted_class = class_names[predicted_index]
confidence = predictions[0][predicted_index] * 100

print()
print("Prediction:", predicted_class)
print("Confidence:", round(float(confidence), 2), "%")