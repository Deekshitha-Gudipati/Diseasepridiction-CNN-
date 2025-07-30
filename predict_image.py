import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load model
model = tf.keras.models.load_model("medical_model.h5")

# Ask for image path
img_path = input("Enter image path: ").strip()

# Load and preprocess image
img = image.load_img(img_path, target_size=(224, 224))
img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

# Predict
pred = model.predict(img_array)[0][0]
print("Confidence Score:", pred)
print("Prediction:", "Pneumonia" if pred > 0.5 else "Normal")
