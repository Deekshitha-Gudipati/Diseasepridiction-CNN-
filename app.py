# STEP 1: Import Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

# STEP 2: Set Base Directory Paths (UPDATE THIS TO YOUR LOCAL PATH)
base_dir = "D:/datasets/chest_xray"  # Replace with your path
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# STEP 3: Image Data Generators
train_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(224, 224), class_mode='binary')
val_data = val_gen.flow_from_directory(val_dir, target_size=(224, 224), class_mode='binary')
test_data = test_gen.flow_from_directory(test_dir, target_size=(224, 224), class_mode='binary')

# STEP 4: Define CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# STEP 5: Compile & Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_data, validation_data=val_data, epochs=10)

# STEP 6: Evaluate & Save
loss, acc = model.evaluate(test_data)
print(f"Test Accuracy: {acc*100:.2f}%")
model.save("medical_model.h5")

