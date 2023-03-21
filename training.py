import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Define the labels
labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Load the training data
train_data = keras.preprocessing.image_dataset_from_directory(
    "./hw4_train",
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=16,
    image_size=(28, 28),
    validation_split=0.25,
    subset="training",
    seed=42
)

# Define the model architecture
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])


# Compile the model
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_data, epochs=10)