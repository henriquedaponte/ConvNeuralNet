# Starter code for CS 165B HW4

import training
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image



# Load the testing data
test_images = []
test_image_paths = []

for i in range(10000):
    image_path = f"./hw4_test/{i}.png"
    image = Image.open(image_path)
    image_data = np.array(image.getdata()).reshape(28, 28, 1)
    test_images.append(image_data)
    test_image_paths.append(image_path)



test_data = np.array(test_images)


# Make predictions on the testing data
predictions = training.model.predict(test_data)

# Write the predictions to a file
with open("prediction.txt", "w") as f:
    for prediction in predictions:
        label = np.argmax(prediction)
        f.write(str(label) + "\n")