import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('/content/traffic_sign_model.keras')

# Function to preprocess a single image
def preprocess_single_image(image_path):
    img = image.load_img(image_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions on a single image
def predict_traffic_sign(image_path):
    img_array = preprocess_single_image(image_path)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    # Map class index to traffic sign label (replace this with your label mapping)
    sign_label = class_index  # Placeholder, replace this with your label mapping
    return sign_label

# Directory containing test images
test_images_dir = '/content/drive/MyDrive/Coding/traffic sign/test image directory'

# Get list of test images
test_images = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith('.jpg')]

# Make predictions for each test image
for img_path in test_images:
    predicted_sign = predict_traffic_sign(img_path)
    print(f"Predicted traffic sign for {img_path}: {predicted_sign}")

    # Display the test image
    img = image.load_img(img_path, target_size=(150, 150))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
