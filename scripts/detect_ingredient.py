# detect_ingredient.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Suppress TensorFlow logging (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the trained model
model = tf.keras.models.load_model('models/ingredient_model.h5')

# Class labels ordered alphabetically for 25 ingredients
class_labels = {
    0: 'apple',
    1: 'banana',
    2: 'beans',
    3: 'beetroot',
    4: 'brinjal',
    5: 'cabbage',
    6: 'carrot',
    7: 'chili pepper',
    8: 'chili powder',
    9: 'curry leaves',
    10: 'curry powder',
    11: 'dhal',
    12: 'garlic',
    13: 'grapes',
    14: 'lemon',
    15: 'mango',
    16: 'onion',
    17: 'orange',
    18: 'pandan leaf',
    19: 'pineapple',
    20: 'potato',
    21: 'salt',
    22: 'tomato',
    23: 'turmeric powder',
    24: 'watermelon'
}

def detect_ingredients(img_path, threshold=0.5):
    """
    Detect multiple ingredients in an image.

    Parameters:
    - img_path (str): Path to the image file.
    - threshold (float): Probability threshold to consider for ingredient detection.

    Returns:
    - list: List of detected ingredient names.
    """
    try:
        logging.info(f"Processing image: {img_path}")
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        predictions = model.predict(img_array)[0]  # Get prediction vector
        logging.info(f"Predictions: {predictions}")  # Log predictions

        detected_ingredients = [
            class_labels[idx] for idx, prob in enumerate(predictions) if prob >= threshold
        ]

        if not detected_ingredients:
            # If no ingredient exceeds the threshold, take the top prediction
            top_idx = np.argmax(predictions)
            detected_ingredients = [class_labels.get(top_idx, "Unknown Ingredient")]

        logging.info(f"Detected Ingredients: {detected_ingredients}")
        return detected_ingredients
    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")
        return []
