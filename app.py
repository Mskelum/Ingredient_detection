# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from scripts.detect_ingredient import detect_ingredients
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Welcome to the Ingredient Detection API!'}), 200

@app.route('/detect', methods=['POST'])
def detect():
    logging.info("Received /detect request")
    if 'image' not in request.files:
        logging.warning("No image file found in the request")
        return jsonify({'error': 'No image file found'}), 400

    # Get the uploaded file
    try:
        file = request.files['image']
        if file.filename == '':
            logging.warning("Empty filename")
            return jsonify({'error': 'No image file found'}), 400

        # Save the image temporarily with a unique filename
        unique_filename = f"temp_image_{uuid.uuid4().hex}.jpg"
        temp_img_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(temp_img_path)
        logging.info(f"Saved image to {temp_img_path}")

        # Call your ingredient detection function
        detected_ingredients = detect_ingredients(temp_img_path)

        # Optionally, delete the image after processing to save space
        os.remove(temp_img_path)
        logging.info(f"Removed temporary image {temp_img_path}")

        return jsonify({'ingredients': detected_ingredients}), 200
    except Exception as e:
        logging.error(f"Error in /detect endpoint: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
