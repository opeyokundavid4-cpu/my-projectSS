import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64

app = Flask(__name__, static_folder='../static')
CORS(app)

# Load model and classes
model = tf.keras.models.load_model("../models/asl_cnn_model.h5")
with open("../data/classes.txt", "r") as f:
    classes = f.read().splitlines()

def preprocess_image(image_b64):
    encoded_data = image_b64.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return send_from_directory('../static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        img = preprocess_image(data['image'])
        prediction = model.predict(img)
        class_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        return jsonify({
            'class': classes[class_idx],
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
