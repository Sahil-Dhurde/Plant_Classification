"""
app.py — Flask Web Server for Plant Leaf Disease Classification.
Provides a modern REST API and serves the dashboard.
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from predict import load_model_and_classes, predict_image

app = Flask(__name__)
CORS(app)

# Configuration for uploads
UPLOAD_FOLDER = os.path.join(config.BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Global model and class names
model = None
class_names = None

def get_model():
    global model, class_names
    if model is None:
        try:
            model, class_names = load_model_and_classes()
            print("[INFO] Model loaded successfully for Flask app.")
        except Exception as e:
            print(f"[ERROR] Could not load model: {e}")
    return model, class_names

@app.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """API endpoint to receive an image and return plant leaf detection result."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Load model lazily
        m, c_names = get_model()
        if m is None:
            return jsonify({'error': 'Model not trained or not found. Run training first.'}), 500

        # Save file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Predict
            cls, conf, top5 = predict_image(m, c_names, filepath, show_plot=False)
            
            # Check if healthy
            is_healthy = 'healthy' in cls.lower()
            
            # Result payload
            result = {
                'class': cls,
                'confidence': float(conf),
                'status': 'Healthy' if is_healthy else 'Diseased',
                'is_healthy': is_healthy,
                'top5': [{'label': t[0], 'confidence': float(t[1])} for t in top5],
                'image_url': f'/static/uploads/{filename}'
            }
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initial model check
    get_model()
    
    print("\n" + "=" * 60)
    print("  🌿 PLANT LEAF CLASSIFIER - WEB SERVER STARTED")
    print("  URL: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=False, port=5000)
