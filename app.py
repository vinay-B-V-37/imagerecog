from flask import Flask, request, jsonify
import os
import shutil
import face_recognition
from firebase_admin import credentials, firestore, initialize_app
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from dotenv import load_dotenv

# Load configuration from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

# Get secret key and Firebase admin key path from environment variables
SECRET_KEY = os.getenv('SECRET_KEY', 'defaultsecretkey')
FIREBASE_ADMIN_KEY_PATH = os.getenv('FIREBASE_ADMIN_KEY_PATH', 'default-path.json')

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Firebase setup
cred = credentials.Certificate(FIREBASE_ADMIN_KEY_PATH)
firebase_app = initialize_app(cred)
db = firestore.client()

@app.route("/")
def hello():
    return "Hello, it's Flask"

# Route to test the app
@app.route("/api/test", methods=["GET"])
def test():
    return "Hello, World!"

# Route to match images using face recognition
@app.route("/api/match-images/", methods=["POST"])
def match_images():
    if 'reference_image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    reference_image = request.files['reference_image']
    group_id = request.form.get("group_id", "uLFXzY5qXGg23xmFoacq")

    if reference_image.content_type not in ["image/jpeg", "image/png"]:
        return jsonify({"error": "Invalid image format. Use JPEG or PNG."}), 400

    # Save reference image locally
    file_location = os.path.join(UPLOAD_FOLDER, reference_image.filename)
    reference_image.save(file_location)

    # Load the reference image and get face encoding
    reference_encoding = face_recognition.face_encodings(face_recognition.load_image_file(file_location))

    if not reference_encoding:
        return jsonify({"error": "No face found in the reference image."}), 400

    # Get image URLs from Firestore using the provided group ID
    image_urls = get_image_urls_from_firestore(group_id)

    # Process the images and find matches
    matching_images = process_images_from_urls_in_batches(reference_encoding[0], image_urls)

    return jsonify({"matching_images": matching_images})

# Helper function to get image URLs from Firestore
def get_image_urls_from_firestore(group_id: str):
    image_urls = []
    docs = db.collection('groups').document(group_id).collection('photos').stream()

    for doc in docs:
        data = doc.to_dict()
        image_urls.append(data.get('photoURL'))  # Assuming 'photoURL' field contains the image URL

    return image_urls

# Face recognition logic to process image URLs in batches
def process_images_from_urls_in_batches(reference_encoding, image_urls, batch_size=5, tolerance=0.6):
    matching_images = []

    for i in range(0, len(image_urls), batch_size):
        batch_urls = image_urls[i:i + batch_size]
        batch_images = []

        for url in batch_urls:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))

            if img.mode != 'RGB':
                img = img.convert('RGB')

            batch_images.append(np.array(img))

        # Get face encodings for all images in the batch
        for img_array in batch_images:
            encodings = face_recognition.face_encodings(img_array)

            # Compare faces in batch with the reference face encoding
            for unknown_encoding in encodings:
                results = face_recognition.compare_faces([reference_encoding], unknown_encoding, tolerance=tolerance)
                if results[0]:
                    matching_images.append(url)
                    break  # Stop checking if one match is found per image

    return matching_images

# Main entry point for the application
if __name__ == "__main__":
    # Check if running in production
    if os.getenv("FLASK_ENV") == "production":
        # Use Gunicorn in production
        from gunicorn.app.base import BaseApplication
        class FlaskApplication(BaseApplication):
            def __init__(self, app, options=None):
                self.app = app
                self.options = options or {}
                super().__init__()
            def load_config(self):
                for key, value in self.options.items():
                    self.cfg.set(key, value)
            def load(self):
                return app
        # Example configuration for Gunicorn
        options = {
            "bind": "0.0.0.0:8000",
            "workers": 4,
            "worker_class": "gthread",
        }
        FlaskApplication(app, options).run()
    else:
        # Run the Flask app in development mode
        app.run(debug=True)
