#!/bin/bash
# Start the Flask application with Gunicorn

echo "Starting Emotion Detector Web App..."

# Navigate to the app directory
cd emotion_detector

# Start with Gunicorn
gunicorn --bind 0.0.0.0:$PORT app:app --timeout 120 --workers 1 --max-requests 1000