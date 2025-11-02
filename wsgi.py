#!/usr/bin/env python3
"""
WSGI entry point for Render deployment
"""
import os
import sys

# Add the emotion_detector directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'emotion_detector'))

# Import the Flask app
from app import app as application

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    application.run(host='0.0.0.0', port=port, debug=False)