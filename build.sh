#!/bin/bash
# Render build script

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p emotion_detector/static/images
mkdir -p emotion_detector/models

# Download required model files (lightweight approach)
echo "Setting up for Render deployment..."

# The app will handle model initialization at runtime