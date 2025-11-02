#!/bin/bash
# Render build script

echo "Starting build process..."

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p emotion_detector/static/images
mkdir -p emotion_detector/models

echo "Build completed successfully!"