#!/bin/bash
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Creating directories..."
mkdir -p emotion_detector/static/images
echo "Build complete!"