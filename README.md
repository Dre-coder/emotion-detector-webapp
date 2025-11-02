# 🎭 FER2013 Emotion Detection System

A Flask-based web application for real-time emotion detection using both DeepFace and custom-trained FER2013 models.

![Emotion Detection Demo](https://img.shields.io/badge/Status-Active-green)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)

## ✨ Features

- **🎥 Real-time webcam emotion detection**
- **📸 Image upload and analysis**
- **🤖 Multiple detection modes:**
  - DeepFace (pre-trained models)
  - FER2013 (custom-trained models)
  - Hybrid (best of both)
- **📊 Confidence scoring and detailed results**
- **🎯 7 emotion categories:** angry, disgust, fear, happy, sad, surprise, neutral
- **⚡ Optimized for speed with OpenCV detector**
- **📱 Responsive web interface**

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Webcam (optional, for real-time detection)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/emotion-detector.git
   cd emotion-detector
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install tf-keras  # For TensorFlow compatibility
   ```

4. **Run the application:**
   ```bash
   cd emotion_detector
   python app.py
   ```

5. **Open your browser:** http://127.0.0.1:5000

## 🎯 Usage Modes

### DeepFace Mode (Default)
```bash
python app.py
# Uses pre-trained DeepFace models
```

### FER2013 Custom Mode
```bash
# Train your model first
python model.py train

# Set environment variable
set MODEL_MODE=custom  # Windows
export MODEL_MODE=custom  # macOS/Linux

# Run app
python app.py
```

### Hybrid Mode
```bash
set MODEL_MODE=hybrid
python app.py
# Tries FER2013 first, falls back to DeepFace
```

## 🏋️ Training Your Own Model

### Option 1: FER2013 Dataset (Recommended)

1. **Download FER2013:**
   - Visit: https://www.kaggle.com/datasets/msambare/fer2013
   - Download and extract to `fer2013/` folder

2. **Train model:**
   ```bash
   python model.py train
   # Choose Random Forest (fast) or SVM (accurate)
   ```

3. **Expected accuracy:** 60-70%

### Folder Structure
```
emotion_detector/
├── app.py                      # Flask web application
├── model.py                    # FER2013 training & prediction
├── requirements.txt            # Python dependencies
├── static/                     # CSS and captured images
├── templates/                  # HTML templates
├── fer2013/                    # FER2013 dataset (download separately)
│   └── fer2013.csv
└── *.pkl                       # Trained models (generated)
```

## 🎮 Web Interface

### Features:
- **📷 Webcam Capture:** Real-time emotion detection
- **📁 File Upload:** Analyze existing images
- **📊 Results Display:** Emotion breakdown with confidence
- **🎛️ Mode Selection:** Switch between detection methods
- **📝 History:** View previous detections

### Supported Image Formats:
- PNG, JPG, JPEG
- Recommended: Clear face visibility, good lighting

## ⚙️ Configuration

### Environment Variables:
```bash
MODEL_MODE=deepface    # Use DeepFace models (default)
MODEL_MODE=custom      # Use FER2013 trained model
MODEL_MODE=hybrid      # Try custom first, fallback to DeepFace
```

### Performance Tuning:
- **DeepFace + OpenCV:** ~3x faster than MTCNN
- **FER2013 models:** ~10x faster than DeepFace
- **Model warm-up:** Enabled for faster subsequent detections

## 📊 Accuracy Comparison

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| DeepFace | ⚡⚡ | 70-75% | General use, no training needed |
| FER2013 Random Forest | ⚡⚡⚡ | 60-65% | Fast processing, resource-limited |
| FER2013 SVM | ⚡⚡ | 65-70% | Balanced speed and accuracy |

## 🔧 Technical Details

### Dependencies:
- **Flask:** Web framework
- **OpenCV:** Image processing
- **scikit-learn:** Machine learning (FER2013 training)
- **pandas/numpy:** Data manipulation
- **DeepFace:** Pre-trained emotion models
- **TensorFlow:** Deep learning backend

### Architecture:
- **Frontend:** HTML5, CSS3, JavaScript
- **Backend:** Flask (Python)
- **Models:** TensorFlow (DeepFace), scikit-learn (FER2013)
- **Database:** SQLite (user history)

## 🐛 Troubleshooting

### Common Issues:

**1. TensorFlow Errors:**
```bash
pip install tf-keras
```

**2. Webcam Not Working:**
- Check browser permissions
- Ensure HTTPS for some browsers
- Try different browsers

**3. Slow Detection:**
- Use FER2013 mode: `MODEL_MODE=custom`
- Enable model warm-up in app.py
- Close other applications using camera

**4. Training Fails:**
- Download FER2013 dataset correctly
- Ensure sufficient disk space (>1GB)
- Try smaller sample size for testing

## 📈 Future Enhancements

- [ ] Real-time video stream processing
- [ ] Multiple face detection
- [ ] Age and gender detection
- [ ] Mobile app version
- [ ] API endpoints for integration
- [ ] Advanced model architectures (CNN, transformers)
- [ ] Cloud deployment ready

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FER2013 Dataset:** Facial Expression Recognition Challenge
- **DeepFace:** Serengil & Ozpinar (2020)
- **OpenCV:** Computer Vision Library
- **Flask Community:** Web framework

## 📞 Support

- **Issues:** GitHub Issues tab
- **Documentation:** This README
- **Training Guide:** See `FER2013_GUIDE.md`

---

**⭐ Star this repo if you found it helpful!**