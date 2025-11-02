from flask import Flask, render_template, request, jsonify
import sqlite3, os, uuid, base64
from datetime import datetime

# Don't import DeepFace immediately - import it when needed
emotion_detector = None

# Model selection - can be 'deepface', 'custom', or 'hybrid'
MODEL_MODE = os.getenv('MODEL_MODE', 'deepface').lower()

app = Flask(__name__)

# Ensure static directory exists
static_dir = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(static_dir, exist_ok=True)

DB_PATH = 'emotion_users.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# initialize database
def init_db():
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS users
                       (id TEXT PRIMARY KEY, name TEXT, image_path TEXT, emotion TEXT, created_at TEXT)''')
        conn.commit()
    finally:
        conn.close()

# Initialize database on startup
init_db()

def warm_up_deepface():
    """Pre-load DeepFace models to reduce first-use delay"""
    try:
        print("Warming up DeepFace models...")
        global DeepFace
        if DeepFace is None:
            from deepface import DeepFace as DF
            DeepFace = DF
        
        # Create a small test image to warm up the models
        import numpy as np
        from PIL import Image
        
        # Create a 100x100 RGB test image
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_path = os.path.join(static_dir, 'warmup_test.jpg')
        Image.fromarray(test_img).save(test_path)
        
        # Run a quick analysis to load models
        DeepFace.analyze(
            img_path=test_path,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )
        
        # Clean up test image
        if os.path.exists(test_path):
            os.remove(test_path)
            
        print("DeepFace models warmed up successfully!")
        
    except Exception as e:
        print(f"Failed to warm up DeepFace: {e}")
        print("Models will be loaded on first use instead.")

# Uncomment the next line if you want to pre-load models on startup (slower startup but faster first use)
# Warm up models on startup for faster detection
warm_up_deepface()

EMOTIONS = ['angry','disgust','fear','happy','sad','surprise','neutral']

# Add a development mode flag for faster testing
DEVELOPMENT_MODE = os.getenv('DEVELOPMENT_MODE', 'false').lower() == 'true'

def analyze_emotion(image_path):
    import time
    start_time = time.time()
    
    global emotion_detector
    
    # In development mode, return a random emotion for faster testing
    if DEVELOPMENT_MODE:
        import random
        emotion = random.choice(EMOTIONS)
        elapsed = time.time() - start_time
        print(f"Development mode: returning random emotion '{emotion}' in {elapsed:.2f}s")
        return emotion
    
    # Initialize emotion detector if needed
    if emotion_detector is None:
        try:
            from model import EmotionDetector
            emotion_detector = EmotionDetector()
            
            # Auto-load FER2013 model if available and in custom/hybrid mode
            if MODEL_MODE in ['custom', 'hybrid']:
                # Look for FER2013 models only
                model_files = [f for f in os.listdir('.') if 
                              f.startswith('fer2013_emotion_model_') 
                              and f.endswith('.pkl')]
                
                if model_files:
                    model_file = model_files[0]
                    print(f"Loading FER2013 model: {model_file}")
                    emotion_detector.load_model(model_file)
                    
        except Exception as e:
            print(f"Error initializing emotion detector: {e}")
            return f"Error: Could not initialize emotion detector - {str(e)}"
    
    # Predict emotion using the integrated model
    try:
        result = emotion_detector.predict_emotion(image_path)
        
        if 'error' in result:
            return result['error']
        
        emotion = result.get('emotion', 'unknown')
        confidence = result.get('confidence', 0)
        source = result.get('source', 'unknown')
        
        # Format result with confidence and source info
        if confidence < 0.3:
            return f"{emotion} (low confidence: {confidence:.1%}, {source})"
        else:
            return f"{emotion} ({source}, {confidence:.1%})"
            
    except Exception as e:
        print(f"Emotion analysis error: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name', 'Unknown')
    file = request.files.get('image')
    
    if not file or file.filename == '':
        return render_template('index.html', error='No file uploaded.')
    
    # Validate file type
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    file_ext = os.path.splitext(file.filename.lower())[1]
    if file_ext not in allowed_extensions:
        return render_template('index.html', error='Invalid file type. Please upload an image.')
    
    try:
        filename = f"{uuid.uuid4().hex}_{file.filename.replace(' ', '_')}"
        full_path = os.path.join(static_dir, filename)
        relative_path = os.path.join('static', filename)
        file.save(full_path)

        emotion = analyze_emotion(full_path)

        conn = get_db_connection()
        try:
            cur = conn.cursor()
            uid = uuid.uuid4().hex
            cur.execute('INSERT INTO users (id, name, image_path, emotion, created_at) VALUES (?, ?, ?, ?, ?)',
                        (uid, name, relative_path, emotion, datetime.utcnow().isoformat()))
            conn.commit()
        finally:
            conn.close()

        return render_template('index.html', name=name, emotion=emotion, image=relative_path)
    
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return render_template('index.html', error=f'An error occurred: {str(e)}')

@app.route('/capture', methods=['POST'])
def capture():
    try:
        # receives a JSON payload with 'name' and 'image' (base64 data URL)
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON payload received.'}), 400
        
        name = data.get('name', 'Unknown')
        img_data = data.get('image')
        if not img_data:
            return jsonify({'error': 'No image data received.'}), 400

        # data URL prefix like "data:image/png;base64,...."
        if ',' in img_data:
            header, img_b64 = img_data.split(',', 1)
        else:
            img_b64 = img_data
        
        try:
            img_bytes = base64.b64decode(img_b64)
        except Exception as e:
            return jsonify({'error': 'Invalid base64 image data.'}), 400

        filename = f"capture_{uuid.uuid4().hex}.png"
        full_path = os.path.join(static_dir, filename)
        relative_path = os.path.join('static', filename)
        
        with open(full_path, 'wb') as f:
            f.write(img_bytes)

        emotion = analyze_emotion(full_path)

        conn = get_db_connection()
        try:
            cur = conn.cursor()
            uid = uuid.uuid4().hex
            cur.execute('INSERT INTO users (id, name, image_path, emotion, created_at) VALUES (?, ?, ?, ?, ?)',
                        (uid, name, relative_path, emotion, datetime.utcnow().isoformat()))
            conn.commit()
        finally:
            conn.close()

        return jsonify({'name': name, 'emotion': emotion, 'image': relative_path})
    
    except Exception as e:
        print(f"Error in capture route: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/users')
def users():
    try:
        conn = get_db_connection()
        try:
            cur = conn.cursor()
            cur.execute('SELECT id, name, image_path, emotion, created_at FROM users ORDER BY created_at DESC LIMIT 50')
            rows = cur.fetchall()
            users = [dict(r) for r in rows]
            return jsonify(users)
        finally:
            conn.close()
    except Exception as e:
        print(f"Error in users route: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/performance')
def performance_info():
    """Provide performance information and tips"""
    global DeepFace
    return jsonify({
        'deepface_loaded': DeepFace is not None,
        'development_mode': DEVELOPMENT_MODE,
        'tips': [
            "First emotion detection will be slower as models need to be downloaded and loaded",
            "Subsequent detections should be much faster",
            "Using multiple detector backends for better accuracy",
            "Set DEVELOPMENT_MODE=true environment variable for instant testing with random emotions"
        ],
        'optimization_settings': {
            'multiple_detectors': ['mtcnn', 'opencv', 'retinaface', 'ssd'],
            'image_preprocessing': True,
            'confidence_threshold': 30,
            'fallback_detection': True
        }
    })

@app.route('/accuracy-tips')
def accuracy_tips():
    """Provide tips for better emotion detection accuracy"""
    return jsonify({
        'photo_tips': [
            "📸 Ensure good lighting - avoid shadows on the face",
            "👤 Face should be clearly visible and not too small",
            "😊 Look directly at the camera for best results",
            "🎯 Avoid extreme angles or tilted faces",
            "📱 Use high-resolution images when possible",
            "🚫 Remove sunglasses or items covering the face",
            "🔍 Ensure the face takes up at least 20% of the image"
        ],
        'emotion_guidelines': [
            "😊 Smile naturally - forced smiles may be detected as different emotions",
            "😢 For sadness: slight frown with lowered eyebrows works best",
            "😠 For anger: furrowed brows and tense jaw",
            "😨 For fear: wide eyes and raised eyebrows",
            "😮 For surprise: wide eyes and open mouth",
            "🤢 For disgust: wrinkled nose and raised upper lip",
            "😐 For neutral: relaxed facial muscles, no strong expression"
        ],
        'technical_limitations': [
            "⚠️ AI models have inherent limitations and may not always be 100% accurate",
            "🎭 Context matters - the same expression might mean different things to different people",
            "🌍 Cultural differences in expressing emotions can affect accuracy",
            "👥 Models are trained on specific datasets which may not represent all populations equally",
            "🔄 Try taking multiple photos if results seem inconsistent"
        ]
    })

@app.route('/model-info')
def model_info():
    """Get information about current model configuration"""
    global emotion_detector
    
    # Check for available FER2013 models
    model_files = [f for f in os.listdir('.') if 
                  f.startswith('fer2013_emotion_model_') 
                  and f.endswith('.pkl')]
    
    return jsonify({
        'current_mode': MODEL_MODE,
        'available_modes': ['deepface', 'custom', 'hybrid'],
        'detector_initialized': emotion_detector is not None,
        'available_models': model_files,
        'training_available': True,
        'datasets': {
            'fer2013': 'Train on FER2013 benchmark dataset (35k+ grayscale images)'
        },
        'mode_descriptions': {
            'deepface': 'Uses pre-trained DeepFace models (good general accuracy)',
            'custom': 'Uses your FER2013-trained model (best benchmark accuracy)',
            'hybrid': 'Tries FER2013 model first, falls back to DeepFace if uncertain'
        }
    })

@app.route('/start-training')
def start_training():
    """Initialize training data structure"""
    try:
        from model import create_training_structure
        folder = create_training_structure()
        
        return jsonify({
            'success': True,
            'message': 'Training data structure created!',
            'folder': folder,
            'instructions': [
                '1. Navigate to the "training_data" folder',
                '2. Add your labeled emotion images to the corresponding emotion folders',
                '3. Each emotion should have at least 20-50 images for good training',
                '4. Use /train-model endpoint to start training once you have data'
            ],
            'folders_created': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/train-model', methods=['POST'])
def train_model_endpoint():
    """Train a custom emotion model"""
    try:
        from model import train_model
        
        # This is a long-running operation, so we'll return immediately
        # and let the training happen in the background
        import threading
        
        def train_in_background():
            train_model()
        
        training_thread = threading.Thread(target=train_in_background)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Training started in background. Check console for progress.',
            'note': 'Training may take several minutes depending on data size.'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
