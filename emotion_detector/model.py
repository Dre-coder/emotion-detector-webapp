"""
Streamlined FER2013 Emotion Detection System
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import time

# Lazy import for DeepFace
DeepFace = None

def lazy_import_deepface():
    global DeepFace
    if DeepFace is None:
        try:
            from deepface import DeepFace as DF
            DeepFace = DF
        except ImportError:
            DeepFace = "unavailable"
    return DeepFace != "unavailable"

class EmotionDetector:
    def __init__(self, model_mode='deepface'):
        self.model_mode = model_mode
        self.custom_model = None
        self.emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.fer2013_emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    
    def check_fer2013_structure(self):
        """Check if FER2013 dataset exists"""
        csv_path = "fer2013/fer2013.csv"
        if os.path.exists(csv_path):
            return "csv"
        
        train_path = "fer2013/train"
        test_path = "fer2013/test"
        if os.path.exists(train_path) and os.path.exists(test_path):
            emotion_folders = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            if all(os.path.exists(os.path.join(train_path, emotion)) for emotion in emotion_folders):
                return "folders"
        
        print("❌ FER2013 dataset not found!")
        print("Download from: https://www.kaggle.com/datasets/msambare/fer2013")
        print("Extract to 'fer2013' folder")
        return None
    
    def load_fer2013_csv(self):
        """Load FER2013 CSV data"""
        try:
            df = pd.read_csv("fer2013/fer2013.csv")
            
            def pixels_to_array(pixel_string):
                return np.array([int(p) for p in pixel_string.split()]).reshape(48, 48)
            
            X = np.array([pixels_to_array(pixels) for pixels in df['pixels']])
            y = [self.fer2013_emotions[emotion_id] for emotion_id in df['emotion'].values]
            
            if 'Usage' in df.columns:
                train_mask = df['Usage'] == 'Training'
                test_mask = df['Usage'] == 'PrivateTest'
                return X[train_mask], np.array(y)[train_mask], X[test_mask], np.array(y)[test_mask]
            else:
                return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None, None, None, None
    
    def load_fer2013_folders(self):
        """Load FER2013 data from folder format"""
        train_path = "fer2013/train"
        test_path = "fer2013/test"
        
        def load_images_from_folders(base_path):
            X, y = [], []
            for emotion in self.emotion_names:
                emotion_path = os.path.join(base_path, emotion)
                if not os.path.exists(emotion_path):
                    continue
                
                images = [f for f in os.listdir(emotion_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for img_file in images:
                    img_path = os.path.join(emotion_path, img_file)
                    try:
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (48, 48))
                            X.append(img)
                            y.append(emotion)
                    except Exception:
                        continue
            return np.array(X), np.array(y)
        
        try:
            X_train, y_train = load_images_from_folders(train_path)
            X_test, y_test = load_images_from_folders(test_path)
            print(f"Loaded {len(X_train)} training and {len(X_test)} test samples")
            return X_train, y_train, X_test, y_test
        except Exception as e:
            print(f"Error loading folders: {e}")
            return None, None, None, None
        """Extract features from 48x48 image"""
        if img_array.shape != (48, 48):
            img_array = cv2.resize(img_array, (48, 48))
        
        features = []
        
        # Statistical features
        features.extend([
            np.mean(img_array), np.std(img_array), np.min(img_array), 
            np.max(img_array), np.median(img_array)
        ])
        
        # Histogram
        hist = cv2.calcHist([img_array], [0], None, [16], [0, 256])
        features.extend(hist.flatten())
        
        # Texture
        features.extend([np.var(img_array), cv2.Laplacian(img_array, cv2.CV_64F).var()])
        
        # Regional features
        h, w = img_array.shape
        regions = [
            img_array[:h//2, :w//2], img_array[:h//2, w//2:],
            img_array[h//2:, :w//2], img_array[h//2:, w//2:]
        ]
        for region in regions:
            features.extend([np.mean(region), np.std(region)])
        
        return np.array(features)
    
    def train_fer2013_model(self, model_type='random_forest', max_samples=None):
        """Train model on FER2013"""
        format_type = self.check_fer2013_structure()
        if not format_type:
            return False
        
        if format_type == "csv":
            X_train, y_train, X_test, y_test = self.load_fer2013_csv()
        else:  # folders format
            X_train, y_train, X_test, y_test = self.load_fer2013_folders()
        
        if X_train is None:
            return False
        
        # Limit samples for testing
        if max_samples and len(X_train) > max_samples:
            indices = np.random.choice(len(X_train), max_samples, replace=False)
            X_train, y_train = X_train[indices], y_train[indices]
        
        # Extract features
        print(f"Extracting features from {len(X_train)} training images...")
        X_train_features = np.array([self.extract_features(img) for img in X_train])
        X_test_features = np.array([self.extract_features(img) for img in X_test])
        
        # Train model
        if model_type == 'svm':
            self.custom_model = SVC(kernel='rbf', probability=True, random_state=42)
        else:  # default to random_forest
            self.custom_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        
        print(f"Training {model_type}...")
        self.custom_model.fit(X_train_features, y_train)
        
        # Evaluate
        y_pred = self.custom_model.predict(X_test_features)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.3f}")
        
        # Save model
        model_filename = f"fer2013_emotion_model_{model_type}.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump({
                'model': self.custom_model,
                'emotions': self.fer2013_emotions,
                'emotion_names': self.emotion_names
            }, f)
        
        print(f"Model saved: {model_filename}")
        return accuracy
    
    def load_model(self, filepath):
        """Load trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.custom_model = model_data['model']
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_emotion(self, image_path):
        """Predict emotion from image"""
        start_time = time.time()
        
        if self.model_mode == 'custom' and self.custom_model:
            return self._predict_fer2013(image_path, start_time)
        elif self.model_mode == 'hybrid':
            result = self._predict_fer2013(image_path, start_time)
            if 'error' in result or result.get('confidence', 0) < 0.6:
                return self._predict_deepface(image_path, start_time)
            return result
        else:
            return self._predict_deepface(image_path, start_time)
    
    def _predict_fer2013(self, image_path, start_time):
        """Predict using FER2013 model"""
        if not self.custom_model:
            return {"error": "No FER2013 model loaded"}
        
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {"error": "Could not load image"}
            
            img = cv2.resize(img, (48, 48))
            features = self.extract_features(img)
            
            prediction = self.custom_model.predict([features])[0]
            probabilities = self.custom_model.predict_proba([features])[0]
            
            return {
                'emotion': prediction,
                'confidence': max(probabilities),
                'all_scores': dict(zip(self.emotion_names, probabilities)),
                'source': 'FER2013',
                'processing_time': time.time() - start_time
            }
        except Exception as e:
            return {"error": f"FER2013 prediction failed: {e}"}
    
    def _predict_deepface(self, image_path, start_time):
        """Predict using DeepFace"""
        if not lazy_import_deepface():
            return {"error": "DeepFace not available"}
        
        try:
            # Use opencv detector for fastest processing
            result = DeepFace.analyze(
                image_path, 
                actions=['emotion'], 
                detector_backend='opencv',  # Fastest detector
                enforce_detection=False
            )
            if isinstance(result, list):
                result = result[0]
            
            emotions = result.get('emotion', {})
            if not emotions:
                return {"error": "No emotions detected"}
            
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            
            return {
                'emotion': dominant_emotion[0],
                'confidence': dominant_emotion[1] / 100.0,
                'all_scores': {k: v/100.0 for k, v in emotions.items()},
                'source': 'DeepFace',
                'processing_time': time.time() - start_time
            }
        except Exception as e:
            return {"error": f"DeepFace failed: {e}"}

def train_fer2013_model():
    """Train FER2013 model"""
    detector = EmotionDetector()
    
    print("🎯 FER2013 Training")
    print("1. Random Forest (fast)")
    print("2. SVM (accurate)")
    
    choice = input("Choose (1-2): ").strip()
    model_type = 'svm' if choice == '2' else 'random_forest'
    
    max_samples = input("Max samples (Enter for all): ").strip()
    max_samples = int(max_samples) if max_samples else None
    
    accuracy = detector.train_fer2013_model(model_type, max_samples)
    if accuracy:
        print(f"Training complete! Accuracy: {accuracy:.1%}")
        print("Set MODEL_MODE='custom' to use this model")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_fer2013_model()
    else:
        print("Usage: python model.py train")