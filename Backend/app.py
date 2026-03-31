import os
import sys

# Disable oneDNN optimizations to prevent potential crashes on some CPUs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import librosa
import numpy as np
from collections import Counter
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import tempfile
import threading
import gdown
import h5py
import json

import tensorflow as tf

# App configuration
MODEL_PATH = "Trained_model.h5"

# Initialize FastAPI
app = FastAPI(
    title="Music Genre Classifier API",
    description="API for classifying music genres from audio files",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to mimic caching
model = None
model_lock = threading.Lock()

def download_model():
    """Download the model if not already present"""
    if not os.path.exists(MODEL_PATH):
        print("Downloading Model from Google Drive...")
        url = "https://drive.google.com/uc?export=download&id=1vc4b2RpeXmnZMn2SOF0snIjos9paVEVH"
        gdown.download(url, MODEL_PATH, quiet=False)

def patch_keras_batch_shape(path):
    """Silently edits the H5 file config to be Keras-3 compliant, avoiding dependency hell"""
    try:
        with h5py.File(path, 'r+') as f:
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    decoded = config_str.decode('utf-8')
                    if '"batch_shape":' in decoded:
                        print("Applying universal Keras 3 patch...")
                        decoded = decoded.replace('"batch_shape":', '"batch_input_shape":')
                        f.attrs['model_config'] = decoded.encode('utf-8')
                elif isinstance(config_str, str) and '"batch_shape":' in config_str:
                    print("Applying universal Keras 3 patch...")
                    f.attrs['model_config'] = config_str.replace('"batch_shape":', '"batch_input_shape":')
    except Exception as e:
        pass

def get_model():
    """Load the model securely using threading locks for safe concurrent initialization"""
    global model
    if model is None:
        with model_lock:
            # Double check inside the lock
            if model is None:
                download_model()
                patch_keras_batch_shape(MODEL_PATH)
                print("Loading TensorFlow Keras model in Inference Mode...")
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                print("Model loaded successfully!")
    return model

def load_and_preprocess_file(file_path, target_shape=(210, 210)):
    """Load and preprocess audio file for prediction"""
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        
        chunk_duration = 4
        overlap_duration = 2
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
        
        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
        
        chunks = []
        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = start + chunk_samples
            
            if end > len(audio_data):
                chunk = np.pad(audio_data[start:], (0, end - len(audio_data)))
            else:
                chunk = audio_data[start:end]

            mel_spec_np = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)

            # Resize to match model input
            resized_spec = tf.image.resize(
                tf.convert_to_tensor(np.expand_dims(mel_spec_np, axis=-1), dtype=tf.float32),
                target_shape
            )

            model_input = tf.reshape(resized_spec, (1, target_shape[0], target_shape[1], 1))
            chunks.append(model_input)
            
        return chunks
    except Exception as e:
        raise Exception(f"Error in preprocessing: {str(e)}")

def model_prediction(chunks, loaded_model):
    """Make predictions on preprocessed chunks"""
    try:
        all_predictions = []
        for chunk in chunks:
            y_pred = loaded_model.predict(chunk, verbose=0)
            predicted_class = np.argmax(y_pred, axis=1)[0]
            all_predictions.append(predicted_class)
            
        prediction_counts = Counter(all_predictions)
        if not all_predictions:
            return {}, 0
            
        most_common_class = prediction_counts.most_common(1)[0][0]
        
        return prediction_counts, most_common_class
    except Exception as e:
        raise Exception(f"Error in prediction: {str(e)}")

# --- FastAPI Endpoints ---

@app.get("/")
async def root():
    return {"message": "Music Genre Classifier API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

class ClassificationResponse(BaseModel):
    predicted_genre: str
    distribution: Dict[str, float]

@app.post("/api/classify", response_model=ClassificationResponse)
async def classify_audio(file: UploadFile = File(...)):
    """API endpoint for music genre classification"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            filepath = tmp_file.name
            
        try:
            chunks = load_and_preprocess_file(filepath)
            
            # Lazy load model
            loaded_model = get_model()
            
            prediction_counts, most_common_class = model_prediction(chunks, loaded_model)
            
            classes = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 
                       'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
            
            predicted_genre = classes[most_common_class] if classes else "Unknown"
            
            # Calculate percentages for API
            total = sum(prediction_counts.values())
            distribution = {classes[idx]: count / total for idx, count in prediction_counts.items()}
            
            return {
                "predicted_genre": predicted_genre,
                "distribution": distribution
            }
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
