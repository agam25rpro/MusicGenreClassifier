import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import torch
import torchaudio
from tensorflow.image import resize
import gdown
import os
import tempfile
from collections import Counter
import plotly.graph_objects as go
import streamlit.components.v1 as components

# Set page config
st.set_page_config(
    page_title="Music Genre Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with animations and styling
st.markdown("""
    <style>
    /* Main theme colors and styles */
    :root {
        --primary-color: #4A90E2;
        --secondary-color: #2C3E50;
        --background-color: #0E1117;
        --card-bg-color: rgba(25, 30, 45, 0.8);
    }

    .main {
        background-color: var(--background-color);
        color: #FFFFFF;
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: var(--primary-color);
    }

    /* Card styling with animations */
    .genre-card {
        background-color: var(--card-bg-color);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        border: 1px solid var(--secondary-color);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .genre-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }

    /* Glowing effect for cards */
    .genre-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            45deg,
            transparent,
            rgba(74, 144, 226, 0.1),
            transparent
        );
        transform: rotate(45deg);
        animation: shine 3s infinite;
    }

    @keyframes shine {
        0% { transform: translateX(-100%) rotate(45deg); }
        100% { transform: translateX(100%) rotate(45deg); }
    }

    /* Title styles */
    .genre-title {
        color: var(--primary-color);
        font-size: 28px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    /* Upload section styling */
    .upload-section {
        background-color: var(--card-bg-color);
        border-radius: 15px;
        padding: 40px;
        margin: 30px 0;
        border: 2px dashed var(--primary-color);
        text-align: center;
        transition: all 0.3s ease;
    }

    .upload-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        border-color: #6BA4E7;
    }

    /* Results container styling */
    .result-container {
        background-color: var(--card-bg-color);
        border-radius: 15px;
        padding: 30px;
        margin: 30px 0;
        border: 1px solid var(--secondary-color);
        animation: fadeIn 0.5s ease-in;
    }

    /* Genre item styling */
    .genre-item {
        background-color: rgba(74, 144, 226, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        transition: transform 0.2s ease;
    }

    .genre-item:hover {
        transform: scale(1.02);
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    /* Floating animation for icons */
    .floating {
        animation: floating 3s ease-in-out infinite;
    }

    @keyframes floating {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    /* Enhanced status text */
    .status-text {
        color: var(--primary-color);
        font-size: 18px;
        text-align: center;
        margin: 15px 0;
        font-weight: 500;
    }

    /* Loading animation */
    .loading-wave {
        width: 300px;
        height: 100px;
        display: flex;
        justify-content: center;
        align-items: flex-end;
        margin: 0 auto;
    }

    .loading-bar {
        width: 20px;
        height: 10px;
        margin: 0 5px;
        background-color: var(--primary-color);
        border-radius: 5px;
        animation: wave 1s infinite;
    }

    @keyframes wave {
        0% { height: 10px; }
        50% { height: 50px; }
        100% { height: 10px; }
    }
    </style>
""", unsafe_allow_html=True)

# Model-related functions (keeping your original functions)
MODEL_PATH = "Trained_model.h5"

@st.cache_resource
def download_model():
    """Download the model if not already present"""
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?export=download&id=1vc4b2RpeXmnZMn2SOF0snIjos9paVEVH"
        gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    """Load the TensorFlow model"""
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load model globally
model = load_model()

# Your original preprocessing and prediction functions remain the same


def load_and_preprocess_file(file_path, target_shape=(210, 210)):
    """
    Load and preprocess audio file for prediction
    """
    try:
        # Load audio file
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        
        # Define chunk parameters
        chunk_duration = 4  # seconds
        overlap_duration = 2  # seconds
        chunk_samples = int(chunk_duration * sample_rate)
        overlap_samples = int(overlap_duration * sample_rate)
        
        # Calculate number of chunks
        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
        
        chunks = []
        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = start + chunk_samples
            
            # Handle the last chunk
            if end > len(audio_data):
                # Pad with zeros if needed
                chunk = np.pad(audio_data[start:], (0, end - len(audio_data)))
            else:
                chunk = audio_data[start:end]
            
            # Convert to tensor and create mel spectrogram
            chunk_tensor = torch.tensor(chunk).float().unsqueeze(0)
            mel_spectrogram = torchaudio.transforms.MelSpectrogram()(chunk_tensor)
            
            # Convert to numpy and prepare for model
            mel_spec_np = mel_spectrogram.numpy()
            
            # Resize spectrogram to target shape
            resized_spec = resize(
                tf.convert_to_tensor(np.expand_dims(mel_spec_np, axis=-1), dtype=tf.float32),
                target_shape
            )
            
            # Reshape for model input
            model_input = tf.reshape(resized_spec, (1, target_shape[0], target_shape[1], 1))
            chunks.append(model_input)
            
        return chunks
    except Exception as e:
        raise Exception(f"Error in preprocessing: {str(e)}")

def model_prediction(chunks):
    """
    Make predictions on preprocessed chunks
    """
    try:
        all_predictions = []
        for chunk in chunks:
            # Make prediction
            y_pred = model.predict(chunk, verbose=0)
            predicted_class = np.argmax(y_pred, axis=1)[0]
            all_predictions.append(predicted_class)
        
        # Count predictions
        prediction_counts = Counter(all_predictions)
        
        # Get most common prediction
        most_common_class = prediction_counts.most_common(1)[0][0]
        
        return prediction_counts, most_common_class
    except Exception as e:
        raise Exception(f"Error in prediction: {str(e)}")

# Enhanced genre descriptions
genre_cards = """
    <div class="genre-card">
        <h2 class="genre-title">🎵 Discover Music Genres</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 25px;">
            <div class="genre-item">
                <h3 style="color: #4A90E2;">Pop 🎤</h3>
                <p>Modern pop music with catchy melodies and contemporary production, perfect for today's listeners.</p>
            </div>
            <div class="genre-item">
                <h3 style="color: #4A90E2;">Rock 🎸</h3>
                <p>From classic rock to modern alternatives, featuring powerful guitars and dynamic rhythms.</p>
            </div>
            <div class="genre-item">
                <h3 style="color: #4A90E2;">Jazz 🎷</h3>
                <p>Smooth improvisations and complex harmonies that define this timeless genre.</p>
            </div>
            <div class="genre-item">
                <h3 style="color: #4A90E2;">Classical 🎻</h3>
                <p>Timeless orchestral pieces that have influenced music for centuries.</p>
            </div>
            <div class="genre-item">
                <h3 style="color: #4A90E2;">Hip-Hop 🎤</h3>
                <p>Urban poetry combined with powerful beats and innovative production techniques.</p>
            </div>
            <div class="genre-item">
                <h3 style="color: #4A90E2;">Electronic 💫</h3>
                <p>Modern electronic music with synthesized sounds and dynamic beats.</p>
            </div>
        </div>
    </div>
"""

def create_genre_chart(genre_distribution):
    """Create an enhanced Plotly pie chart for genre distribution"""
    colors = ['#4A90E2', '#2ECC71', '#E74C3C', '#F1C40F', '#9B59B6', 
              '#3498DB', '#E67E22', '#1ABC9C', '#34495E', '#95A5A6']
    
    fig = go.Figure(data=[go.Pie(
        labels=list(genre_distribution.keys()),
        values=list(genre_distribution.values()),
        hole=.3,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textfont=dict(size=14, color='white'),
        hoverinfo='label+value',
        textposition='outside',
        rotation=90
    )])
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=30, b=30, l=30, r=30),
        height=500,
        font=dict(color='white'),
        annotations=[dict(
            text='Genre Distribution',
            x=0.5,
            y=0.5,
            font_size=16,
            showarrow=False
        )]
    )
    
    return fig

def main():
    """Main application function"""
    st.markdown("""
        <h1 style='text-align: center; color: #4A90E2; margin-bottom: 30px;'>
            <span class='floating'>🎵</span> 
            Music Genre Predictor
            <span class='floating'>🎸</span>
        </h1>
    """, unsafe_allow_html=True)
    
    # Display genre descriptions
    st.markdown(genre_cards, unsafe_allow_html=True)
    
    # Upload section with enhanced styling
    st.markdown("""
        <div class='upload-section'>
            <h3 style='color: #4A90E2; margin-bottom: 20px;'>Upload Your Music</h3>
            <p style='color: #FFFFFF; margin-bottom: 20px;'>Drop your MP3 file here or click to upload</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["mp3"])
    
    if uploaded_file:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Add loading animation
            status_text.markdown("""
                <div class='loading-wave'>
                    <div class='loading-bar'></div>
                    <div class='loading-bar' style='animation-delay: 0.1s'></div>
                    <div class='loading-bar' style='animation-delay: 0.2s'></div>
                    <div class='loading-bar' style='animation-delay: 0.3s'></div>
                    <div class='loading-bar' style='animation-delay: 0.4s'></div>
                </div>
            """, unsafe_allow_html=True)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                filepath = tmp_file.name
            
            status_text.markdown("<p class='status-text'>🎵 Processing your audio...</p>", unsafe_allow_html=True)
            progress_bar.progress(25)
            
            chunks = load_and_preprocess_file(filepath)
            progress_bar.progress(50)
            
            prediction_counts, most_common_class = model_prediction(chunks)
            progress_bar.progress(75)
            
            classes = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 
                      'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
            
            genre_distribution = {classes[class_idx]: count 
                                for class_idx, count in prediction_counts.items()}
            
            progress_bar.progress(100)
            status_text.markdown("<p class='status-text'>✨ Analysis complete!</p>", unsafe_allow_html=True)
            
            # Enhanced results display
            st.markdown("""
                <div class='result-container'>
                    <h2 style='text-align: center; color: #4A90E2; margin-bottom: 30px;'>
                        Analysis Results 🎼
                    </h2>
            """, unsafe_allow_html=True)
            
            # Display predicted genre with animation
            st.markdown(f"""
                <div style='text-align: center; animation: fadeIn 0.5s ease-in;'>
                    <h3 style='color: #4A90E2; font-size: 24px; margin-bottom: 20px;'>
                        Predicted Genre: {classes[most_common_class]} 
                        <span class='floating'>🎵</span>
                    </h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Show enhanced chart
            st.plotly_chart(create_genre_chart(genre_distribution), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Cleanup
            os.remove(filepath)
            progress_bar.empty()
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()