# Music Genre Classifier

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Vercel](https://img.shields.io/badge/Vercel-%23000000.svg?style=for-the-badge&logo=vercel&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue?style=for-the-badge)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge)

A complete, AI-powered music genre classification system that classifies audio files into one of ten genres using a deep learning model trained on Mel spectrogram representations. Upload any MP3 file and receive an accurate genre prediction along with interactive visualizations showing the confidence distribution across all supported genres.

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Architecture](#architecture)
  - [Frontend](#frontend)
  - [Backend and Model Serving](#backend-and-model-serving)
- [Model Architecture](#model-architecture)
- [Data Handling and Audio Preprocessing](#data-handling-and-audio-preprocessing)
  - [Audio Chunking](#audio-chunking)
  - [Mel Spectrogram Generation](#mel-spectrogram-generation)
  - [Majority Voting for Final Prediction](#majority-voting-for-final-prediction)
- [Supported Genres](#supported-genres)
- [Features](#features)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Music Genre Classifier** is a full-stack machine learning web application that allows users to upload an MP3 audio file and receive real-time genre classification results. The system uses a TensorFlow-based Convolutional Neural Network (CNN) trained on Mel spectrogram images derived from audio data. It segments audio into overlapping chunks, generates a Mel spectrogram for each chunk, classifies each one independently, and uses a majority voting strategy to determine the final genre prediction. The results are displayed as an interactive Plotly donut chart alongside the predicted genre label.

The application features a decoupled architecture: a sleek, minimal Vanilla JS frontend designed for Vercel, and a high-performance FastAPI backend that serves the predictive model. The backend is also deployed on Hugging Face Spaces for public API access.

---

## Demo

Try out the live application here: [Music Genre Classifier Live Demo](https://huggingface.co/spaces/Agamrampal/Music)

Below are screenshots of the deployed application interface:

![Upload Interface](assets/upload_interface.png)

![Visualization Example](assets/visualization_example.png)

---

## Architecture

The project follows a decoupled architecture cleanly separating the UI in the `Frontend` directory from the model serving logic in the `Backend` directory.

### Frontend

- **Framework**: Vanilla HTML, CSS, and JavaScript. No heavy frontend frameworks required.
- **Styling**: Engineered with a strict two-color minimalist aesthetic (Deep Black and Electric Cyan) using custom CSS. Fully responsive layout.
- **Visualization**: Chart.js is used to render an interactive doughnut chart showing the genre probability distribution dynamically upon receiving the API JSON payload.

### Backend

- **API Framework**: FastAPI, chosen for its asynchronous efficiency and highly reliable ASGI servers (Uvicorn).
- **Model Loading**: The TensorFlow model (`Trained_model.h5`) is safely downloaded from Google Drive on first run and is lazily-loaded with thread-locking to prevent initialization crashes.
- **Inference**: Librosa is used for audio processing and converting into Mel spectrogram representations natively without relying on heavy Torch dependencies. TensorFlow processes these tensors natively.
- **Deployment**: The backend API (`/api/classify`) easily deploys to Hugging Face Spaces via a Docker configuration.

---

## Model Architecture

The core of the classifier is a TensorFlow/Keras Convolutional Neural Network (CNN). Rather than working directly on raw audio waveforms, the model operates on 2D Mel spectrogram images, treating genre classification as an image classification task. Key characteristics of this approach include:

1. **Input Representation**: Each audio chunk is converted into a single-channel Mel spectrogram image, resized to a fixed spatial resolution of 210 x 210 pixels. This ensures a uniform input shape regardless of the original audio's sample rate or duration.
2. **Convolutional Layers**: The CNN applies a series of convolutional and pooling layers to extract hierarchical spatial features from the spectrogram, capturing patterns in frequency and time that are characteristic of different genres.
3. **Classification Output**: The final layer produces a probability distribution over 10 genre classes using a softmax activation. The class with the highest probability for each chunk becomes that chunk's prediction.
4. **Model Storage**: The trained model weights are stored as an `.h5` file on Google Drive and are automatically fetched and cached at application startup.

---

## Data Handling and Audio Preprocessing

The data handling pipeline is a critical component that bridges the gap between a raw MP3 upload and the tensor-formatted input the CNN expects. The pipeline is implemented in the `load_and_preprocess_file` function.

### Audio Chunking

Rather than feeding the entire audio track as a single input, the system segments the audio into smaller overlapping chunks to capture genre-relevant patterns across different parts of the track:

- **Chunk Duration**: 4 seconds.
- **Overlap Duration**: 2 seconds (50% overlap between consecutive chunks).
- The number of chunks is calculated dynamically based on the total length of the audio file.
- If the final chunk is shorter than 4 seconds, it is zero-padded to maintain a consistent length.

### Mel Spectrogram Generation

For each audio chunk, the following steps are performed:

1. **Audio Loading**: The raw audio file is loaded using `librosa.load()`, which decodes the MP3 and returns a NumPy array of audio samples along with the sample rate.
2. **Tensor Conversion**: The chunk is converted into a PyTorch tensor.
3. **Mel Spectrogram Transform**: The `torchaudio.transforms.MelSpectrogram()` transform is applied to convert the time-domain audio signal into a frequency-domain Mel spectrogram representation.
4. **Resizing**: The resulting spectrogram is resized to a fixed 210 x 210 pixel resolution using `tensorflow.image.resize` to match the model's expected input dimensions.
5. **Reshaping**: The spectrogram is reshaped into the format `(1, 210, 210, 1)` (batch size, height, width, channels) required by the TensorFlow model.

### Majority Voting for Final Prediction

Each chunk is classified independently by the CNN. The final genre prediction is determined by a **majority voting** strategy:

- The `Counter` class from Python's `collections` module tallies the predictions across all chunks.
- The genre that receives the most votes (the most common prediction) is selected as the overall genre for the track.
- The full vote distribution is visualized as a Plotly donut chart, providing transparency into how confident the model was across the entire track.

---

## Supported Genres

The model classifies audio into the following 10 genres:

- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

---

## Features

- **10-Class Genre Classification**: Covers a wide spectrum of popular music genres.
- **Overlapping Chunk Analysis**: Processes audio in 4-second overlapping windows for robust predictions.
- **Majority Voting**: Aggregates per-chunk predictions for a reliable final result.
- **Interactive Visualizations**: Plotly-based donut charts display the full genre probability distribution.
- **Real-Time Feedback**: Progress bar and animated loading indicators provide instant user feedback during processing.
- **Automated Model Download**: The trained model is fetched from Google Drive and cached locally on first launch.

---

## Installation and Setup

```bash
# Clone the repository
git clone https://github.com/agam25rpro/MusicGenreClassifier.git

# Navigate to the project directory
cd MusicGenreClassifier

# --- Backend Setup ---
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r Backend/requirements.txt

# Run the FastAPI server locally (will start on port 8000)
python Backend/app.py

# --- Frontend Setup ---
# Open the frontend in any web browser without needing a build step
# Simply open Frontend/index.html in your browser!
```

---

## Usage

1. Ensure your backend is running, or that the Hugging Face Space API endpoint is active.
2. Open `Frontend/index.html` in your web browser.
3. Click the upload button and select an MP3 file from your device.
4. Wait for the processing pipeline to complete (audio chunking, spectrogram generation, and model inference).
5. View the predicted genre and explore the interactive donut chart showing the classification distribution across all chunks.

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Add some feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Submit a Pull Request on GitHub.

---

## License

This project is licensed under the Apache License. See the [LICENSE](LICENSE.txt) file for details.
