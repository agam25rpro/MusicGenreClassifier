/**
 * Music Genre Classifier Frontend Logic
 * Connects to Hugging Face Spaces FastAPI Server
 */

// --- Configuration ---
const API_URL = 'https://agamrampal-music.hf.space/api/classify';

// --- DOM Elements ---
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const filePreview = document.getElementById('file-preview');
const fileNameDisplay = document.getElementById('file-name');
const removeBtn = document.getElementById('remove-file');
const analyzeBtn = document.getElementById('analyze-btn');
const errorMessage = document.getElementById('error-message');

const uploadSection = document.getElementById('upload-section');
const resultsSection = document.getElementById('results-section');
const winningGenreText = document.getElementById('winning-genre');
const resetBtn = document.getElementById('reset-btn');

let selectedFile = null;
let confidenceChart = null;

// --- Event Listeners (Drag & Drop + File Selection) ---

dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    
    if (e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
    }
});

removeBtn.addEventListener('click', clearFileSelection);
resetBtn.addEventListener('click', resetApp);
analyzeBtn.addEventListener('click', analyzeAudio);

// --- Functions ---

function handleFileSelection(file) {
    // Validate file type
    const validTypes = ['audio/mpeg', 'audio/wav', 'audio/mp3', 'audio/x-wav'];
    const validExtensions = ['.mp3', '.wav'];
    
    const isValidType = validTypes.includes(file.type);
    const isValidExt = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    
    if (!isValidType && !isValidExt) {
        showError('Please upload a valid .mp3 or .wav audio file.');
        return;
    }

    selectedFile = file;
    fileNameDisplay.textContent = file.name;
    
    dropzone.classList.add('hidden');
    filePreview.classList.remove('hidden');
    analyzeBtn.disabled = false;
    hideError();
}

function clearFileSelection() {
    selectedFile = null;
    fileInput.value = '';
    
    dropzone.classList.remove('hidden');
    filePreview.classList.add('hidden');
    analyzeBtn.disabled = true;
    hideError();
}

function resetApp() {
    clearFileSelection();
    resultsSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    
    // Destroy chart if exists
    if (confidenceChart) {
        confidenceChart.destroy();
        confidenceChart = null;
    }
}

function showError(msg) {
    errorMessage.textContent = msg;
    errorMessage.classList.remove('hidden');
}

function hideError() {
    errorMessage.classList.add('hidden');
}

// --- API Communication and Chart Rendering ---

async function analyzeAudio() {
    if (!selectedFile) return;

    // Set Loading State
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'ANALYZING SOUND WAVES...';
    hideError();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData,
            // Huggingface spaces usually handle CORS well natively, but keep fetch straightforward
            headers: {
                'Accept': 'application/json'
            }
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();
        
        // Success - reveal results
        displayResults(data);

    } catch (error) {
        console.error("Classification Error:", error);
        showError(`Analysis failed: ${error.message}. Is the Hugging Face Space running?`);
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'ANALYZE GENRE';
    }
}

function displayResults(data) {
    // Hide upload, show results
    uploadSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');
    analyzeBtn.textContent = 'ANALYZE GENRE'; // Reset for next time

    // Setup Text
    // Remove the musical note emoji if Python sent it, keep it sharp via UI
    let genre = data.predicted_genre.replace('🎵 ', '');
    winningGenreText.textContent = genre;

    // Structure chart data
    const labels = Object.keys(data.distribution);
    const rawValues = Object.values(data.distribution);
    // Convert decimals to percentages (0.45 -> 45)
    const percentages = rawValues.map(val => (val * 100).toFixed(1));

    // Colors matching strictly to our two-color minimalist theme + monochrome tiers
    // Using opacity variations of our Cyan accent and grayscale to stay on-brand
    const colors = [
        '#00FFFF', // Primary Cyan
        '#00CCCC',
        '#009999',
        '#006666',
        '#FFFFFF', // White
        '#CCCCCC',
        '#999999',
        '#666666',
        '#333333',
        '#1A1A1A'
    ];

    renderChart(labels, percentages, colors);
}

function renderChart(labels, data, colors) {
    const ctx = document.getElementById('confidence-chart').getContext('2d');
    
    if (confidenceChart) {
        confidenceChart.destroy();
    }

    Chart.defaults.color = '#999999';
    Chart.defaults.font.family = "'Space Grotesk', sans-serif";

    confidenceChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderColor: '#0A0A0A', // Deep black border to match background
                borderWidth: 2,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%', // Sleek thin doughnut
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#FFFFFF',
                        padding: 15,
                        usePointStyle: true,
                        pointStyle: 'rect'
                    }
                },
                tooltip: {
                    backgroundColor: '#141414',
                    titleColor: '#00FFFF',
                    bodyColor: '#FFFFFF',
                    borderColor: '#333333',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            return ` ${context.label}: ${context.raw}%`;
                        }
                    }
                }
            }
        }
    });
}
