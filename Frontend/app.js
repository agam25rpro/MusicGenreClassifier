/**
 * Premium Music Genre Classifier Logic
 */

const API_URL = 'https://agamrampal-music.hf.space/api/classify';

// DOM Nodes
const dropzoneInner = document.querySelector('.dropzone-inner');
const fileInput = document.getElementById('file-input');
const filePreview = document.getElementById('file-preview');
const fileNameDisplay = document.getElementById('file-name');
const removeBtn = document.getElementById('remove-file');
const analyzeBtn = document.getElementById('analyze-btn');
const analyzerText = document.querySelector('.analyzer-text');
const errorMessage = document.getElementById('error-message');

const uploadSection = document.getElementById('upload-section');
const resultsSection = document.getElementById('results-section');
const winningGenreText = document.getElementById('winning-genre');
const resetBtn = document.getElementById('reset-btn');

let selectedFile = null;
let confidenceChart = null;

// ======================================
// INTERSECTION OBSERVERS (Micro-Interactions)
// ======================================

document.addEventListener("DOMContentLoaded", () => {
    // Scroll reveal observer
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                
                // Trigger children if it's the timeline
                const stepCards = entry.target.querySelectorAll('.step-card');
                if (stepCards.length > 0) {
                    stepCards.forEach(card => card.classList.add('visible'));
                }
            }
        });
    }, {
        threshold: 0.15,
        rootMargin: "0px 0px -50px 0px"
    });

    document.querySelectorAll('.scroll-reveal').forEach(el => observer.observe(el));
});

// ======================================
// FILE UPLOAD LOGIC
// ======================================

// The input automatically handles clicking. We just listen.
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileSelection(e.target.files[0]);
});

// Drag & Drop specific CSS logic
dropzoneInner.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzoneInner.classList.add('dragover');
});

dropzoneInner.addEventListener('dragleave', () => {
    dropzoneInner.classList.remove('dragover');
});

dropzoneInner.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzoneInner.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleFileSelection(e.dataTransfer.files[0]);
});

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation(); // Prevent re-triggering dropzone click
    clearFileSelection();
});

resetBtn.addEventListener('click', resetApp);
analyzeBtn.addEventListener('click', analyzeAudio);

// ======================================
// FUNCTIONS
// ======================================

function handleFileSelection(file) {
    const validTypes = ['audio/mpeg', 'audio/wav', 'audio/mp3', 'audio/x-wav'];
    const validExts = ['.mp3', '.wav'];
    
    if (!validTypes.includes(file.type) && !validExts.some(e => file.name.toLowerCase().endsWith(e))) {
        showError('Please upload a valid .mp3 or .wav audio file.');
        return;
    }

    selectedFile = file;
    fileNameDisplay.textContent = file.name;
    
    // UI State Swap
    filePreview.classList.remove('hidden');
    dropzoneInner.querySelector('#dropzone-content').classList.add('opacity-0');
    
    // Activate Button into Premium State
    analyzeBtn.disabled = false;
    analyzerText.textContent = 'ANALYZE AUDIO PATTERNS';
    analyzerText.classList.replace('text-slate-400', 'text-white');
    // Button naturally picks up hover gradients in CSS because it's no longer disabled
    hideError();
}

function clearFileSelection() {
    selectedFile = null;
    fileInput.value = '';
    
    filePreview.classList.add('hidden');
    dropzoneInner.querySelector('#dropzone-content').classList.remove('opacity-0');
    
    // Reset Button
    analyzeBtn.disabled = true;
    analyzerText.textContent = 'Select a Track';
    analyzerText.classList.replace('text-white', 'text-slate-400');
    hideError();
}

function resetApp() {
    clearFileSelection();
    
    // Fade out Results, Fade in Upload
    resultsSection.classList.add('hidden');
    uploadSection.classList.remove('opacity-0', 'pointer-events-none', 'hidden');
    
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

// ======================================
// API & CHARTING
// ======================================

async function analyzeAudio() {
    if (!selectedFile) return;

    // Premium Loading State
    analyzeBtn.disabled = true;
    analyzerText.innerHTML = `<span class="inline-flex items-center gap-2"><svg class="animate-spin -ml-1 mr-3 w-5 h-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg> Compiling Heatmaps...</span>`;
    
    // We force hover gradient background color even when disabled during API call for aesthetics
    const hoverBg = analyzeBtn.querySelector('.hover-gradient-bg');
    hoverBg.classList.add('opacity-100');

    hideError();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData,
            headers: { 'Accept': 'application/json' }
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();
        
        displayResults(data);

    } catch (error) {
        console.error("Classification Error:", error);
        showError(`Analysis failed: ${error.message}. Is your backend active?`);
        analyzeBtn.disabled = false;
        analyzerText.textContent = 'RE-ANALYZE GENRE';
        hoverBg.classList.remove('opacity-100');
    }
}

function displayResults(data) {
    // Smooth transition between sections
    uploadSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');
    resultsSection.classList.add('fade-in-up'); // Re-trigger animation

    // Reset button states for future runs
    const hoverBg = analyzeBtn.querySelector('.hover-gradient-bg');
    hoverBg.classList.remove('opacity-100');

    // Display robust genre string
    let genre = data.predicted_genre.replace('🎵 ', '');
    winningGenreText.textContent = genre;

    // Destructure JSON payload
    const labels = Object.keys(data.distribution);
    const rawValues = Object.values(data.distribution);
    const percentages = rawValues.map(val => (val * 100).toFixed(1));

    // NEW PREMIUN COLOR PALETTE (Spotify/Linear themed deep violet/cyan structure)
    const premiumColors = [
        '#06B6D4', // Cyan 500
        '#8B5CF6', // Violet 500
        '#3B82F6', // Blue 500
        '#D946EF', // Fuchsia 500
        '#0284C7', // Sky 600
        '#7C3AED', // Violet 600
        '#1E293B', // Slate 800
        '#334155', // Slate 700
        '#475569', // Slate 600
        '#64748B', // Slate 500
    ];

    renderChart(labels, percentages, premiumColors);
}

function renderChart(labels, data, premiumColors) {
    const ctx = document.getElementById('confidence-chart').getContext('2d');
    
    if (confidenceChart) confidenceChart.destroy();

    // Sync Chart.js fonts perfectly with our Tailwind fonts
    Chart.defaults.color = '#94A3B8'; // Slate 400
    Chart.defaults.font.family = "'Inter', sans-serif";

    confidenceChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: premiumColors,
                borderColor: '#020617', // Match Slate 950 deep bg perfectly
                borderWidth: 4,
                hoverOffset: 6,
                hoverBorderColor: '#00FFFF', // Cyan glow offset on hover
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '75%', // Extremely thin sharp doughnut 
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#E2E8F0', // Slate 200
                        padding: 20,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        font: { size: 13, weight: '500' }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)', // Slate 900 Glassmorphic
                    titleColor: '#06B6D4', // Cyan Title
                    bodyColor: '#F8FAFC',
                    bodyFont: { size: 14, family: "'Space Grotesk', sans-serif" },
                    borderColor: 'rgba(30, 41, 59, 1)',
                    borderWidth: 1,
                    padding: 16,
                    cornerRadius: 12, // Rounded Tooltips
                    callbacks: {
                        label: function(context) {
                            return ` Confidence: ${context.raw}%`;
                        }
                    }
                }
            }
        }
    });
}
