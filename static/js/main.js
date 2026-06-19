document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const selectedState = document.getElementById('selected-state');
    const selectedFileName = document.getElementById('selected-file-name');
    const removeBtn = document.getElementById('remove-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnSpinner = document.getElementById('btn-spinner');
    
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsContent = document.getElementById('results-content');
    const diagnosisAlert = document.getElementById('diagnosis-alert');
    const diagnosisLabel = document.getElementById('diagnosis-label');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceValue = document.getElementById('confidence-value');
    
    const imgOriginal = document.getElementById('img-original');
    const imgHeatmap = document.getElementById('img-heatmap');
    
    // Slider elements
    const slider = document.getElementById('comparison-slider');
    const overlay = document.getElementById('overlay-container');
    const handle = document.getElementById('slider-handle');

    let selectedFile = null;
    let isDraggingSlider = false;

    // --- Drag and Drop / Selection Event Listeners ---
    
    // Click dropzone to trigger input
    dropzone.addEventListener('click', (e) => {
        // Prevent click if we click the remove button
        if (e.target === removeBtn) return;
        if (selectedFile) return; // Ignore if file already selected
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileSelection(fileInput.files[0]);
        }
    });

    // Drag-over highlights
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        if (selectedFile) return;
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (selectedFile) return;
        
        if (e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    // Remove file selection
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetFileSelection();
    });

    function handleFileSelection(file) {
        // Simple file validation
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
        if (!validTypes.includes(file.type)) {
            alert('Invalid file format. Please upload a chest X-ray image in JPEG or PNG format.');
            return;
        }

        selectedFile = file;
        selectedFileName.textContent = file.name;
        
        // UI updates
        dropzone.querySelector('.dropzone-prompt').style.display = 'none';
        selectedState.style.display = 'flex';
        analyzeBtn.removeAttribute('disabled');
    }

    function resetFileSelection() {
        selectedFile = null;
        fileInput.value = '';
        
        // UI updates
        selectedState.style.display = 'none';
        dropzone.querySelector('.dropzone-prompt').style.display = 'block';
        analyzeBtn.setAttribute('disabled', 'true');
    }

    // --- Prediction API Request ---
    
    analyzeBtn.addEventListener('click', () => {
        if (!selectedFile) return;

        // Set Loading State
        analyzeBtn.setAttribute('disabled', 'true');
        removeBtn.style.display = 'none';
        btnSpinner.style.display = 'inline-block';
        
        // Show temporary diagnostic placeholder
        resultsPlaceholder.querySelector('h3').textContent = 'Analyzing Radiograph...';
        resultsPlaceholder.querySelector('p').textContent = 'Loading neural network models, performing CLAHE preprocessing, and running localization layers...';
        resultsPlaceholder.style.display = 'flex';
        resultsContent.style.display = 'none';

        const formData = new FormData();
        formData.append('image', selectedFile);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Server error'); });
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                renderResults(data);
            } else {
                throw new Error(data.error || 'Diagnostic engine failed.');
            }
        })
        .catch(err => {
            console.error(err);
            alert(`Analysis Error: ${err.message}`);
            // Reset placeholders
            resultsPlaceholder.querySelector('h3').textContent = 'Analysis Failed';
            resultsPlaceholder.querySelector('p').textContent = err.message || 'An error occurred during deep learning inference.';
        })
        .finally(() => {
            // Remove Loading State
            analyzeBtn.removeAttribute('disabled');
            removeBtn.style.display = 'inline-block';
            btnSpinner.style.display = 'none';
        });
    });

    function renderResults(data) {
        // Hide placeholder, show content
        resultsPlaceholder.style.display = 'none';
        resultsContent.style.display = 'block';

        // Set diagnosis alert theme classes
        diagnosisAlert.className = 'alert-box'; // reset
        if (data.label === 'PNEUMONIA') {
            diagnosisAlert.classList.add('pneumonia');
            diagnosisLabel.textContent = 'PNEUMONIA DETECTED';
        } else {
            diagnosisAlert.classList.add('normal');
            diagnosisLabel.textContent = 'NORMAL / NO SIGN OF PNEUMONIA';
        }

        // Set metrics values (convert probability to percentage)
        // Note: For binary model output (0 to 1), confidence represents pneumonia probability.
        // Let's print the percentage
        const probPercent = (data.confidence * 100).toFixed(2);
        confidenceValue.textContent = `${probPercent}%`;
        confidenceBar.style.width = `${probPercent}%`;

        // Update comparison images
        imgOriginal.src = data.original_image;
        imgHeatmap.src = data.heatmap_image;

        // Reset the slider to initial 50% split
        updateSliderPosition(50);
    }

    // --- Interactive Comparison Slider Logic ---

    // Trigger sliding action
    const startSliderDrag = (e) => {
        isDraggingSlider = true;
        e.preventDefault();
    };

    const stopSliderDrag = () => {
        isDraggingSlider = false;
    };

    const moveSlider = (clientX) => {
        if (!isDraggingSlider) return;

        const rect = slider.getBoundingClientRect();
        let posX = clientX - rect.left;

        // Clamp positions to boundary
        if (posX < 0) posX = 0;
        if (posX > rect.width) posX = rect.width;

        const percentage = (posX / rect.width) * 100;
        updateSliderPosition(percentage);
    };

    function updateSliderPosition(percentage) {
        handle.style.left = `${percentage}%`;
        overlay.style.width = `${percentage}%`;
    }

    // Mouse Listeners
    handle.addEventListener('mousedown', startSliderDrag);
    window.addEventListener('mouseup', stopSliderDrag);
    window.addEventListener('mousemove', (e) => {
        if (isDraggingSlider) moveSlider(e.clientX);
    });

    // Touch Listeners (Mobile Friendly)
    handle.addEventListener('touchstart', startSliderDrag);
    window.addEventListener('touchend', stopSliderDrag);
    window.addEventListener('touchmove', (e) => {
        if (isDraggingSlider && e.touches.length > 0) {
            moveSlider(e.touches[0].clientX);
        }
    });

    // Allow clicking/tapping directly on the slider container to move the slider position
    slider.addEventListener('click', (e) => {
        // Prevent click events originating from sliding actions or direct handles from running twice
        if (e.target.closest('#slider-handle')) return;
        
        isDraggingSlider = true;
        moveSlider(e.clientX);
        isDraggingSlider = false;
    });
});
