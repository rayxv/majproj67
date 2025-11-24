// Barcode Scanner JavaScript

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startButton = document.getElementById('startButton');
const captureButton = document.getElementById('captureButton');
const cameraDropdownButton = document.getElementById('cameraDropdownButton');
const cameraDropdown = document.getElementById('cameraDropdown');
const cameraList = document.getElementById('cameraList');
const currentCameraElement = document.getElementById('currentCamera');
const cameraInfo = document.getElementById('cameraInfo');
const statusElement = document.getElementById('status');
const productCard = document.getElementById('productCard');

// Product info elements
const productName = document.getElementById('productName');
const productBrand = document.getElementById('productBrand');
const productSugar = document.getElementById('productSugar');
const productCalories = document.getElementById('productCalories');
const productFat = document.getElementById('productFat');
const productCarbs = document.getElementById('productCarbs');
const productProtein = document.getElementById('productProtein');
const productIngredients = document.getElementById('productIngredients');
const barcodeType = document.getElementById('barcodeType');
const nutriScore = document.getElementById('nutriScore');

// Analysis elements
const analysisSection = document.getElementById('analysisSection');
const healthAnalysis = document.getElementById('healthAnalysis');
const healthWarnings = document.getElementById('healthWarnings');
const allergens = document.getElementById('allergens');
const analysisSource = document.getElementById('analysisSource');

let stream = null;
let devices = [];
let currentDeviceId = null;

// Camera dropdown toggle
cameraDropdownButton.addEventListener('click', (e) => {
    e.stopPropagation();
    cameraDropdown.classList.toggle('hidden');
});

document.addEventListener('click', () => {
    cameraDropdown.classList.add('hidden');
});

cameraDropdown.addEventListener('click', (e) => {
    e.stopPropagation();
});

// Populate camera list
async function populateCameraList() {
    try {
        devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        cameraList.innerHTML = '';

        if (videoDevices.length === 0) {
            cameraList.innerHTML = '<div class="px-4 py-2 text-sm text-gray-700">No cameras found</div>';
            return;
        }

        videoDevices.forEach((device, index) => {
            const label = device.label || `Camera ${index + 1}`;
            const btn = document.createElement('button');
            btn.className = 'block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100';
            btn.textContent = label;
            btn.dataset.deviceId = device.deviceId;

            btn.addEventListener('click', () => {
                currentDeviceId = device.deviceId;
                currentCameraElement.textContent = label;
                cameraDropdown.classList.add('hidden');
                startCamera();
            });

            cameraList.appendChild(btn);
            if (index === 0 && !currentDeviceId) {
                currentDeviceId = device.deviceId;
                currentCameraElement.textContent = label;
            }
        });

        cameraInfo.textContent = `${videoDevices.length} camera(s) found`;
    } catch (err) {
        cameraInfo.textContent = 'Could not access camera devices';
        cameraInfo.className = 'text-sm text-red-500';
    }
}

// Start camera
async function startCamera() {
    try {
        if (stream) stream.getTracks().forEach(track => track.stop());
        await populateCameraList();

        const constraints = currentDeviceId
            ? { video: { deviceId: { exact: currentDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false }
            : { video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false };

        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;

        startButton.textContent = 'Restart Camera';
        captureButton.disabled = false;
        captureButton.classList.remove('opacity-50', 'cursor-not-allowed');
        statusElement.textContent = 'Camera started. Point at a barcode and click Scan Barcode';
        statusElement.className = 'text-green-600 mb-4';
    } catch (err) {
        statusElement.textContent = 'Error accessing camera. Please grant camera permission.';
        statusElement.className = 'text-red-600 mb-4';
    }
}

// Update Nutri-Score with source indicator
function updateNutriScoreWithSource(score, source, details) {
    if (!score || score === '') {
        nutriScore.style.display = 'none';
        return;
    }
    
    nutriScore.textContent = score.toUpperCase();
    nutriScore.className = `nutri-score nutri-score-${score.toLowerCase()}`;
    nutriScore.style.display = 'inline-flex';
    
    if (source === 'gemini' || source === 'calculated') {
        nutriScore.title = `Calculated Nutri-Score: ${details || 'Based on nutritional data'}`;
        const indicator = document.createElement('span');
        indicator.innerHTML = '*';
        indicator.style.fontSize = '0.7em';
        indicator.style.verticalAlign = 'super';
        indicator.style.color = '#6b7280';
        nutriScore.appendChild(indicator);
    } else {
        nutriScore.title = 'Official Nutri-Score';
    }
}

// Event listeners
startButton.addEventListener('click', startCamera);

captureButton.addEventListener('click', async () => {
    if (!stream) {
        statusElement.textContent = 'Please start the camera first';
        statusElement.className = 'text-red-600';
        return;
    }

    productCard.style.display = 'none';
    statusElement.textContent = 'Scanning barcode...';
    statusElement.className = 'text-blue-600';

    try {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob, 'barcode.jpg');

            try {
                const response = await fetch('/scan', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                console.log('API Response:', result);

                if (result.status === 'success') {
                    statusElement.textContent = `Found barcode: ${result.barcode}`;
                    statusElement.className = 'text-green-600';

                    const product = result.product;
                    
                    // Basic product info
                    productName.textContent = product.product_name || 'Product name not available';
                    productBrand.textContent = product.brands || 'Brand not specified';
                    productSugar.textContent = product.sugar || 'Sugar not specified';
                    productCalories.textContent = product.nutriments?.['energy-kcal_100g'] ?? '-';
                    productFat.textContent = product.nutriments?.fat_100g ?? '-';
                    productCarbs.textContent = product.nutriments?.carbohydrates_100g ?? '-';
                    productProtein.textContent = product.nutriments?.proteins_100g ?? '-';
                    productIngredients.textContent = product.ingredients_text || product.ingredients || 'Ingredients not available';
                    
                    barcodeType.textContent = result.barcode_type || 'Barcode';
                    updateNutriScoreWithSource(product.nutriscore_grade, product.nutriscore_source, product.nutriscore_details);

                    // Analysis section
                    const analysis = product.ingredient_analysis || {};
                    analysisSource.textContent = analysis.source ? `via ${analysis.source}` : '';
                    
                    healthAnalysis.innerHTML = analysis.analysis || 'No analysis available';
                    
                    // Health warnings
                    healthWarnings.innerHTML = '';
                    if (analysis.health_warnings && analysis.health_warnings.length) {
                        analysis.health_warnings.forEach(warning => {
                            const badge = document.createElement('span');
                            badge.className = 'warning-badge';
                            badge.innerHTML = `
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                                    <path fill-rule="evenodd" d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z" clip-rule="evenodd" />
                                </svg>
                                ${warning}
                            `;
                            healthWarnings.appendChild(badge);
                        });
                    } else {
                        healthWarnings.innerHTML = '<span class="text-sm text-gray-500">No significant warnings detected</span>';
                    }
                    
                    // Allergens
                    allergens.innerHTML = '';
                    if (analysis.allergens && analysis.allergens.length) {
                        analysis.allergens.forEach(allergen => {
                            const chip = document.createElement('span');
                            chip.className = 'allergen-chip';
                            chip.textContent = allergen;
                            allergens.appendChild(chip);
                        });
                    } else {
                        allergens.innerHTML = '<span class="text-sm text-gray-500">No common allergens detected</span>';
                    }
                    
                    productCard.style.display = 'block';
                    analysisSection.style.display = 'block';
                    
                } else if (result.status === 'not_found') {
                    statusElement.textContent = `Barcode ${result.barcode} found, but product not in database.`;
                    statusElement.className = 'text-yellow-600';
                } else {
                    throw new Error(result.message || 'Unknown error');
                }
            } catch (err) {
                console.error('Error:', err);
                statusElement.textContent = `Error: ${err.message}`;
                statusElement.className = 'text-red-600';
            }
        }, 'image/jpeg', 0.8);
    } catch (err) {
        console.error('Error:', err);
        statusElement.textContent = `Error: ${err.message}`;
        statusElement.className = 'text-red-600';
    }
});

// Auto-start camera if in secure context
if (window.isSecureContext) {
    startButton.click();
}