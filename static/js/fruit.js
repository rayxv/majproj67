// Fruit Recognition JavaScript

const detectionStatusEl = document.getElementById('detection-status');
const infoEl = document.getElementById('nutritional-info');
const scanButton = document.getElementById('scan-button');
const cameraStatusEl = document.getElementById('camera-status');
const videoFeedEl = document.getElementById('video-feed');

// Initialize Camera Status Display
function initializeCameraStatus() {
    // ACTIVE_INDEX is defined in the HTML template via Flask
    if (typeof ACTIVE_INDEX !== 'undefined' && ACTIVE_INDEX >= 0) {
        cameraStatusEl.classList.remove('text-red-600');
        cameraStatusEl.classList.add('text-green-600');
        cameraStatusEl.textContent = `Live: Device ${ACTIVE_INDEX} (Active Camera)`;
    } else {
        cameraStatusEl.textContent = 'Failed to initialize any camera device.';
    }
}

// Markdown Table to HTML Converter
function markdownTableToHtml(markdown) {
    const lines = markdown.trim().split('\n');
    if (lines.length < 3) return markdown;
    if (!lines[2].includes('---')) return markdown;

    const header = lines[0].split('|').map(h => h.trim()).filter(Boolean);
    const dataRows = lines.slice(3).map(line => 
        line.split('|').map(d => d.trim()).filter(Boolean)
    );

    let html = '<div class="overflow-x-auto"><table class="min-w-full divide-y divide-gray-200 rounded-lg overflow-hidden">';
    
    html += '<thead class="bg-green-100/70">';
    html += '<tr>' + header.map(h => `<th scope="col" class="px-3 py-2 text-left text-xs font-bold text-green-700 uppercase tracking-wider">${h}</th>`).join('') + '</tr>';
    html += '</thead>';

    html += '<tbody class="bg-white divide-y divide-gray-200">';
    dataRows.forEach((row, index) => {
        html += `<tr class="${index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}">`;
        html += row.map(d => `<td class="px-3 py-2 whitespace-nowrap text-sm text-gray-800">${d}</td>`).join('');
        html += '</tr>';
    });
    html += '</tbody></table></div>';
    
    return html;
}

// Perform Scan
async function performScan() {
    scanButton.disabled = true;
    scanButton.textContent = 'SCANNING... (Capturing Frame)';
    detectionStatusEl.textContent = 'Capturing frame and running CNN classification...';
    
    // Show loading spinner
    infoEl.innerHTML = `
        <div class="flex items-center justify-center h-full">
            <div class="spinner"></div>
            <span class="ml-3 text-gray-600">Running local ML model...</span>
        </div>
    `;
    
    try {
        const response = await fetch('/scan_and_get_info');
        const result = await response.json();
        
        const fruitName = result.detected_fruit;
        const nutritionalInfo = result.info;

        if (fruitName !== "Unknown" && fruitName !== "Error") {
            detectionStatusEl.textContent = `✓ ${fruitName.toUpperCase()} recognized!`;
            detectionStatusEl.classList.add('text-green-700', 'font-bold');
        } else {
            detectionStatusEl.textContent = `⚠ Classification failed or Unknown item detected.`;
            detectionStatusEl.classList.add('text-yellow-700');
        }
        
        if (nutritionalInfo.startsWith('Error:')) {
            infoEl.innerHTML = `<div class="error-message">${nutritionalInfo}</div>`;
        } else {
            // Parse the markdown table
            const firstPipeIndex = nutritionalInfo.indexOf('|');
            let introduction = '';
            let markdownTable = nutritionalInfo;

            if (firstPipeIndex !== -1) {
                introduction = nutritionalInfo.substring(0, firstPipeIndex).trim();
                markdownTable = nutritionalInfo.substring(firstPipeIndex).trim();
            }
            
            const htmlTable = markdownTableToHtml(markdownTable);
            infoEl.innerHTML = `
                <p class="mb-3 font-medium text-gray-700">${introduction}</p>
                ${htmlTable}
            `;
            infoEl.classList.add('fade-in');
        }

    } catch (error) {
        console.error("Error during scan:", error);
        detectionStatusEl.textContent = " Fatal error during communication.";
        detectionStatusEl.classList.add('text-red-700');
        infoEl.innerHTML = `
            <div class="error-message">
                <strong>Error:</strong> Failed to communicate with the server. 
                Check the Flask console for Python errors.
            </div>
        `;
    } finally {
        scanButton.disabled = false;
        scanButton.textContent = 'SCAN AGAIN';
    }
}

// Event Listeners
scanButton.addEventListener('click', performScan);

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeCameraStatus();
    
    // Check if video feed is loading
    videoFeedEl.addEventListener('error', () => {
        cameraStatusEl.textContent = 'Error: Unable to load video feed';
        cameraStatusEl.classList.remove('text-green-600');
        cameraStatusEl.classList.add('text-red-600');
    });
    
    videoFeedEl.addEventListener('load', () => {
        console.log('Video feed loaded successfully');
    });
});