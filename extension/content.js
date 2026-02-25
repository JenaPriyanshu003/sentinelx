if (typeof window.sentinelxInjected === 'undefined') {
    window.sentinelxInjected = true;

    let stream = null;
    let intervalId = null;
    let isScanning = false;

    // UI Elements
    let widget, statusText, startBtn, stopBtn, confidenceBarContainer, confidenceFill;

    // Create floating widget
    function injectWidget() {
        if (document.getElementById('sentinelx-widget-container')) return;

        widget = document.createElement('div');
        widget.id = 'sentinelx-widget-container';

        widget.innerHTML = `
        <div id="sentinelx-header">
            <span class="logo">SentinelX Lens</span>
            <span id="sentinelx-close">✖</span>
        </div>
        <div id="sentinelx-status">Ready to Scan</div>
        <div id="sentinelx-confidence-bar-container">
            <div id="sentinelx-confidence-fill"></div>
        </div>
    `;

        document.body.appendChild(widget);

        // Grab refs
        statusText = document.getElementById('sentinelx-status');
        confidenceBarContainer = document.getElementById('sentinelx-confidence-bar-container');
        confidenceFill = document.getElementById('sentinelx-confidence-fill');
        const closeBtn = document.getElementById('sentinelx-close');

        // Attach listeners
        closeBtn.addEventListener('click', () => {
            stopCapture();
            widget.remove();
        });
    }

    function updateUI(status, label = '', confidence = 0) {
        if (!statusText) return;

        // Reset classes
        statusText.className = '';
        confidenceFill.className = '';

        if (status === 'READY') {
            statusText.innerText = 'Ready to Scan';
            confidenceBarContainer.style.display = 'none';
        } else if (status === 'INIT') {
            statusText.innerText = 'Requesting Screen...';
            statusText.className = 'scanning';
        } else if (status === 'SCANNING') {
            statusText.innerText = 'Scanning... (Looking for faces)';
            statusText.className = 'scanning';
            confidenceBarContainer.style.display = 'block';
            confidenceFill.style.width = '0%';
        } else if (status === 'RESULT') {
            // label: 'Real' or 'Fake'
            const isFake = label.toLowerCase() === 'fake';
            statusText.innerText = `${isFake ? '⚠ Deepfake Detected' : '✓ Authentic'} (${confidence.toFixed(1)}%)`;
            statusText.className = isFake ? 'fake' : 'real';

            confidenceFill.classList.add(isFake ? 'fake-fill' : 'real-fill');
            confidenceFill.style.width = `${confidence}%`;
        } else if (status === 'ERROR') {
            statusText.innerText = 'Error: ' + label;
            statusText.className = 'fake';
            stopCapture();
        }
    }

    async function startCapture() {
        try {
            updateUI('INIT');

            // This triggers the browser's native screen sharing prompt
            stream = await navigator.mediaDevices.getDisplayMedia({
                video: {
                    displaySurface: "browser",
                    frameRate: { ideal: 5, max: 10 }
                },
                audio: false
            });

            // Set up the hidden video processor
            setupProcessor();

            // Handle stream manual stop by user from the native Chrome UI
            stream.getVideoTracks()[0].addEventListener('ended', stopCapture);

            updateUI('SCANNING');
            isScanning = true;

        } catch (err) {
            console.error("Capture Error:", err);
            updateUI('READY'); // Reset if user cancels prompt
        }
    }

    function setupProcessor() {
        const video = document.createElement('video');
        video.srcObject = stream;
        video.play();

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        // To prevent aggressive memory bloat, limit resolution slightly if 4k
        const MAX_DIM = 1280;

        video.addEventListener('loadedmetadata', () => {
            let width = video.videoWidth;
            let height = video.videoHeight;

            if (width > MAX_DIM) {
                height = Math.floor(height * (MAX_DIM / width));
                width = MAX_DIM;
            }

            canvas.width = width;
            canvas.height = height;

            // Sample at 5 FPS (every 200ms)
            intervalId = setInterval(() => {
                if (!isScanning) return;

                ctx.drawImage(video, 0, 0, width, height);

                // Extract frame as high-compressed JPEG strictly in memory
                const frameBase64 = canvas.toDataURL('image/jpeg', 0.6);
                sendToBackend(frameBase64);
            }, 200);
        });
    }

    // Simple sliding window for temporal smoothing (last 5 results)
    let history = [];

    function sendToBackend(base64Frame) {
        try {
            chrome.runtime.sendMessage(
                { action: "ANALYZE_FRAME", frame: base64Frame },
                (response) => {
                    if (chrome.runtime.lastError) {
                        console.error("Extension error:", chrome.runtime.lastError.message);
                        return;
                    }
                    if (response && response.error) {
                        console.error("Backend error:", response.error);
                        return;
                    }

                    const result = response.result;
                    if (!result) return;

                    if (result.status === 'success') {
                        // Sliding Window calculation for temporal smoothing
                        const score_fake = result.score_fake;
                        history.push(score_fake);
                        if (history.length > 5) history.shift();

                        const avg_fake = history.reduce((a, b) => a + b, 0) / history.length;
                        const avg_real = 100 - avg_fake;

                        const isFake = avg_fake > avg_real;
                        const conf = Math.max(avg_fake, avg_real);

                        updateUI('RESULT', isFake ? 'Fake' : 'Real', conf);
                    } else if (result.status === 'no_face') {
                        // Keep previous state but dim it, or say looking for faces
                        if (statusText.className !== 'scanning') {
                            statusText.innerText = "Looking for faces...";
                            statusText.className = "scanning";
                            confidenceFill.style.width = '0%';
                        }
                    }
                }); // wait for background response

        } catch (error) {
            console.error("Frame Dispatch Error:", error);
        }
    }

    function stopCapture() {
        isScanning = false;
        history = [];
        if (intervalId) {
            clearInterval(intervalId);
            intervalId = null;
        }
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        updateUI('READY');
    }

    // Listen for messages from popup
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.action === "START_SCAN") {
            if (!document.getElementById('sentinelx-widget-container')) {
                injectWidget();
            }
            startCapture();
            sendResponse({ status: "started" });
        } else if (request.action === "STOP_SCAN") {
            stopCapture();
            if (widget) widget.remove();
            sendResponse({ status: "stopped" });
        } else if (request.action === "GET_STATUS") {
            sendResponse({ isScanning: isScanning });
        }
    });

} // End of injection guard
