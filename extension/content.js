if (typeof window.sentinelxInjected === 'undefined') {
    window.sentinelxInjected = true;

    // Fix 1: Block auto-scan on video-call sites (manual scan still works)
    const AUTO_SCAN_BLOCKLIST = ['zoom.us', 'teams.microsoft.com', 'whereby.com', 'webex.com'];
    const isBlocklisted = AUTO_SCAN_BLOCKLIST.some(h => location.hostname.includes(h));

    // Streak counter: require N consecutive fake-majority windows before alerting
    const STREAK_NEEDED = 3;
    let fakeStreak = 0;

    let stream = null;
    let intervalId = null;
    let isScanning = false;
    let autoVideoElement = null; // Track natively hooked video tag

    // UI Elements
    let widget, statusText, confidenceBarContainer, confidenceFill;

    // Create floating widget
    function injectWidget() {
        if (document.getElementById('sentinelx-widget-container')) return;

        widget = document.createElement('div');
        widget.id = 'sentinelx-widget-container';
        widget.className = 'sentinelx-animate-in'; // New smooth animation

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
            statusText.innerText = 'Initializing...';
            statusText.className = 'scanning';
        } else if (status === 'SCANNING') {
            statusText.innerText = 'Live Scanning...';
            statusText.className = 'scanning';
            confidenceBarContainer.style.display = 'block';
            confidenceFill.style.width = '0%';
        } else if (status === 'RESULT') {
            // label: 'Real', 'Fake', or 'Uncertain'
            const isFake = label === 'Fake';
            const isUncertain = label === 'Uncertain';
            if (isUncertain) {
                statusText.innerText = `~ Uncertain (${confidence.toFixed(1)}%)`;
                statusText.className = 'scanning';
                confidenceFill.classList.add('real-fill');
            } else {
                statusText.innerText = `${isFake ? '⚠ Deepfake' : '✓ Authentic'} (${confidence.toFixed(1)}%)`;
                statusText.className = isFake ? 'fake' : 'real';
                confidenceFill.classList.add(isFake ? 'fake-fill' : 'real-fill');
            }
            confidenceBarContainer.style.display = 'block';
            confidenceFill.style.width = `${confidence}%`;
        } else if (status === 'ERROR') {
            statusText.innerText = 'Error: ' + label;
            statusText.className = 'fake';
            stopCapture();
        }
    }

    // Start Capture (Manual via popup OR Auto via Video tag)
    async function startCapture(targetVideoElement = null) {
        if (isScanning) return;
        try {
            updateUI('INIT');
            isScanning = true;

            if (targetVideoElement) {
                // AUTO-MODE: We have a direct DOM video tag
                autoVideoElement = targetVideoElement;
                setupProcessor(autoVideoElement, false);
                updateUI('SCANNING');
            } else {
                // MANUAL-MODE: Browser screen-share fallback
                stream = await navigator.mediaDevices.getDisplayMedia({
                    video: { displaySurface: "browser", frameRate: { ideal: 5, max: 10 } },
                    audio: false
                });

                const hiddenVideo = document.createElement('video');
                hiddenVideo.srcObject = stream;
                hiddenVideo.play();
                setupProcessor(hiddenVideo, true);

                // Handle stream manual stop by user from the native Chrome UI
                stream.getVideoTracks()[0].addEventListener('ended', stopCapture);
                updateUI('SCANNING');
            }
        } catch (err) {
            console.error("Capture Error:", err);
            isScanning = false;
            updateUI('READY'); // Reset if user cancels prompt
        }
    }

    /**
     * Frame quality gate — returns true if the frame is usable.
     * Skips frames that are too dark (low brightness) or too blurry (low variance).
     * This prevents bad-camera / low-light frames from polluting the sliding window.
     */
    function isFrameUsable(ctx, width, height) {
        // Sample a small 64x64 region from the center for speed
        const sw = Math.min(64, width), sh = Math.min(64, height);
        const sx = Math.floor((width - sw) / 2);
        const sy = Math.floor((height - sh) / 2);
        const data = ctx.getImageData(sx, sy, sw, sh).data; // RGBA

        let sum = 0, sumSq = 0;
        const pixels = sw * sh;
        for (let i = 0; i < data.length; i += 4) {
            // Luminance approximation
            const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            sum += lum;
            sumSq += lum * lum;
        }
        const mean = sum / pixels;
        const variance = sumSq / pixels - mean * mean;

        // Too dark: mean brightness below 30/255 (looser for darker rooms)
        if (mean < 30) { console.log('[SentinelX] Frame too dark, skipping.'); return false; }
        // Too blurry/flat: variance below 150 (looser for compressed Meet streams)
        if (variance < 150) { console.log('[SentinelX] Frame too blurry/flat, skipping.'); return false; }

        return true;
    }

    function setupProcessor(videoElement, isStream) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const MAX_DIM = 1280;

        const startSampling = () => {
            let width = videoElement.videoWidth || videoElement.clientWidth;
            let height = videoElement.videoHeight || videoElement.clientHeight;

            // Wait for valid dimensions if they are 0
            if (width === 0 || height === 0) {
                setTimeout(startSampling, 500);
                return;
            }

            if (width > MAX_DIM) {
                height = Math.floor(height * (MAX_DIM / width));
                width = MAX_DIM;
            }

            canvas.width = width;
            canvas.height = height;

            // Store current sampling dimensions for the coordinate mapper
            videoElement.setAttribute('data-sentinelx-sample-w', width);
            videoElement.setAttribute('data-sentinelx-sample-h', height);

            // Sample at 5 FPS (every 200ms)
            intervalId = setInterval(() => {
                if (!isScanning) return;

                // If the DOM video element was paused, we pause scanning
                if (!isStream && (videoElement.paused || videoElement.ended)) {
                    updateUI('READY');
                    return;
                }

                // If it resumed, show scanning
                if (!isStream && !videoElement.paused && statusText.className !== 'scanning' && statusText.className !== 'real' && statusText.className !== 'fake') {
                    updateUI('SCANNING');
                }

                try {
                    ctx.drawImage(videoElement, 0, 0, width, height);

                    // Quality gate: skip dark / blurry frames
                    if (!isFrameUsable(ctx, width, height)) {
                        if (statusText && statusText.className !== 'fake' && statusText.className !== 'real') {
                            statusText.innerText = '⟳ Low quality — skipping frame';
                            statusText.className = 'scanning';
                        }
                        return; // Don't touch history or streak — preserve last good verdict
                    }

                    const frameBase64 = canvas.toDataURL('image/jpeg', 0.6);
                    sendToBackend(frameBase64, videoElement);
                } catch (e) {
                    console.error("Canvas draw error (CORS/Tainted usually):", e);
                }
            }, 200);
        };

        if (videoElement.videoWidth) {
            startSampling();
        } else {
            videoElement.addEventListener('loadedmetadata', startSampling);
        }
    }

    // Simple sliding window for temporal smoothing (last 5 results)
    let history = [];

    function renderFaceBox(video, box, prediction, confidence) {
        let boxId = video.getAttribute('data-sentinelx-box-id');
        let boxEl;

        if (!boxId) {
            boxId = 'box-' + Math.random().toString(36).substr(2, 9);
            video.setAttribute('data-sentinelx-box-id', boxId);
            boxEl = document.createElement('div');
            boxEl.id = boxId;
            boxEl.className = 'sentinelx-face-box';
            boxEl.innerHTML = '<div class="sentinelx-face-label"></div>';
            document.body.appendChild(boxEl);
        } else {
            boxEl = document.getElementById(boxId);
        }

        if (!boxEl) return;

        if (!box) {
            boxEl.style.display = 'none';
            return;
        }

        // Calculate positioning
        const rect = video.getBoundingClientRect();
        const videoNativeW = video.videoWidth || video.clientWidth;
        const videoNativeH = video.videoHeight || video.clientHeight;

        // Scaling factors
        // The box coordinates from backend are relative to the canvas dimensions we sent
        const sampleW = parseFloat(video.getAttribute('data-sentinelx-sample-w')) || videoNativeW;
        const sampleH = parseFloat(video.getAttribute('data-sentinelx-sample-h')) || videoNativeH;

        const scaleX = rect.width / sampleW;
        const scaleY = rect.height / sampleH;

        const [x, y, w, h] = box;

        boxEl.style.display = 'block';
        boxEl.style.left = (rect.left + window.scrollX + (x * scaleX)) + 'px';
        boxEl.style.top = (rect.top + window.scrollY + (y * scaleY)) + 'px';
        boxEl.style.width = (w * scaleX) + 'px';
        boxEl.style.height = (h * scaleY) + 'px';

        // Update classes and label
        boxEl.classList.remove('fake', 'real');
        const label = boxEl.querySelector('.sentinelx-face-label');

        if (prediction === 'Fake') {
            boxEl.classList.add('fake');
            label.innerText = `⚠ Deepfake Detected (${confidence.toFixed(1)}%)`;
        } else if (prediction === 'Real') {
            boxEl.classList.add('real');
            label.innerText = `✓ Authentic (${confidence.toFixed(1)}%)`;
        } else {
            label.innerText = `Analyzing...`;
        }
    }

    function sendToBackend(base64Frame, videoElement = null) {
        if (!chrome.runtime?.id) {
            console.log("[SentinelX] Extension context invalidated. Stopping scan.");
            stopCapture();
            return;
        }
        try {
            chrome.runtime.sendMessage(
                { action: "ANALYZE_FRAME", frame: base64Frame },
                (response) => {
                    if (chrome.runtime.lastError) return;
                    if (!response || response.error) return;

                    const result = response.result;
                    if (!result) return;

                    if (result.status === 'success') {
                        // Temporal smoothing
                        const score_fake = result.score_fake;
                        history.push(score_fake);
                        if (history.length > 5) history.shift();

                        const avg_fake = history.reduce((a, b) => a + b, 0) / history.length;
                        const avg_real = 100 - avg_fake;

                        const pred = avg_fake > avg_real ? 'Fake' : 'Real';
                        const conf = avg_fake > avg_real ? avg_fake : avg_real;

                        if (pred === 'Fake') {
                            fakeStreak++;
                            if (fakeStreak >= STREAK_NEEDED) {
                                if (videoElement) renderFaceBox(videoElement, result.face_box, pred, conf);
                                updateUI('RESULT', 'Fake', avg_fake);
                            } else {
                                if (videoElement) renderFaceBox(videoElement, result.face_box, 'Analyzing', avg_fake);
                                statusText.innerText = `Analyzing... (${avg_fake.toFixed(1)}%)`;
                                statusText.className = 'scanning';
                            }
                        } else {
                            fakeStreak = 0;
                            if (videoElement) renderFaceBox(videoElement, result.face_box, 'Real', avg_real);
                            updateUI('RESULT', 'Real', avg_real);
                        }
                    } else if (result.status === 'no_face') {
                        if (videoElement) renderFaceBox(videoElement, null);
                        if (statusText.className !== 'scanning') {
                            statusText.innerText = "Scanning Feed...";
                            statusText.className = "scanning";
                            confidenceFill.style.width = '0%';
                        }
                    }
                });
        } catch (error) {
            console.error("Frame Dispatch Error:", error.message);
        }
    }

    function stopCapture() {
        isScanning = false;
        autoVideoElement = null;
        history = [];
        fakeStreak = 0;

        // Cleanup face boxes
        document.querySelectorAll('.sentinelx-face-box').forEach(box => box.remove());
        document.querySelectorAll('[data-sentinelx-box-id]').forEach(v => v.removeAttribute('data-sentinelx-box-id'));

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

    // --- PHASE 7: AUTO-SCAN MUTATION OBSERVER ---
    // Automatically detect playing <video> tags on the page and hook into them seamlessly.
    function checkForVideos() {
        // Fix 1: Don't auto-scan on video-call sites
        if (isBlocklisted) {
            console.log("[SentinelX] Video-call site detected. Auto-scan disabled. Use popup to scan manually.");
            return;
        }
        if (isScanning) return; // Already scanning
        const videos = document.querySelectorAll('video');
        for (let vid of videos) {
            // Fix 2: Only hook videos larger than 300x200 (ignore tiny participant tiles)
            if (!vid.paused && vid.readyState >= 2 && vid.clientWidth > 300 && vid.clientHeight > 200) {
                console.log("[SentinelX] Auto-detected playing video. Deploying Shield...");
                injectWidget();
                startCapture(vid);
                return; // Only hook the first major playing video
            }

            // If it's not playing yet, attach a play listener
            if (!vid.hasAttribute('data-sentinelx-hooked')) {
                vid.setAttribute('data-sentinelx-hooked', 'true');
                vid.addEventListener('playing', () => {
                    // Fix 1 & 2: Re-check blocklist and size on play event too
                    if (!isScanning && !isBlocklisted && vid.clientWidth > 300 && vid.clientHeight > 200) {
                        console.log("[SentinelX] Video play event detected. Deploying Shield...");
                        injectWidget();
                        startCapture(vid);
                    }
                });
            }
        }
    }

    // Run initial check
    setTimeout(checkForVideos, 1000);

    // Watch for dynamically added videos (like YouTube navigation or Twitter feed scrolling)
    const observer = new MutationObserver(() => {
        if (!isScanning) checkForVideos();
    });
    observer.observe(document.body, { childList: true, subtree: true });


    // --- LISTEN FOR MANUAL MESSAGES FROM POPUP ---
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.action === "START_SCAN") {
            if (!document.getElementById('sentinelx-widget-container')) injectWidget();
            startCapture(); // Manual start via screen-share
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
