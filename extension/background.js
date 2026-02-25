// Minimal Background Service Worker
// SentinelX relies purely on content-script interactions injected from the popup
// to abide by Manifest V3 tight security standards and performance profiles.

chrome.runtime.onInstalled.addListener(() => {
    console.log("SentinelX Extension Installed Successfully.");
});

const API_ENDPOINT = "http://127.0.0.1:8000/predict/frame";

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "ANALYZE_FRAME") {
        fetch(API_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame_base64: request.frame })
        })
            .then(response => {
                if (!response.ok) throw new Error("Backend returned " + response.status);
                return response.json();
            })
            .then(data => sendResponse({ result: data }))
            .catch(error => sendResponse({ error: error.message }));

        return true; // Keep message channel open for async response
    }
});
