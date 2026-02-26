// Initialize toggle state on load
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0] && !tabs[0].url.startsWith('chrome://')) {
        chrome.tabs.sendMessage(tabs[0].id, { action: "GET_STATUS" }, (response) => {
            if (!chrome.runtime.lastError && response) {
                document.getElementById('scan-toggle').checked = response.isScanning;
            }
        });
    }
});

// Toggle the live scanning overlay on the active tab
document.getElementById('scan-toggle').addEventListener('change', async (e) => {
    const isEnabled = e.target.checked;
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab || tab.url.startsWith('chrome://')) {
        // Can't inject into this tab
        e.target.checked = false;
        alert("SentinelX Shield cannot be injected into system pages.");
        return;
    }

    const action = isEnabled ? "START_SCAN" : "STOP_SCAN";

    // Attempt to message the content script first to toggle the UI cleanly
    chrome.tabs.sendMessage(tab.id, { action }, (response) => {
        // If the request fails, the content script hasn't been injected yet
        if (chrome.runtime.lastError) {
            console.log("Injecting SentinelX Scripts...");

            chrome.scripting.insertCSS({
                target: { tabId: tab.id },
                files: ["content.css"]
            });

            chrome.scripting.executeScript({
                target: { tabId: tab.id },
                files: ["content.js"]
            }, () => {
                // Scripts injected, re-trigger the action but wait a moment for listener to attach
                setTimeout(() => {
                    chrome.tabs.sendMessage(tab.id, { action });
                }, 100);
            });
        }
    });
});

// Open the bundled React dashboard as a new Chrome tab
document.getElementById('dashboard-btn').addEventListener('click', () => {
    const dashboardUrl = chrome.runtime.getURL('dashboard/index.html');
    chrome.tabs.create({ url: dashboardUrl });
});


