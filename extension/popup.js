// Toggle the live scanning overlay on the active tab
document.getElementById('toggle-btn').addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab || tab.url.startsWith('chrome://')) {
        alert("SentinelX Shield cannot be injected into system pages.");
        return;
    }

    // Attempt to message the content script first to toggle the UI cleanly
    chrome.tabs.sendMessage(tab.id, { action: "TOGGLE_WIDGET" }, (response) => {
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
                // Scripts injected, re-trigger the action
                chrome.tabs.sendMessage(tab.id, { action: "TOGGLE_WIDGET" });
            });
        }
    });
});

// Open the bundled React dashboard as a new Chrome tab
document.getElementById('dashboard-btn').addEventListener('click', () => {
    const dashboardUrl = chrome.runtime.getURL('dashboard/index.html');
    chrome.tabs.create({ url: dashboardUrl });
});
