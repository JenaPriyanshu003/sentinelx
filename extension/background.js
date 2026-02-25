// Minimal Background Service Worker
// SentinelX relies purely on content-script interactions injected from the popup
// to abide by Manifest V3 tight security standards and performance profiles.

chrome.runtime.onInstalled.addListener(() => {
    console.log("SentinelX Extension Installed Successfully.");
});
