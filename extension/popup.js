document.addEventListener('DOMContentLoaded', () => {
    // Load saved URL
    chrome.storage.sync.get(['serverUrl'], (result) => {
        if (result.serverUrl) {
            document.getElementById('serverUrl').value = result.serverUrl;
        }
    });
});

document.getElementById('toggleBtn').addEventListener('click', () => {
    const btn = document.getElementById('toggleBtn');
    const status = document.getElementById('status');
    const url = document.getElementById('serverUrl').value;

    // Save URL
    chrome.storage.sync.set({serverUrl: url});
    
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
        if (!tabs[0]?.id) return;
        
        // Pass URL to content script
        chrome.tabs.sendMessage(tabs[0].id, {
            action: "toggle", 
            serverUrl: url
        }, (response) => {
            if (response?.active) {
                btn.innerText = "Stop AI";
                btn.classList.add('stop');
                status.innerText = "Thinking...";
            } else {
                btn.innerText = "Start AI";
                btn.classList.remove('stop');
                status.innerText = "Idle";
            }
        });
    });
});
