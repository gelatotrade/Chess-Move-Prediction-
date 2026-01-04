/**
 * Chess Move Predictor - Popup Script
 */

document.addEventListener('DOMContentLoaded', async () => {
    const elements = {
        enableToggle: document.getElementById('enableToggle'),
        statusIndicator: document.getElementById('statusIndicator'),
        serverStatus: document.getElementById('serverStatus'),
        statsSection: document.getElementById('statsSection'),
        gamesAnalyzed: document.getElementById('gamesAnalyzed'),
        movesAnalyzed: document.getElementById('movesAnalyzed'),
        numMoves: document.getElementById('numMoves'),
        showArrows: document.getElementById('showArrows'),
        showPanel: document.getElementById('showPanel'),
        autoUpdate: document.getElementById('autoUpdate'),
        serverUrl: document.getElementById('serverUrl'),
        testConnection: document.getElementById('testConnection')
    };

    // Load saved settings
    const settings = await loadSettings();
    applySettings(settings);

    // Check server connection
    await checkServerConnection();

    // Event listeners
    elements.enableToggle.addEventListener('change', async (e) => {
        const enabled = e.target.checked;
        await chrome.runtime.sendMessage({ type: 'SET_ENABLED', enabled });
        await saveSettings({ ...settings, enabled });
    });

    elements.numMoves.addEventListener('change', async (e) => {
        const numMoves = parseInt(e.target.value);
        await saveSettings({ ...settings, numMoves });
        notifyContentScript({ type: 'SETTINGS_CHANGED', settings: { numMoves } });
    });

    elements.showArrows.addEventListener('change', async (e) => {
        const showArrows = e.target.checked;
        await saveSettings({ ...settings, showArrows });
        notifyContentScript({ type: 'SETTINGS_CHANGED', settings: { showArrows } });
    });

    elements.showPanel.addEventListener('change', async (e) => {
        const showPanel = e.target.checked;
        await saveSettings({ ...settings, showPanel });
        notifyContentScript({ type: 'SETTINGS_CHANGED', settings: { showPanel } });
    });

    elements.autoUpdate.addEventListener('change', async (e) => {
        const autoUpdate = e.target.checked;
        await saveSettings({ ...settings, autoUpdate });
        notifyContentScript({ type: 'SETTINGS_CHANGED', settings: { autoUpdate } });
    });

    elements.serverUrl.addEventListener('change', async (e) => {
        const serverUrl = e.target.value;
        await saveSettings({ ...settings, serverUrl });
    });

    elements.testConnection.addEventListener('click', async () => {
        elements.testConnection.disabled = true;
        elements.testConnection.textContent = 'Testing...';
        await checkServerConnection();
        elements.testConnection.disabled = false;
        elements.testConnection.textContent = 'Test Connection';
    });

    // Functions
    async function loadSettings() {
        return new Promise((resolve) => {
            chrome.storage.sync.get({
                enabled: true,
                numMoves: 5,
                showArrows: true,
                showPanel: true,
                autoUpdate: true,
                serverUrl: 'http://localhost:8000'
            }, resolve);
        });
    }

    async function saveSettings(settings) {
        return new Promise((resolve) => {
            chrome.storage.sync.set(settings, resolve);
        });
    }

    function applySettings(settings) {
        elements.enableToggle.checked = settings.enabled;
        elements.numMoves.value = settings.numMoves;
        elements.showArrows.checked = settings.showArrows;
        elements.showPanel.checked = settings.showPanel;
        elements.autoUpdate.checked = settings.autoUpdate;
        elements.serverUrl.value = settings.serverUrl;
    }

    async function checkServerConnection() {
        const serverUrl = elements.serverUrl.value;

        elements.serverStatus.innerHTML = `
            <div class="loading-spinner"></div>
            <span>Connecting...</span>
        `;
        elements.statusIndicator.className = 'status-indicator';

        try {
            const response = await fetch(`${serverUrl}/health`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });

            if (response.ok) {
                const data = await response.json();
                elements.serverStatus.innerHTML = `
                    <span class="success">Connected - Model ${data.model_loaded ? 'loaded' : 'loading'}</span>
                `;
                elements.serverStatus.className = 'info-content success';
                elements.statusIndicator.className = 'status-indicator connected';
                elements.statusIndicator.querySelector('.status-text').textContent = 'Connected';
                elements.statsSection.style.display = 'grid';
            } else {
                throw new Error('Server error');
            }
        } catch (error) {
            elements.serverStatus.innerHTML = `
                <span class="error">Cannot connect to server</span>
            `;
            elements.serverStatus.className = 'info-content error';
            elements.statusIndicator.className = 'status-indicator disconnected';
            elements.statusIndicator.querySelector('.status-text').textContent = 'Disconnected';
            elements.statsSection.style.display = 'none';
        }
    }

    async function notifyContentScript(message) {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (tab && tab.url?.includes('chess.com')) {
            chrome.tabs.sendMessage(tab.id, message).catch(() => {});
        }
    }

    // Load stats
    async function loadStats() {
        const stats = await new Promise((resolve) => {
            chrome.storage.local.get({
                gamesAnalyzed: 0,
                movesAnalyzed: 0
            }, resolve);
        });

        elements.gamesAnalyzed.textContent = stats.gamesAnalyzed;
        elements.movesAnalyzed.textContent = stats.movesAnalyzed;
    }

    loadStats();
});
