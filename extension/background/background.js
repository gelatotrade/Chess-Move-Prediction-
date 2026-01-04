/**
 * Chess Move Predictor - Background Service Worker
 *
 * Handles:
 * 1. Extension state management
 * 2. API communication
 * 3. Message passing between components
 */

const API_URL = 'http://localhost:8000';

// Extension state
let extensionState = {
    enabled: true,
    connected: false,
    activeGames: new Map()
};

// Check API health
async function checkHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        extensionState.connected = response.ok;
        return response.ok;
    } catch (error) {
        extensionState.connected = false;
        return false;
    }
}

// Periodic health check
setInterval(checkHealth, 30000);

// Initial health check
checkHealth();

// Handle messages from content scripts and popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    switch (request.type) {
        case 'GET_STATE':
            sendResponse(extensionState);
            break;

        case 'SET_ENABLED':
            extensionState.enabled = request.enabled;
            // Notify all tabs
            chrome.tabs.query({}, (tabs) => {
                tabs.forEach(tab => {
                    chrome.tabs.sendMessage(tab.id, {
                        type: 'STATE_CHANGED',
                        state: extensionState
                    }).catch(() => {});
                });
            });
            sendResponse({ success: true });
            break;

        case 'CHECK_HEALTH':
            checkHealth().then(connected => {
                sendResponse({ connected });
            });
            return true; // Keep channel open for async response

        case 'PREDICT':
            fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(request.data)
            })
            .then(res => res.json())
            .then(data => sendResponse({ success: true, data }))
            .catch(error => sendResponse({ success: false, error: error.message }));
            return true;

        case 'NEW_GAME':
            const gameId = `game_${Date.now()}`;
            extensionState.activeGames.set(gameId, {
                tabId: sender.tab?.id,
                startTime: Date.now(),
                moves: []
            });
            sendResponse({ gameId });
            break;

        case 'GAME_MOVE':
            const game = extensionState.activeGames.get(request.gameId);
            if (game) {
                game.moves.push({
                    fen: request.fen,
                    move: request.move,
                    timestamp: Date.now()
                });
            }
            sendResponse({ success: true });
            break;

        default:
            sendResponse({ error: 'Unknown message type' });
    }
});

// Handle extension icon click
chrome.action.onClicked.addListener((tab) => {
    // Toggle enabled state
    extensionState.enabled = !extensionState.enabled;

    // Update icon
    const iconPath = extensionState.enabled ? 'icons/icon' : 'icons/icon-disabled';
    chrome.action.setIcon({
        path: {
            16: `${iconPath}16.png`,
            32: `${iconPath}32.png`,
            48: `${iconPath}48.png`,
            128: `${iconPath}128.png`
        }
    });

    // Notify content script
    chrome.tabs.sendMessage(tab.id, {
        type: 'STATE_CHANGED',
        state: extensionState
    }).catch(() => {});
});

// Handle tab updates
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === 'complete' && tab.url?.includes('chess.com')) {
        // Send current state to new page
        chrome.tabs.sendMessage(tabId, {
            type: 'STATE_CHANGED',
            state: extensionState
        }).catch(() => {});
    }
});

// Clean up game data when tab closes
chrome.tabs.onRemoved.addListener((tabId) => {
    for (const [gameId, game] of extensionState.activeGames) {
        if (game.tabId === tabId) {
            extensionState.activeGames.delete(gameId);
        }
    }
});

console.log('Chess Move Predictor: Background service worker started');
