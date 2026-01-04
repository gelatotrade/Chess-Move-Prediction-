/**
 * Chess Move Predictor - Content Script
 *
 * This script runs on Chess.com pages and provides:
 * 1. Board state detection
 * 2. Move prediction overlay
 * 3. Win probability visualization
 * 4. Real-time updates
 */

(function() {
    'use strict';

    // ============= Configuration =============

    const CONFIG = {
        API_URL: 'http://localhost:8000',
        WS_URL: 'ws://localhost:8000/ws',
        UPDATE_INTERVAL: 500,
        ARROW_COLORS: {
            best: '#22c55e',      // Green
            good: '#3b82f6',      // Blue
            average: '#f59e0b',   // Orange
            poor: '#ef4444'       // Red
        },
        OVERLAY_ID: 'chess-predictor-overlay',
        PANEL_ID: 'chess-predictor-panel'
    };

    // ============= State =============

    let state = {
        enabled: true,
        connected: false,
        currentFen: null,
        playerColor: null,
        gameId: null,
        predictions: [],
        winProbability: 0.5,
        winProbabilityHistory: [],
        settings: {
            showArrows: true,
            showPanel: true,
            numMoves: 5,
            autoUpdate: true
        }
    };

    // ============= Utility Functions =============

    /**
     * Debounce function calls
     */
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * Convert square notation to coordinates
     */
    function squareToCoords(square) {
        const file = square.charCodeAt(0) - 'a'.charCodeAt(0);
        const rank = parseInt(square[1]) - 1;
        return { file, rank };
    }

    /**
     * Convert move UCI to from/to squares
     */
    function parseMoveUci(uci) {
        return {
            from: uci.substring(0, 2),
            to: uci.substring(2, 4),
            promotion: uci.length > 4 ? uci[4] : null
        };
    }

    /**
     * Get color for probability value
     */
    function getProbabilityColor(prob) {
        if (prob >= 0.3) return CONFIG.ARROW_COLORS.best;
        if (prob >= 0.15) return CONFIG.ARROW_COLORS.good;
        if (prob >= 0.05) return CONFIG.ARROW_COLORS.average;
        return CONFIG.ARROW_COLORS.poor;
    }

    /**
     * Get color for win probability
     */
    function getWinProbabilityColor(prob) {
        if (prob >= 0.6) return '#22c55e';  // Green - winning
        if (prob >= 0.45) return '#6b7280'; // Gray - even
        return '#ef4444';                    // Red - losing
    }

    // ============= Board Detection =============

    /**
     * Extract FEN from Chess.com board
     */
    function extractFEN() {
        // Try to find the board element
        const board = document.querySelector('chess-board, wc-chess-board, .board');
        if (!board) return null;

        // Method 1: Check for data attribute
        const fenAttr = board.getAttribute('data-fen') ||
                        board.getAttribute('fen');
        if (fenAttr) return fenAttr;

        // Method 2: Parse from piece positions
        return parseBoardFromDOM();
    }

    /**
     * Parse board state from DOM elements
     */
    function parseBoardFromDOM() {
        const pieces = document.querySelectorAll('.piece');
        if (pieces.length === 0) return null;

        // Initialize empty board
        const board = Array(8).fill(null).map(() => Array(8).fill(null));
        const pieceMap = {
            'wp': 'P', 'wn': 'N', 'wb': 'B', 'wr': 'R', 'wq': 'Q', 'wk': 'K',
            'bp': 'p', 'bn': 'n', 'bb': 'b', 'br': 'r', 'bq': 'q', 'bk': 'k'
        };

        pieces.forEach(piece => {
            // Parse piece class (e.g., "piece wp square-52")
            const classes = piece.className.split(' ');
            let pieceType = null;
            let square = null;

            classes.forEach(cls => {
                if (pieceMap[cls]) {
                    pieceType = pieceMap[cls];
                } else if (cls.startsWith('square-')) {
                    const num = parseInt(cls.replace('square-', ''));
                    const file = (num % 10) - 1;
                    const rank = Math.floor(num / 10) - 1;
                    square = { file, rank };
                }
            });

            if (pieceType && square && square.file >= 0 && square.file < 8 &&
                square.rank >= 0 && square.rank < 8) {
                board[7 - square.rank][square.file] = pieceType;
            }
        });

        // Convert to FEN
        return boardToFEN(board);
    }

    /**
     * Convert board array to FEN string
     */
    function boardToFEN(board) {
        let fen = '';

        for (let rank = 0; rank < 8; rank++) {
            let empty = 0;
            for (let file = 0; file < 8; file++) {
                const piece = board[rank][file];
                if (piece) {
                    if (empty > 0) {
                        fen += empty;
                        empty = 0;
                    }
                    fen += piece;
                } else {
                    empty++;
                }
            }
            if (empty > 0) fen += empty;
            if (rank < 7) fen += '/';
        }

        // Add default game state (simplified - actual turn detection would be more complex)
        fen += ' w KQkq - 0 1';

        return fen;
    }

    /**
     * Detect player color
     */
    function detectPlayerColor() {
        const board = document.querySelector('chess-board, wc-chess-board, .board');
        if (!board) return null;

        // Check if board is flipped
        const isFlipped = board.classList.contains('flipped') ||
                          board.getAttribute('data-flipped') === 'true';

        return isFlipped ? 'black' : 'white';
    }

    /**
     * Detect current turn
     */
    function detectCurrentTurn() {
        // Check clock highlighting
        const whiteClock = document.querySelector('.clock-white.clock-player-turn');
        const blackClock = document.querySelector('.clock-black.clock-player-turn');

        if (whiteClock) return 'white';
        if (blackClock) return 'black';

        return null;
    }

    // ============= API Communication =============

    /**
     * Fetch predictions from API
     */
    async function fetchPredictions(fen) {
        try {
            const response = await fetch(`${CONFIG.API_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    fen: fen,
                    top_k: state.settings.numMoves
                })
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Prediction error:', error);
            return null;
        }
    }

    /**
     * Check API connection
     */
    async function checkConnection() {
        try {
            const response = await fetch(`${CONFIG.API_URL}/health`);
            state.connected = response.ok;
            return response.ok;
        } catch {
            state.connected = false;
            return false;
        }
    }

    // ============= Overlay Rendering =============

    /**
     * Create or get the main overlay container
     */
    function getOverlayContainer() {
        let container = document.getElementById(CONFIG.OVERLAY_ID);

        if (!container) {
            container = document.createElement('div');
            container.id = CONFIG.OVERLAY_ID;
            document.body.appendChild(container);
        }

        return container;
    }

    /**
     * Create arrow SVG for move visualization
     */
    function createArrow(fromSquare, toSquare, color, opacity) {
        const board = document.querySelector('chess-board, wc-chess-board, .board');
        if (!board) return null;

        const rect = board.getBoundingClientRect();
        const squareSize = rect.width / 8;

        const from = squareToCoords(fromSquare);
        const to = squareToCoords(toSquare);

        // Adjust for board orientation
        const isFlipped = state.playerColor === 'black';
        const fromX = isFlipped ? (7 - from.file) : from.file;
        const fromY = isFlipped ? from.rank : (7 - from.rank);
        const toX = isFlipped ? (7 - to.file) : to.file;
        const toY = isFlipped ? to.rank : (7 - to.rank);

        // Calculate arrow coordinates
        const x1 = rect.left + (fromX + 0.5) * squareSize;
        const y1 = rect.top + (fromY + 0.5) * squareSize;
        const x2 = rect.left + (toX + 0.5) * squareSize;
        const y2 = rect.top + (toY + 0.5) * squareSize;

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.style.cssText = `
            position: fixed;
            left: 0;
            top: 0;
            width: 100vw;
            height: 100vh;
            pointer-events: none;
            z-index: 9999;
        `;

        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        marker.id = `arrowhead-${fromSquare}-${toSquare}`;
        marker.setAttribute('markerWidth', '10');
        marker.setAttribute('markerHeight', '7');
        marker.setAttribute('refX', '9');
        marker.setAttribute('refY', '3.5');
        marker.setAttribute('orient', 'auto');

        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', '0 0, 10 3.5, 0 7');
        polygon.setAttribute('fill', color);

        marker.appendChild(polygon);
        defs.appendChild(marker);
        svg.appendChild(defs);

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x2);
        line.setAttribute('y2', y2);
        line.setAttribute('stroke', color);
        line.setAttribute('stroke-width', Math.max(squareSize * 0.15, 8));
        line.setAttribute('stroke-opacity', opacity);
        line.setAttribute('marker-end', `url(#arrowhead-${fromSquare}-${toSquare})`);
        line.setAttribute('stroke-linecap', 'round');

        svg.appendChild(line);

        return svg;
    }

    /**
     * Render move arrows on board
     */
    function renderArrows(predictions) {
        // Remove existing arrows
        document.querySelectorAll('.chess-predictor-arrow').forEach(el => el.remove());

        if (!state.settings.showArrows || !predictions.length) return;

        predictions.forEach((move, index) => {
            const { from, to } = parseMoveUci(move.move);
            const color = getProbabilityColor(move.probability);
            const opacity = 0.8 - (index * 0.12);

            const arrow = createArrow(from, to, color, opacity);
            if (arrow) {
                arrow.classList.add('chess-predictor-arrow');
                getOverlayContainer().appendChild(arrow);
            }
        });
    }

    /**
     * Create the side panel UI
     */
    function createPanel() {
        let panel = document.getElementById(CONFIG.PANEL_ID);

        if (panel) return panel;

        panel = document.createElement('div');
        panel.id = CONFIG.PANEL_ID;
        panel.innerHTML = `
            <div class="predictor-header">
                <span class="predictor-title">Move Predictor</span>
                <div class="predictor-status"></div>
                <button class="predictor-toggle">_</button>
            </div>
            <div class="predictor-content">
                <div class="win-probability-section">
                    <div class="win-probability-label">Win Probability</div>
                    <div class="win-probability-bar">
                        <div class="win-probability-fill"></div>
                    </div>
                    <div class="win-probability-value">50%</div>
                </div>
                <div class="win-probability-chart">
                    <canvas id="win-prob-canvas"></canvas>
                </div>
                <div class="moves-section">
                    <div class="moves-label">Top Moves</div>
                    <div class="moves-list"></div>
                </div>
                <div class="settings-section">
                    <label class="setting-item">
                        <input type="checkbox" id="setting-arrows" checked>
                        <span>Show arrows</span>
                    </label>
                    <label class="setting-item">
                        <input type="checkbox" id="setting-auto" checked>
                        <span>Auto-update</span>
                    </label>
                    <label class="setting-item">
                        <span>Moves:</span>
                        <select id="setting-num-moves">
                            <option value="3">3</option>
                            <option value="5" selected>5</option>
                            <option value="10">10</option>
                        </select>
                    </label>
                </div>
            </div>
        `;

        document.body.appendChild(panel);

        // Add event listeners
        panel.querySelector('.predictor-toggle').addEventListener('click', () => {
            panel.classList.toggle('minimized');
        });

        panel.querySelector('#setting-arrows').addEventListener('change', (e) => {
            state.settings.showArrows = e.target.checked;
            updateOverlay();
        });

        panel.querySelector('#setting-auto').addEventListener('change', (e) => {
            state.settings.autoUpdate = e.target.checked;
        });

        panel.querySelector('#setting-num-moves').addEventListener('change', (e) => {
            state.settings.numMoves = parseInt(e.target.value);
            updateOverlay();
        });

        return panel;
    }

    /**
     * Update the side panel with new data
     */
    function updatePanel(predictions, winProbability) {
        const panel = document.getElementById(CONFIG.PANEL_ID);
        if (!panel) return;

        // Update connection status
        const status = panel.querySelector('.predictor-status');
        status.className = `predictor-status ${state.connected ? 'connected' : 'disconnected'}`;
        status.title = state.connected ? 'Connected' : 'Disconnected';

        // Update win probability
        const probValue = panel.querySelector('.win-probability-value');
        const probFill = panel.querySelector('.win-probability-fill');
        const probPercent = Math.round(winProbability * 100);

        probValue.textContent = `${probPercent}%`;
        probValue.style.color = getWinProbabilityColor(winProbability);
        probFill.style.width = `${probPercent}%`;
        probFill.style.background = getWinProbabilityColor(winProbability);

        // Update moves list
        const movesList = panel.querySelector('.moves-list');
        movesList.innerHTML = predictions.map((move, i) => `
            <div class="move-item" data-move="${move.move}">
                <span class="move-rank">${i + 1}.</span>
                <span class="move-san">${move.san}</span>
                <div class="move-stats">
                    <span class="move-prob" style="color: ${getProbabilityColor(move.probability)}">
                        ${(move.probability * 100).toFixed(1)}%
                    </span>
                    <span class="move-eval ${move.eval_change >= 0 ? 'positive' : 'negative'}">
                        ${move.eval_change >= 0 ? '+' : ''}${(move.eval_change * 100).toFixed(1)}%
                    </span>
                </div>
                <div class="move-win-bar">
                    <div class="move-win-fill" style="width: ${move.win_probability * 100}%; background: ${getWinProbabilityColor(move.win_probability)}"></div>
                </div>
            </div>
        `).join('');

        // Add hover effects
        movesList.querySelectorAll('.move-item').forEach(item => {
            item.addEventListener('mouseenter', () => {
                const move = item.dataset.move;
                highlightMove(move);
            });
            item.addEventListener('mouseleave', () => {
                renderArrows(state.predictions);
            });
        });

        // Update chart
        updateWinProbabilityChart();
    }

    /**
     * Highlight a specific move
     */
    function highlightMove(moveUci) {
        document.querySelectorAll('.chess-predictor-arrow').forEach(el => el.remove());

        const { from, to } = parseMoveUci(moveUci);
        const arrow = createArrow(from, to, '#8b5cf6', 0.9); // Purple for highlighted
        if (arrow) {
            arrow.classList.add('chess-predictor-arrow');
            getOverlayContainer().appendChild(arrow);
        }
    }

    /**
     * Update the win probability chart
     */
    function updateWinProbabilityChart() {
        const canvas = document.getElementById('win-prob-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const history = state.winProbabilityHistory;

        // Set canvas size
        canvas.width = canvas.offsetWidth * 2;
        canvas.height = canvas.offsetHeight * 2;
        ctx.scale(2, 2);

        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;

        // Clear canvas
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, width, height);

        if (history.length < 2) return;

        // Draw 50% line
        ctx.strokeStyle = '#4a4a6a';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw probability curve
        ctx.strokeStyle = '#8b5cf6';
        ctx.lineWidth = 2;
        ctx.beginPath();

        const step = width / (history.length - 1);
        history.forEach((prob, i) => {
            const x = i * step;
            const y = height - (prob * height);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // Fill area
        ctx.lineTo(width, height);
        ctx.lineTo(0, height);
        ctx.closePath();
        ctx.fillStyle = 'rgba(139, 92, 246, 0.2)';
        ctx.fill();
    }

    // ============= Main Update Loop =============

    /**
     * Update the overlay with new predictions
     */
    async function updateOverlay() {
        if (!state.enabled) return;

        const fen = extractFEN();
        if (!fen || fen === state.currentFen) return;

        state.currentFen = fen;
        state.playerColor = detectPlayerColor();

        const predictions = await fetchPredictions(fen);

        if (predictions) {
            state.predictions = predictions.top_moves || [];
            state.winProbability = predictions.win_probability;
            state.winProbabilityHistory.push(state.winProbability);

            // Keep history to reasonable size
            if (state.winProbabilityHistory.length > 100) {
                state.winProbabilityHistory.shift();
            }

            renderArrows(state.predictions);
            updatePanel(state.predictions, state.winProbability);
        }
    }

    /**
     * Debounced update function
     */
    const debouncedUpdate = debounce(updateOverlay, CONFIG.UPDATE_INTERVAL);

    /**
     * Start the update loop
     */
    function startUpdateLoop() {
        // Initial update
        updateOverlay();

        // Set up mutation observer for board changes
        const observer = new MutationObserver((mutations) => {
            if (state.settings.autoUpdate) {
                debouncedUpdate();
            }
        });

        // Observe the whole document for chess board changes
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['class', 'style']
        });

        // Also poll periodically
        setInterval(() => {
            if (state.settings.autoUpdate) {
                updateOverlay();
            }
        }, 2000);
    }

    // ============= Initialization =============

    /**
     * Initialize the extension
     */
    async function init() {
        console.log('Chess Move Predictor: Initializing...');

        // Check if we're on a game page
        if (!window.location.pathname.includes('/game/') &&
            !window.location.pathname.includes('/play/')) {
            console.log('Chess Move Predictor: Not a game page');
            return;
        }

        // Check API connection
        const connected = await checkConnection();
        if (!connected) {
            console.warn('Chess Move Predictor: Cannot connect to API server');
        }

        // Create UI elements
        createPanel();
        getOverlayContainer();

        // Start update loop
        startUpdateLoop();

        console.log('Chess Move Predictor: Initialized');
    }

    // Wait for page load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    // Handle window resize
    window.addEventListener('resize', debounce(() => {
        renderArrows(state.predictions);
    }, 100));

    // Export for debugging
    window.ChessPredictor = {
        getState: () => state,
        updateOverlay,
        extractFEN,
        checkConnection
    };

})();
