// State
let isAIActive = false;
let engineProcessing = false;
const PIECE_SELECTOR = '.piece';
const HIGHLIGHT_SELECTOR = '.highlight';

let serverUrl = "https://chess.bsi.deno.dev"; // Default

// DOM Selectors (Chess.com specific)
const BOARD_SELECTOR = 'chess-board';

// Toggle Listener
chrome.runtime.onMessage.addListener((request, _sender, sendResponse) => {
    if (request.action === "toggle") {
        isAIActive = !isAIActive;
        
        if (request.serverUrl) {
            serverUrl = request.serverUrl;
            if (serverUrl.endsWith('/')) serverUrl = serverUrl.slice(0, -1);
        }

        console.log(`AI Active: ${isAIActive} (Server: ${serverUrl})`);
        
        if (isAIActive) {
            playNextMove();
        }
        sendResponse({active: isAIActive});
    }
    return true;
});

// Main Loop - Observer
const observer = new MutationObserver(() => {
    if (!isAIActive || engineProcessing) return;
    debounce(playNextMove, 1000);
});

// Wait for board to appear
const waitForBoard = setInterval(() => {
    const boardElement = document.querySelector(BOARD_SELECTOR) || document.body;
    if (boardElement) {
        observer.observe(boardElement, { childList: true, subtree: true });
        clearInterval(waitForBoard);
    }
}, 1000);

let debounceTimer;
function debounce(func, delay) {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(func, delay);
}

async function playNextMove() {
    if (!isAIActive || engineProcessing) return;
    
    engineProcessing = true;
    try {
        const fen = await getFEN();
        console.log("Current FEN:", fen);
        
        if (!fen) {
            console.log("Could not detect board/FEN");
            return;
        }

        const move = await fetchBestMove(fen);
        console.log("AI suggests:", move);
        
        if (move) {
            makeMove(move);
        }
    } catch (e) {
        console.error("AI Error:", e);
    } finally {
        engineProcessing = false;
    }
}

function getFEN() {
    return new Promise((resolve) => {
        const script = document.createElement('script');
        script.textContent = `
            (function() {
                try {
                    let fen = "";
                    const game = document.querySelector('chess-board')?.game;
                    if (game) {
                        fen = game.getFEN();
                    } else if (globalThis.chesscom && globalThis.chesscom.game) {
                        fen = globalThis.chesscom.game.getFEN();
                    }
                    globalThis.postMessage({ type: "FROM_PAGE", text: fen }, "*");
                } catch(e) { 
                    globalThis.postMessage({ type: "FROM_PAGE", text: "" }, "*");
                }
            })();
        `;
        
        const listener = (event) => {
            if (event.source != globalThis) return;
            if (event.data.type && (event.data.type == "FROM_PAGE")) {
                globalThis.removeEventListener("message", listener);
                resolve(event.data.text);
            }
        };

        globalThis.addEventListener("message", listener);
        (document.head || document.documentElement).appendChild(script);
        script.remove();
    });
}

// 2. Fetch Move from Deno Server
async function fetchBestMove(fen) {
    try {
        const endpoint = `${serverUrl}/analyze`;
        const req = await fetch(endpoint, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ fen: fen })
        });
        const res = await req.json();
        return res.move;
    } catch (e) {
        console.error("Server connection failed:", e);
        displayHint("Connection Error!");
        return null;
    }
}

// 3. Make Move (Simulate Clicks)
function makeMove(moveSan) {
    console.log("Executing move:", moveSan);
    displayHint(moveSan);
}

function displayHint(move) {
    let hint = document.getElementById('ai-hint');
    if (!hint) {
        hint = document.createElement('div');
        hint.id = 'ai-hint';
        hint.style.position = 'fixed';
        hint.style.top = '10px';
        hint.style.left = '50%';
        hint.style.transform = 'translateX(-50%)';
        hint.style.zIndex = '9999';
        hint.style.background = 'rgba(0,0,0,0.8)';
        hint.style.color = '#0f0';
        hint.style.padding = '10px 20px';
        hint.style.borderRadius = '5px';
        hint.style.fontSize = '24px';
        hint.style.fontWeight = 'bold';
        document.body.appendChild(hint);
    }
    hint.innerText = `AI Suggests: ${move}`;
}
