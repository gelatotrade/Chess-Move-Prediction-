"""
Chess Move Prediction API Server

FastAPI-based backend server for the Chrome extension.
Provides REST API endpoints for:
1. Move prediction
2. Position analysis
3. Game tracking
4. Model information
"""

import os
import sys
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import chess

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

from predictor import ChessPredictor, PositionTracker, create_predictor
from board_encoder import get_move_encoder, PositionEvaluator, get_board_encoder


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global predictor instance
predictor: Optional[ChessPredictor] = None
active_trackers: Dict[str, PositionTracker] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global predictor

    logger.info("Initializing chess prediction model...")
    predictor = create_predictor()
    logger.info("Model loaded successfully!")

    yield

    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Chess Move Prediction API",
    description="Neural network-based chess move prediction service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for Chrome extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for extension
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============= Request/Response Models =============

class PredictionRequest(BaseModel):
    """Request for move prediction."""
    fen: str = Field(..., description="FEN notation of current position")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of moves to return")
    temperature: float = Field(default=1.0, ge=0.1, le=2.0, description="Softmax temperature")


class MoveInfo(BaseModel):
    """Information about a recommended move."""
    move: str
    san: str
    probability: float
    win_probability: float
    eval_change: float
    is_capture: bool
    is_check: bool
    piece_moved: str


class PredictionResponse(BaseModel):
    """Response from move prediction."""
    fen: str
    turn: str
    win_probability: float
    top_moves: List[MoveInfo]
    material_balance: int
    is_check: bool
    phase: str


class EvaluateMoveRequest(BaseModel):
    """Request to evaluate a specific move."""
    fen: str
    move: str  # UCI format (e.g., "e2e4")


class MoveEvaluationResponse(BaseModel):
    """Response from move evaluation."""
    move: str
    san: str
    probability: float
    rank: int
    current_win_prob: float
    new_win_prob: float
    eval_change: float
    is_best: bool


class GameUpdateRequest(BaseModel):
    """Request to update game state."""
    game_id: str
    move: str  # UCI format


class GameUpdateResponse(BaseModel):
    """Response from game update."""
    move_number: int
    move_san: str
    move_probability: float
    move_rank: int
    was_best_move: bool
    win_prob_before: float
    win_prob_after: float
    eval_change: float
    best_move_was: Optional[str]
    current_analysis: PredictionResponse


class NewGameRequest(BaseModel):
    """Request to start tracking a new game."""
    game_id: str
    fen: str = Field(default=chess.STARTING_FEN)


class GameHistoryResponse(BaseModel):
    """Game history and statistics."""
    game_id: str
    moves: List[Dict[str, Any]]
    win_probability_curve: List[float]
    white_accuracy: float
    black_accuracy: float


class ContinuationRequest(BaseModel):
    """Request for move continuation."""
    fen: str
    depth: int = Field(default=5, ge=1, le=10)


class ContinuationResponse(BaseModel):
    """Continuation line response."""
    moves: List[str]
    final_win_probability: float


# ============= API Endpoints =============

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "name": "Chess Move Prediction API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_moves(request: PredictionRequest):
    """
    Predict best moves for a position.

    Args:
        request: Prediction request with FEN

    Returns:
        Top move recommendations with probabilities
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        board = chess.Board(request.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {e}")

    if board.is_game_over():
        return PredictionResponse(
            fen=request.fen,
            turn='white' if board.turn == chess.WHITE else 'black',
            win_probability=0.5,
            top_moves=[],
            material_balance=0,
            is_check=False,
            phase='endgame'
        )

    try:
        analysis = predictor.analyze_position(board, request.top_k)

        return PredictionResponse(
            fen=analysis.fen,
            turn=analysis.turn,
            win_probability=analysis.win_probability,
            top_moves=[
                MoveInfo(
                    move=m.uci,
                    san=m.san,
                    probability=m.probability,
                    win_probability=m.win_probability,
                    eval_change=m.evaluation_change,
                    is_capture=m.is_capture,
                    is_check=m.is_check,
                    piece_moved=m.piece_moved
                )
                for m in analysis.top_moves
            ],
            material_balance=analysis.material_balance,
            is_check=analysis.is_check,
            phase=analysis.phase
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=MoveEvaluationResponse)
async def evaluate_move(request: EvaluateMoveRequest):
    """
    Evaluate a specific move.

    Args:
        request: Move evaluation request

    Returns:
        Move evaluation with rank and probability
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        board = chess.Board(request.fen)
        move = chess.Move.from_uci(request.move)

        if move not in board.legal_moves:
            raise HTTPException(status_code=400, detail="Illegal move")

        eval_result = predictor.evaluate_move(board, move)

        return MoveEvaluationResponse(**eval_result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/game/new")
async def new_game(request: NewGameRequest):
    """
    Start tracking a new game.

    Args:
        request: New game request with ID and initial FEN

    Returns:
        Confirmation and initial analysis
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    tracker = PositionTracker(predictor)
    tracker.reset(request.fen)
    active_trackers[request.game_id] = tracker

    # Get initial analysis
    analysis = predictor.analyze_position(tracker.board)

    return {
        "game_id": request.game_id,
        "fen": request.fen,
        "initial_analysis": PredictionResponse(
            fen=analysis.fen,
            turn=analysis.turn,
            win_probability=analysis.win_probability,
            top_moves=[
                MoveInfo(
                    move=m.uci,
                    san=m.san,
                    probability=m.probability,
                    win_probability=m.win_probability,
                    eval_change=m.evaluation_change,
                    is_capture=m.is_capture,
                    is_check=m.is_check,
                    piece_moved=m.piece_moved
                )
                for m in analysis.top_moves
            ],
            material_balance=analysis.material_balance,
            is_check=analysis.is_check,
            phase=analysis.phase
        )
    }


@app.post("/game/update", response_model=GameUpdateResponse)
async def update_game(request: GameUpdateRequest):
    """
    Update game with a new move.

    Args:
        request: Game update request

    Returns:
        Move analysis and updated position
    """
    if request.game_id not in active_trackers:
        raise HTTPException(status_code=404, detail="Game not found")

    tracker = active_trackers[request.game_id]

    try:
        move = chess.Move.from_uci(request.move)
        if move not in tracker.board.legal_moves:
            raise HTTPException(status_code=400, detail="Illegal move")

        result = tracker.update(move)
        entry = result['entry']
        analysis = result['current_analysis']

        return GameUpdateResponse(
            move_number=entry['move_number'],
            move_san=entry['move_san'],
            move_probability=entry['move_probability'],
            move_rank=entry['move_rank'],
            was_best_move=entry['was_best_move'],
            win_prob_before=entry['win_prob_before'],
            win_prob_after=entry['win_prob_after'],
            eval_change=entry['eval_change'],
            best_move_was=entry['best_move_was'],
            current_analysis=PredictionResponse(
                fen=analysis.fen,
                turn=analysis.turn,
                win_probability=analysis.win_probability,
                top_moves=[
                    MoveInfo(
                        move=m.uci,
                        san=m.san,
                        probability=m.probability,
                        win_probability=m.win_probability,
                        eval_change=m.evaluation_change,
                        is_capture=m.is_capture,
                        is_check=m.is_check,
                        piece_moved=m.piece_moved
                    )
                    for m in analysis.top_moves
                ],
                material_balance=analysis.material_balance,
                is_check=analysis.is_check,
                phase=analysis.phase
            )
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/game/{game_id}/history", response_model=GameHistoryResponse)
async def get_game_history(game_id: str):
    """
    Get full game history and statistics.

    Args:
        game_id: Game identifier

    Returns:
        Game history with statistics
    """
    if game_id not in active_trackers:
        raise HTTPException(status_code=404, detail="Game not found")

    tracker = active_trackers[game_id]
    accuracy = tracker.get_accuracy()

    return GameHistoryResponse(
        game_id=game_id,
        moves=tracker.history,
        win_probability_curve=tracker.get_win_probability_curve(),
        white_accuracy=accuracy['white_accuracy'],
        black_accuracy=accuracy['black_accuracy']
    )


@app.delete("/game/{game_id}")
async def delete_game(game_id: str):
    """Delete a tracked game."""
    if game_id in active_trackers:
        del active_trackers[game_id]
    return {"status": "deleted", "game_id": game_id}


@app.post("/continuation", response_model=ContinuationResponse)
async def get_continuation(request: ContinuationRequest):
    """
    Get a suggested continuation line.

    Args:
        request: Continuation request

    Returns:
        List of suggested moves
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        board = chess.Board(request.fen)
        continuation = predictor.suggest_continuation(board, request.depth)

        # Get final position win probability
        for move in continuation:
            board.push(move)
        _, final_win_prob = predictor.predict(board, top_k=1)

        return ContinuationResponse(
            moves=[m.uci() for m in continuation],
            final_win_probability=final_win_prob
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============= WebSocket for Real-time Updates =============

class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, game_id: str):
        await websocket.accept()
        if game_id not in self.active_connections:
            self.active_connections[game_id] = []
        self.active_connections[game_id].append(websocket)

    def disconnect(self, websocket: WebSocket, game_id: str):
        if game_id in self.active_connections:
            self.active_connections[game_id].remove(websocket)
            if not self.active_connections[game_id]:
                del self.active_connections[game_id]

    async def broadcast(self, game_id: str, message: dict):
        if game_id in self.active_connections:
            for connection in self.active_connections[game_id]:
                await connection.send_json(message)


manager = ConnectionManager()


@app.websocket("/ws/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    """
    WebSocket endpoint for real-time game updates.

    Clients can connect to receive live move predictions and updates.
    """
    await manager.connect(websocket, game_id)

    try:
        while True:
            data = await websocket.receive_json()

            if data.get('type') == 'predict':
                fen = data.get('fen')
                if fen and predictor:
                    try:
                        board = chess.Board(fen)
                        analysis = predictor.analyze_position(board)

                        await websocket.send_json({
                            'type': 'prediction',
                            'fen': fen,
                            'win_probability': analysis.win_probability,
                            'top_moves': [
                                {
                                    'move': m.uci,
                                    'san': m.san,
                                    'probability': m.probability,
                                    'win_probability': m.win_probability
                                }
                                for m in analysis.top_moves[:5]
                            ]
                        })
                    except Exception as e:
                        await websocket.send_json({
                            'type': 'error',
                            'message': str(e)
                        })

            elif data.get('type') == 'update_move':
                move = data.get('move')
                if game_id in active_trackers and move:
                    try:
                        tracker = active_trackers[game_id]
                        chess_move = chess.Move.from_uci(move)
                        result = tracker.update(chess_move)

                        # Broadcast to all connected clients
                        await manager.broadcast(game_id, {
                            'type': 'move_update',
                            'move': move,
                            'result': result['entry'],
                            'current_fen': tracker.board.fen()
                        })
                    except Exception as e:
                        await websocket.send_json({
                            'type': 'error',
                            'message': str(e)
                        })

    except WebSocketDisconnect:
        manager.disconnect(websocket, game_id)


# ============= Main Entry Point =============

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chess Move Prediction API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    if args.reload:
        uvicorn.run("app:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)
