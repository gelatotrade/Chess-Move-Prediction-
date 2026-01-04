"""
Chess Move Prediction Model Package

This package provides a neural network-based chess move prediction system
with the following components:

- chess_nn: Neural network architecture (CNN + Transformer)
- board_encoder: Board state encoding utilities
- predictor: High-level prediction interface
- training: Training pipeline and utilities
"""

from .chess_nn import ChessNet, create_model
from .board_encoder import (
    BoardEncoder,
    MoveEncoder,
    PositionEvaluator,
    get_board_encoder,
    get_move_encoder,
    encode_fen
)
from .predictor import (
    ChessPredictor,
    PositionTracker,
    MoveRecommendation,
    PositionAnalysis,
    create_predictor,
    predict_move
)

__all__ = [
    'ChessNet',
    'create_model',
    'BoardEncoder',
    'MoveEncoder',
    'PositionEvaluator',
    'get_board_encoder',
    'get_move_encoder',
    'encode_fen',
    'ChessPredictor',
    'PositionTracker',
    'MoveRecommendation',
    'PositionAnalysis',
    'create_predictor',
    'predict_move'
]
