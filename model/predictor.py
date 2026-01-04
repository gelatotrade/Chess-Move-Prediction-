"""
Chess Move Predictor

High-level interface for chess move prediction using the trained neural network.
Provides easy-to-use methods for:
1. Getting best move recommendations
2. Evaluating position win probability
3. Analyzing multiple candidate moves
"""

import os
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
import chess

from chess_nn import ChessNet, create_model
from board_encoder import (
    BoardEncoder, MoveEncoder,
    get_board_encoder, get_move_encoder,
    PositionEvaluator
)


@dataclass
class MoveRecommendation:
    """A recommended chess move with analysis."""
    move: chess.Move
    uci: str
    san: str
    probability: float
    win_probability: float
    evaluation_change: float
    is_capture: bool
    is_check: bool
    piece_moved: str


@dataclass
class PositionAnalysis:
    """Complete analysis of a chess position."""
    fen: str
    turn: str
    win_probability: float
    top_moves: List[MoveRecommendation]
    material_balance: int
    is_check: bool
    is_checkmate: bool
    is_stalemate: bool
    phase: str  # 'opening', 'middlegame', 'endgame'


class ChessPredictor:
    """
    Main predictor class for chess move prediction.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = 'large',
        device: str = None
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model checkpoint
            model_type: Model type if creating new model
            device: Device to run on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load or create model
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path)
        else:
            self.model = create_model(model_type, self.device)

        self.model.eval()

        # Encoders
        self.board_encoder = get_board_encoder()
        self.move_encoder = get_move_encoder()
        self.position_evaluator = PositionEvaluator(
            self.board_encoder, self.move_encoder
        )

    def _load_model(self, path: str) -> ChessNet:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Get model config
        config = checkpoint.get('config', None)
        if config:
            model = create_model(config.model_type, self.device)
        else:
            model = create_model('large', self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def encode_board(self, board: chess.Board) -> torch.Tensor:
        """Encode board for neural network."""
        encoded = self.board_encoder.encode_board(board)
        tensor = torch.from_numpy(encoded).unsqueeze(0).to(self.device)
        return tensor

    @torch.no_grad()
    def predict(
        self,
        board: chess.Board,
        top_k: int = 5,
        temperature: float = 1.0
    ) -> Tuple[List[MoveRecommendation], float]:
        """
        Predict best moves for position.

        Args:
            board: Chess board position
            top_k: Number of moves to return
            temperature: Softmax temperature

        Returns:
            List of move recommendations and win probability
        """
        # Encode board
        board_tensor = self.encode_board(board)

        # Get model predictions
        policy, value = self.model(board_tensor)

        # Get legal moves mask
        legal_mask = torch.from_numpy(
            self.move_encoder.get_legal_move_mask(board)
        ).to(self.device)

        # Mask illegal moves
        policy = policy.squeeze()
        policy = policy.masked_fill(~legal_mask, float('-inf'))

        # Apply temperature and softmax
        probs = F.softmax(policy / temperature, dim=-1)

        # Get top moves
        top_probs, top_indices = torch.topk(probs, min(top_k, legal_mask.sum().item()))

        # Win probability (convert from [-1, 1] to [0, 1])
        win_prob = (value.item() + 1) / 2
        if board.turn == chess.BLACK:
            win_prob = 1 - win_prob

        # Build recommendations
        recommendations = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            move = self.move_encoder.decode_move(idx, board)
            if move is None:
                continue

            # Calculate evaluation change
            board.push(move)
            _, new_value = self.model(self.encode_board(board))
            board.pop()

            new_win_prob = (new_value.item() + 1) / 2
            if board.turn == chess.BLACK:
                new_win_prob = 1 - new_win_prob

            eval_change = new_win_prob - win_prob

            # Get piece moved
            piece = board.piece_at(move.from_square)
            piece_name = chess.piece_name(piece.piece_type) if piece else 'unknown'

            recommendations.append(MoveRecommendation(
                move=move,
                uci=move.uci(),
                san=board.san(move),
                probability=prob,
                win_probability=new_win_prob,
                evaluation_change=eval_change,
                is_capture=board.is_capture(move),
                is_check=board.gives_check(move),
                piece_moved=piece_name
            ))

        return recommendations, win_prob

    def analyze_position(self, board: chess.Board, top_k: int = 5) -> PositionAnalysis:
        """
        Perform complete analysis of a position.

        Args:
            board: Chess board position
            top_k: Number of moves to analyze

        Returns:
            Complete position analysis
        """
        # Get predictions
        recommendations, win_prob = self.predict(board, top_k)

        # Get position features
        features = self.position_evaluator.get_position_features(board)

        # Determine game phase
        if board.fullmove_number <= 10:
            phase = 'opening'
        elif features['is_endgame']:
            phase = 'endgame'
        else:
            phase = 'middlegame'

        return PositionAnalysis(
            fen=board.fen(),
            turn='white' if board.turn == chess.WHITE else 'black',
            win_probability=win_prob,
            top_moves=recommendations,
            material_balance=features['material_balance'],
            is_check=board.is_check(),
            is_checkmate=board.is_checkmate(),
            is_stalemate=board.is_stalemate(),
            phase=phase
        )

    def get_best_move(self, board: chess.Board) -> chess.Move:
        """Get the single best move for position."""
        recommendations, _ = self.predict(board, top_k=1)
        if recommendations:
            return recommendations[0].move

        # Fallback to random legal move
        legal_moves = list(board.legal_moves)
        if legal_moves:
            return legal_moves[0]
        return None

    def evaluate_move(
        self,
        board: chess.Board,
        move: chess.Move
    ) -> Dict[str, Any]:
        """
        Evaluate a specific move.

        Args:
            board: Current position
            move: Move to evaluate

        Returns:
            Evaluation dictionary
        """
        # Get current evaluation
        _, current_win_prob = self.predict(board, top_k=1)

        # Make move
        board.push(move)
        _, new_win_prob = self.predict(board, top_k=1)
        board.pop()

        # Get move probability from policy
        board_tensor = self.encode_board(board)
        policy, _ = self.model(board_tensor)

        move_idx = self.move_encoder.encode_move(move)
        legal_mask = torch.from_numpy(
            self.move_encoder.get_legal_move_mask(board)
        ).to(self.device)

        policy = policy.squeeze()
        policy = policy.masked_fill(~legal_mask, float('-inf'))
        probs = F.softmax(policy, dim=-1)
        move_prob = probs[move_idx].item()

        # Rank among all moves
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        rank = (sorted_indices == move_idx).nonzero().item() + 1

        return {
            'move': move.uci(),
            'san': board.san(move),
            'probability': move_prob,
            'rank': rank,
            'current_win_prob': current_win_prob,
            'new_win_prob': new_win_prob,
            'eval_change': new_win_prob - current_win_prob,
            'is_best': rank == 1
        }

    def compare_moves(
        self,
        board: chess.Board,
        moves: List[chess.Move]
    ) -> List[Dict[str, Any]]:
        """Compare multiple moves."""
        return [self.evaluate_move(board, move) for move in moves]

    def suggest_continuation(
        self,
        board: chess.Board,
        depth: int = 5
    ) -> List[chess.Move]:
        """
        Suggest a continuation line from current position.

        Args:
            board: Current position
            depth: Number of moves to look ahead

        Returns:
            List of suggested moves
        """
        continuation = []
        temp_board = board.copy()

        for _ in range(depth):
            if temp_board.is_game_over():
                break

            move = self.get_best_move(temp_board)
            if move is None:
                break

            continuation.append(move)
            temp_board.push(move)

        return continuation


class PositionTracker:
    """
    Tracks position history and probability changes during a game.
    """

    def __init__(self, predictor: ChessPredictor):
        self.predictor = predictor
        self.history = []
        self.board = chess.Board()

    def reset(self, fen: str = chess.STARTING_FEN):
        """Reset to new game."""
        self.history = []
        self.board = chess.Board(fen)

    def update(self, move: chess.Move) -> Dict[str, Any]:
        """
        Update with a new move and return analysis.

        Args:
            move: Move that was played

        Returns:
            Analysis including probability changes
        """
        # Get pre-move analysis
        pre_analysis = self.predictor.analyze_position(self.board)
        move_eval = self.predictor.evaluate_move(self.board, move)

        # Make move
        san = self.board.san(move)
        self.board.push(move)

        # Get post-move analysis
        post_analysis = self.predictor.analyze_position(self.board)

        # Record in history
        entry = {
            'move_number': self.board.fullmove_number,
            'side': 'black' if self.board.turn == chess.WHITE else 'white',  # Just played
            'move_uci': move.uci(),
            'move_san': san,
            'move_probability': move_eval['probability'],
            'move_rank': move_eval['rank'],
            'win_prob_before': pre_analysis.win_probability,
            'win_prob_after': post_analysis.win_probability,
            'eval_change': move_eval['eval_change'],
            'was_best_move': move_eval['is_best'],
            'best_move_was': pre_analysis.top_moves[0].san if pre_analysis.top_moves else None
        }

        self.history.append(entry)

        return {
            'entry': entry,
            'current_analysis': post_analysis
        }

    def get_win_probability_curve(self) -> List[float]:
        """Get win probability over the game."""
        if not self.history:
            return []

        curve = [self.history[0]['win_prob_before']]
        for entry in self.history:
            curve.append(entry['win_prob_after'])

        return curve

    def get_accuracy(self) -> Dict[str, float]:
        """Calculate accuracy for each side."""
        white_correct = 0
        white_total = 0
        black_correct = 0
        black_total = 0

        for entry in self.history:
            if entry['side'] == 'white':
                white_total += 1
                if entry['was_best_move']:
                    white_correct += 1
            else:
                black_total += 1
                if entry['was_best_move']:
                    black_correct += 1

        return {
            'white_accuracy': white_correct / white_total if white_total else 0,
            'black_accuracy': black_correct / black_total if black_total else 0,
            'white_moves': white_total,
            'black_moves': black_total
        }


def create_predictor(model_path: Optional[str] = None) -> ChessPredictor:
    """
    Factory function to create a chess predictor.

    Args:
        model_path: Optional path to trained model

    Returns:
        ChessPredictor instance
    """
    return ChessPredictor(model_path=model_path)


# Convenience functions for quick predictions
_default_predictor = None


def get_default_predictor() -> ChessPredictor:
    """Get or create default predictor."""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = ChessPredictor()
    return _default_predictor


def predict_move(fen: str, top_k: int = 5) -> List[Dict]:
    """
    Quick prediction from FEN string.

    Args:
        fen: FEN notation
        top_k: Number of moves to return

    Returns:
        List of move recommendations as dictionaries
    """
    predictor = get_default_predictor()
    board = chess.Board(fen)
    recommendations, win_prob = predictor.predict(board, top_k)

    return {
        'fen': fen,
        'win_probability': win_prob,
        'moves': [
            {
                'move': r.uci,
                'san': r.san,
                'probability': r.probability,
                'win_probability': r.win_probability,
                'eval_change': r.evaluation_change
            }
            for r in recommendations
        ]
    }


if __name__ == '__main__':
    # Demo
    predictor = ChessPredictor()
    board = chess.Board()

    print("Analyzing starting position...")
    analysis = predictor.analyze_position(board)

    print(f"\nPosition: {analysis.fen}")
    print(f"Turn: {analysis.turn}")
    print(f"Win probability: {analysis.win_probability:.2%}")
    print(f"Phase: {analysis.phase}")
    print(f"\nTop moves:")

    for i, move in enumerate(analysis.top_moves, 1):
        print(f"  {i}. {move.san} ({move.probability:.1%}) - "
              f"Win: {move.win_probability:.1%}, "
              f"Change: {move.evaluation_change:+.1%}")
