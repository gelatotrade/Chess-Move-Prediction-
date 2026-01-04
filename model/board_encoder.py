"""
Chess Board Encoder

This module handles the conversion between chess board states (FEN notation,
python-chess boards) and tensor representations for the neural network.

The encoding uses 19 planes of 8x8:
- 12 planes: Piece positions (6 pieces x 2 colors)
- 1 plane: All white pieces
- 1 plane: All black pieces
- 1 plane: Turn indicator (all 1s if white to move)
- 2 planes: Castling rights (kingside, queenside)
- 1 plane: En passant square
- 1 plane: Fifty-move counter (normalized)
"""

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
import chess


# Piece to index mapping
PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

# Move encoding: from_square * 73 + move_type
# Move types: 56 queen moves + 8 knight moves + 9 underpromotions = 73
MOVE_TYPES = 73
NUM_SQUARES = 64
TOTAL_MOVES = 4672


class BoardEncoder:
    """
    Encodes chess board states into tensor format for neural network input.
    """

    def __init__(self, history_length: int = 8):
        """
        Initialize encoder.

        Args:
            history_length: Number of previous positions to include
        """
        self.history_length = history_length
        self.num_planes = 19  # Base planes per position

    def encode_board(self, board: chess.Board) -> np.ndarray:
        """
        Encode a single board position into a tensor.

        Args:
            board: python-chess Board object

        Returns:
            numpy array of shape (19, 8, 8)
        """
        planes = np.zeros((19, 8, 8), dtype=np.float32)

        # Encode pieces for both colors
        for piece_type in chess.PIECE_TYPES:
            # White pieces
            for square in board.pieces(piece_type, chess.WHITE):
                rank, file = divmod(square, 8)
                planes[PIECE_TO_INDEX[piece_type], rank, file] = 1.0

            # Black pieces
            for square in board.pieces(piece_type, chess.BLACK):
                rank, file = divmod(square, 8)
                planes[PIECE_TO_INDEX[piece_type] + 6, rank, file] = 1.0

        # All white pieces
        for square in board.occupied_co[chess.WHITE]:
            rank, file = divmod(square, 8)
            planes[12, rank, file] = 1.0

        # All black pieces
        for square in board.occupied_co[chess.BLACK]:
            rank, file = divmod(square, 8)
            planes[13, rank, file] = 1.0

        # Turn indicator
        if board.turn == chess.WHITE:
            planes[14] = 1.0

        # Castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            planes[15, 0, 4:8] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            planes[15, 0, 0:5] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            planes[16, 7, 4:8] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            planes[16, 7, 0:5] = 1.0

        # En passant
        if board.ep_square is not None:
            rank, file = divmod(board.ep_square, 8)
            planes[17, rank, file] = 1.0

        # Fifty-move counter (normalized)
        planes[18] = board.halfmove_clock / 100.0

        return planes

    def encode_board_history(
        self,
        board: chess.Board,
        history: List[chess.Board]
    ) -> np.ndarray:
        """
        Encode current board with move history.

        Args:
            board: Current position
            history: List of previous board states

        Returns:
            numpy array with history planes
        """
        # Start with current position
        all_planes = [self.encode_board(board)]

        # Add history (most recent first)
        for hist_board in reversed(history[-self.history_length:]):
            all_planes.append(self.encode_board(hist_board))

        # Pad if needed
        while len(all_planes) < self.history_length + 1:
            all_planes.append(np.zeros((19, 8, 8), dtype=np.float32))

        return np.concatenate(all_planes, axis=0)

    def encode_batch(self, boards: List[chess.Board]) -> torch.Tensor:
        """
        Encode a batch of boards.

        Args:
            boards: List of Board objects

        Returns:
            Tensor of shape (batch_size, 19, 8, 8)
        """
        encoded = np.stack([self.encode_board(b) for b in boards])
        return torch.from_numpy(encoded)


class MoveEncoder:
    """
    Handles encoding and decoding of chess moves.

    Encoding scheme:
    - From square: 0-63
    - Move type: 0-72 (56 queen moves + 8 knight moves + 9 underpromotions)
    - Total: 64 * 73 = 4672 possible moves
    """

    # Direction vectors for queen-like moves
    QUEEN_DIRECTIONS = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1)
    ]

    # Knight move offsets
    KNIGHT_MOVES = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]

    # Underpromotion pieces
    UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

    def __init__(self):
        self._build_move_tables()

    def _build_move_tables(self):
        """Build lookup tables for move encoding/decoding."""
        self.move_to_index = {}
        self.index_to_move = {}

        idx = 0

        for from_sq in range(64):
            from_rank, from_file = divmod(from_sq, 8)

            # Queen-like moves (56 types: 8 directions x 7 max distance)
            for dir_idx, (dr, df) in enumerate(self.QUEEN_DIRECTIONS):
                for dist in range(1, 8):
                    to_rank = from_rank + dr * dist
                    to_file = from_file + df * dist

                    if 0 <= to_rank < 8 and 0 <= to_file < 8:
                        to_sq = to_rank * 8 + to_file
                        move_type = dir_idx * 7 + (dist - 1)
                        move_idx = from_sq * MOVE_TYPES + move_type

                        # Regular move
                        key = (from_sq, to_sq, None)
                        self.move_to_index[key] = move_idx
                        self.index_to_move[move_idx] = key

                        # Queen promotion
                        if (from_rank == 6 and to_rank == 7) or (from_rank == 1 and to_rank == 0):
                            key_promo = (from_sq, to_sq, chess.QUEEN)
                            self.move_to_index[key_promo] = move_idx

            # Knight moves (8 types)
            for knight_idx, (dr, df) in enumerate(self.KNIGHT_MOVES):
                to_rank = from_rank + dr
                to_file = from_file + df

                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_sq = to_rank * 8 + to_file
                    move_type = 56 + knight_idx
                    move_idx = from_sq * MOVE_TYPES + move_type

                    key = (from_sq, to_sq, None)
                    self.move_to_index[key] = move_idx
                    self.index_to_move[move_idx] = key

            # Underpromotions (9 types: 3 pieces x 3 directions)
            if from_rank == 6 or from_rank == 1:  # Pawn on 7th/2nd rank
                for promo_idx, piece in enumerate(self.UNDERPROMOTIONS):
                    for dir_idx, df in enumerate([-1, 0, 1]):  # Capture left, push, capture right
                        to_rank = 7 if from_rank == 6 else 0
                        to_file = from_file + df

                        if 0 <= to_file < 8:
                            to_sq = to_rank * 8 + to_file
                            move_type = 64 + promo_idx * 3 + dir_idx
                            move_idx = from_sq * MOVE_TYPES + move_type

                            key = (from_sq, to_sq, piece)
                            self.move_to_index[key] = move_idx
                            self.index_to_move[move_idx] = key

    def encode_move(self, move: chess.Move) -> int:
        """
        Encode a chess move to an index.

        Args:
            move: python-chess Move object

        Returns:
            Move index (0-4671)
        """
        key = (move.from_square, move.to_square, move.promotion)

        if key in self.move_to_index:
            return self.move_to_index[key]

        # Try without promotion
        key_no_promo = (move.from_square, move.to_square, None)
        return self.move_to_index.get(key_no_promo, 0)

    def decode_move(self, index: int, board: chess.Board) -> Optional[chess.Move]:
        """
        Decode a move index to a chess move.

        Args:
            index: Move index
            board: Current board (to validate move)

        Returns:
            Chess move or None if invalid
        """
        if index not in self.index_to_move:
            return None

        from_sq, to_sq, promotion = self.index_to_move[index]

        # Check for pawn promotion
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = to_sq // 8
            if to_rank == 0 or to_rank == 7:
                if promotion is None:
                    promotion = chess.QUEEN

        move = chess.Move(from_sq, to_sq, promotion=promotion)

        if move in board.legal_moves:
            return move

        return None

    def get_legal_move_mask(self, board: chess.Board) -> np.ndarray:
        """
        Get a boolean mask of legal moves.

        Args:
            board: Current board

        Returns:
            Boolean array of shape (4672,)
        """
        mask = np.zeros(TOTAL_MOVES, dtype=bool)

        for move in board.legal_moves:
            idx = self.encode_move(move)
            mask[idx] = True

        return mask

    def get_top_moves(
        self,
        board: chess.Board,
        policy: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[chess.Move, float]]:
        """
        Get top K legal moves with their probabilities.

        Args:
            board: Current board
            policy: Policy distribution from network
            top_k: Number of moves to return

        Returns:
            List of (move, probability) tuples
        """
        legal_mask = self.get_legal_move_mask(board)

        # Mask illegal moves
        masked_policy = policy.copy()
        masked_policy[~legal_mask] = -np.inf

        # Softmax over legal moves
        exp_policy = np.exp(masked_policy - np.max(masked_policy))
        probs = exp_policy / exp_policy.sum()

        # Get top K
        top_indices = np.argsort(probs)[-top_k:][::-1]

        result = []
        for idx in top_indices:
            if legal_mask[idx]:
                move = self.decode_move(idx, board)
                if move:
                    result.append((move, probs[idx]))

        return result


class PositionEvaluator:
    """
    Evaluates chess positions and provides analysis.
    """

    def __init__(self, board_encoder: BoardEncoder, move_encoder: MoveEncoder):
        self.board_encoder = board_encoder
        self.move_encoder = move_encoder

    def get_material_balance(self, board: chess.Board) -> int:
        """Calculate material balance (positive = white advantage)."""
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }

        balance = 0
        for piece_type in chess.PIECE_TYPES:
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))
            balance += piece_values[piece_type] * (white_count - black_count)

        return balance

    def get_position_features(self, board: chess.Board) -> Dict:
        """
        Extract high-level features from position.

        Returns:
            Dictionary of position features
        """
        features = {
            'material_balance': self.get_material_balance(board),
            'white_king_safety': self._evaluate_king_safety(board, chess.WHITE),
            'black_king_safety': self._evaluate_king_safety(board, chess.BLACK),
            'center_control': self._evaluate_center_control(board),
            'development': self._evaluate_development(board),
            'pawn_structure': self._evaluate_pawn_structure(board),
            'is_check': board.is_check(),
            'is_endgame': self._is_endgame(board),
            'legal_moves_count': len(list(board.legal_moves))
        }

        return features

    def _evaluate_king_safety(self, board: chess.Board, color: chess.Color) -> float:
        """Evaluate king safety for given color."""
        king_square = board.king(color)
        if king_square is None:
            return 0.0

        # Count pawns near king
        king_rank, king_file = divmod(king_square, 8)
        pawn_shield = 0

        for df in [-1, 0, 1]:
            if 0 <= king_file + df < 8:
                check_rank = king_rank + (1 if color == chess.WHITE else -1)
                if 0 <= check_rank < 8:
                    square = check_rank * 8 + king_file + df
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        pawn_shield += 1

        return pawn_shield / 3.0

    def _evaluate_center_control(self, board: chess.Board) -> float:
        """Evaluate control of center squares."""
        center_squares = [27, 28, 35, 36]  # d4, e4, d5, e5
        extended_center = [18, 19, 20, 21, 26, 29, 34, 37, 42, 43, 44, 45]

        white_control = 0
        black_control = 0

        for square in center_squares:
            white_control += len(board.attackers(chess.WHITE, square))
            black_control += len(board.attackers(chess.BLACK, square))

        for square in extended_center:
            white_control += 0.5 * len(board.attackers(chess.WHITE, square))
            black_control += 0.5 * len(board.attackers(chess.BLACK, square))

        total = white_control + black_control
        if total == 0:
            return 0.0

        return (white_control - black_control) / total

    def _evaluate_development(self, board: chess.Board) -> float:
        """Evaluate piece development."""
        white_dev = 0
        black_dev = 0

        # Knights developed
        white_dev += 2 - len(board.pieces(chess.KNIGHT, chess.WHITE) & chess.BB_RANK_1)
        black_dev += 2 - len(board.pieces(chess.KNIGHT, chess.BLACK) & chess.BB_RANK_8)

        # Bishops developed
        white_dev += 2 - len(board.pieces(chess.BISHOP, chess.WHITE) & chess.BB_RANK_1)
        black_dev += 2 - len(board.pieces(chess.BISHOP, chess.BLACK) & chess.BB_RANK_8)

        return (white_dev - black_dev) / 8.0

    def _evaluate_pawn_structure(self, board: chess.Board) -> float:
        """Evaluate pawn structure quality."""
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)

        white_score = len(white_pawns)
        black_score = len(black_pawns)

        # Penalize doubled pawns
        for file in range(8):
            file_mask = chess.BB_FILES[file]
            white_on_file = len(white_pawns & file_mask)
            black_on_file = len(black_pawns & file_mask)

            if white_on_file > 1:
                white_score -= (white_on_file - 1) * 0.5
            if black_on_file > 1:
                black_score -= (black_on_file - 1) * 0.5

        return (white_score - black_score) / 8.0

    def _is_endgame(self, board: chess.Board) -> bool:
        """Determine if position is an endgame."""
        # No queens or minor pieces limited
        if not board.pieces(chess.QUEEN, chess.WHITE) and not board.pieces(chess.QUEEN, chess.BLACK):
            return True

        # Count total material
        total_pieces = (
            len(board.pieces(chess.KNIGHT, chess.WHITE)) +
            len(board.pieces(chess.KNIGHT, chess.BLACK)) +
            len(board.pieces(chess.BISHOP, chess.WHITE)) +
            len(board.pieces(chess.BISHOP, chess.BLACK)) +
            len(board.pieces(chess.ROOK, chess.WHITE)) +
            len(board.pieces(chess.ROOK, chess.BLACK)) +
            len(board.pieces(chess.QUEEN, chess.WHITE)) +
            len(board.pieces(chess.QUEEN, chess.BLACK))
        )

        return total_pieces <= 6


# Singleton instances
_board_encoder = None
_move_encoder = None


def get_board_encoder() -> BoardEncoder:
    """Get singleton board encoder."""
    global _board_encoder
    if _board_encoder is None:
        _board_encoder = BoardEncoder()
    return _board_encoder


def get_move_encoder() -> MoveEncoder:
    """Get singleton move encoder."""
    global _move_encoder
    if _move_encoder is None:
        _move_encoder = MoveEncoder()
    return _move_encoder


def encode_fen(fen: str) -> torch.Tensor:
    """
    Convenience function to encode a FEN string.

    Args:
        fen: FEN notation string

    Returns:
        Tensor of shape (1, 19, 8, 8)
    """
    board = chess.Board(fen)
    encoder = get_board_encoder()
    encoded = encoder.encode_board(board)
    return torch.from_numpy(encoded).unsqueeze(0)


if __name__ == '__main__':
    # Test encoding
    board = chess.Board()
    encoder = BoardEncoder()
    move_encoder = MoveEncoder()

    # Encode starting position
    encoded = encoder.encode_board(board)
    print(f"Encoded shape: {encoded.shape}")
    print(f"Non-zero elements: {np.count_nonzero(encoded)}")

    # Test move encoding
    for move in list(board.legal_moves)[:5]:
        idx = move_encoder.encode_move(move)
        decoded = move_encoder.decode_move(idx, board)
        print(f"Move: {move}, Index: {idx}, Decoded: {decoded}")

    # Test legal move mask
    mask = move_encoder.get_legal_move_mask(board)
    print(f"Legal moves: {np.sum(mask)}")
