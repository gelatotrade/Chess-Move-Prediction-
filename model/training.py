"""
Chess Neural Network Training Pipeline

This module provides a complete training pipeline for the chess move prediction model.
It handles:
1. Data loading from PGN files
2. Preprocessing and augmentation
3. Training with mixed precision
4. Evaluation and validation
5. Model checkpointing
"""

import os
import io
import random
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Generator
from dataclasses import dataclass
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler
import chess
import chess.pgn

from chess_nn import ChessNet, create_model
from board_encoder import BoardEncoder, MoveEncoder, get_board_encoder, get_move_encoder


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    # Model settings
    model_type: str = 'large'
    num_blocks: int = 20
    channels: int = 256

    # Training settings
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000

    # Policy loss weight
    policy_weight: float = 1.0
    value_weight: float = 1.0

    # Data settings
    train_data_path: str = 'data/train'
    val_data_path: str = 'data/val'
    num_workers: int = 4

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_every: int = 1000
    log_every: int = 100

    # Mixed precision
    use_amp: bool = True


class ChessDataset(Dataset):
    """
    Dataset for loading chess positions from PGN files.
    """

    def __init__(
        self,
        pgn_files: List[str],
        board_encoder: BoardEncoder,
        move_encoder: MoveEncoder,
        positions_per_game: int = 10,
        max_positions: Optional[int] = None
    ):
        """
        Initialize dataset.

        Args:
            pgn_files: List of PGN file paths
            board_encoder: Board encoder instance
            move_encoder: Move encoder instance
            positions_per_game: Number of positions to sample per game
            max_positions: Maximum total positions to load
        """
        self.board_encoder = board_encoder
        self.move_encoder = move_encoder
        self.positions_per_game = positions_per_game
        self.max_positions = max_positions

        self.positions = []
        self._load_pgn_files(pgn_files)

    def _load_pgn_files(self, pgn_files: List[str]):
        """Load positions from PGN files."""
        logger.info(f"Loading {len(pgn_files)} PGN files...")

        for pgn_file in pgn_files:
            if self.max_positions and len(self.positions) >= self.max_positions:
                break

            try:
                with open(pgn_file, 'r') as f:
                    while True:
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break

                        self._process_game(game)

                        if self.max_positions and len(self.positions) >= self.max_positions:
                            break

            except Exception as e:
                logger.warning(f"Error loading {pgn_file}: {e}")

        logger.info(f"Loaded {len(self.positions)} positions")

    def _process_game(self, game: chess.pgn.Game):
        """Extract training positions from a game."""
        result = game.headers.get('Result', '*')

        # Determine game result
        if result == '1-0':
            white_value = 1.0
        elif result == '0-1':
            white_value = -1.0
        elif result == '1/2-1/2':
            white_value = 0.0
        else:
            return  # Unknown result

        board = game.board()
        moves = list(game.mainline_moves())

        if len(moves) < 10:
            return  # Skip very short games

        # Sample positions
        sample_indices = random.sample(
            range(len(moves)),
            min(self.positions_per_game, len(moves))
        )

        for idx in sample_indices:
            # Replay to position
            temp_board = game.board()
            for i, move in enumerate(moves):
                if i == idx:
                    # Get value for current player
                    value = white_value if temp_board.turn == chess.WHITE else -white_value

                    self.positions.append({
                        'board': temp_board.copy(),
                        'move': move,
                        'value': value
                    })
                    break
                temp_board.push(move)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            board_tensor: Encoded board state
            move_idx: Target move index
            value: Game outcome value
        """
        pos = self.positions[idx]

        board_tensor = self.board_encoder.encode_board(pos['board'])
        move_idx = self.move_encoder.encode_move(pos['move'])
        value = pos['value']

        return (
            torch.from_numpy(board_tensor),
            torch.tensor(move_idx, dtype=torch.long),
            torch.tensor(value, dtype=torch.float32)
        )


class StreamingChessDataset(IterableDataset):
    """
    Streaming dataset for large PGN files.
    More memory efficient for training on large datasets.
    """

    def __init__(
        self,
        pgn_files: List[str],
        board_encoder: BoardEncoder,
        move_encoder: MoveEncoder,
        buffer_size: int = 10000
    ):
        self.pgn_files = pgn_files
        self.board_encoder = board_encoder
        self.move_encoder = move_encoder
        self.buffer_size = buffer_size

    def __iter__(self) -> Generator:
        buffer = deque(maxlen=self.buffer_size)

        for pgn_file in self.pgn_files:
            try:
                with open(pgn_file, 'r') as f:
                    while True:
                        game = chess.pgn.read_game(f)
                        if game is None:
                            break

                        # Add positions to buffer
                        positions = self._extract_positions(game)
                        buffer.extend(positions)

                        # Yield shuffled samples
                        while len(buffer) > self.buffer_size // 2:
                            idx = random.randint(0, len(buffer) - 1)
                            pos = buffer[idx]
                            del buffer[idx]
                            yield self._encode_position(pos)

            except Exception as e:
                logger.warning(f"Error in streaming {pgn_file}: {e}")

        # Yield remaining samples
        while buffer:
            pos = buffer.popleft()
            yield self._encode_position(pos)

    def _extract_positions(self, game: chess.pgn.Game) -> List[Dict]:
        """Extract positions from game."""
        result = game.headers.get('Result', '*')

        if result == '1-0':
            white_value = 1.0
        elif result == '0-1':
            white_value = -1.0
        elif result == '1/2-1/2':
            white_value = 0.0
        else:
            return []

        positions = []
        board = game.board()

        for move in game.mainline_moves():
            if random.random() < 0.1:  # Sample 10% of positions
                value = white_value if board.turn == chess.WHITE else -white_value
                positions.append({
                    'board': board.copy(),
                    'move': move,
                    'value': value
                })
            board.push(move)

        return positions

    def _encode_position(self, pos: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a position."""
        board_tensor = self.board_encoder.encode_board(pos['board'])
        move_idx = self.move_encoder.encode_move(pos['move'])

        return (
            torch.from_numpy(board_tensor),
            torch.tensor(move_idx, dtype=torch.long),
            torch.tensor(pos['value'], dtype=torch.float32)
        )


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing the training pipeline.
    Generates random positions with heuristic evaluations.
    """

    def __init__(self, size: int = 10000):
        self.size = size
        self.board_encoder = get_board_encoder()
        self.move_encoder = get_move_encoder()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Generate random position
        board = chess.Board()

        # Make random moves
        num_moves = random.randint(0, 40)
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            board.push(random.choice(legal_moves))

        if board.is_game_over():
            board = chess.Board()

        # Get random legal move
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            board = chess.Board()
            legal_moves = list(board.legal_moves)

        move = random.choice(legal_moves)

        # Heuristic value
        from board_encoder import PositionEvaluator
        evaluator = PositionEvaluator(self.board_encoder, self.move_encoder)
        material = evaluator.get_material_balance(board) / 3900.0  # Normalize
        value = np.clip(material, -1, 1)

        board_tensor = self.board_encoder.encode_board(board)
        move_idx = self.move_encoder.encode_move(move)

        return (
            torch.from_numpy(board_tensor),
            torch.tensor(move_idx, dtype=torch.long),
            torch.tensor(value, dtype=torch.float32)
        )


class ChessTrainer:
    """
    Trainer class for chess neural network.
    """

    def __init__(
        self,
        model: ChessNet,
        config: TrainingConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None

        # Metrics
        self.step = 0
        self.best_val_loss = float('inf')

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0

        for batch in train_loader:
            board, target_move, target_value = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()

            if self.config.use_amp:
                with autocast():
                    policy, value = self.model(board)
                    policy_loss = F.cross_entropy(policy, target_move)
                    value_loss = F.mse_loss(value.squeeze(), target_value)
                    loss = (self.config.policy_weight * policy_loss +
                            self.config.value_weight * value_loss)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                policy, value = self.model(board)
                policy_loss = F.cross_entropy(policy, target_move)
                value_loss = F.mse_loss(value.squeeze(), target_value)
                loss = (self.config.policy_weight * policy_loss +
                        self.config.value_weight * value_loss)

                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
            self.step += 1

            if self.step % self.config.log_every == 0:
                logger.info(
                    f"Step {self.step}: Loss={loss.item():.4f}, "
                    f"Policy={policy_loss.item():.4f}, Value={value_loss.item():.4f}"
                )

            if self.step % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_{self.step}.pt")

        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        correct_moves = 0
        total_samples = 0
        num_batches = 0

        for batch in val_loader:
            board, target_move, target_value = [x.to(self.device) for x in batch]

            policy, value = self.model(board)

            policy_loss = F.cross_entropy(policy, target_move)
            value_loss = F.mse_loss(value.squeeze(), target_value)
            loss = (self.config.policy_weight * policy_loss +
                    self.config.value_weight * value_loss)

            # Accuracy
            pred_moves = policy.argmax(dim=-1)
            correct_moves += (pred_moves == target_move).sum().item()
            total_samples += target_move.size(0)

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'accuracy': correct_moves / total_samples
        }

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """Full training loop."""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Device: {self.device}")

        for epoch in range(1, self.config.num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.config.num_epochs}")

            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                        f"Policy: {train_metrics['policy_loss']:.4f}, "
                        f"Value: {train_metrics['value_loss']:.4f}")

            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader)
                logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                            f"Policy: {val_metrics['policy_loss']:.4f}, "
                            f"Value: {val_metrics['value_loss']:.4f}, "
                            f"Accuracy: {val_metrics['accuracy']:.4f}")

                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pt')
                    logger.info("Saved best model!")

            # Update learning rate
            self.scheduler.step()

            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch}.pt")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Loaded checkpoint from {path}")


def train_from_pgn(
    pgn_files: List[str],
    output_dir: str = 'output',
    model_type: str = 'large',
    epochs: int = 50,
    batch_size: int = 256
):
    """
    Train model from PGN files.

    Args:
        pgn_files: List of PGN file paths
        output_dir: Output directory
        model_type: Model size
        epochs: Number of epochs
        batch_size: Batch size
    """
    # Create config
    config = TrainingConfig(
        model_type=model_type,
        batch_size=batch_size,
        num_epochs=epochs,
        checkpoint_dir=os.path.join(output_dir, 'checkpoints')
    )

    # Create encoders
    board_encoder = get_board_encoder()
    move_encoder = get_move_encoder()

    # Split into train/val
    random.shuffle(pgn_files)
    split_idx = int(0.9 * len(pgn_files))
    train_files = pgn_files[:split_idx]
    val_files = pgn_files[split_idx:]

    # Create datasets
    train_dataset = ChessDataset(
        train_files, board_encoder, move_encoder,
        positions_per_game=20
    )

    val_dataset = ChessDataset(
        val_files, board_encoder, move_encoder,
        positions_per_game=10
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # Create model
    model = create_model(model_type)

    # Create trainer
    trainer = ChessTrainer(model, config)

    # Train
    trainer.train(train_loader, val_loader)


def demo_training():
    """Demo training with synthetic data."""
    logger.info("Running demo training with synthetic data")

    config = TrainingConfig(
        model_type='small',
        batch_size=32,
        num_epochs=5,
        log_every=10,
        save_every=100
    )

    # Create synthetic dataset
    train_dataset = SyntheticDataset(size=1000)
    val_dataset = SyntheticDataset(size=200)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    # Create model
    model = create_model('small')

    # Create trainer
    trainer = ChessTrainer(model, config)

    # Train
    trainer.train(train_loader, val_loader)

    logger.info("Demo training complete!")


if __name__ == '__main__':
    demo_training()
