"""
Chess Move Prediction Neural Network

This module implements a deep neural network for chess move prediction using
a combination of Convolutional Neural Networks (CNN) for board representation
and attention mechanisms for move selection.

The model architecture:
1. Board Encoder: CNN-based feature extraction from board state
2. Move Predictor: Transformer-based move prediction with attention
3. Value Head: Win probability estimation
4. Policy Head: Move probability distribution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class ResidualBlock(nn.Module):
    """
    Residual block with skip connections for deep feature extraction.
    Based on AlphaZero architecture.
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Improves feature recalibration.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        squeeze = x.view(batch, channels, -1).mean(dim=2)
        excitation = F.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        excitation = excitation.view(batch, channels, 1, 1)
        return x * excitation


class ResidualSEBlock(nn.Module):
    """
    Residual block with Squeeze-and-Excitation attention.
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out)


class BoardEncoder(nn.Module):
    """
    Encodes the chess board state into a feature representation.

    Input: 19 planes of 8x8 (piece positions, castling rights, en passant, etc.)
    Output: Feature map of shape (batch, 256, 8, 8)
    """

    def __init__(self, input_channels: int = 19, num_blocks: int = 20, channels: int = 256):
        super().__init__()

        # Initial convolution
        self.conv_input = nn.Conv2d(input_channels, channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(channels)

        # Residual tower with SE blocks
        self.residual_tower = nn.ModuleList([
            ResidualSEBlock(channels) if i % 4 == 0 else ResidualBlock(channels)
            for i in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_input(self.conv_input(x)))

        for block in self.residual_tower:
            x = block(x)

        return x


class PolicyHead(nn.Module):
    """
    Policy head for move probability distribution.

    Outputs probabilities for all possible moves (4672 possible moves in chess).
    Uses the format: from_square (64) * to_square (64) + promotion (8) = 4672
    """

    def __init__(self, input_channels: int = 256, num_moves: int = 4672):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, 80, 1, bias=False)
        self.bn = nn.BatchNorm2d(80)
        self.fc = nn.Linear(80 * 8 * 8, num_moves)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ValueHead(nn.Module):
    """
    Value head for win probability estimation.

    Outputs a single value in range [-1, 1] representing:
    -1: Black wins
     0: Draw
    +1: White wins
    """

    def __init__(self, input_channels: int = 256):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, 32, 1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block for capturing long-range dependencies in move sequences.
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, dim_feedforward: int = 1024):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        x = self.norm2(x + self.ffn(x))

        return x


class ChessTransformer(nn.Module):
    """
    Transformer-based chess model that processes board states and move history.
    """

    def __init__(self, d_model: int = 256, num_layers: int = 6):
        super().__init__()

        self.board_encoder = BoardEncoder(channels=d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model)
            for _ in range(num_layers)
        ])

        self.policy_head = PolicyHead(input_channels=d_model)
        self.value_head = ValueHead(input_channels=d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode board
        features = self.board_encoder(x)  # (batch, 256, 8, 8)

        # Reshape for transformer
        batch_size = features.size(0)
        seq = features.view(batch_size, 256, 64).permute(0, 2, 1)  # (batch, 64, 256)
        seq = seq + self.pos_embedding

        # Apply transformer layers
        for layer in self.transformer_layers:
            seq = layer(seq)

        # Reshape back for heads
        features = seq.permute(0, 2, 1).view(batch_size, 256, 8, 8)

        # Get policy and value
        policy = self.policy_head(features)
        value = self.value_head(features)

        return policy, value


class ChessNet(nn.Module):
    """
    Main chess neural network combining CNN and Transformer architectures.

    This is the primary model used for move prediction and win probability estimation.

    Architecture Overview:
    1. Input Layer: 19 planes representing board state
    2. Board Encoder: 20 residual blocks with SE attention
    3. Policy Head: Move probability distribution (4672 outputs)
    4. Value Head: Win probability (-1 to 1)

    The model can predict:
    - Best move for current position
    - Win probability for current player
    - Top N moves with their probabilities
    """

    def __init__(
        self,
        input_channels: int = 19,
        num_blocks: int = 20,
        channels: int = 256,
        num_moves: int = 4672,
        use_transformer: bool = True
    ):
        super().__init__()

        self.use_transformer = use_transformer

        self.board_encoder = BoardEncoder(input_channels, num_blocks, channels)

        if use_transformer:
            self.pos_embedding = nn.Parameter(torch.randn(1, 64, channels) * 0.02)
            self.transformer_layers = nn.ModuleList([
                TransformerBlock(channels) for _ in range(4)
            ])

        self.policy_head = PolicyHead(channels, num_moves)
        self.value_head = ValueHead(channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Board state tensor of shape (batch, 19, 8, 8)

        Returns:
            policy: Move probabilities of shape (batch, 4672)
            value: Win probability of shape (batch, 1)
        """
        features = self.board_encoder(x)

        if self.use_transformer:
            batch_size = features.size(0)
            seq = features.view(batch_size, -1, 64).permute(0, 2, 1)
            seq = seq + self.pos_embedding

            for layer in self.transformer_layers:
                seq = layer(seq)

            features = seq.permute(0, 2, 1).view(batch_size, -1, 8, 8)

        policy = self.policy_head(features)
        value = self.value_head(features)

        return policy, value

    def predict_move(
        self,
        board_tensor: torch.Tensor,
        legal_moves_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict the best move with probabilities.

        Args:
            board_tensor: Board state
            legal_moves_mask: Boolean mask for legal moves
            temperature: Softmax temperature for exploration

        Returns:
            move_probs: Probability distribution over moves
            value: Win probability
            top_moves: Indices of top moves
        """
        self.eval()
        with torch.no_grad():
            policy, value = self.forward(board_tensor)

            if legal_moves_mask is not None:
                policy = policy.masked_fill(~legal_moves_mask, float('-inf'))

            move_probs = F.softmax(policy / temperature, dim=-1)
            top_moves = torch.topk(move_probs, k=5, dim=-1)

        return move_probs, value, top_moves


class MCTS:
    """
    Monte Carlo Tree Search for improved move selection.
    Uses the neural network for position evaluation.
    """

    def __init__(self, model: ChessNet, num_simulations: int = 800, c_puct: float = 1.4):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

        # Tree storage
        self.Qsa = {}  # Q values for state-action pairs
        self.Nsa = {}  # Visit counts for state-action pairs
        self.Ns = {}   # Visit counts for states
        self.Ps = {}   # Policy from neural network

    def search(self, board_tensor: torch.Tensor, legal_moves: List[int]) -> np.ndarray:
        """
        Perform MCTS search from current position.

        Args:
            board_tensor: Current board state
            legal_moves: List of legal move indices

        Returns:
            Action probabilities after search
        """
        for _ in range(self.num_simulations):
            self._simulate(board_tensor.clone(), legal_moves.copy())

        # Get visit counts
        s = self._hash_state(board_tensor)
        counts = np.array([
            self.Nsa.get((s, a), 0) for a in range(4672)
        ])

        # Convert to probabilities
        probs = counts / counts.sum()
        return probs

    def _simulate(self, board_tensor: torch.Tensor, legal_moves: List[int]) -> float:
        """Run a single simulation."""
        s = self._hash_state(board_tensor)

        if s not in self.Ps:
            # Leaf node - evaluate with neural network
            policy, value = self.model(board_tensor.unsqueeze(0))
            self.Ps[s] = F.softmax(policy, dim=-1).squeeze().cpu().numpy()
            self.Ns[s] = 0
            return value.item()

        # Select action with highest UCB
        best_ucb = -float('inf')
        best_action = legal_moves[0]

        for a in legal_moves:
            if (s, a) in self.Qsa:
                q = self.Qsa[(s, a)]
                n = self.Nsa[(s, a)]
                ucb = q + self.c_puct * self.Ps[s][a] * np.sqrt(self.Ns[s]) / (1 + n)
            else:
                ucb = self.c_puct * self.Ps[s][a] * np.sqrt(self.Ns[s] + 1e-8)

            if ucb > best_ucb:
                best_ucb = ucb
                best_action = a

        # For now, return current value (full implementation would update board)
        return 0.0

    def _hash_state(self, board_tensor: torch.Tensor) -> str:
        """Create a hash for the board state."""
        return board_tensor.cpu().numpy().tobytes()


def create_model(
    model_type: str = 'large',
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> ChessNet:
    """
    Factory function to create chess models of different sizes.

    Args:
        model_type: 'small', 'medium', 'large', or 'xl'
        device: Device to place model on

    Returns:
        Initialized ChessNet model
    """
    configs = {
        'small': {'num_blocks': 10, 'channels': 128, 'use_transformer': False},
        'medium': {'num_blocks': 15, 'channels': 192, 'use_transformer': True},
        'large': {'num_blocks': 20, 'channels': 256, 'use_transformer': True},
        'xl': {'num_blocks': 30, 'channels': 384, 'use_transformer': True}
    }

    config = configs.get(model_type, configs['large'])
    model = ChessNet(**config)
    model = model.to(device)

    return model


if __name__ == '__main__':
    # Test model creation
    model = create_model('large')
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    dummy_input = torch.randn(1, 19, 8, 8)
    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()

    policy, value = model(dummy_input)
    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")
