"""Rotary Position Embeddings (RoPE) for 2D wind farm positions."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE2DPositionalEncoding(nn.Module):
    """
    2D Rotary Position Embeddings for wind farm coordinates.

    RoPE encodes position by rotating query and key vectors. When Q·K is
    computed, the rotation angles subtract, naturally encoding relative position.

    For 2D positions, we split the head dimension into two halves:
    - First half: rotated by angle proportional to x-position
    - Second half: rotated by angle proportional to y-position

    This is applied INSIDE the attention mechanism by transforming Q and K
    before the dot product.

    Key properties:
    - Relative position encoded in dot product (no explicit bias)
    - Decays with distance (like a soft locality bias)
    - No learned parameters for position encoding itself

    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    Extended to 2D following approaches from vision transformers.
    """

    def __init__(
        self,
        head_dim: int,
        max_position: float = 50.0,
        base: float = 10000.0,
    ):
        """
        Args:
            head_dim: Dimension per attention head (must be divisible by 4)
            max_position: Maximum expected position value (in rotor diameters)
            base: Base for frequency computation (like in standard RoPE)
        """
        super().__init__()

        if head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}")

        self.head_dim = head_dim
        self.max_position = max_position

        # Each spatial dimension gets half the head_dim
        # Within each half, we have pairs for rotation (so divide by 2 again)
        dim_per_axis = head_dim // 2

        # Frequency bands for rotation (standard RoPE frequencies)
        # Lower frequencies = longer range interactions
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_per_axis, 2).float() / dim_per_axis))
        self.register_buffer("inv_freq", inv_freq)

    def _compute_rotation_angles(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotation angles for x and y positions.

        Args:
            positions: (batch, n_tokens, 2) normalized positions

        Returns:
            angles_x, angles_y: (batch, n_tokens, dim_per_axis/2) rotation angles
        """
        x = positions[..., 0]  # (batch, n)
        y = positions[..., 1]  # (batch, n)

        # Scale positions to reasonable angle range
        x_scaled = x / self.max_position * math.pi
        y_scaled = y / self.max_position * math.pi

        # Compute angles: position * frequency for each frequency band
        # (batch, n, 1) * (dim/4,) -> (batch, n, dim/4)
        angles_x = x_scaled.unsqueeze(-1) * self.inv_freq
        angles_y = y_scaled.unsqueeze(-1) * self.inv_freq

        return angles_x, angles_y

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of x."""
        x1 = x[..., ::2]   # Even indices
        x2 = x[..., 1::2]  # Odd indices
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def apply_rotary_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D rotary embeddings to query and key tensors.

        Args:
            q: (batch, n_heads, n_tokens, head_dim) query tensor
            k: (batch, n_heads, n_tokens, head_dim) key tensor
            positions: (batch, n_tokens, 2) wind-relative positions

        Returns:
            q_rot, k_rot: Rotated query and key tensors, same shape as input
        """
        batch, n_heads, n_tokens, head_dim = q.shape
        half_dim = head_dim // 2

        # Split into x and y portions
        q_x, q_y = q[..., :half_dim], q[..., half_dim:]
        k_x, k_y = k[..., :half_dim], k[..., half_dim:]

        # Compute rotation angles
        angles_x, angles_y = self._compute_rotation_angles(positions)

        # Expand angles for all heads: (batch, n, dim/4) -> (batch, 1, n, dim/4)
        angles_x = angles_x.unsqueeze(1)
        angles_y = angles_y.unsqueeze(1)

        # Duplicate angles for sin/cos pairs: (batch, 1, n, dim/4) -> (batch, 1, n, dim/2)
        cos_x = torch.cos(angles_x).repeat(1, 1, 1, 2)
        sin_x = torch.sin(angles_x).repeat(1, 1, 1, 2)
        cos_y = torch.cos(angles_y).repeat(1, 1, 1, 2)
        sin_y = torch.sin(angles_y).repeat(1, 1, 1, 2)

        # Apply rotation: x' = x*cos - rotate(x)*sin
        q_x_rot = q_x * cos_x + self._rotate_half(q_x) * sin_x
        k_x_rot = k_x * cos_x + self._rotate_half(k_x) * sin_x
        q_y_rot = q_y * cos_y + self._rotate_half(q_y) * sin_y
        k_y_rot = k_y * cos_y + self._rotate_half(k_y) * sin_y

        # Concatenate back
        q_rot = torch.cat([q_x_rot, q_y_rot], dim=-1)
        k_rot = torch.cat([k_x_rot, k_y_rot], dim=-1)

        return q_rot, k_rot

class RoPEMultiheadAttention(nn.Module):
    """
    Multi-head attention with 2D Rotary Position Embeddings.

    This replaces nn.MultiheadAttention when using RoPE, because RoPE
    needs to be applied to Q and K before the attention computation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        max_position: float = 50.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert self.head_dim % 4 == 0, f"head_dim ({self.head_dim}) must be divisible by 4 for 2D RoPE"

        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # RoPE module
        self.rope = RoPE2DPositionalEncoding(
            head_dim=self.head_dim,
            max_position=max_position,
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_tokens, embed_dim) input tensor
            positions: (batch, n_tokens, 2) wind-relative positions
            key_padding_mask: (batch, n_tokens) where True = padding
            attn_mask: Optional additional attention mask

        Returns:
            output: (batch, n_tokens, embed_dim)
            attn_weights: (batch, n_heads, n_tokens, n_tokens)
        """
        batch, n_tokens, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, n_heads, n_tokens, head_dim)
        q = q.view(batch, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = self.rope.apply_rotary_emb(q, k, positions)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply masks
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        if key_padding_mask is not None:
            # (batch, n_tokens) -> (batch, 1, 1, n_tokens)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape back: (batch, n_heads, n_tokens, head_dim) -> (batch, n_tokens, embed_dim)
        output = output.transpose(1, 2).contiguous().view(batch, n_tokens, self.embed_dim)
        output = self.out_proj(output)

        return output, attn_weights
