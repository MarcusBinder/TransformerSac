import os
import random
import time
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from collections import deque


class AbsolutePositionalEncoding(nn.Module):
    """
    Original absolute positional encoding for turbine (x, y) coordinates.
    
    Transforms 2D positions into a higher-dimensional embedding space
    using a small MLP. This is ADDED to the token embedding.
    
    Input positions should be:
    1. Normalized by rotor diameter (physics-meaningful scale)
    2. Transformed to wind-relative coordinates (wind from 270°)
    """
    
    def __init__(self, pos_dim: int = 2, embed_dim: int = 32):
        """
        Args:
            pos_dim: Input position dimension (2 for x, y)
            embed_dim: Output embedding dimension
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(pos_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_turbines, 2) normalized wind-relative coordinates
        
        Returns:
            Position embeddings: (batch, n_turbines, embed_dim)
        """
        return self.encoder(positions)

class RelativePositionalBias(nn.Module):
    """
    Relative positional bias for attention.
    
    Computes a learned bias for each pair of positions based on their
    relative displacement. This bias is ADDED to attention logits.
    
    Physics intuition:
    - rel_pos[i,j] = pos[j] - pos[i] tells us "j is X upwind, Y lateral from i"
    - The learned bias can encode "pay more attention to upwind turbines"
    - Translation invariant: same relative geometry → same bias
    
    For wind farm control:
    - Positive x in wind-relative coords = upwind
    - The model can learn that upwind turbines are important (wake sources)
    """
    
    def __init__(
        self, 
        num_heads: int,
        hidden_dim: int = 64,
        per_head: bool = True,
        pos_dim: int = 2
    ):
        """
        Args:
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension of bias MLP
            per_head: If True, each head gets its own bias. If False, shared.
            pos_dim: Dimension of position vectors (2 for x, y)
        """
        super().__init__()
        self.num_heads = num_heads
        self.per_head = per_head
        
        output_dim = num_heads if per_head else 1
        
        # MLP: relative_position (2D) → bias value(s)
        self.bias_mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self, 
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute relative position bias matrix.
        
        Args:
            positions: (batch, n_tokens, 2) wind-relative positions
            key_padding_mask: (batch, n_tokens) where True = padding
        
        Returns:
            bias: (batch, num_heads, n_tokens, n_tokens) attention bias
                  Add this to attention logits before softmax.
        """
        batch_size, n_tokens, _ = positions.shape
        
        # Compute pairwise relative positions
        # pos_i: (batch, n, 1, 2), pos_j: (batch, 1, n, 2)
        pos_i = positions.unsqueeze(2)  # (batch, n, 1, 2)
        pos_j = positions.unsqueeze(1)  # (batch, 1, n, 2)
        
        # rel_pos[i,j] = pos[j] - pos[i]: "displacement from i to j"
        # If j is upwind of i (positive x), rel_pos has positive x component
        rel_pos = pos_j - pos_i  # (batch, n, n, 2)
        
        # Reshape for MLP: (batch * n * n, 2)
        rel_pos_flat = rel_pos.reshape(-1, 2)
        
        # Compute bias values
        bias_flat = self.bias_mlp(rel_pos_flat)  # (batch*n*n, num_heads or 1)
        
        # Reshape back
        if self.per_head:
            bias = bias_flat.reshape(batch_size, n_tokens, n_tokens, self.num_heads)
            bias = bias.permute(0, 3, 1, 2)  # (batch, num_heads, n, n)
        else:
            bias = bias_flat.reshape(batch_size, n_tokens, n_tokens, 1)
            bias = bias.permute(0, 3, 1, 2)  # (batch, 1, n, n)
            bias = bias.expand(-1, self.num_heads, -1, -1)  # (batch, num_heads, n, n)
        
        # Apply masking: set bias to large negative for padded positions
        if key_padding_mask is not None:
            # Expand mask: (batch, n) → (batch, 1, 1, n) for keys
            # and (batch, 1, n, 1) for queries
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n)
            mask_q = key_padding_mask.unsqueeze(1).unsqueeze(3)  # (batch, 1, n, 1)
            
            # Zero out bias for padded positions (they'll be masked in attention anyway)
            # This prevents any gradient flow through padded positions
            combined_mask = mask_k | mask_q  # (batch, 1, n, n)
            bias = bias.masked_fill(combined_mask, 0.0)
        
        return bias

class Sinusoidal2DPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding extended to 2D coordinates.
    
    Uses multiple frequencies for both x and y dimensions, similar to
    NeRF-style positional encoding. This captures both coarse and fine
    spatial structure without learning the frequency bands.
    
    A final linear layer projects to the desired output dimension.
    """
    
    def __init__(self, embed_dim: int = 32, num_frequencies: int = 8, max_freq_log2: int = 6):
        """
        Args:
            embed_dim: Output embedding dimension
            num_frequencies: Number of frequency bands
            max_freq_log2: Log2 of maximum frequency (default: 2^6 = 64 cycles per unit)
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.raw_dim = 4 * num_frequencies  # sin/cos for x and y
        self.embed_dim = embed_dim
        
        # Frequency bands: 2^0, 2^1, ..., 2^(max_freq_log2)
        freq_bands = 2.0 ** torch.linspace(0, max_freq_log2, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)
        
        # Project to desired dimension
        self.proj = nn.Linear(self.raw_dim, embed_dim)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_turbines, 2) normalized coordinates
        
        Returns:
            Sinusoidal embeddings: (batch, n_turbines, embed_dim)
        """
        # positions: (batch, n, 2)
        x = positions[..., 0:1]  # (batch, n, 1)
        y = positions[..., 1:2]  # (batch, n, 1)
        
        # Multiply by frequencies: (batch, n, num_freq)
        x_freq = x * self.freq_bands * math.pi
        y_freq = y * self.freq_bands * math.pi
        
        # Compute sin and cos
        raw_embeddings = torch.cat([
            torch.sin(x_freq),
            torch.cos(x_freq),
            torch.sin(y_freq),
            torch.cos(y_freq),
        ], dim=-1)
        
        return self.proj(raw_embeddings)

class PolarPositionalEncoding(nn.Module):
    """
    Positional encoding using polar coordinates (r, θ).
    
    In wind-relative coordinates (wind from 270°):
    - θ = 0° means directly upwind
    - θ = 180° means directly downwind
    - r = distance from farm centroid
    
    This naturally aligns with wake physics where effects depend on
    both distance and angle relative to wind.
    """
    
    def __init__(self, embed_dim: int = 32):
        """
        Args:
            embed_dim: Output embedding dimension
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # MLP on (r, θ, sin(θ), cos(θ)) for better angle representation
        self.encoder = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_turbines, 2) wind-relative Cartesian coordinates
        
        Returns:
            Polar embeddings: (batch, n_turbines, embed_dim)
        """
        x = positions[..., 0]
        y = positions[..., 1]
        
        # Convert to polar
        r = torch.sqrt(x**2 + y**2 + 1e-8)
        theta = torch.atan2(y, x)  # Angle from positive x-axis (upwind direction)
        
        # Create input features: (r, θ, sin(θ), cos(θ))
        polar_features = torch.stack([
            r,
            theta,
            torch.sin(theta),
            torch.cos(theta),
        ], dim=-1)
        
        return self.encoder(polar_features)

class RelativePolarBias(nn.Module):
    """
    Relative positional bias using polar coordinates.
    
    For each pair (i, j), computes:
    - Δr = distance from i to j
    - θ_ij = angle from i to j relative to wind direction
    
    The bias MLP learns how attention should depend on:
    - How far apart turbines are
    - Whether j is upwind/downwind/lateral from i
    """
    
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int = 64,
        per_head: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.per_head = per_head
        
        output_dim = num_heads if per_head else 1
        
        # Input: (Δr, θ, sin(θ), cos(θ))
        self.bias_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_tokens, 2) wind-relative positions
            key_padding_mask: (batch, n_tokens) where True = padding
        
        Returns:
            bias: (batch, num_heads, n_tokens, n_tokens)
        """
        batch_size, n_tokens, _ = positions.shape
        
        # Compute pairwise relative positions
        pos_i = positions.unsqueeze(2)  # (batch, n, 1, 2)
        pos_j = positions.unsqueeze(1)  # (batch, 1, n, 2)
        rel_pos = pos_j - pos_i  # (batch, n, n, 2)
        
        # Convert to polar
        dx = rel_pos[..., 0]
        dy = rel_pos[..., 1]
        
        r = torch.sqrt(dx**2 + dy**2 + 1e-8)
        theta = torch.atan2(dy, dx)
        
        # Stack polar features
        polar_features = torch.stack([
            r,
            theta,
            torch.sin(theta),
            torch.cos(theta),
        ], dim=-1)  # (batch, n, n, 4)
        
        # Apply MLP
        polar_flat = polar_features.reshape(-1, 4)
        bias_flat = self.bias_mlp(polar_flat)
        
        if self.per_head:
            bias = bias_flat.reshape(batch_size, n_tokens, n_tokens, self.num_heads)
            bias = bias.permute(0, 3, 1, 2)
        else:
            bias = bias_flat.reshape(batch_size, n_tokens, n_tokens, 1)
            bias = bias.permute(0, 3, 1, 2)
            bias = bias.expand(-1, self.num_heads, -1, -1)
        
        if key_padding_mask is not None:
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(2)
            mask_q = key_padding_mask.unsqueeze(1).unsqueeze(3)
            combined_mask = mask_k | mask_q
            bias = bias.masked_fill(combined_mask, 0.0)
        
        return bias

class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi) for 2D spatial positions.
    
    Simple linear penalty based on distance:
        bias[i,j] = -slope * distance(i, j)
    
    No learned parameters! Just an inductive bias that nearby turbines
    should attend more to each other.
    
    Each attention head gets a different slope (geometric sequence),
    allowing different heads to focus on different distance scales.
    
    Reference: Press et al., "Train Short, Test Long" (2022)
    """
    
    def __init__(self, num_heads: int, max_distance: float = 20.0):
        """
        Args:
            num_heads: Number of attention heads
            max_distance: Expected maximum distance (in rotor diameters) for slope scaling
        """
        super().__init__()
        self.num_heads = num_heads
        
        # Geometric sequence of slopes (like original ALiBi)
        # Slopes: 2^(-8/n), 2^(-8*2/n), ..., 2^(-8)
        slopes = torch.tensor([
            2 ** (-8 * (i + 1) / num_heads) for i in range(num_heads)
        ])
        self.register_buffer("slopes", slopes.view(1, num_heads, 1, 1))
    
    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_tokens, 2) normalized positions
            key_padding_mask: (batch, n_tokens) where True = padding
        
        Returns:
            bias: (batch, num_heads, n_tokens, n_tokens)
        """
        batch_size, n_tokens, _ = positions.shape
        
        # Compute pairwise distances
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        diff = pos_j - pos_i
        distances = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # (batch, n, n)
        
        # Apply linear penalty with per-head slopes
        distances = distances.unsqueeze(1)  # (batch, 1, n, n)
        bias = -self.slopes * distances  # (batch, num_heads, n, n)
        
        if key_padding_mask is not None:
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(2)
            mask_q = key_padding_mask.unsqueeze(1).unsqueeze(3)
            combined_mask = mask_k | mask_q
            bias = bias.masked_fill(combined_mask, 0.0)
        
        return bias

class DirectionalALiBiPositionalBias(nn.Module):
    """
    Directional ALiBi: Different slopes for upwind vs downwind.
    
    In wind-relative coordinates (wind from negative x):
    - Upwind (positive x direction): Use upwind_slope
    - Downwind (negative x direction): Use downwind_slope
    
    This encodes the physical asymmetry: upwind turbines affect
    downwind ones, but not vice versa.
    
    Learned slopes allow the model to discover the right asymmetry.
    """
    
    def __init__(self, num_heads: int):
        """
        Args:
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        
        # Learnable slopes for upwind and downwind (per head)
        # Initialize with ALiBi-style geometric sequence
        init_slopes = torch.tensor([
            2 ** (-8 * (i + 1) / num_heads) for i in range(num_heads)
        ])
        
        self.upwind_slopes = nn.Parameter(init_slopes.clone())
        self.downwind_slopes = nn.Parameter(init_slopes.clone() * 0.5)  # Less penalty downwind
        self.lateral_slopes = nn.Parameter(init_slopes.clone() * 0.3)  # Even less for lateral
    
    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_tokens, 2) wind-relative positions
            key_padding_mask: (batch, n_tokens) where True = padding
        
        Returns:
            bias: (batch, num_heads, n_tokens, n_tokens)
        """
        batch_size, n_tokens, _ = positions.shape
        
        # Compute pairwise relative positions
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        rel_pos = pos_j - pos_i  # (batch, n, n, 2)
        
        dx = rel_pos[..., 0]  # Positive = j is upwind of i
        dy = rel_pos[..., 1]
        
        # Distances
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
        
        # Classify direction: upwind (dx > 0), downwind (dx < 0), lateral (|dy| > |dx|)
        is_upwind = dx > torch.abs(dy)      # Predominantly upwind
        is_downwind = dx < -torch.abs(dy)   # Predominantly downwind
        is_lateral = ~is_upwind & ~is_downwind  # Lateral
        
        # Apply different slopes based on direction
        slopes_upwind = self.upwind_slopes.view(1, self.num_heads, 1, 1)
        slopes_downwind = self.downwind_slopes.view(1, self.num_heads, 1, 1)
        slopes_lateral = self.lateral_slopes.view(1, self.num_heads, 1, 1)
        
        dist = dist.unsqueeze(1)  # (batch, 1, n, n)
        is_upwind = is_upwind.unsqueeze(1).float()
        is_downwind = is_downwind.unsqueeze(1).float()
        is_lateral = is_lateral.unsqueeze(1).float()
        
        # Weighted combination of slopes
        bias = -(
            slopes_upwind * dist * is_upwind +
            slopes_downwind * dist * is_downwind +
            slopes_lateral * dist * is_lateral
        )
        
        if key_padding_mask is not None:
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(2)
            mask_q = key_padding_mask.unsqueeze(1).unsqueeze(3)
            combined_mask = mask_k | mask_q
            bias = bias.masked_fill(combined_mask, 0.0)
        
        return bias

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

