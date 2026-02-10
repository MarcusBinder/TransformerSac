"""
Contains various positional encoding and relative positional bias
Only related to positional encodings and biases for wind farm transformer models.
"""
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

# from torch_geometric.nn import GATv2Conv
# from torch_geometric.data import Data, Batch


class AbsolutePositionalEncoding(nn.Module): # I dont like this one
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

class RelativePositionalBias(nn.Module): # This is okay
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

class Sinusoidal2DPositionalEncoding(nn.Module): # I dont like this one
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

class PolarPositionalEncoding(nn.Module): # I dont like this one
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

class RelativePolarBias(nn.Module): # This is okay
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

class ALiBiPositionalBias(nn.Module): # Did not show good perforamnce in inital testing.
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

class RelativePositionalBiasAdvanced(nn.Module):
    """
    Improved relative position bias for wind farm attention.
    
    Key insight: In wind-relative coordinates, the relative position (dx, dy) 
    has clear physical meaning:
    - dx > 0: j is downwind of i (j in i's wake)
    - dx < 0: j is upwind of i (i in j's wake)
    - |dy|: lateral separation
    
    This encoder explicitly decomposes into:
    1. Distance component: How much to attend based on distance
    2. Angular component: How much to attend based on direction
    3. Asymmetry: Upwind/downwind relationships
    
    Improvements over vanilla RelativePositionalBias:
    - Richer feature representation (not just raw dx, dy)
    - Separate pathways for distance and angle
    - Physics-motivated asymmetry
    - Proper normalization
    """
    
    def __init__(
        self, 
        num_heads: int, 
        hidden_dim: int = 64,
        characteristic_distance: float = 5.0,  # In rotor diameters
        use_physics_asymmetry: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.characteristic_distance = characteristic_distance
        self.use_physics_asymmetry = use_physics_asymmetry
        
        # Feature dimension for MLP input
        # (dx_norm, dy_norm, dist_norm, angle, sin, cos, dist_decay)
        self.feature_dim = 7
        if use_physics_asymmetry:
            # Add asymmetry features: (is_upwind, is_downwind, is_lateral)
            self.feature_dim += 3
        
        # Main bias MLP (per-head output)
        self.bias_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
        
        # Initialize final layer to small values (start with weak bias)
        nn.init.zeros_(self.bias_mlp[-1].weight)
        nn.init.zeros_(self.bias_mlp[-1].bias)
        
        # Learnable distance decay slopes (ALiBi-inspired, but learned)
        # Different heads can learn different distance sensitivities
        init_slopes = torch.tensor([
            2 ** (-8 * (i + 1) / num_heads) for i in range(num_heads)
        ])
        self.distance_slopes = nn.Parameter(init_slopes)
    
    def forward(
        self, 
        positions: torch.Tensor, 
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_tokens, 2) - wind-relative positions in rotor diameters
            key_padding_mask: (batch, n_tokens) where True = padding
        
        Returns:
            bias: (batch, num_heads, n_tokens, n_tokens)
        """
        batch_size, n_tokens, _ = positions.shape
        device = positions.device
        
        # Compute pairwise relative positions: rel_pos[i,j] = pos[j] - pos[i]
        # "From i's perspective, where is j?"
        pos_i = positions.unsqueeze(2)  # (batch, n, 1, 2)
        pos_j = positions.unsqueeze(1)  # (batch, 1, n, 2)
        rel_pos = pos_j - pos_i         # (batch, n, n, 2)
        
        dx = rel_pos[..., 0]  # Positive = j is downwind of i
        dy = rel_pos[..., 1]  # Lateral displacement
        
        # Distance (with epsilon for numerical stability)
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
        
        # Normalized features
        dx_norm = dx / (dist + 1e-8)  # Unit direction x
        dy_norm = dy / (dist + 1e-8)  # Unit direction y
        dist_norm = dist / self.characteristic_distance  # Normalized distance
        
        # Angular features
        angle = torch.atan2(dy, dx)  # Angle from i to j
        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)
        
        # Soft distance decay (log scale, bounded)
        dist_decay = torch.log1p(dist_norm)  # Smooth, bounded
        
        # Stack features
        features = [dx_norm, dy_norm, dist_norm, angle, sin_angle, cos_angle, dist_decay]
        
        if self.use_physics_asymmetry:
            # Soft asymmetry indicators (smooth approximation)
            # is_upwind: j is upwind of i (dx < 0 and mostly aligned with wind)
            # is_downwind: j is downwind of i (dx > 0 and mostly aligned)
            # is_lateral: j is roughly perpendicular to wind
            
            # Use tanh for smooth transitions
            upwind_score = torch.tanh(-dx_norm * 3)  # High when dx < 0
            downwind_score = torch.tanh(dx_norm * 3)  # High when dx > 0
            lateral_score = 1 - torch.abs(dx_norm)    # High when |dy| >> |dx|
            
            features.extend([upwind_score, downwind_score, lateral_score])
        
        # Stack: (batch, n, n, feature_dim)
        features = torch.stack(features, dim=-1)
        
        # Flatten for MLP: (batch * n * n, feature_dim)
        features_flat = features.reshape(-1, self.feature_dim)
        
        # MLP: (batch * n * n, num_heads)
        bias_flat = self.bias_mlp(features_flat)
        
        # Reshape: (batch, n, n, num_heads) -> (batch, num_heads, n, n)
        bias = bias_flat.reshape(batch_size, n_tokens, n_tokens, self.num_heads)
        bias = bias.permute(0, 3, 1, 2)
        
        # Add learned distance decay (ALiBi-style)
        dist_expanded = dist.unsqueeze(1)  # (batch, 1, n, n)
        distance_bias = -self.distance_slopes.view(1, -1, 1, 1) * dist_expanded
        bias = bias + distance_bias
        
        # Zero out padded positions
        if key_padding_mask is not None:
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n)
            mask_q = key_padding_mask.unsqueeze(1).unsqueeze(3)  # (batch, 1, n, 1)
            bias = bias.masked_fill(mask_k | mask_q, 0.0)
        
        return bias

class RelativePositionalBiasFactorized(nn.Module):
    """
    Alternative: Factorized relative bias with separate distance and angle networks.
    
    bias(i,j) = distance_bias(dist_ij) * angle_weight(angle_ij)
    
    This factorization makes it easier to learn interpretable patterns:
    - Distance network: "Closer turbines are more relevant"
    - Angle network: "Upwind turbines are more relevant"
    """
    
    def __init__(
        self, 
        num_heads: int, 
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.num_heads = num_heads
        
        # Distance network: dist -> per-head bias
        self.distance_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
        
        # Angle network: (sin, cos) -> per-head weight
        self.angle_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
            nn.Sigmoid(),  # Output in [0, 1] as a multiplicative weight
        )
        
        # Learnable base decay (per head)
        self.base_decay = nn.Parameter(torch.ones(num_heads) * 0.1)
    
    def forward(
        self, 
        positions: torch.Tensor, 
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, n_tokens, _ = positions.shape
        
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        rel_pos = pos_j - pos_i
        
        dx, dy = rel_pos[..., 0], rel_pos[..., 1]
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
        
        # Distance features
        dist_input = dist.unsqueeze(-1)  # (batch, n, n, 1)
        dist_bias = self.distance_net(dist_input.reshape(-1, 1))
        dist_bias = dist_bias.reshape(batch_size, n_tokens, n_tokens, self.num_heads)
        
        # Angle features
        sin_angle = dy / (dist + 1e-8)
        cos_angle = dx / (dist + 1e-8)
        angle_input = torch.stack([sin_angle, cos_angle], dim=-1)  # (batch, n, n, 2)
        angle_weight = self.angle_net(angle_input.reshape(-1, 2))
        angle_weight = angle_weight.reshape(batch_size, n_tokens, n_tokens, self.num_heads)
        
        # Combine: base decay + learned adjustment, modulated by angle
        base = -self.base_decay.view(1, 1, 1, -1) * dist.unsqueeze(-1)
        bias = (base + dist_bias) * angle_weight
        
        # Reshape to (batch, heads, n, n)
        bias = bias.permute(0, 3, 1, 2)
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2) | key_padding_mask.unsqueeze(1).unsqueeze(3)
            bias = bias.masked_fill(mask, 0.0)
        
        return bias

class RelativePositionalBiasWithWind(nn.Module):
    """
    Variant that takes wind direction as explicit input.
    
    Useful if positions are NOT pre-transformed to wind-relative frame.
    The network learns to interpret positions given the current wind direction.
    """
    
    def __init__(
        self, 
        num_heads: int, 
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_heads = num_heads
        
        # Wind direction embedding
        self.wind_embed_dim = 16
        self.wind_embed = nn.Sequential(
            nn.Linear(2, self.wind_embed_dim),  # (sin, cos)
            nn.GELU(),
        )
        
        # Relative position MLP (takes wind embedding as context)
        # Input: (dx, dy, dist, sin, cos, wind_embed)
        self.bias_mlp = nn.Sequential(
            nn.Linear(5 + self.wind_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
    
    def forward(
        self, 
        positions: torch.Tensor,          # (batch, n, 2) - global frame
        wind_direction: torch.Tensor,     # (batch,) - degrees
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, n_tokens, _ = positions.shape
        
        # Wind direction embedding
        wd_rad = wind_direction * math.pi / 180
        wd_features = torch.stack([torch.sin(wd_rad), torch.cos(wd_rad)], dim=-1)
        wd_embed = self.wind_embed(wd_features)  # (batch, wind_embed_dim)
        
        # Expand wind embedding for all pairs
        wd_embed = wd_embed.unsqueeze(1).unsqueeze(1).expand(-1, n_tokens, n_tokens, -1)
        
        # Relative positions
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        rel_pos = pos_j - pos_i
        
        dx, dy = rel_pos[..., 0], rel_pos[..., 1]
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
        sin_angle = dy / (dist + 1e-8)
        cos_angle = dx / (dist + 1e-8)
        
        # Combine features
        features = torch.stack([dx, dy, dist, sin_angle, cos_angle], dim=-1)
        features = torch.cat([features, wd_embed], dim=-1)
        
        # MLP
        bias_flat = self.bias_mlp(features.reshape(-1, 5 + self.wind_embed_dim))
        bias = bias_flat.reshape(batch_size, n_tokens, n_tokens, self.num_heads)
        bias = bias.permute(0, 3, 1, 2)
        
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2) | key_padding_mask.unsqueeze(1).unsqueeze(3)
            bias = bias.masked_fill(mask, 0.0)
        
        return bias

class SpatialContextEmbedding(nn.Module):
    """
    Per-turbine embedding based on spatial neighborhood structure.

    For each turbine, computes physics-meaningful summary features
    from the relative positions of all other turbines:
      - Upstream / downstream / lateral neighbor counts at multiple distance thresholds
      - Nearest neighbor distances in each directional sector
      - Angular sector occupancy (turbine count + min distance per sector)
      - Global stats (total neighbors, mean distance)

    These features describe a turbine's "role" in the farm (edge vs interior,
    upstream exposure, downstream influence footprint) without being tied to
    any absolute position. A turbine on the upwind edge of a 5-turbine row
    and the upwind edge of a 20-turbine grid get similar feature vectors.

    Output is added to token embeddings, same as profile encodings.

    Pathway: ADDITIVE (added to token embeddings)
    """
    def __init__(
        self,
        embed_dim: int = 128,
        distance_thresholds: Tuple[float, ...] = (3.0, 5.0, 8.0, 12.0),
        n_angular_sectors: int = 8,
    ):
        """
        Args:
            embed_dim: Output dimension (should match transformer embed_dim)
            distance_thresholds: Distance shells (in rotor diameters) for
                                 counting upstream/downstream/lateral neighbors
            n_angular_sectors: Number of angular bins for sector features
        """
        super().__init__()
        self.distance_thresholds = distance_thresholds
        self.n_angular_sectors = n_angular_sectors
        
        # Feature count breakdown:
        #   3 * len(thresholds)  — upstream/downstream/lateral counts per threshold
        #   3                    — nearest upstream/downstream/lateral distances
        #   2 * n_sectors        — per-sector count + min distance
        #   2                    — total neighbors, mean distance
        n_threshold_feats = 3 * len(distance_thresholds)
        n_nearest_feats = 3
        n_sector_feats = 2 * n_angular_sectors
        n_global_feats = 2
        
        self.feature_dim = n_threshold_feats + n_nearest_feats + n_sector_feats + n_global_feats
        
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Sector boundaries (fixed)
        sector_edges = torch.linspace(-math.pi, math.pi, n_angular_sectors + 1)
        self.register_buffer("sector_lo", sector_edges[:-1])
        self.register_buffer("sector_hi", sector_edges[1:])
    
    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            positions: (B, N, 2) wind-relative, normalized by rotor diameter
            key_padding_mask: (B, N) True = padding

        Returns:
            embeddings: (B, N, embed_dim)
        """
        B, N, _ = positions.shape
        device = positions.device
        
        # Pairwise relative positions
        pos_i = positions.unsqueeze(2)  # (B, N, 1, 2)
        pos_j = positions.unsqueeze(1)  # (B, 1, N, 2)
        rel = pos_j - pos_i             # (B, N, N, 2) "from i, where is j?"
        
        dx = rel[..., 0]  # positive = j is upstream of i (wind-relative)
        dy = rel[..., 1]
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)  # (B, N, N)
        angle = torch.atan2(dy, dx)                # (B, N, N)
        
        # Mask: exclude self-pairs and padded turbines
        self_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)  # (1, N, N)
        if key_padding_mask is not None:
            pad_mask = key_padding_mask.unsqueeze(1)  # (B, 1, N) - j is padding
            valid = ~(self_mask | pad_mask)
        else:
            valid = ~self_mask
        
        # Replace invalid distances with large number
        LARGE = 1e6
        dist_masked = dist.masked_fill(~valid, LARGE)

        # Directional classification (relative to wind)
        is_upstream = dx > torch.abs(dy) * 0.5
        is_downstream = dx < -torch.abs(dy) * 0.5
        is_lateral = ~is_upstream & ~is_downstream

        features = []

        # --- Threshold features: neighbor counts ---
        for thresh in self.distance_thresholds:
            within = (dist_masked < thresh) & valid
            features.append((within & is_upstream).float().sum(dim=-1, keepdim=True))
            features.append((within & is_downstream).float().sum(dim=-1, keepdim=True))
            features.append((within & is_lateral).float().sum(dim=-1, keepdim=True))

        # --- Nearest distances per direction ---
        scale = 10.0
        up_dist = dist_masked.masked_fill(~(is_upstream & valid), LARGE)
        dn_dist = dist_masked.masked_fill(~(is_downstream & valid), LARGE)
        lat_dist = dist_masked.masked_fill(~(is_lateral & valid), LARGE)

        features.append(
            (up_dist.min(dim=-1).values / scale).clamp(max=1.0).unsqueeze(-1)
        )
        features.append(
            (dn_dist.min(dim=-1).values / scale).clamp(max=1.0).unsqueeze(-1)
        )
        features.append(
            (lat_dist.min(dim=-1).values / scale).clamp(max=1.0).unsqueeze(-1)
        )

        # --- Angular sector features ---
        for s in range(self.n_angular_sectors):
            in_sector = (
                (angle >= self.sector_lo[s])
                & (angle < self.sector_hi[s])
                & valid
            )
            features.append(in_sector.float().sum(dim=-1, keepdim=True))
            sector_dist = dist_masked.masked_fill(~in_sector, LARGE)
            features.append(
                (sector_dist.min(dim=-1).values / scale)
                .clamp(max=1.0)
                .unsqueeze(-1)
            )

        # --- Global features ---
        max_thresh = max(self.distance_thresholds)
        n_neighbors = (
            ((dist_masked < max_thresh) & valid).float().sum(dim=-1, keepdim=True)
        )
        valid_count = valid.float().sum(dim=-1, keepdim=True).clamp(min=1)
        mean_dist = (
            dist_masked.masked_fill(~valid, 0.0).sum(dim=-1, keepdim=True)
            / valid_count
            / scale
        )
        features.append(n_neighbors)
        features.append(mean_dist.clamp(max=1.0))

        feat_vec = torch.cat(features, dim=-1)  # (B, N, feature_dim)
        return self.encoder(feat_vec)
    
class NeighborhoodAggregationEmbedding(nn.Module):
    """
    Learned aggregation of pairwise spatial features into per-token embeddings.
    
    For each turbine i, attends over all other turbines j using spatial
    features (distance, angle, upstream/downstream) as keys, and aggregates
    into a single context vector.
    
    This is essentially one round of message passing on the spatial graph
    before the main transformer processes the tokens.

    Pathway: ADDITIVE (added to token embeddings)
    """
    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        pairwise_dim: int = 6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        # Pairwise features → key/value
        # Input: (dist_norm, cos_angle, sin_angle, dx_norm, dy_norm, log_dist)
        self.pairwise_dim = pairwise_dim
        self.kv_proj = nn.Linear(pairwise_dim, 2 * embed_dim)
        
        # Learnable query per head (not per turbine — structure-agnostic)
        self.query = nn.Parameter(torch.randn(1, 1, n_heads, self.head_dim) * 0.02)
        
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, positions, key_padding_mask=None):
        """
        positions: (B, N, 2) wind-relative
        Returns: (B, N, embed_dim) per-turbine spatial context
        """
        B, N, _ = positions.shape
        
        pos_i = positions.unsqueeze(2)  # (B, N, 1, 2)
        pos_j = positions.unsqueeze(1)  # (B, 1, N, 2)
        rel = pos_j - pos_i             # (B, N, N, 2)
        
        dx, dy = rel[..., 0], rel[..., 1]
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
        angle = torch.atan2(dy, dx)
        
        # Pairwise features
        feats = torch.stack([
            dist,                           #  distance, already normalized by rotor diameter
            torch.cos(angle),                # direction cosine
            torch.sin(angle),                # direction sine
            dx / (dist + 1e-8),              # unit dx
            dy / (dist + 1e-8),              # unit dy
            torch.log1p(dist),               # log distance (soft)
        ], dim=-1)  # (B, N, N, 6)
        
        # Project to keys and values
        kv = self.kv_proj(feats)  # (B, N, N, 2*embed_dim)
        k, v = kv.chunk(2, dim=-1)
        
        # Reshape for multi-head attention
        k = k.view(B, N, N, self.n_heads, self.head_dim)  # (B, N, N, H, D)
        v = v.view(B, N, N, self.n_heads, self.head_dim)
        
        # Query broadcasts: (1, 1, H, D) → (B, N, H, D)
        q = self.query.expand(B, N, -1, -1)
        
        # Attention: for each turbine i, attend over all j
        # q: (B, N, H, D), k: (B, N, N, H, D)
        attn_logits = torch.einsum('bnhd,bnmhd->bnmh', q, k) / (self.head_dim ** 0.5)
        
        # Mask self and padding
        self_mask = torch.eye(N, device=positions.device, dtype=torch.bool)
        attn_logits = attn_logits.masked_fill(self_mask.unsqueeze(0).unsqueeze(-1), float('-inf'))
        
        if key_padding_mask is not None:
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
            attn_logits = attn_logits.masked_fill(pad_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_logits, dim=2)  # over j dimension
        
        # Aggregate: (B, N, N, H, D) weighted sum over N → (B, N, H, D)
        context = torch.einsum('bnmh,bnmhd->bnhd', attn_weights, v)
        context = context.reshape(B, N, self.embed_dim)
        
        return self.out_proj(context)
    
class WakeKernelBias(nn.Module):
    """
    Physics-motivated attention bias with very few learnable parameters.

    Models the wake interaction pattern:
      - Distance decay: closer turbines get more attention
      - Upstream bonus: turbines directly upstream matter most
      - Lateral Gaussian falloff: turbines perpendicular to wind matter less

    Only ~3 * num_heads learnable parameters -> essentially impossible to overfit.
    The inductive bias is correct for wake physics; the model just learns the
    right scale/strength per attention head.

    Pathway: ATTENTION BIAS (added to attention logits, like RelativePositionalBias)
    """

    def __init__(self, num_heads: int):
        """
        Args:
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads

        # Per-head learnable parameters
        self.dist_decay = nn.Parameter(torch.ones(num_heads) * 0.3)
        self.lateral_width = nn.Parameter(torch.ones(num_heads) * 2.0)
        self.upstream_bonus = nn.Parameter(torch.ones(num_heads) * 1.0)

    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            positions: (B, N, 2) wind-relative positions (rotor-diameter normalized)
            key_padding_mask: (B, N) True = padding

        Returns:
            bias: (B, num_heads, N, N) — add to attention logits before softmax
        """
        B, N, _ = positions.shape

        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        rel = pos_j - pos_i

        dx = rel[..., 0]  # positive = j upstream of i (wind-relative)
        dy = rel[..., 1]
        dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

        # Expand for heads: (B, 1, N, N)
        dx = dx.unsqueeze(1)
        dy = dy.unsqueeze(1)
        dist = dist.unsqueeze(1)

        decay = self.dist_decay.abs().view(1, -1, 1, 1)
        width = (self.lateral_width.abs() + 0.1).view(1, -1, 1, 1)
        bonus = self.upstream_bonus.view(1, -1, 1, 1)

        # bias = distance_decay + upstream_bonus + lateral_penalty
        bias = -decay * dist
        bias = bias + bonus * torch.tanh(dx)
        bias = bias - (dy ** 2) / (2 * width ** 2)

        # Mask padded positions
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2) | \
                   key_padding_mask.unsqueeze(1).unsqueeze(3)
            bias = bias.masked_fill(mask, 0.0)

        return bias


class DenseGATv2Layer(nn.Module):
    """
    GATv2-style attention operating on dense (B, N, N) pairs.

    Implements the "dynamic attention" of Brody et al. (2022):
        e_ij = a^T · LeakyReLU(W_l·h_i + W_r·h_j + W_e·edge_ij)
        a_ij = softmax_j(e_ij)
        h_i' = Σ_j a_ij · W_r·h_j

    All operations are fully batched — no loops over B or N.
    """
    def __init__(self, embed_dim: int, n_heads: int, edge_dim: int, dropout: float = 0.0):
        super().__init__()
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim**-0.5  # optional scaling for stability

        # Node projections (left = query-side, right = key/value-side)
        self.W_l = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_r = nn.Linear(embed_dim, embed_dim, bias=False)

        # Edge projection
        self.W_e = nn.Linear(edge_dim, embed_dim, bias=False)

        # Per-head attention vector
        self.attn = nn.Parameter(torch.empty(n_heads, self.head_dim))
        nn.init.xavier_normal_(self.attn.unsqueeze(-1))

        # Output projection (after concatenating heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_feats: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            h:          (B, N, D) node embeddings
            edge_feats: (B, N, N, edge_dim) pairwise edge features
            attn_mask:  (B, N, N) bool — True = attend, False = mask out

        Returns:
            out: (B, N, D) updated node embeddings
        """
        B, N, D = h.shape
        H, Dh = self.n_heads, self.head_dim

        l = self.W_l(h)  # (B, N, D)
        r = self.W_r(h)  # (B, N, D)

        # Pairwise interaction: (B, N, 1, D) + (B, 1, N, D) + (B, N, N, D)
        pair = l.unsqueeze(2) + r.unsqueeze(1) + self.W_e(edge_feats)
        pair = F.leaky_relu(pair, negative_slope=0.2)

        # Reshape to multi-head: (B, N, N, H, Dh)
        pair = pair.view(B, N, N, H, Dh)

        # Attention logits: dot with per-head vector → (B, N, N, H)
        logits = (pair * self.attn).sum(dim=-1)

        # Apply mask (padding + self-loops + optional distance cutoff)
        if attn_mask is not None:
            logits = logits.masked_fill(~attn_mask.unsqueeze(-1), float("-inf"))

        weights = F.softmax(logits, dim=2)  # softmax over source dim j
        weights = self.attn_dropout(weights)

        # Values: use W_r·h (standard GATv2) reshaped to (B, N, H, Dh)
        values = r.view(B, N, H, Dh)

        # Weighted aggregation: (B, N_i, N_j, H) × (B, N_j, H, Dh) → (B, N_i, H, Dh)
        out = torch.einsum("bijn,bjnh->binh", weights, values)
        out = out.reshape(B, N, D)

        return self.out_proj(out)
    
class GATPositionalEncoder(nn.Module):
    """
    Dense Graph Attention Network that produces per-turbine spatial context
    embeddings from positions (and optionally wind conditions).

    Uses a fully vectorized GATv2-style attention mechanism operating on
    dense (B, N, N) pair matrices. No PyG dependency required.

    For wind farms of typical scale (10–100 turbines), dense attention is
    both faster and simpler than sparse message passing, since the graphs
    are fully or near-fully connected anyway.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        edge_dim: int = 8,
        dropout: float = 0.0,
        use_wind_context: bool = False,
        distance_cutoff: Optional[float] = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.edge_dim = edge_dim
        self.distance_cutoff = distance_cutoff
        self.use_wind_context = use_wind_context

        # --- Raw edge features (8-dim) → projected edge features ---
        raw_edge_dim = 8
        self.edge_proj = nn.Sequential(
            nn.Linear(raw_edge_dim, edge_dim * 2),
            nn.GELU(),
            nn.Linear(edge_dim * 2, edge_dim),
        )

        # --- Initial node embedding ---
        if use_wind_context:
            self.node_init = nn.Sequential(
                nn.Linear(3, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            self.node_init_embed = nn.Parameter(
                torch.randn(1, 1, embed_dim) * 0.02
            )

        # --- GAT layers (dense) ---
        self.gat_layers = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.ffns = nn.ModuleList()

        for _ in range(n_layers):
            self.gat_layers.append(
                DenseGATv2Layer(embed_dim, n_heads, edge_dim, dropout=dropout)
            )
            self.norms1.append(nn.LayerNorm(embed_dim))
            self.norms2.append(nn.LayerNorm(embed_dim))
            self.ffns.append(
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(embed_dim * 2, embed_dim),
                )
            )

        self.final_norm = nn.LayerNorm(embed_dim)

    def _compute_dense_edge_feats(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise edge features for all (i, j) pairs in one vectorized pass.

        Args:
            positions: (B, N, 2) wind-relative turbine positions

        Returns:
            edge_feats: (B, N, N, edge_dim) projected edge features
        """
        # Pairwise differences: (B, N, N, 2)
        diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        dx = diff[..., 0]  # (B, N, N)
        dy = diff[..., 1]  # (B, N, N)

        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
        angle = torch.atan2(dy, dx)
        unit_dx = dx / (dist + 1e-8)
        unit_dy = dy / (dist + 1e-8)

        # Stack raw features: (B, N, N, 8) — same features as original
        raw_feats = torch.stack(
            [
                unit_dx,
                unit_dy,
                dist / 10.0,
                torch.cos(angle),
                torch.sin(angle),
                torch.log1p(dist),
                torch.tanh(unit_dx * 2),
                1.0 - torch.abs(unit_dx),
            ],
            dim=-1,
        )

        # Single batched projection: (B, N, N, 8) → (B, N, N, edge_dim)
        return self.edge_proj(raw_feats)

    def _build_attn_mask(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Build dense attention mask: (B, N, N) where True = attend.

        Handles: no self-loops, padding exclusion, optional distance cutoff.
        """
        B, N, _ = positions.shape
        device = positions.device

        # No self-attention (no self-loops)
        mask = ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)  # (1, N, N)

        # Exclude padding positions from both source and target
        if key_padding_mask is not None:
            valid = ~key_padding_mask  # (B, N) True = real turbine
            # valid_i AND valid_j: (B, N, 1) & (B, 1, N) → (B, N, N)
            mask = mask & valid.unsqueeze(1) & valid.unsqueeze(2)

        # Optional distance cutoff (sparse connectivity for large farms)
        if self.distance_cutoff is not None:
            dist = torch.cdist(positions, positions)  # (B, N, N)
            mask = mask & (dist < self.distance_cutoff)

        return mask

    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        wind_speed: Optional[torch.Tensor] = None,
        wind_direction: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            positions:        (B, N, 2) wind-relative turbine positions
            key_padding_mask: (B, N) True = padding token
            wind_speed:       (B,) free-stream wind speed [m/s]
            wind_direction:   (B,) wind direction [degrees]

        Returns:
            out: (B, N, embed_dim) spatial context embedding per turbine
        """
        B, N, _ = positions.shape

        # --- Dense edge features (single batched pass) ---
        edge_feats = self._compute_dense_edge_feats(positions)  # (B, N, N, edge_dim)

        # --- Attention mask ---
        attn_mask = self._build_attn_mask(positions, key_padding_mask)  # (B, N, N)

        # --- Initialize node embeddings ---
        if self.use_wind_context and wind_speed is not None:
            wd_rad = wind_direction * (math.pi / 180.0)
            wind_feats = torch.stack(
                [
                    wind_speed / 15.0,
                    torch.cos(wd_rad),
                    torch.sin(wd_rad),
                ],
                dim=-1,
            )  # (B, 3)
            h = self.node_init(wind_feats).unsqueeze(1).expand(-1, N, -1)  # (B, N, D)
        else:
            h = self.node_init_embed.expand(B, N, -1)  # (B, N, D)

        # --- Message passing ---
        for gat, norm1, norm2, ffn in zip(
            self.gat_layers, self.norms1, self.norms2, self.ffns
        ):
            h = h + gat(norm1(h), edge_feats, attn_mask)  # GAT + residual
            h = h + ffn(norm2(h))                          # FFN + residual

        h = self.final_norm(h)

        # --- Zero out padding positions ---
        if key_padding_mask is not None:
            h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return h