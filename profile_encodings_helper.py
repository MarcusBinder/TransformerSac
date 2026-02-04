"""
Containts various positional encoding and bias modules.
Only related to the profiles.

Currently split into 2 types:
CNN based encoders:
- PyWakeProfileEncoder -> multi-scale CNN with residuals
- PyWakeProfileEncoderDilated -> dilated convs for large receptive field
- PyWakeProfileEncoderWithAttention -> CNN + self-attention

Fourier based encoders:
- FourierProfileEncoder -> Fourier decomposition of profiles
- FourierProfileEncoderWithContext -> Fourier + wind direction context
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

# =============================================================
# CNN based encoders for PyWake profiles
# ============================================================

class ResidualConvBlock(nn.Module):
    """Residual block with circular padding for 1D angular data."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 5,
        dilation: int = 1,
    ):
        super().__init__()
        # Circular padding amount
        self.pad = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=0, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=0, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection (1x1 conv if channels change)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, length)"""
        residual = self.skip(x)
        
        # Circular padding for wraparound
        x = F.pad(x, (self.pad, self.pad), mode='circular')
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        
        x = F.pad(x, (self.pad, self.pad), mode='circular')
        x = self.conv2(x)
        x = self.bn2(x)
        
        return F.gelu(x + residual)


class PyWakeProfileEncoder(nn.Module):
    """
    Improved CNN encoder for PyWake profiles.
    
    Key improvements:
    1. Multi-scale pyramid: Extract features at multiple resolutions
    2. Residual connections: Better gradient flow
    3. Don't pool all the way to 1: Keep some angular bins
    4. Dilated convolutions: Larger receptive field without excessive pooling
    
    Architecture:
        Input (360) → ResBlock → Pool (90) → ResBlock → Pool (30) → ResBlock → Pool (8)
        Features from each scale are projected and combined
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_channels: int = 64,
        n_angular_bins: int = 8,  # Keep this many angular bins (not 1!)
    ):
        super().__init__()
        self.n_angular_bins = n_angular_bins
        
        # Scale 1: Full resolution (360)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=9, padding=4, padding_mode='circular'),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.res1 = ResidualConvBlock(hidden_channels, hidden_channels, kernel_size=5)
        
        # Scale 2: 1/4 resolution (90)
        self.pool1 = nn.MaxPool1d(4)  # 360 → 90
        self.res2 = ResidualConvBlock(hidden_channels, hidden_channels * 2, kernel_size=5)
        
        # Scale 3: 1/12 resolution (30)
        self.pool2 = nn.MaxPool1d(3)  # 90 → 30
        self.res3 = ResidualConvBlock(hidden_channels * 2, hidden_channels * 4, kernel_size=5)
        
        # Final pooling to n_angular_bins
        self.final_pool = nn.AdaptiveAvgPool1d(n_angular_bins)  # 30 → 8
        
        # Multi-scale feature projections
        # Project each scale to a fixed size, then combine
        self.proj1 = nn.Linear(hidden_channels * n_angular_bins, embed_dim)
        self.proj2 = nn.Linear(hidden_channels * 2 * n_angular_bins, embed_dim)
        self.proj3 = nn.Linear(hidden_channels * 4 * n_angular_bins, embed_dim)
        
        # Combine multi-scale features
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            profiles: (batch, n_turbines, n_directions) e.g., (32, 10, 360)
        
        Returns:
            embeddings: (batch, n_turbines, embed_dim)
        """
        batch_size, n_turbines, n_dirs = profiles.shape
        
        # Flatten batch and turbines: (batch * n_turbines, 1, n_directions)
        x = profiles.view(batch_size * n_turbines, 1, n_dirs)
        
        # Scale 1: (batch * n_turb, hidden, 360)
        x1 = self.conv1(x)
        x1 = self.res1(x1)
        
        # Scale 2: (batch * n_turb, hidden * 2, 90)
        x2 = self.pool1(x1)
        x2 = self.res2(x2)
        
        # Scale 3: (batch * n_turb, hidden * 4, 30)
        x3 = self.pool2(x2)
        x3 = self.res3(x3)
        
        # Pool each scale to same number of bins
        x1_pooled = self.final_pool(x1)  # (batch * n_turb, hidden, 8)
        x2_pooled = self.final_pool(x2)  # (batch * n_turb, hidden * 2, 8)
        x3_pooled = self.final_pool(x3)  # (batch * n_turb, hidden * 4, 8)
        
        # Flatten and project each scale
        f1 = self.proj1(x1_pooled.flatten(1))  # (batch * n_turb, embed_dim)
        f2 = self.proj2(x2_pooled.flatten(1))
        f3 = self.proj3(x3_pooled.flatten(1))
        
        # Fuse multi-scale features
        combined = torch.cat([f1, f2, f3], dim=-1)
        out = self.fusion(combined)
        
        # Reshape: (batch, n_turbines, embed_dim)
        return out.view(batch_size, n_turbines, -1)


class PyWakeProfileEncoderDilated(nn.Module):
    """
    Alternative: Dilated convolutions for large receptive field without pooling.
    
    Uses exponentially increasing dilation rates to cover the full 360° with
    fewer layers and no intermediate pooling.
    
    Receptive field with dilations [1, 2, 4, 8, 16]:
    - Each layer adds (kernel_size - 1) * dilation to receptive field
    - With kernel=5: 4 + 8 + 16 + 32 + 64 = 124 per side = 248 total
    - Covers most of 360° profile
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_channels: int = 64,
        kernel_size: int = 5,
        dilations: Tuple[int, ...] = (1, 2, 4, 8, 16),
    ):
        super().__init__()
        
        # Input projection
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size, padding=kernel_size//2, padding_mode='circular'),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        
        # Dilated residual blocks
        self.dilated_blocks = nn.ModuleList([
            ResidualConvBlock(hidden_channels, hidden_channels, kernel_size, dilation=d)
            for d in dilations
        ])
        
        # Global average pool + projection
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        batch_size, n_turbines, n_dirs = profiles.shape
        x = profiles.view(batch_size * n_turbines, 1, n_dirs)
        
        x = self.input_conv(x)
        for block in self.dilated_blocks:
            x = block(x)
        
        out = self.output(x)
        return out.view(batch_size, n_turbines, -1)


class PyWakeProfileEncoderWithAttention(nn.Module):
    """
    Alternative: Lightweight attention over angular positions.
    
    After initial CNN processing, use self-attention to capture
    long-range angular dependencies (e.g., two peaks 180° apart).
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_channels: int = 32,
        n_attention_heads: int = 4,
    ):
        super().__init__()
        
        # CNN to reduce dimensionality: 360 → 36
        self.cnn = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=9, padding=4, padding_mode='circular'),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.MaxPool1d(5),  # 360 → 72
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.MaxPool1d(2),  # 72 → 36
        )
        
        # Self-attention over angular positions
        # Each of the 36 positions becomes a token
        self.pos_embed = nn.Parameter(torch.randn(1, 36, hidden_channels) * 0.02)
        self.attention = nn.MultiheadAttention(
            hidden_channels, n_attention_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_channels)
        
        # Output projection
        self.output = nn.Sequential(
            nn.Flatten(),  # (batch * n_turb, 36 * hidden)
            nn.Linear(36 * hidden_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        batch_size, n_turbines, n_dirs = profiles.shape
        x = profiles.view(batch_size * n_turbines, 1, n_dirs)
        
        # CNN: (batch * n_turb, hidden, 36)
        x = self.cnn(x)
        
        # Transpose for attention: (batch * n_turb, 36, hidden)
        x = x.transpose(1, 2)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Self-attention
        x_attn, _ = self.attention(x, x, x)
        x = self.norm(x + x_attn)
        
        # Output
        out = self.output(x)
        return out.view(batch_size, n_turbines, -1)


# ===============================================================
# Fourier based encoders for PyWake profiles
# ======


class FourierProfileEncoder(nn.Module):
    """
    Encode circular profiles via Fourier decomposition.
    
    For wind farm profiles (receptivity/influence), the key information is:
    - DC component (harmonic 0): Total integrated wake exposure
    - Harmonic 1: Primary directional bias (most important in wind-relative frame)
    - Harmonics 2-4: Finer structure (multiple wake sources, complex layouts)
    - Higher harmonics: Usually noise
    
    Using Fourier features instead of CNN pooling:
    - Preserves angular structure in a compact representation
    - Naturally handles circular wraparound
    - Interpretable components
    - No learned pooling that might destroy information
    
    Args:
        embed_dim: Output embedding dimension
        n_harmonics: Number of Fourier harmonics to extract (default 8)
                    Higher = more angular detail, but diminishing returns
        use_phase: If True, use (magnitude, phase). If False, use (real, imag).
                  Phase representation is more interpretable but has discontinuities.
        learnable_weights: If True, learn per-harmonic importance weights
    """
    
    def __init__(
        self, 
        embed_dim: int = 128, 
        n_harmonics: int = 8,
        use_phase: bool = False, # False beacuse if we did true, then it MIGHT be unstable, due to phase discontinuities
        learnable_weights: bool = True,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.use_phase = use_phase
        
        # Feature dimension: DC (1) + harmonics (2 each for real/imag or mag/phase)
        self.feature_dim = 1 + 2 * n_harmonics
        
        # Optional learnable importance weights per harmonic
        if learnable_weights:
            # Initialize with slight decay for higher harmonics
            init_weights = torch.ones(n_harmonics + 1)
            init_weights[1:] = torch.exp(-0.1 * torch.arange(1, n_harmonics + 1).float())
            self.harmonic_weights = nn.Parameter(init_weights)
        else:
            self.register_buffer('harmonic_weights', torch.ones(n_harmonics + 1))
        
        # Project Fourier features to embedding
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            profiles: (batch, n_turbines, n_directions) 
                     Angular profiles, e.g., (32, 10, 360)
                     Assumed to be in wind-relative frame (index 0 = upstream)
        
        Returns:
            embeddings: (batch, n_turbines, embed_dim)
        """
        batch_size, n_turbines, n_dirs = profiles.shape
        
        # Reshape for FFT: (batch * n_turbines, n_directions)
        x = profiles.reshape(batch_size * n_turbines, n_dirs)
        
        # Real FFT along angular dimension
        # Output shape: (batch * n_turbines, n_dirs // 2 + 1) complex
        fft = torch.fft.rfft(x, dim=-1)
        
        # Extract first n_harmonics + 1 (including DC)
        fft = fft[:, :self.n_harmonics + 1]
        
        # Apply learnable harmonic weights
        fft = fft * self.harmonic_weights.to(fft.device)
        
        if self.use_phase:
            # Magnitude and phase representation
            magnitude = torch.abs(fft)
            phase = torch.angle(fft)
            
            # DC has no meaningful phase, use 0
            features = torch.cat([
                magnitude[:, 0:1],           # DC magnitude
                magnitude[:, 1:],            # Harmonic magnitudes  
                phase[:, 1:],                # Harmonic phases (skip DC)
            ], dim=-1)
        else:
            # Real and imaginary representation (more stable for gradients)
            features = torch.cat([
                fft[:, 0:1].real,            # DC (real only, imag is always 0)
                fft[:, 1:].real,             # Harmonic real parts
                fft[:, 1:].imag,             # Harmonic imaginary parts
            ], dim=-1)
        
        # Project to embedding dimension
        embeddings = self.proj(features)
        
        # Reshape back: (batch, n_turbines, embed_dim)
        return embeddings.reshape(batch_size, n_turbines, -1)
    
    def get_interpretable_features(self, profiles: torch.Tensor) -> dict:
        """
        Extract interpretable Fourier features for analysis/debugging.
        
        Returns dict with:
        - dc: Total integrated value (mean of profile)
        - h1_magnitude: Strength of primary directional bias
        - h1_direction: Peak direction of primary bias (in indices, 0=upstream)
        - higher_harmonics: Magnitudes of h2, h3, ...
        """
        batch_size, n_turbines, n_dirs = profiles.shape
        x = profiles.reshape(-1, n_dirs)
        
        fft = torch.fft.rfft(x, dim=-1)
        
        dc = fft[:, 0].real / n_dirs  # Normalize to get mean
        
        h1 = fft[:, 1]
        h1_magnitude = torch.abs(h1) * 2 / n_dirs  # Normalize
        h1_direction = -torch.angle(h1) * n_dirs / (2 * math.pi)  # Convert to index
        h1_direction = h1_direction % n_dirs  # Wrap to [0, n_dirs)
        
        higher_mags = torch.abs(fft[:, 2:self.n_harmonics + 1]) * 2 / n_dirs
        
        return {
            'dc': dc.reshape(batch_size, n_turbines),
            'h1_magnitude': h1_magnitude.reshape(batch_size, n_turbines),
            'h1_direction': h1_direction.reshape(batch_size, n_turbines),
            'higher_harmonics': higher_mags.reshape(batch_size, n_turbines, -1),
        }


class FourierProfileEncoderWithContext(nn.Module):
    """
    Extended Fourier encoder that also takes wind direction as context.
    
    This allows the encoder to learn direction-dependent transformations,
    which can be useful if profiles are NOT pre-rotated to wind-relative frame.
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        n_harmonics: int = 8,
        wind_dir_embed_dim: int = 16,
    ):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.feature_dim = 1 + 2 * n_harmonics
        
        # Wind direction embedding (sin/cos encoding)
        self.wind_embed = nn.Sequential(
            nn.Linear(2, wind_dir_embed_dim),
            nn.GELU(),
        )
        
        # Combined projection
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim + wind_dir_embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(
        self, 
        profiles: torch.Tensor,
        wind_direction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            profiles: (batch, n_turbines, n_directions)
            wind_direction: (batch,) wind direction in degrees
        """
        batch_size, n_turbines, n_dirs = profiles.shape
        
        # Fourier features
        x = profiles.reshape(batch_size * n_turbines, n_dirs)
        fft = torch.fft.rfft(x, dim=-1)[:, :self.n_harmonics + 1]
        
        fourier_features = torch.cat([
            fft[:, 0:1].real,
            fft[:, 1:].real,
            fft[:, 1:].imag,
        ], dim=-1)
        
        # Wind direction embedding (broadcast to all turbines)
        wd_rad = wind_direction * math.pi / 180
        wd_encoding = torch.stack([torch.sin(wd_rad), torch.cos(wd_rad)], dim=-1)
        wd_embed = self.wind_embed(wd_encoding)  # (batch, wind_dir_embed_dim)
        wd_embed = wd_embed.unsqueeze(1).expand(-1, n_turbines, -1)  # (batch, n_turb, dim)
        wd_embed = wd_embed.reshape(batch_size * n_turbines, -1)
        
        # Combine and project
        combined = torch.cat([fourier_features, wd_embed], dim=-1)
        embeddings = self.proj(combined)
        
        return embeddings.reshape(batch_size, n_turbines, -1)

