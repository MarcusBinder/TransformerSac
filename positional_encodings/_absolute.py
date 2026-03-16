"""Absolute positional encodings for wind farm turbine positions."""
import math

import torch
import torch.nn as nn


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
