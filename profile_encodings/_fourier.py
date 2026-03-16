"""Fourier-based profile encoders for wind farm profiles."""

import math

import torch
import torch.nn as nn




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
    - Works with ANY n_profile_directions

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
        use_phase: bool = False,        # Always false gave best results.
        learnable_weights: bool = True, # Inital test showed that learnable weights can help, but not by much. Worth further tuning.
        **kwargs,
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
                     Angular profiles, e.g., (32, 10, 72) or (32, 10, 360)
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
        # Note: For small n_dirs, we might have fewer harmonics available
        n_available = min(self.n_harmonics + 1, fft.shape[-1])
        fft = fft[:, :n_available]

        # Pad with zeros if we have fewer harmonics than expected
        if n_available < self.n_harmonics + 1:
            padding = torch.zeros(
                fft.shape[0], self.n_harmonics + 1 - n_available,
                dtype=fft.dtype, device=fft.device
            )
            fft = torch.cat([fft, padding], dim=-1)

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

        n_higher = min(self.n_harmonics - 1, fft.shape[-1] - 2)
        if n_higher > 0:
            higher_mags = torch.abs(fft[:, 2:2 + n_higher]) * 2 / n_dirs
        else:
            higher_mags = torch.zeros(x.shape[0], 0, device=x.device)

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

    Works with any n_profile_directions.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_harmonics: int = 8,
        wind_dir_embed_dim: int = 16,
        **kwargs,
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
        fft = torch.fft.rfft(x, dim=-1)

        # Handle case where n_dirs is small
        n_available = min(self.n_harmonics + 1, fft.shape[-1])
        fft = fft[:, :n_available]

        # Pad if needed
        if n_available < self.n_harmonics + 1:
            padding = torch.zeros(
                fft.shape[0], self.n_harmonics + 1 - n_available,
                dtype=fft.dtype, device=fft.device
            )
            fft = torch.cat([fft, padding], dim=-1)

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
