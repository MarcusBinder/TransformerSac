"""CNN-based profile encoders for wind farm profiles."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from profile_encodings._blocks import ResidualConvBlock


class CNNProfileEncoder(nn.Module):
    """
    Improved CNN encoder for profiles.

    Key improvements:
    1. Multi-scale pyramid: Extract features at multiple resolutions
    2. Residual connections: Better gradient flow
    3. Don't pool all the way to 1: Keep some angular bins
    4. Adaptive pooling: Works with ANY n_profile_directions

    Architecture:
        Input (N) → ResBlock → Pool (N/4) → ResBlock → Pool (N/12) → ResBlock → Pool (n_angular_bins)
        Features from each scale are projected and combined
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_channels: int = 64,
        n_angular_bins: int = 8,  # Keep this many angular bins (not 1!)
        **kwargs,
    ):
        super().__init__()
        self.n_angular_bins = n_angular_bins

        # Scale 1: Full resolution
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=9, padding=4, padding_mode='circular'),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )
        self.res1 = ResidualConvBlock(hidden_channels, hidden_channels, kernel_size=5)

        # Scale 2: ~1/4 resolution (use adaptive pooling)
        self.pool1 = nn.AdaptiveAvgPool1d(None)  # Will be set dynamically
        self.res2 = ResidualConvBlock(hidden_channels, hidden_channels * 2, kernel_size=5)

        # Scale 3: ~1/12 resolution (use adaptive pooling)
        self.pool2 = nn.AdaptiveAvgPool1d(None)  # Will be set dynamically
        self.res3 = ResidualConvBlock(hidden_channels * 2, hidden_channels * 4, kernel_size=5)

        # Final pooling to n_angular_bins
        self.final_pool = nn.AdaptiveAvgPool1d(n_angular_bins)

        # Multi-scale feature projections
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
            profiles: (batch, n_turbines, n_directions) e.g., (32, 10, 72)

        Returns:
            embeddings: (batch, n_turbines, embed_dim)
        """
        batch_size, n_turbines, n_dirs = profiles.shape

        # Compute adaptive pool sizes based on actual input size
        # Scale 2: ~1/4 of input, minimum 8
        scale2_size = max(n_dirs // 4, self.n_angular_bins)
        # Scale 3: ~1/3 of scale2, minimum n_angular_bins
        scale3_size = max(scale2_size // 3, self.n_angular_bins)

        # Flatten batch and turbines: (batch * n_turbines, 1, n_directions)
        x = profiles.view(batch_size * n_turbines, 1, n_dirs)

        # Scale 1: (batch * n_turb, hidden, n_dirs)
        x1 = self.conv1(x)
        x1 = self.res1(x1)

        # Scale 2: (batch * n_turb, hidden * 2, scale2_size)
        x2 = F.adaptive_avg_pool1d(x1, scale2_size)
        x2 = self.res2(x2)

        # Scale 3: (batch * n_turb, hidden * 4, scale3_size)
        x3 = F.adaptive_avg_pool1d(x2, scale3_size)
        x3 = self.res3(x3)

        # Pool each scale to same number of bins
        x1_pooled = self.final_pool(x1)  # (batch * n_turb, hidden, n_angular_bins)
        x2_pooled = self.final_pool(x2)  # (batch * n_turb, hidden * 2, n_angular_bins)
        x3_pooled = self.final_pool(x3)  # (batch * n_turb, hidden * 4, n_angular_bins)

        # Flatten and project each scale
        f1 = self.proj1(x1_pooled.flatten(1))  # (batch * n_turb, embed_dim)
        f2 = self.proj2(x2_pooled.flatten(1))
        f3 = self.proj3(x3_pooled.flatten(1))

        # Fuse multi-scale features
        combined = torch.cat([f1, f2, f3], dim=-1)
        out = self.fusion(combined)

        # Reshape: (batch, n_turbines, embed_dim)
        return out.view(batch_size, n_turbines, -1)


class DilatedProfileEncoder(nn.Module):
    """
    Alternative: Dilated convolutions for large receptive field without pooling.

    Uses exponentially increasing dilation rates to cover the full angular range
    with fewer layers and no intermediate pooling.

    This encoder naturally works with any input size since it uses:
    - Circular padding (handles wraparound)
    - Global adaptive pooling at the end
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_channels: int = 64,
        kernel_size: int = 5,
        dilations: Tuple[int, ...] = (1, 2, 4, 8, 16),
        **kwargs,
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


class AttentionProfileEncoder(nn.Module):
    """
    Alternative: Lightweight attention over angular positions.

    After initial CNN processing, use self-attention to capture
    long-range angular dependencies (e.g., two peaks 180° apart).

    Now supports arbitrary n_profile_directions through adaptive pooling.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_channels: int = 32,
        n_attention_heads: int = 4,
        n_attention_tokens: int = 16,  # Number of angular tokens after pooling
        **kwargs,
    ):
        super().__init__()
        self.n_attention_tokens = n_attention_tokens
        self.hidden_channels = hidden_channels

        # CNN to reduce dimensionality adaptively
        self.cnn = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=7, padding=3, padding_mode='circular'),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        )

        # Adaptive pooling to fixed number of tokens
        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_attention_tokens)

        # Learnable position embedding for the fixed number of tokens
        self.pos_embed = nn.Parameter(torch.randn(1, n_attention_tokens, hidden_channels) * 0.02)

        # Self-attention over angular positions
        self.attention = nn.MultiheadAttention(
            hidden_channels, n_attention_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_channels)

        # Output projection
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_attention_tokens * hidden_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        batch_size, n_turbines, n_dirs = profiles.shape
        x = profiles.view(batch_size * n_turbines, 1, n_dirs)

        # CNN: (batch * n_turb, hidden, n_dirs)
        x = self.cnn(x)

        # Adaptive pool to fixed size: (batch * n_turb, hidden, n_attention_tokens)
        x = self.adaptive_pool(x)

        # Transpose for attention: (batch * n_turb, n_attention_tokens, hidden)
        x = x.transpose(1, 2)

        # Add positional embedding
        x = x + self.pos_embed

        # Self-attention
        x_attn, _ = self.attention(x, x, x)
        x = self.norm(x + x_attn)

        # Output
        out = self.output(x)
        return out.view(batch_size, n_turbines, -1)


class MultiResolutionProfileEncoder(nn.Module):
    """
    Multi-resolution 1D conv on circular profiles.

    Each scale uses a different kernel size, capturing features
    at different angular resolutions. Circular padding ensures
    the 360° wraparound is handled correctly.

    Kernel sizes [3, 7, 15, 31] roughly correspond to angular
    resolutions of [3°, 7°, 15°, 31°] — from individual turbine
    wakes to broad directional sectors.
    """
    def __init__(self, embed_dim=128,
                 scales=[3, 7, 15, 31],
                 channels_per_scale=16,
                 **kwargs,
                 ):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList()

        for k in scales:
            self.convs.append(nn.Sequential(
                # Circular padding: pad with wraparound
                # Conv1d with kernel k on 1-channel input
                nn.Conv1d(1, channels_per_scale, kernel_size=k, padding=0),
                nn.GELU(),
                nn.Conv1d(channels_per_scale, channels_per_scale, kernel_size=1),
            ))

        total_features = channels_per_scale * len(scales)
        self.proj = nn.Sequential(
            nn.Linear(total_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _circular_pad(self, x, pad_size):
        """Wrap-around padding for circular signals."""
        return torch.cat([x[:, :, -pad_size:], x, x[:, :, :pad_size]], dim=-1)

    def forward(self, profiles):
        B, T, D = profiles.shape
        x = profiles.reshape(B * T, 1, D)  # (B*T, 1, 360)

        scale_features = []
        for conv, k in zip(self.convs, self.scales):
            pad = k // 2
            x_padded = self._circular_pad(x, pad)       # (B*T, 1, 360+2*pad)
            h = conv(x_padded)                            # (B*T, C, 360)
            # Global average + max pool over angles
            avg = h.mean(dim=-1)                          # (B*T, C)
            mx = h.max(dim=-1).values                     # (B*T, C)
            scale_features.append(avg + mx)

        features = torch.cat(scale_features, dim=-1)      # (B*T, C*n_scales)
        return self.proj(features).reshape(B, T, -1)
