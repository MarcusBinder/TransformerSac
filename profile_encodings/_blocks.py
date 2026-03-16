"""Shared building blocks for CNN-based profile encoders."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """Residual block with circular padding for 1D angular data."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        **kwargs,
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
