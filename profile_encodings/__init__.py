"""Profile encoding modules for wind farm transformer models.

Two encoder families:
- CNN-based: CNNProfileEncoder, DilatedProfileEncoder, AttentionProfileEncoder,
             MultiResolutionProfileEncoder
- Fourier-based: FourierProfileEncoder, FourierProfileEncoderWithContext,
                 TancikProfileEncoder (Random Fourier Features)

Shared building block: ResidualConvBlock
"""

from profile_encodings._blocks import ResidualConvBlock
from profile_encodings._cnn import (
    AttentionProfileEncoder,
    CNNProfileEncoder,
    DilatedProfileEncoder,
    MultiResolutionProfileEncoder,
)
from profile_encodings._fourier import (
    FourierProfileEncoder,
    FourierProfileEncoderWithContext,
    TancikProfileEncoder,
)

__all__ = [
    "ResidualConvBlock",
    "CNNProfileEncoder",
    "DilatedProfileEncoder",
    "AttentionProfileEncoder",
    "MultiResolutionProfileEncoder",
    "FourierProfileEncoder",
    "FourierProfileEncoderWithContext",
    "TancikProfileEncoder",
]
