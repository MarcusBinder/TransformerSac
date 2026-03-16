"""Standalone test script for profile encoders.

Usage:
    python profile_encodings/test_encoders.py
"""

import sys
import time
from pathlib import Path

# Ensure the repo root is on sys.path when run as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from profile_encodings import (
    CNNProfileEncoder,
    DilatedProfileEncoder,
    AttentionProfileEncoder,
    FourierProfileEncoder,
    TancikProfileEncoder,
)


def main():
    torch.manual_seed(42)

    # Test setup
    batch_size = 32
    n_turbines = 10
    n_directions = int(360/2)
    embed_dim = 128

    profiles = torch.randn(batch_size, n_turbines, n_directions)

    encoders = {
        "Original": CNNProfileEncoder(embed_dim),
        "ProfileEncoderWithAttention": AttentionProfileEncoder(embed_dim),
        "Dilated": DilatedProfileEncoder(embed_dim),
        "FourierProfileEncoder": FourierProfileEncoder(embed_dim),
        "TancikProfileEncoder": TancikProfileEncoder(embed_dim),
    }

    print(f"Input shape: {profiles.shape}\n")

    for name, encoder in encoders.items():
        # Count parameters
        n_params = sum(p.numel() for p in encoder.parameters())

        # Time forward pass
        encoder.eval()
        with torch.no_grad():
            start = time.time()
            for _ in range(100):
                out = encoder(profiles)
            elapsed = (time.time() - start) / 100 * 1000

        print(f"{name}:")
        print(f"  Parameters: {n_params:,}")
        print(f"  Output shape: {out.shape}")
        print(f"  Forward time: {elapsed:.2f} ms")
        print()


if __name__ == "__main__":
    main()
