"""Positional encoding and relative bias modules for wind farm transformer models.

Five families:
- Absolute encodings (added to token embeddings):
    AbsolutePositionalEncoding, Sinusoidal2DPositionalEncoding, PolarPositionalEncoding
- Relative positional biases (added to attention logits):
    RelativePositionalBias, RelativePolarBias, ALiBiPositionalBias,
    DirectionalALiBiPositionalBias, RelativePositionalBiasAdvanced,
    RelativePositionalBiasFactorized, RelativePositionalBiasWithWind, WakeKernelBias
- Rotary Position Embeddings (applied to Q/K inside attention):
    RoPE2DPositionalEncoding, RoPEMultiheadAttention
- Spatial context (neighborhood aggregation):
    SpatialContextEmbedding, NeighborhoodAggregationEmbedding
- Graph Attention Network:
    DenseGATv2Layer, GATPositionalEncoder
"""

from positional_encodings._absolute import (
    AbsolutePositionalEncoding,
    PolarPositionalEncoding,
    Sinusoidal2DPositionalEncoding,
)
from positional_encodings._bias import (
    ALiBiPositionalBias,
    DirectionalALiBiPositionalBias,
    RelativePositionalBias,
    RelativePositionalBiasAdvanced,
    RelativePositionalBiasFactorized,
    RelativePositionalBiasWithWind,
    RelativePolarBias,
    WakeKernelBias,
)
from positional_encodings._gat import (
    DenseGATv2Layer,
    GATPositionalEncoder,
)
from positional_encodings._rope import (
    RoPE2DPositionalEncoding,
    RoPEMultiheadAttention,
)
from positional_encodings._spatial import (
    NeighborhoodAggregationEmbedding,
    SpatialContextEmbedding,
)

__all__ = [
    # Absolute encodings
    "AbsolutePositionalEncoding",
    "Sinusoidal2DPositionalEncoding",
    "PolarPositionalEncoding",
    # Relative biases
    "RelativePositionalBias",
    "RelativePolarBias",
    "ALiBiPositionalBias",
    "DirectionalALiBiPositionalBias",
    "RelativePositionalBiasAdvanced",
    "RelativePositionalBiasFactorized",
    "RelativePositionalBiasWithWind",
    "WakeKernelBias",
    # RoPE
    "RoPE2DPositionalEncoding",
    "RoPEMultiheadAttention",
    # Spatial context
    "SpatialContextEmbedding",
    "NeighborhoodAggregationEmbedding",
    # GAT
    "DenseGATv2Layer",
    "GATPositionalEncoder",
]
