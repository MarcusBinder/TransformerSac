"""
Neural network architectures for Transformer-SAC wind farm control.

Contains the actor (policy), critic (Q-function), and TQC critic networks,
plus factory functions for positional and profile encodings.
"""

import json
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Args

from positional_encodings._attn import MaskedScaledAttention, neighbour_allow_mask
from positional_encodings import (
    AbsolutePositionalEncoding,
    RelativePositionalBias,
    Sinusoidal2DPositionalEncoding,
    PolarPositionalEncoding,
    RelativePolarBias,
    ALiBiPositionalBias,
    DirectionalALiBiPositionalBias,
    RelativePositionalBiasAdvanced,
    RelativePositionalBiasFactorized,
    RelativePositionalBiasWithWind,
    SpatialContextEmbedding,
    NeighborhoodAggregationEmbedding,
    WakeKernelBias,
    GATPositionalEncoder,
)

from profile_encodings import (
    CNNProfileEncoder,
    DilatedProfileEncoder,
    AttentionProfileEncoder,
    FourierProfileEncoder,
    MultiResolutionProfileEncoder,
    FourierProfileEncoderWithContext,
    TancikProfileEncoder,
)


# =============================================================================
# POSITIONAL AND PROFILE ENCODING
# =============================================================================

# Type alias for encoding type
VALID_POS_ENCODING_TYPES = [
    None,                  # No positional encoding
    # === Additive (added to token embeddings) ===
    "absolute_mlp",         # Original: MLP on (x,y) → add to token
    "sinusoidal_2d",        # NeRF-style multi-frequency encoding
    "polar_mlp",            # MLP on (r, θ) polar coordinates
    "spatial_context",      # Embedding of spatial context (e.g. local density)
    "neighborhood_agg",     # Embedding based on local neighborhood (e.g. via GNN)
    "gat_encoder",          # Graph Attention Network encoder for positions

    # === Attention Bias (added to attention logits) ===
    "relative_mlp",         # MLP on pairwise rel pos → attention bias (per-head)
    "relative_mlp_shared",  # MLP on pairwise rel pos → attention bias (shared)
    "relative_polar",       # MLP on pairwise (Δr, Δθ) → attention bias (per-head)
    "relative_polar_shared",       # MLP on pairwise (Δr, Δθ) → attention bias (shared)
    "alibi",                # Linear distance penalty (no learned params)
    "alibi_directional",    # ALiBi with upwind/downwind asymmetry
    "RelativePositionalBiasAdvanced",  # Advanced relative bias with distance and angle features
    "RelativePositionalBiasFactorized", # Factorized relative bias for efficiency
    "RelativePositionalBiasWithWind",   # Relative bias incorporating wind direction
    "wake_kernel",                      # Wake kernel bias based on physics-inspired functions of relative position
    # === Combined ===
    "absolute_plus_relative",  # Both absolute embedding AND relative bias
]


def create_positional_encoding(
    encoding_type: Optional[str],  # Now Optional
    embed_dim: int,
    pos_embed_dim: int,
    num_heads: int,
    rel_pos_hidden_dim: int = 64,
    rel_pos_per_head: bool = True,
    embedding_mode: str = "concat",  # "add" or "concat" for absolute types
) -> Tuple[Optional[nn.Module], Optional[nn.Module], Union[str, bool]]:
    """
    Factory function to create positional encoding modules.

    Args:
        encoding_type: One of VALID_POS_ENCODING_TYPES
        embed_dim: Main transformer embedding dimension
        pos_embed_dim: Dimension for absolute position embedding
        num_heads: Number of attention heads (for relative bias)
        rel_pos_hidden_dim: Hidden dim for relative position MLP
        rel_pos_per_head: Whether relative bias is per-head

    Returns:
        (pos_encoder, rel_pos_bias, embedding_mode)
        - pos_encoder: Module for absolute position embedding (or None)
        - rel_pos_bias: Module for relative position bias (or None)
        - embedding_mode: "none", "add" or "concat"
            - "none": No position embedding added to tokens (bias)
            - "add": Position embedding directly added to tokens (like LLMs)
            - "concat": Position embedding concatenated to tokens and projected
            """
    if encoding_type not in VALID_POS_ENCODING_TYPES:
        raise ValueError(
            f"Unknown pos_encoding_type: {encoding_type}. "
            f"Valid options: {VALID_POS_ENCODING_TYPES}"
        )


    # =========================================================================
    # No Positional Encoding
    # =========================================================================
    if encoding_type is None:
        return None, None, False

    # =========================================================================
    # Additive Encodings (added to token embeddings)
    # =========================================================================

    elif encoding_type == "absolute_mlp":
        out_dim = embed_dim if embedding_mode == "add" else pos_embed_dim
        # Original approach: MLP embedding added to tokens
        pos_encoder = AbsolutePositionalEncoding(pos_dim=2, embed_dim=out_dim)
        rel_pos_bias = None
        embedding_mode = embedding_mode

    elif encoding_type == "sinusoidal_2d":
        # Sinusoidal 2D encoding (frequency bands are fixed, projection is learned)
        out_dim = embed_dim if embedding_mode == "add" else pos_embed_dim
        pos_encoder = Sinusoidal2DPositionalEncoding(
            embed_dim=out_dim,
            num_frequencies=8,  # 8 frequency bands
            max_freq_log2=6,    # Max frequency 2^6 = 64
        )
        rel_pos_bias = None
        embedding_mode = embedding_mode

    elif encoding_type == "polar_mlp":
        # Polar coordinate encoding
        out_dim = embed_dim if embedding_mode == "add" else pos_embed_dim
        pos_encoder = PolarPositionalEncoding(embed_dim=out_dim)
        rel_pos_bias = None
        embedding_mode = embedding_mode

    elif encoding_type == "spatial_context":
        out_dim = embed_dim if embedding_mode == "add" else pos_embed_dim
        pos_encoder = SpatialContextEmbedding(embed_dim=out_dim)
        rel_pos_bias = None
        embedding_mode = embedding_mode

    elif encoding_type == "neighborhood_agg":
        out_dim = embed_dim if embedding_mode == "add" else pos_embed_dim
        pos_encoder = NeighborhoodAggregationEmbedding(embed_dim=out_dim)
        rel_pos_bias = None
        embedding_mode = embedding_mode

    elif encoding_type == "gat_encoder":
        # Graph Attention Network encoder for positions
        out_dim = embed_dim if embedding_mode == "add" else pos_embed_dim
        pos_encoder = GATPositionalEncoder(embed_dim=out_dim,
                                           n_heads=num_heads,
                                           n_layers=2,
                                           edge_dim=8,
                                           use_wind_context=False,
                                           distance_cutoff=15.0,
                                           )
        rel_pos_bias = None
        embedding_mode = embedding_mode


    elif encoding_type == "relative_mlp":
        # Relative position bias added to attention (per-head)
        pos_encoder = None
        rel_pos_bias = RelativePositionalBias(
            num_heads=num_heads,
            hidden_dim=rel_pos_hidden_dim,
            per_head=True,
            pos_dim=2
        )
        embedding_mode = False

    elif encoding_type == "relative_mlp_shared":
        # Relative position bias (shared across heads)
        pos_encoder = None
        rel_pos_bias = RelativePositionalBias(
            num_heads=num_heads,
            hidden_dim=rel_pos_hidden_dim,
            per_head=False,
            pos_dim=2
        )
        embedding_mode = False

    elif encoding_type == "relative_polar":
        # Relative position bias using polar coordinates
        pos_encoder = None
        rel_pos_bias = RelativePolarBias(
            num_heads=num_heads,
            hidden_dim=rel_pos_hidden_dim,
            per_head=True,
        )
        embedding_mode = False

    elif encoding_type == "relative_polar_shared":
        # Relative polar bias (shared across heads)
        pos_encoder = None
        rel_pos_bias = RelativePolarBias(
            num_heads=num_heads,
            hidden_dim=rel_pos_hidden_dim,
            per_head=False,
        )
        embedding_mode = False

    elif encoding_type == "alibi":
        # ALiBi: Simple linear distance penalty (no learned params)
        pos_encoder = None
        rel_pos_bias = ALiBiPositionalBias(num_heads=num_heads)
        embedding_mode = False

    elif encoding_type == "alibi_directional":
        # Directional ALiBi with upwind/downwind asymmetry
        pos_encoder = None
        rel_pos_bias = DirectionalALiBiPositionalBias(num_heads=num_heads)
        embedding_mode = False

    elif encoding_type == "RelativePositionalBiasAdvanced":
        # Advanced relative bias with distance and angle features
        pos_encoder = None
        rel_pos_bias = RelativePositionalBiasAdvanced(
            num_heads=num_heads,
            hidden_dim=rel_pos_hidden_dim,
            characteristic_distance=5.0,
            use_physics_asymmetry=True,
        )
        embedding_mode = False

    elif encoding_type == "RelativePositionalBiasFactorized":
        # Factorized relative bias for efficiency
        pos_encoder = None
        rel_pos_bias = RelativePositionalBiasFactorized(
            num_heads=num_heads,
            hidden_dim=rel_pos_hidden_dim,
        )
        embedding_mode = False

    elif encoding_type == "RelativePositionalBiasWithWind":
        # Relative bias incorporating wind direction
        # NOT YET IMPLEMENTED
        raise NotImplementedError(
            "RelativePositionalBiasWithWind requires wind direction as input. See TODO."
        )

    elif encoding_type == "wake_kernel":
        # Wake kernel bias based on physics-inspired functions of relative position
        pos_encoder = None
        rel_pos_bias = WakeKernelBias(num_heads=num_heads)
        embedding_mode = False

    # =========================================================================
    # Combined Encodings
    # =========================================================================

    elif encoding_type == "absolute_plus_relative":
        # Both absolute embedding AND relative bias
        out_dim = embed_dim if embedding_mode == "add" else pos_embed_dim
        pos_encoder = AbsolutePositionalEncoding(pos_dim=2, embed_dim=out_dim)
        rel_pos_bias = RelativePositionalBias(
            num_heads=num_heads,
            hidden_dim=rel_pos_hidden_dim,
            per_head=rel_pos_per_head,
            pos_dim=2
        )
        embedding_mode = embedding_mode

    else:
        raise ValueError(f"Encoding type '{encoding_type}' not implemented yet.")


    return pos_encoder, rel_pos_bias, embedding_mode


# Backward compatibility alias
PositionalEncoding = AbsolutePositionalEncoding


# Type alias for encoding type
VALID_PROFILE_ENCODING_TYPES = [
    None,                  # No positional encoding
    # === CNN Based ===
    "CNNProfileEncoder",                # CNN encoder for PyWake profiles
    "DilatedProfileEncoder",            # Dilated convolutions for large receptive field without pooling
    "AttentionProfileEncoder",          # Lightweight attention over angular positions
    "MultiResolutionProfileEncoder",     # Multi-resolution CNN encoder for profiles (captures both local and global patterns)
    # === Fourier Based ===
    "FourierProfileEncoder",                # Encode circular profiles via Fourier decomposition.
    "FourierProfileEncoderWithContext",     # Needs wind direction as input. Not yet implemented
    "TancikProfileEncoder",                 # Random Fourier Features (Tancik et al., NeurIPS 2020)
]

def create_profile_encoding(
    profile_type: Optional[str],  # Optional
    embed_dim: int,
    hidden_channels: int,
    use_influence: bool = True,  # If False, only build the receptivity encoder (returns None for influence)
    **encoder_kwargs,  # Flexible kwargs for different encoder types (e.g. n_harmonics for Fourier, scales for MultiResolution)
) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    """
    Factory function to create pywake-profile encoding modules.

    Args:
        profile_type: One of VALID_PROFILE_ENCODING_TYPES
        embed_dim: Embedding dimension
        hidden_channels: Hidden channels in profile encoder MLP

    Returns:
        (recep_encoder, influence_encoder)
        - recep_encoder: The receptivity encoder module (or None)
        - influence_encoder: The influence encoder module (or None)
    """
    if profile_type not in VALID_PROFILE_ENCODING_TYPES:
        raise ValueError(
            f"Unknown profile_type: {profile_type}. "
            f"Valid options: {VALID_PROFILE_ENCODING_TYPES}"
        )


    # =========================================================================
    # No Profile Encoding
    # =========================================================================
    if profile_type is None:
        return None, None

    # =========================================================================
    # Profile Encodings
    # =========================================================================

    # Default configs per encoder type, overridden by encoder_kwargs
    ENCODER_DEFAULTS = {
        "FourierProfileEncoder": dict(n_harmonics=8, use_phase=False, learnable_weights=True),
        "TancikProfileEncoder": dict(n_features=128, sigma=1.0),
        "MultiResolutionProfileEncoder": dict(scales=[3, 7, 15, 31], channels_per_scale=16),
        "AttentionProfileEncoder": dict(n_attention_heads=4),
    }


    defaults = dict(ENCODER_DEFAULTS.get(profile_type, {}))  # Copy to avoid mutating ENCODER_DEFAULTS
    defaults.update(encoder_kwargs)  # user overrides win
    defaults.pop("embed_dim", None)        # Avoid duplicate kwargs
    defaults.pop("hidden_channels", None)  # Avoid duplicate kwargs


    ENCODER_CLASSES = {
        "CNNProfileEncoder": CNNProfileEncoder,
        "DilatedProfileEncoder": DilatedProfileEncoder,
        "AttentionProfileEncoder": AttentionProfileEncoder,
        "MultiResolutionProfileEncoder": MultiResolutionProfileEncoder,
        "FourierProfileEncoder": FourierProfileEncoder,
        "TancikProfileEncoder": TancikProfileEncoder,
    }

    cls = ENCODER_CLASSES.get(profile_type)
    if cls is None:
        raise ValueError(f"Unknown profile_type: {profile_type}")

    recep_encoder = cls(embed_dim=embed_dim, hidden_channels=hidden_channels, **defaults)
    influence_encoder = (
        cls(embed_dim=embed_dim, hidden_channels=hidden_channels, **defaults)
        if use_influence else None
    )

    return recep_encoder, influence_encoder



# =============================================================================
# TRANSFORMER BLOCKS
# =============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Standard transformer encoder layer with pre-norm (more stable for RL).

    Returns attention weights for visualization/debugging of learned
    wake interaction patterns.

    Architecture:
        x -> LayerNorm -> MultiheadAttention -> + -> LayerNorm -> FFN -> +
             (skip connection)                      (skip connection)

    Supports optional attention bias for relative positional encoding.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        attn_logit_scale: str = "none",   # v5: "none" | "logn"
        attn_softmax: str = "softmax",    # v5: "softmax" | "entmax15"
        film_dim: int = 0,                # Stage 10: per-token FiLM from a cond vector (0 = off)
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Pre-norm layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Stage 10 (cond_mode film): zero-initialised per-token FiLM on the two
        # pre-norm outputs, x * (1 + gamma) + beta with (gamma, beta) =
        # film(cond). Zero init makes the layer forward-identical to the plain
        # layer at construction (an identity-preserving warm start from a
        # nominal checkpoint with strict=False). Only built when film_dim > 0,
        # so the cond-off state_dict is unchanged.
        self.film_dim = int(film_dim)
        if self.film_dim > 0:
            self.film1 = nn.Linear(self.film_dim, 2 * embed_dim)
            self.film2 = nn.Linear(self.film_dim, 2 * embed_dim)
            for lin in (self.film1, self.film2):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

        # Multi-head attention (custom: supports log-N scaling + local masking;
        # identical to nn.MultiheadAttention when flags are off)
        self.attn = MaskedScaledAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            logit_scale=attn_logit_scale,
            softmax_type=attn_softmax,
        )

        # Feed-forward network
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        local_allow: Optional[torch.Tensor] = None,  # v5: (batch, n, n) bool, True = allowed
        need_weights: bool = False,
        cond: Optional[torch.Tensor] = None,  # Stage 10: (batch, n_tokens, film_dim) for FiLM
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_tokens, embed_dim)
            key_padding_mask: (batch, n_tokens) where True = ignore this position
            attn_bias: (batch, n_heads, n_tokens, n_tokens) optional bias added
                       to attention logits (for relative positional encoding)
            local_allow: (batch, n, n) optional bool mask; True = query may attend key
            cond: (batch, n_tokens, film_dim) per-token conditioning for FiLM (film_dim > 0 only)

        Returns:
            x: Transformed tensor, same shape as input
            attn_weights: (batch, n_heads, n_tokens, n_tokens) attention weights (or None)
        """
        # Self-attention with pre-norm. The custom attention takes the (B,H,N,N) bias
        # directly (no reshape) and applies optional log-N scaling + local masking.
        x_norm = self.norm1(x)
        if self.film_dim > 0:
            g, b = self.film1(cond).chunk(2, dim=-1)
            x_norm = x_norm * (1 + g) + b
        attn_out, attn_weights = self.attn(
            x_norm,
            key_padding_mask=key_padding_mask,
            attn_bias=attn_bias,
            local_allow=local_allow,
            need_weights=need_weights,
        )
        x = x + attn_out

        # FFN with pre-norm
        x_norm = self.norm2(x)
        if self.film_dim > 0:
            g, b = self.film2(cond).chunk(2, dim=-1)
            x_norm = x_norm * (1 + g) + b
        x = x + self.mlp(x_norm)

        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Stack of transformer encoder layers.

    Processes per-turbine tokens and allows each turbine to attend to
    all other turbines, learning spatial wake interaction patterns.

    Supports optional attention bias for relative positional encoding.
    The same bias is applied to all layers (position relationships don't change).

    Future extension point: This could be replaced with a SpatioTemporalEncoder
    for Option B (temporal attention across timesteps).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        attn_logit_scale: str = "none",   # v5
        attn_softmax: str = "softmax",    # v5
        film_dim: int = 0,                # Stage 10 FiLM conditioning (0 = off)
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout,
                                    attn_logit_scale=attn_logit_scale, attn_softmax=attn_softmax,
                                    film_dim=film_dim)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)  # Final layer norm

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        local_allow: Optional[torch.Tensor] = None,  # v5
        need_weights: bool = False,
        cond: Optional[torch.Tensor] = None,  # Stage 10 FiLM
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (batch, n_tokens, embed_dim)
            key_padding_mask: (batch, n_tokens) where True = padding
            attn_bias: (batch, n_heads, n_tokens, n_tokens) optional attention bias
            local_allow: (batch, n, n) optional bool local-attention mask (same for all layers)
            need_weights: If True, return attention weights (expensive). Default False.
            cond: (batch, n_tokens, film_dim) FiLM conditioning (film_dim > 0 only)

        Returns:
            x: Transformed tensor
            all_attn_weights: List of attention weights from each layer (empty if need_weights=False)
        """
        all_attn_weights = []

        for layer in self.layers:
            x, attn_weights = layer(x, key_padding_mask, attn_bias,
                                    local_allow=local_allow, need_weights=need_weights,
                                    cond=cond)
            if need_weights:
                all_attn_weights.append(attn_weights)

        x = self.norm(x)

        return x, all_attn_weights


# =============================================================================
# PER-SENSOR OBSERVATION ENCODERS (change_wd_4, --obs_encoder_mode per_sensor)
# =============================================================================
# Hypothesis (Marcus): the shared obs-encoder MLP mixes the four sensor
# histories (ws/wd/yaw/power) in its very first Linear, so a low-resolution ws
# channel can be drowned by the wider-range channels before any nonlinearity.
# Encoding each sensor group with its OWN small MLP forces the network to build
# a per-sensor representation first; the groups only meet after concat.
#
# Contract: the per-turbine obs must be exactly 4 contiguous groups of
# history_length ([ws x H, wd x H, yaw x H, power x H] — the hard_2 layout, no
# probes/TI). The asserts below double as a guard against accidentally
# combining per_sensor with an expanding --obs_encoding (desired hard failure).

class PerSensorObsEncoder(nn.Module):
    """4 per-sensor MLPs (H -> E/4 each), concatenated to E. Drop-in for the
    shared obs_encoder: same input/output shapes, forward untouched."""

    def __init__(self, obs_dim_per_turbine: int, embed_dim: int, history_length: int):
        super().__init__()
        assert obs_dim_per_turbine == 4 * history_length, (
            f"per_sensor obs encoder expects obs_dim == 4*history_length "
            f"(4 sensor groups), got obs_dim={obs_dim_per_turbine}, "
            f"history_length={history_length}. Incompatible with expanding "
            f"--obs_encoding modes."
        )
        assert embed_dim % 4 == 0, f"embed_dim must be divisible by 4, got {embed_dim}"
        self.history_length = history_length
        e4 = embed_dim // 4
        self.sensor_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(history_length, e4),
                nn.ReLU(),
                nn.Linear(e4, e4),
            )
            for _ in range(4)
        ])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: (..., 4*H) -> 4 x (..., H) -> concat 4 x (..., E/4) = (..., E)
        groups = obs.split(self.history_length, dim=-1)
        return torch.cat([mlp(g) for mlp, g in zip(self.sensor_mlps, groups)], dim=-1)


class PerSensorObsActionEncoder(nn.Module):
    """Critic variant: input is already cat([obs, action], dim=-1). The 4 sensor
    groups plus the action group each get their own MLP (-> E/4), then a final
    Linear fuses the 5E/4 concat back to E."""

    def __init__(self, obs_dim_per_turbine: int, action_dim_per_turbine: int,
                 embed_dim: int, history_length: int):
        super().__init__()
        assert obs_dim_per_turbine == 4 * history_length, (
            f"per_sensor obs-action encoder expects obs_dim == 4*history_length, "
            f"got obs_dim={obs_dim_per_turbine}, history_length={history_length}."
        )
        assert embed_dim % 4 == 0, f"embed_dim must be divisible by 4, got {embed_dim}"
        self.history_length = history_length
        self.action_dim = action_dim_per_turbine
        e4 = embed_dim // 4

        def _mlp(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, e4),
                nn.ReLU(),
                nn.Linear(e4, e4),
            )

        self.sensor_mlps = nn.ModuleList([_mlp(history_length) for _ in range(4)])
        self.action_mlp = _mlp(action_dim_per_turbine)
        self.fuse = nn.Linear(5 * e4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 4*H + A); split off the action tail, then the 4 sensor groups
        obs, action = x.split([4 * self.history_length, self.action_dim], dim=-1)
        groups = obs.split(self.history_length, dim=-1)
        parts = [mlp(g) for mlp, g in zip(self.sensor_mlps, groups)]
        parts.append(self.action_mlp(action))
        return self.fuse(torch.cat(parts, dim=-1))


# =============================================================================
# ACTOR NETWORK
# =============================================================================

def _read_attn_cfg(args):
    """v5 attention flags from args, with backward-compatible defaults."""
    if args is None:
        return "none", "softmax", "none", 10.0, 5, 40.0
    return (
        getattr(args, "attn_logit_scale", "none"),
        getattr(args, "attn_softmax", "softmax"),
        getattr(args, "attn_local", "none"),
        float(getattr(args, "attn_local_radius_D", 10.0)),
        int(getattr(args, "attn_local_k", 5)),
        float(getattr(args, "attn_local_cone_deg", 40.0)),
    )


COND_IN_DIM_DEFAULT = 12   # legacy e width (6 turbine + 6 farm keys) for checkpoints without args.cond_in_dim; Stage 11+ carries 13
COND_SOURCES = ("none", "turbine", "turbine_farm")
COND_MODES = ("concat", "film")
COND_CRITIC_MODES = ("raw", "none")


def _read_cond_cfg(args, cond_source=None, cond_latent_dim=None, cond_mode=None,
                   cond_critic=None, cond_in_dim=None):
    """Stage-10 conditioning flags: explicit kwargs win, else args, else off.

    Returns (source, latent_dim, mode, critic_mode, in_dim). Old checkpoints
    have no cond_* keys -> ("none", 0, "concat", "raw", 12) = today's networks.
    """
    g = (lambda k, d: getattr(args, k, d)) if args is not None else (lambda k, d: d)
    source = cond_source if cond_source is not None else g("cond_source", "none")
    latent = int(cond_latent_dim if cond_latent_dim is not None else g("cond_latent_dim", 0))
    mode = cond_mode if cond_mode is not None else g("cond_mode", "concat")
    critic = cond_critic if cond_critic is not None else g("cond_critic", "raw")
    in_dim = int(cond_in_dim if cond_in_dim is not None else g("cond_in_dim", COND_IN_DIM_DEFAULT))
    if source not in COND_SOURCES:
        raise ValueError(f"cond_source must be one of {COND_SOURCES}, got {source!r}")
    if mode not in COND_MODES:
        raise ValueError(f"cond_mode must be one of {COND_MODES}, got {mode!r}")
    if critic not in COND_CRITIC_MODES:
        raise ValueError(f"cond_critic must be one of {COND_CRITIC_MODES}, got {critic!r}")
    if latent < 0:
        raise ValueError("cond_latent_dim must be >= 0")
    return source, latent, mode, critic, in_dim


class ParamEncoder(nn.Module):
    """mu: hidden parameter vector e (…, cond_in_dim) -> latent z (…, latent_dim).

    RMA's env-factor encoder. Applied per token (shared across turbines), so
    the latent is equivariant like everything else. dz = 0 is handled by the
    caller as the identity (UP-OSI: the policy sees raw e)."""

    def __init__(self, cond_in_dim: int, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.out_dim = latent_dim

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)


def masked_mse(pred: torch.Tensor, target: torch.Tensor,
               key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean squared error over REAL tokens only (mask True = padding), averaged
    over the last dim too. Used by the phase-2 adaptation fit."""
    err = (pred - target) ** 2
    if key_padding_mask is None:
        return err.mean()
    valid = (~key_padding_mask).unsqueeze(-1).to(err.dtype)
    n = valid.sum() * err.shape[-1]
    return (err * valid).sum() / n.clamp(min=1.0)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class TransformerActor(nn.Module):
    """
    Transformer-based actor (policy) network for wind farm control.

    Architecture:
    1. Per-turbine observations → embedding via MLP
    2. Add positional encoding (method depends on pos_encoding_type):
       - "absolute_mlp": Position embedding concatenated to token embedding
       - "relative_mlp": Position used to compute attention bias
    3. Project to embed_dim
    4. ADD receptivity profile encoding (if enabled)
    5. Process through transformer (turbines attend to each other)
    6. Per-turbine action heads (shared weights across turbines)
    The shared action head ensures permutation equivariance:
    swapping two turbines' inputs swaps their outputs.
    """

    def __init__(
        self,
        obs_dim_per_turbine: int,
        action_dim_per_turbine: int = 1,
        embed_dim: int = 128,
        pos_embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
        # Positional encoding settings
        pos_encoding_type: str = "absolute_mlp",
        rel_pos_hidden_dim: int = 64,
        rel_pos_per_head: bool = True,
        pos_embedding_mode: str = "concat",  # "add" or "concat" for absolute types
        # Receptivity profile settings
        profile_encoding: Optional[str] = None,
        profile_encoder_hidden: int = 128,
        n_profile_directions: int = 360,
        profile_fusion_type: str = "add",  # "add" or "joint"
        profile_embed_mode: str = "add",   # "add" or "concat"
        # Shared profile encoders (optional - if None, creates own)
        shared_recep_encoder: Optional[nn.Module] = None,
        shared_influence_encoder: Optional[nn.Module] = None,
        args: Optional[Args] = None,  # For flexible encoder kwargs (e.g. Fourier n_harmonics, MultiRes scales
        # Stage 10 parameter conditioning (None -> read from args; defaults = off)
        cond_source: Optional[str] = None,
        cond_latent_dim: Optional[int] = None,
        cond_mode: Optional[str] = None,
        cond_in_dim: Optional[int] = None,
    ):
        """
        Args:
            obs_dim_per_turbine: Observation dimension per turbine
            action_dim_per_turbine: Action dimension per turbine (1 for yaw)
            embed_dim: Transformer hidden dimension
            pos_embed_dim: Positional encoding dimension (for absolute types)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            mlp_ratio: FFN expansion ratio
            dropout: Dropout rate
            action_scale: Scale for tanh output
            action_bias: Bias for tanh output
            pos_encoding_type: Type of positional encoding (see VALID_POS_ENCODING_TYPES)
            rel_pos_hidden_dim: Hidden dimension for relative position MLP
            rel_pos_per_head: Whether relative bias is per-head
            profile_encoding: Type of profile encoding (see VALID_PROFILE_ENCODING_TYPES)
            profile_encoder_hidden: Hidden dimension in profile encoder
            n_profile_directions: Number of directions in profile
        """
        super().__init__()

        self.obs_dim_per_turbine = obs_dim_per_turbine
        self.action_dim_per_turbine = action_dim_per_turbine
        # Stage 10 conditioning: cond_on => forward_trunk REQUIRES cond (the
        # hidden-parameter vector e, (batch, n_turb, cond_in_dim), or a latent
        # when cond_is_latent). cond_dim = what reaches the trunk: the latent
        # size dz, or cond_in_dim when dz == 0 (identity encoder, UP-OSI).
        (self.cond_source, self.cond_latent_dim, self.cond_mode,
         _cc, self.cond_in_dim) = _read_cond_cfg(
            args, cond_source, cond_latent_dim, cond_mode, None, cond_in_dim)
        self.cond_on = self.cond_source != "none"
        self.cond_dim = (self.cond_latent_dim or self.cond_in_dim) if self.cond_on else 0
        self.param_encoder: Optional[nn.Module] = None   # created after fc_logstd (seeded-init order)
        # Entropy aggregation over turbines for the SAC log-prob ("sum" | "mean").
        # "mean" keeps per-turbine entropy pressure size-invariant (see config.entropy_agg).
        self.entropy_agg = getattr(args, "entropy_agg", "sum") if args is not None else "sum"
        self.embed_dim = embed_dim
        self.pos_encoding_type = pos_encoding_type

        self.profile_encoding = profile_encoding
        self.profile_fusion_type = profile_fusion_type
        self.profile_embed_mode = profile_embed_mode

        assert profile_fusion_type in ("add", "joint"), \
            f"Invalid profile_fusion_type: {profile_fusion_type}"
        assert profile_embed_mode in ("add", "concat"), \
            f"Invalid profile_embed_mode: {profile_embed_mode}"

        # Create positional encoding modules based on type
        self.pos_encoder, self.rel_pos_bias, self.embedding_mode = \
            create_positional_encoding(
                encoding_type=pos_encoding_type,
                embed_dim=embed_dim,
                pos_embed_dim=pos_embed_dim,
                num_heads=num_heads,
                rel_pos_hidden_dim=rel_pos_hidden_dim,
                rel_pos_per_head=rel_pos_per_head,
                embedding_mode=pos_embedding_mode,
            )


        # Whether to use the (redundant) influence rose; False => single receptivity encoder
        self.use_influence = getattr(args, "profile_use_influence", True) if args is not None else True

        # Receptivity profile encoder (optional)
        # Use shared encoders if provided, otherwise create new ones
        if shared_recep_encoder is not None:
            self.recep_encoder = shared_recep_encoder
            self.influence_encoder = shared_influence_encoder if self.use_influence else None
        else:
            encoder_kwargs = json.loads(args.profile_encoder_kwargs)
            if "hidden_channels" in encoder_kwargs:  # silently popping it wasted a whole sweep once
                raise ValueError(
                    "hidden_channels in --profile_encoder_kwargs is ignored; "
                    "use the --profile_encoder_hidden flag instead."
                )
            self.recep_encoder, self.influence_encoder = \
                create_profile_encoding(
                    profile_type=profile_encoding,
                    embed_dim=embed_dim,
                    hidden_channels=profile_encoder_hidden,
                    use_influence=self.use_influence,
                    **encoder_kwargs,
                )



        if profile_encoding is not None and self.use_influence and profile_fusion_type == "joint":
            # self.profile_fusion = nn.Sequential(
            #     nn.Linear(2 * embed_dim, embed_dim),
            #     nn.LayerNorm(embed_dim),
            #     nn.GELU(),
            #     nn.Linear(embed_dim, embed_dim),
            # )
            self.profile_fusion = nn.Linear(2 * embed_dim, embed_dim)

        if profile_encoding is not None and profile_embed_mode == "concat":
            self.profile_proj = nn.Linear(2 * embed_dim, embed_dim)


        # Observation encoder (shared across turbines). change_wd_4: "per_sensor"
        # swaps in one small MLP per sensor group (see PerSensorObsEncoder);
        # forward() is untouched because the input/output shapes are identical.
        # Stage 10 concat: z is appended to each token's obs features, so only
        # the first Linear's input width changes (same state_dict keys).
        concat_dim = self.cond_dim if (self.cond_on and self.cond_mode == "concat") else 0
        film_dim = self.cond_dim if (self.cond_on and self.cond_mode == "film") else 0
        if getattr(args, "obs_encoder_mode", "shared") == "per_sensor":
            if concat_dim:
                raise ValueError("cond_mode concat is incompatible with --obs_encoder_mode per_sensor (use film)")
            self.obs_encoder = PerSensorObsEncoder(
                obs_dim_per_turbine, embed_dim,
                history_length=int(getattr(args, "history_length")),
            )
        else:
            self.obs_encoder = nn.Sequential(
                nn.Linear(obs_dim_per_turbine + concat_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        # Input projection: only needed when concatenating position embedding
        if self.embedding_mode == "concat":
            self.input_proj = nn.Linear(embed_dim + pos_embed_dim, embed_dim)
        else:
            self.input_proj = nn.Identity()


        # Standard transformer (with optional attention bias + v5 log-N scaling / local attention)
        (_ls, _sm, self.attn_local, self.attn_local_radius_D,
         self.attn_local_k, self.attn_local_cone_deg) = _read_attn_cfg(args)
        self.transformer = TransformerEncoder(
            embed_dim, num_heads, num_layers, mlp_ratio, dropout,
            attn_logit_scale=_ls, attn_softmax=_sm, film_dim=film_dim,
        )

        # Action heads (shared across turbines)
        self.fc_mean = nn.Linear(embed_dim, action_dim_per_turbine)
        self.fc_logstd = nn.Linear(embed_dim, action_dim_per_turbine)

        # Stage 10: parameter encoder mu AFTER the existing modules so the
        # seeded init of everything above is unchanged (dz = 0 -> identity).
        if self.cond_on and self.cond_latent_dim > 0:
            self.param_encoder = ParamEncoder(self.cond_in_dim, self.cond_latent_dim)

        # Action scaling
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias_val", torch.tensor(action_bias, dtype=torch.float32))

    def encode_cond(self, e: torch.Tensor) -> torch.Tensor:
        """z = mu(e) (identity when cond_latent_dim == 0). Phase 2 regresses this."""
        if self.param_encoder is None:
            return e
        return self.param_encoder(e)

    def _resolve_cond(self, cond: Optional[torch.Tensor], cond_is_latent: bool) -> Optional[torch.Tensor]:
        """Check the cond contract and return z (or None when cond is off).

        cond is ALWAYS a tensor when the source is on and ALWAYS None when
        off; mixing the two is a hard error (a stray None would also be a
        second torch.compile graph, see get_action)."""
        if not self.cond_on:
            if cond is not None:
                raise ValueError("cond passed to an actor whose cond_source is 'none'")
            return None
        if cond is None:
            raise ValueError(f"cond_source={self.cond_source!r} requires cond (batch, n_turb, {self.cond_in_dim})")
        if cond_is_latent or self.param_encoder is None:
            if cond.shape[-1] != self.cond_dim:
                raise ValueError(f"cond latent must have {self.cond_dim} features, got {cond.shape[-1]}")
            return cond
        if cond.shape[-1] != self.cond_in_dim:
            raise ValueError(f"cond must have {self.cond_in_dim} features, got {cond.shape[-1]}")
        return self.param_encoder(cond)

    def forward_trunk(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
        need_weights: bool = False,  # Whether to return attention weights for debugging
        cond: Optional[torch.Tensor] = None,      # Stage 10: e (batch, n_turb, cond_in_dim) or latent z
        cond_is_latent: bool = False,             # True: cond is already z (phase-2 adaptation output)
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Trunk forward: everything up to (and including) the transformer, i.e.
        forward() minus the action heads. The PPO trainer's shared-trunk value
        head reads these per-turbine embeddings directly.

        Args:
            obs: (batch, n_turbines, obs_dim_per_turbine)
            positions: (batch, n_turbines, 2) wind-relative normalized positions
            key_padding_mask: (batch, n_turbines) where True = padding
            recep_profile: (batch, n_turbines, n_directions) receptivity profiles (optional)
            influence_profile: (batch, n_turbines, n_directions) influence profiles (optional)
            need_weights: If True, compute and return attention weights for all layers
            cond: hidden-parameter vector e per token (required iff cond_source != "none")
            cond_is_latent: cond is the latent z (skip the parameter encoder)

        Returns:
            h: (batch, n_turbines, embed_dim) final-LayerNorm'd token embeddings
            attn_weights: List of attention weights from each layer
        """
        batch_size, n_turbines, _ = obs.shape

        # Stage 10: resolve the conditioning latent (None when cond is off)
        z = self._resolve_cond(cond, cond_is_latent)
        film_cond = None
        if z is not None:
            if self.cond_mode == "concat":
                # cat promotes to obs dtype (fp32 under AMP: e stays fp32 end to end)
                obs = torch.cat([obs, z.to(obs.dtype)], dim=-1)
            else:
                film_cond = z

        # Encode observations
        h = self.obs_encoder(obs)  # (batch, n_turb, embed_dim)

        # Apply positional encoding based on type
        if self.embedding_mode == "concat" and self.pos_encoder is not None:
            # Absolute encoding: concatenate position embedding
            pos_embed = self.pos_encoder(positions)  # (batch, n_turb, pos_embed_dim)
            h = torch.cat([h, pos_embed], dim=-1)  # (batch, n_turb, embed_dim + pos_embed_dim)
        elif self.embedding_mode == "add" and self.pos_encoder is not None:
            # Absolute encoding: add position embedding
            pos_embed = self.pos_encoder(positions)
            h = h + pos_embed  # (batch, n_turb, embed_dim)


        # Project to embed_dim
        h = self.input_proj(h)  # (batch, n_turb, embed_dim)

        # Profile encoding (after projection, like positional encoding in LLMs)
        if self.recep_encoder and recep_profile is not None:
            recep_embed = self.recep_encoder(recep_profile)  # (batch, n_turb, embed_dim)

            # Step 1: profile embedding -- fuse with influence, or use receptivity alone
            if self.use_influence and influence_profile is not None:
                influence_embed = self.influence_encoder(influence_profile)  # (batch, n_turb, embed_dim)
                if self.profile_fusion_type == "joint":
                    profile_embed = self.profile_fusion(
                        torch.cat([recep_embed, influence_embed], dim=-1)
                    )  # (batch, n_turb, embed_dim)
                else:  # "add"
                    profile_embed = recep_embed + influence_embed  # (batch, n_turb, embed_dim)
            else:  # single-rose: receptivity only
                profile_embed = recep_embed

            # Step 2: Integrate profile embedding into token representation
            if self.profile_embed_mode == "concat":
                h = self.profile_proj(torch.cat([h, profile_embed], dim=-1))
            else:  # "add"
                h = h + profile_embed





        # Compute relative position bias if using relative encoding
        attn_bias = None
        if self.rel_pos_bias is not None:
            attn_bias = self.rel_pos_bias(positions, key_padding_mask)

        # v5: local-attention mask from (rotor-diameter) positions
        local_allow = None
        if self.attn_local != "none":
            local_allow = neighbour_allow_mask(
                positions, key_padding_mask, self.attn_local,
                radius_D=self.attn_local_radius_D, k=self.attn_local_k,
                cone_deg=self.attn_local_cone_deg)

        h, attn_weights = self.transformer(h, key_padding_mask, attn_bias,
                                           local_allow=local_allow, need_weights=need_weights,
                                           cond=film_cond)

        return h, attn_weights

    def forward(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
        need_weights: bool = False,  # Whether to return attention weights for debugging
        cond: Optional[torch.Tensor] = None,
        cond_is_latent: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning action distribution parameters.

        Args:
            obs: (batch, n_turbines, obs_dim_per_turbine)
            positions: (batch, n_turbines, 2) wind-relative normalized positions
            key_padding_mask: (batch, n_turbines) where True = padding
            recep_profile: (batch, n_turbines, n_directions) receptivity profiles (optional)
            influence_profile: (batch, n_turbines, n_directions) influence profiles (optional)
            need_weights: If True, compute and return attention weights for all layers

        Returns:
            mean: (batch, n_turbines, action_dim) action means
            log_std: (batch, n_turbines, action_dim) action log stds
            attn_weights: List of attention weights from each layer
        """
        h, attn_weights = self.forward_trunk(
            obs, positions, key_padding_mask,
            recep_profile, influence_profile, need_weights,
            cond=cond, cond_is_latent=cond_is_latent,
        )

        # Action distribution parameters
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h)

        # Constrain log_std to reasonable range
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std, attn_weights

    def get_action(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        cond: Optional[torch.Tensor] = None,
        cond_is_latent: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Sample action from policy with log probability.

        Args:
            obs: (batch, n_turbines, obs_dim)
            positions: (batch, n_turbines, 2) wind-relative normalized positions
            key_padding_mask: (batch, n_turbines) where True = padding
            deterministic: If True, return mean action
            recep_profile: (batch, n_turbines, n_directions) receptivity profiles (optional)
            influence_profile: (batch, n_turbines, n_directions) influence profiles (optional)
            need_weights: If True, return attention weights (expensive). Default False.

        Returns:
            action: (batch, n_turbines, action_dim) sampled actions
            log_prob: (batch, 1) log probability of actions
            mean_action: (batch, n_turbines, action_dim) mean actions
            attn_weights: List of attention weights (empty if need_weights=False)
        """
        # Single-rose mode: never pass the (unused) influence tensor into the compiled
        # forward -- a phantom static input corrupts the reduce-overhead cudagraph-trees
        # allocator on eval-time recapture. None is not a graph input.
        if not self.use_influence:
            influence_profile = None
        mean, log_std, attn_weights = self.forward(obs, positions, key_padding_mask,
                                                   recep_profile, influence_profile,
                                                   need_weights=need_weights,
                                                   cond=cond, cond_is_latent=cond_is_latent)
        std = log_std.exp()

        # Sample from Gaussian
        normal = torch.distributions.Normal(mean, std)
        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()  # Reparameterization trick

        # Apply tanh squashing
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias_val

        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)

        # Mask out padded positions before aggregating
        if key_padding_mask is not None:
            # key_padding_mask: (batch, n_turbines), True = padding
            mask = ~key_padding_mask.unsqueeze(-1)  # (batch, n_turb, 1), True = real
            mask_f = mask.float()
            log_prob = log_prob * mask_f
            n_real_dims = mask_f.sum(dim=(-2, -1)) * log_prob.shape[-1]  # (batch,) real action dims
        else:
            n_real_dims = log_prob.new_full(
                (log_prob.shape[0],), float(log_prob.shape[-2] * log_prob.shape[-1])
            )

        # Aggregate over turbines and action dims -> (batch, 1)
        log_prob = log_prob.sum(dim=(-2, -1), keepdim=False)
        if self.entropy_agg == "mean":
            # Per-(turbine,action) MEAN: keeps entropy O(1) regardless of farm size N.
            log_prob = log_prob / n_real_dims.clamp(min=1.0)
        log_prob = log_prob.unsqueeze(-1)

        # Mean action (for logging)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias_val

        return action, log_prob, mean_action, attn_weights


# =============================================================================
# CRITIC NETWORK
# =============================================================================

class TransformerCritic(nn.Module):
    """
    Transformer-based critic (Q-function) network.

    Architecture:
    1. Concatenate per-turbine observations and actions
    2. Encode via MLP
    3. Add positional encoding (if using additive type)
    4. Project to embed_dim
    5. profile_encoding: Type of profile encoding (see VALID_PROFILE_ENCODING_TYPES)
    6. Process through transformer
    7. Pool over turbines (masked mean) → single Q-value

    The pooling operation aggregates information from all turbines
    into a single scalar Q-value for the entire farm.
    """

    def __init__(
        self,
        obs_dim_per_turbine: int,
        action_dim_per_turbine: int = 1,
        embed_dim: int = 128,
        pos_embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        # Positional encoding settings
        pos_encoding_type: str = "absolute_mlp",
        rel_pos_hidden_dim: int = 64,
        rel_pos_per_head: bool = True,
        pos_embedding_mode: str = "concat",  # "add" or "concat" for absolute types
        # PyWake profile settings
        profile_encoding: Optional[str] = None,
        profile_encoder_hidden: int = 128,
        n_profile_directions: int = 360,
        profile_fusion_type: str = "add",  # "add" or "joint"
        profile_embed_mode: str = "add",
        # Shared profile encoders (optional - if None, creates own)
        shared_recep_encoder: Optional[nn.Module] = None,
        shared_influence_encoder: Optional[nn.Module] = None,
        args: Optional[Args] = None,  # For flexible encoder kwargs (e.g. Fourier n_harmonics, MultiRes scales
        # DroQ settings (dropout + LayerNorm in critic MLPs)
        droq_dropout: float = 0.0,
        droq_layer_norm: bool = False,
        cond_critic: Optional[str] = None,   # Stage 10: "raw" | "none" (None -> args)
        cond_in_dim: Optional[int] = None
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.pos_encoding_type = pos_encoding_type
        self.profile_encoding = profile_encoding
        self.profile_fusion_type = profile_fusion_type
        self.profile_embed_mode = profile_embed_mode
        # v9: "pool" (masked-mean embeddings -> farm-Q) or "vdn" (per-turbine q_head -> masked-sum).
        self.critic_agg = getattr(args, "critic_agg", "pool") if args is not None else "pool"

        # Create positional encoding modules based on type
        self.pos_encoder, self.rel_pos_bias, self.embedding_mode = \
            create_positional_encoding(
                encoding_type=pos_encoding_type,
                embed_dim=embed_dim,
                pos_embed_dim=pos_embed_dim,
                num_heads=num_heads,
                rel_pos_hidden_dim=rel_pos_hidden_dim,
                rel_pos_per_head=rel_pos_per_head,
                embedding_mode=pos_embedding_mode,
            )

        # Whether to use the (redundant) influence rose; False => single receptivity encoder
        self.use_influence = getattr(args, "profile_use_influence", True) if args is not None else True

        # PyWake profile encoder (optional)
        # Use shared encoders if provided, otherwise create new ones
        if shared_recep_encoder is not None:
            self.recep_encoder = shared_recep_encoder
            self.influence_encoder = shared_influence_encoder if self.use_influence else None
        else:
            encoder_kwargs = json.loads(args.profile_encoder_kwargs)
            if "hidden_channels" in encoder_kwargs:  # silently popping it wasted a whole sweep once
                raise ValueError(
                    "hidden_channels in --profile_encoder_kwargs is ignored; "
                    "use the --profile_encoder_hidden flag instead."
                )
            self.recep_encoder, self.influence_encoder = \
                create_profile_encoding(
                    profile_type=profile_encoding,
                    embed_dim=embed_dim,
                    hidden_channels=profile_encoder_hidden,
                    use_influence=self.use_influence,
                    **encoder_kwargs,
                )


        # Observation + action encoder (no DroQ here — applied only in q_head per Hiraoka et al.)
        # change_wd_4 "per_sensor": per-sensor MLPs + a dedicated action MLP
        # (see PerSensorObsActionEncoder); same input/output shapes, forward untouched.
        # Stage 10 asymmetric critic: raw e per token when cond_critic == "raw"
        # and the actor is conditioned (cond_source != "none"); otherwise 0.
        (_cs, _cl, _cm, _cc, _cin) = _read_cond_cfg(args, cond_critic=cond_critic, cond_in_dim=cond_in_dim)
        self.cond_critic_dim = _cin if (_cs != "none" and _cc == "raw") else 0
        if getattr(args, "obs_encoder_mode", "shared") == "per_sensor":
            if self.cond_critic_dim:
                raise ValueError("cond_critic raw is incompatible with --obs_encoder_mode per_sensor (use cond_critic none)")
            self.obs_action_encoder = PerSensorObsActionEncoder(
                obs_dim_per_turbine, action_dim_per_turbine, embed_dim,
                history_length=int(getattr(args, "history_length")),
            )
        else:
            self.obs_action_encoder = nn.Sequential(
                nn.Linear(obs_dim_per_turbine + action_dim_per_turbine + self.cond_critic_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        # Input projection: only needed when concatenating position embedding
        if self.embedding_mode == "concat":
            self.input_proj = nn.Linear(embed_dim + pos_embed_dim, embed_dim)
        else:
            self.input_proj = nn.Identity()

        if profile_encoding is not None and self.use_influence and profile_fusion_type == "joint":
            # self.profile_fusion = nn.Sequential(
            #     nn.Linear(2 * embed_dim, embed_dim),
            #     nn.LayerNorm(embed_dim),
            #     nn.GELU(),
            #     nn.Linear(embed_dim, embed_dim),
            # ))
            self.profile_fusion = nn.Linear(2 * embed_dim, embed_dim)

        if profile_encoding is not None and profile_embed_mode == "concat":
            self.profile_proj = nn.Linear(2 * embed_dim, embed_dim)

        # Transformer encoder (+ v5 log-N scaling / local attention)
        (_ls, _sm, self.attn_local, self.attn_local_radius_D,
         self.attn_local_k, self.attn_local_cone_deg) = _read_attn_cfg(args)
        self.transformer = TransformerEncoder(
            embed_dim, num_heads, num_layers, mlp_ratio, dropout,
            attn_logit_scale=_ls, attn_softmax=_sm,
        )

        # Q-value head (after pooling)
        q_head_layers: list[nn.Module] = [nn.Linear(embed_dim, embed_dim)]
        if droq_layer_norm:
            q_head_layers.append(nn.LayerNorm(embed_dim))
        q_head_layers.append(nn.ReLU())
        if droq_dropout > 0.0:
            q_head_layers.append(nn.Dropout(droq_dropout))
        q_head_layers.append(nn.Linear(embed_dim, 1))
        self.q_head = nn.Sequential(*q_head_layers)

    def forward_trunk(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,   # Stage 10: raw e (batch, n_turb, cond_in_dim)
    ) -> torch.Tensor:
        """
        Everything up to (and including) the transformer: encode obs+action,
        positional/profile encoding, transformer. Returns per-turbine embeddings
        h of shape (batch, n_turbines, embed_dim) — no aggregation, no q_head.
        """
        batch_size = obs.shape[0]

        # Concatenate obs and action (+ raw e for the asymmetric critic)
        if self.cond_critic_dim:
            if cond is None:
                raise ValueError(f"cond_critic raw requires cond (batch, n_turb, {self.cond_critic_dim})")
            x = torch.cat([obs, action, cond.to(obs.dtype)], dim=-1)
        else:
            if cond is not None:
                raise ValueError("cond passed to a critic without conditioning (cond_critic none / cond_source none)")
            x = torch.cat([obs, action], dim=-1)

        # Encode
        h = self.obs_action_encoder(x)

        # Apply positional encoding based on type
        if self.embedding_mode == "concat" and self.pos_encoder is not None:
            pos_embed = self.pos_encoder(positions)
            h = torch.cat([h, pos_embed], dim=-1)
        elif self.embedding_mode == "add" and self.pos_encoder is not None:
            pos_embed = self.pos_encoder(positions)
            h = h + pos_embed

        # Project to embed_dim
        h = self.input_proj(h)

        # Profile encoding (after projection, like positional encoding in LLMs)
        if self.recep_encoder and recep_profile is not None:
            recep_embed = self.recep_encoder(recep_profile)  # (batch, n_turb, embed_dim)

            # Step 1: profile embedding -- fuse with influence, or use receptivity alone
            if self.use_influence and influence_profile is not None:
                influence_embed = self.influence_encoder(influence_profile)  # (batch, n_turb, embed_dim)
                if self.profile_fusion_type == "joint":
                    profile_embed = self.profile_fusion(
                        torch.cat([recep_embed, influence_embed], dim=-1)
                    )
                else:
                    profile_embed = recep_embed + influence_embed
            else:  # single-rose: receptivity only
                profile_embed = recep_embed

            # Step 2: Integrate into token representation
            if self.profile_embed_mode == "concat":
                h = self.profile_proj(torch.cat([h, profile_embed], dim=-1))
            else:
                h = h + profile_embed


        # Compute relative position bias if using relative encoding
        attn_bias = None
        if self.rel_pos_bias is not None:
            attn_bias = self.rel_pos_bias(positions, key_padding_mask)

        # v5: local-attention mask from (rotor-diameter) positions
        local_allow = None
        if self.attn_local != "none":
            local_allow = neighbour_allow_mask(
                positions, key_padding_mask, self.attn_local,
                radius_D=self.attn_local_radius_D, k=self.attn_local_k,
                cone_deg=self.attn_local_cone_deg)

        # Transformer (no need for attention weights in critic)
        h, _ = self.transformer(h, key_padding_mask, attn_bias,
                                local_allow=local_allow, need_weights=False)

        return h

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Q-value for observation-action pair.

        Args:
            obs: (batch, n_turbines, obs_dim)
            action: (batch, n_turbines, action_dim)
            positions: (batch, n_turbines, 2) wind-relative normalized positions
            key_padding_mask: (batch, n_turbines) where True = padding
            recep_profile: (batch, n_turbines, n_directions) receptivity profiles (optional)
            influence_profile: (batch, n_turbines, n_directions) influence profiles (optional)

        Returns:
            q_value: (batch, 1) Q-value for the entire farm
        """
        h = self.forward_trunk(obs, action, positions, key_padding_mask,
                               recep_profile, influence_profile, cond=cond)

        if self.critic_agg == "vdn":
            # v9 value decomposition: per-turbine Q-head, then masked-SUM over turbines.
            # q_head applies to the last dim, so it broadcasts over (batch, n_turb, embed).
            # Padded turbines' embeddings are non-zero, so zero their per-turbine Q before summing.
            q_per = self.q_head(h)  # (batch, n_turb, q_out)  [q_out=1 for SAC, n_quantiles for TQC]
            if key_padding_mask is not None:
                valid = (~key_padding_mask).unsqueeze(-1).float()  # (batch, n_turb, 1)
                q_per = q_per * valid
            q = q_per.sum(dim=1)  # (batch, q_out) — un-diluted per-turbine credit
        else:
            # "pool" (standard): masked-mean of turbine embeddings -> single farm-Q.
            if key_padding_mask is not None:
                mask = ~key_padding_mask.unsqueeze(-1)  # (batch, n_turb, 1), True = real
                mask_f = mask.float()
                h = h * mask_f
                h_sum = h.sum(dim=1)  # (batch, embed_dim)
                n_real = mask_f.sum(dim=1).clamp(min=1)  # (batch, 1)
                h_pooled = h_sum / n_real
            else:
                h_pooled = h.mean(dim=1)  # (batch, embed_dim)
            q = self.q_head(h_pooled)  # (batch, 1)

        return q


# =============================================================================
# TQC CRITIC (Truncated Quantile Critics)
# =============================================================================

def _make_tqc_head(
    embed_dim: int,
    n_quantiles: int,
    droq_dropout: float = 0.0,
    droq_layer_norm: bool = False,
) -> nn.Sequential:
    """Quantile head MLP: Linear -> [LayerNorm] -> ReLU -> [Dropout] -> Linear.

    Same layer order as TransformerCritic's q_head so RNG consumption (and thus
    seeded init) is unchanged for the independent TQC critic.
    """
    layers: list[nn.Module] = [nn.Linear(embed_dim, embed_dim)]
    if droq_layer_norm:
        layers.append(nn.LayerNorm(embed_dim))
    layers.append(nn.ReLU())
    if droq_dropout > 0.0:
        layers.append(nn.Dropout(droq_dropout))
    layers.append(nn.Linear(embed_dim, n_quantiles))
    return nn.Sequential(*layers)


class TransformerTQCCritic(nn.Module):
    """
    TQC critic: N independent TransformerCritic networks, each outputting
    M quantiles instead of a single Q-value.

    Forward returns (n_critics, batch, n_quantiles).
    """

    def __init__(self, n_critics: int, n_quantiles: int, **critic_kwargs):
        super().__init__()
        self.n_critics = n_critics
        self.n_quantiles = n_quantiles
        self.critics = nn.ModuleList([
            TransformerCritic(**critic_kwargs) for _ in range(n_critics)
        ])
        # Override each critic's q_head to output n_quantiles instead of 1
        # Read DroQ settings (without removing — TransformerCritic also uses them)
        droq_dropout = critic_kwargs.get("droq_dropout", 0.0)
        droq_layer_norm = critic_kwargs.get("droq_layer_norm", False)

        for critic in self.critics:
            critic.q_head = _make_tqc_head(
                critic.embed_dim, n_quantiles, droq_dropout, droq_layer_norm)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns (n_critics, batch, n_quantiles)."""
        return torch.stack([
            c(obs, action, positions, key_padding_mask,
              recep_profile, influence_profile, cond=cond)
            for c in self.critics
        ], dim=0)


class TransformerTQCSharedCritic(nn.Module):
    """
    Shared-trunk TQC critic: ONE TransformerCritic trunk feeding n_critics small
    quantile heads. TQC's quantile truncation handles overestimation without
    independent ensembles, so the heads may be correlated — the payoff is ~1/N
    the trunk compute and params vs TransformerTQCCritic.

    Same output contract as TransformerTQCCritic: (n_critics, batch, n_quantiles).
    Checkpoints are NOT interchangeable with the independent TQC critic.
    """

    def __init__(self, n_critics: int, n_quantiles: int, **critic_kwargs):
        super().__init__()
        self.n_critics = n_critics
        self.n_quantiles = n_quantiles
        self.trunk = TransformerCritic(**critic_kwargs)
        self.critic_agg = self.trunk.critic_agg
        # Drop the trunk's unused scalar q_head so it never shows up in
        # state_dict/parameters (only the agg branch reads it, which we bypass).
        self.trunk.q_head = nn.Identity()
        droq_dropout = critic_kwargs.get("droq_dropout", 0.0)
        droq_layer_norm = critic_kwargs.get("droq_layer_norm", False)
        self.heads = nn.ModuleList([
            _make_tqc_head(self.trunk.embed_dim, n_quantiles,
                           droq_dropout, droq_layer_norm)
            for _ in range(n_critics)
        ])

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns (n_critics, batch, n_quantiles)."""
        h = self.trunk.forward_trunk(obs, action, positions, key_padding_mask,
                                     recep_profile, influence_profile, cond=cond)
        # Python loop over heads (N tiny MLPs): measured faster than a vmap
        # ensemble under reduce-overhead cudagraphs (see trainer compile block).
        if self.critic_agg == "vdn":
            # Per-turbine quantiles -> zero padded turbines -> masked-SUM.
            valid = None
            if key_padding_mask is not None:
                valid = (~key_padding_mask).unsqueeze(-1).float()  # (batch, n_turb, 1)
            outs = []
            for head in self.heads:
                q_per = head(h)  # (batch, n_turb, n_quantiles)
                if valid is not None:
                    q_per = q_per * valid
                outs.append(q_per.sum(dim=1))  # (batch, n_quantiles)
            return torch.stack(outs, dim=0)
        else:
            # "pool": masked-mean pool ONCE, then all heads on the pooled embedding.
            if key_padding_mask is not None:
                mask = ~key_padding_mask.unsqueeze(-1)  # (batch, n_turb, 1), True = real
                mask_f = mask.float()
                h = h * mask_f
                h_sum = h.sum(dim=1)  # (batch, embed_dim)
                n_real = mask_f.sum(dim=1).clamp(min=1)  # (batch, 1)
                h_pooled = h_sum / n_real
            else:
                h_pooled = h.mean(dim=1)  # (batch, embed_dim)
            return torch.stack([head(h_pooled) for head in self.heads], dim=0)


# =============================================================================
# ADAPTATION MODULE (Stage 10 phase 2: RMA phi / UP-OSI OSI / Lee student encoder)
# =============================================================================

class AdaptationModule(nn.Module):
    """phi: per-token [obs | commanded-action history] (+ positions, profiles)
    -> z_hat (batch, n_turb, out_dim), the estimate of the actor's latent z =
    mu(e) (or raw e when dz == 0).

    Reuses the actor trunk class with conditioning forced OFF over the widened
    token (obs_dim + action_hist_dim), drops the action heads (Identity, the
    PPO shared-trunk precedent) and adds a Linear head. A transformer over
    turbines on purpose: ct_gain and the farm closure are only identifiable
    through the DOWNSTREAM turbines' histories, not from one turbine's own.
    """

    def __init__(self, obs_dim_per_turbine: int, action_hist_dim: int, out_dim: int,
                 **trunk_kwargs):
        super().__init__()
        trunk_kwargs = dict(trunk_kwargs)
        trunk_kwargs["cond_source"] = "none"      # explicit kwarg beats args.cond_source
        trunk_kwargs.pop("cond_latent_dim", None)
        trunk_kwargs.pop("cond_mode", None)
        trunk_kwargs.pop("cond_in_dim", None)
        trunk_kwargs.pop("action_scale", None)
        trunk_kwargs.pop("action_bias", None)
        self.obs_dim_per_turbine = int(obs_dim_per_turbine)
        self.action_hist_dim = int(action_hist_dim)
        self.out_dim = int(out_dim)
        self.trunk = TransformerActor(
            obs_dim_per_turbine=self.obs_dim_per_turbine + self.action_hist_dim,
            action_dim_per_turbine=1, **trunk_kwargs,
        )
        self.trunk.fc_mean = nn.Identity()
        self.trunk.fc_logstd = nn.Identity()
        self.head = nn.Linear(self.trunk.embed_dim, self.out_dim)

    def forward(
        self,
        obs: torch.Tensor,
        action_hist: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = torch.cat([obs, action_hist.to(obs.dtype)], dim=-1)
        h, _ = self.trunk.forward_trunk(x, positions, key_padding_mask,
                                        recep_profile, influence_profile)
        return self.head(h)


def quantile_huber_loss(
    quantiles_pred: torch.Tensor,
    target: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """
    Quantile regression loss with Huber penalty.

    Args:
        quantiles_pred: (batch, n_quantiles) predicted quantile values
        target: (batch, 1) target Q-values
        taus: (n_quantiles,) quantile midpoints
        kappa: Huber loss threshold
    Returns:
        Scalar loss
    """
    # Pairwise TD errors: (batch, 1, n_quantiles) - (batch, 1, 1)
    td_error = target.unsqueeze(-1) - quantiles_pred.unsqueeze(1)
    huber = torch.where(
        td_error.abs() <= kappa,
        0.5 * td_error.pow(2),
        kappa * (td_error.abs() - 0.5 * kappa),
    )
    quantile_weight = (taus - (td_error < 0).float()).abs()
    return (quantile_weight * huber).mean()
