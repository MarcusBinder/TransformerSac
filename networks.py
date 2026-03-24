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
    influence_encoder = cls(embed_dim=embed_dim, hidden_channels=hidden_channels, **defaults)

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
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Pre-norm layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
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
        need_weights: bool = False,  # NEW
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_tokens, embed_dim)
            key_padding_mask: (batch, n_tokens) where True = ignore this position
            attention: (batch, n_heads, n_tokens, n_tokens) optional bias to add
                       to attention logits (for relative positional encoding)

        Returns:
            x: Transformed tensor, same shape as input
            attn_weights: (batch, n_heads, n_tokens, n_tokens) attention weights
        """
        # Self-attention with pre-norm
        x_norm = self.norm1(x)

        # If we have attention bias, we need to use it as attn_mask
        # PyTorch's MultiheadAttention adds attn_mask to attention logits
        if attn_bias is not None:
            # attn_mask in PyTorch MHA: (batch * num_heads, tgt_len, src_len) or (tgt_len, src_len)
            # We need to reshape our bias: (batch, num_heads, n, n) → (batch * num_heads, n, n)
            batch_size, num_heads, n, _ = attn_bias.shape
            attn_mask = attn_bias.reshape(batch_size * num_heads, n, n)
        else:
            attn_mask = None

        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            average_attn_weights=False,  # Return per-head weights
            need_weights=need_weights,  # Only compute if needed!
        )
        x = x + attn_out

        # FFN with pre-norm
        x = x + self.mlp(self.norm2(x))

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
        dropout: float = 0.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)  # Final layer norm

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (batch, n_tokens, embed_dim)
            key_padding_mask: (batch, n_tokens) where True = padding
            attn_bias: (batch, n_heads, n_tokens, n_tokens) optional attention bias
            need_weights: If True, return attention weights (expensive). Default False.

        Returns:
            x: Transformed tensor
            all_attn_weights: List of attention weights from each layer (empty if need_weights=False)
        """
        all_attn_weights = []

        for layer in self.layers:
            x, attn_weights = layer(x, key_padding_mask, attn_bias, need_weights=need_weights)
            if need_weights:
                all_attn_weights.append(attn_weights)

        x = self.norm(x)

        return x, all_attn_weights


# =============================================================================
# ACTOR NETWORK
# =============================================================================

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


        # Receptivity profile encoder (optional)
        # Use shared encoders if provided, otherwise create new ones
        if shared_recep_encoder is not None and shared_influence_encoder is not None:
            self.recep_encoder = shared_recep_encoder
            self.influence_encoder = shared_influence_encoder
        else:
            encoder_kwargs = json.loads(args.profile_encoder_kwargs)
            self.recep_encoder, self.influence_encoder = \
                create_profile_encoding(
                    profile_type=profile_encoding,
                    embed_dim=embed_dim,
                    hidden_channels=profile_encoder_hidden,
                    **encoder_kwargs,
                )



        if profile_encoding is not None and profile_fusion_type == "joint":
            # self.profile_fusion = nn.Sequential(
            #     nn.Linear(2 * embed_dim, embed_dim),
            #     nn.LayerNorm(embed_dim),
            #     nn.GELU(),
            #     nn.Linear(embed_dim, embed_dim),
            # )
            self.profile_fusion = nn.Linear(2 * embed_dim, embed_dim)

        if profile_encoding is not None and profile_embed_mode == "concat":
            self.profile_proj = nn.Linear(2 * embed_dim, embed_dim)


        # Observation encoder (shared across turbines)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim_per_turbine, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Input projection: only needed when concatenating position embedding
        if self.embedding_mode == "concat":
            self.input_proj = nn.Linear(embed_dim + pos_embed_dim, embed_dim)
        else:
            self.input_proj = nn.Identity()


        # Standard transformer (with optional attention bias)
        self.transformer = TransformerEncoder(
            embed_dim, num_heads, num_layers, mlp_ratio, dropout
        )

        # Action heads (shared across turbines)
        self.fc_mean = nn.Linear(embed_dim, action_dim_per_turbine)
        self.fc_logstd = nn.Linear(embed_dim, action_dim_per_turbine)

        # Action scaling
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias_val", torch.tensor(action_bias, dtype=torch.float32))

    def forward(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
        need_weights: bool = False,  # Whether to return attention weights for debugging
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
        batch_size, n_turbines, _ = obs.shape

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
        if self.recep_encoder and recep_profile is not None and influence_profile is not None:
            recep_embed = self.recep_encoder(recep_profile)  # (batch, n_turb, embed_dim)
            influence_embed = self.influence_encoder(influence_profile)  # (batch, n_turb, embed_dim)

            # Step 1: Fuse receptivity + influence into a single profile embedding
            if self.profile_fusion_type == "joint":
                profile_embed = self.profile_fusion(
                    torch.cat([recep_embed, influence_embed], dim=-1)
                )  # (batch, n_turb, embed_dim)
            else:  # "add"
                profile_embed = recep_embed + influence_embed  # (batch, n_turb, embed_dim)

            # Step 2: Integrate profile embedding into token representation
            if self.profile_embed_mode == "concat":
                h = self.profile_proj(torch.cat([h, profile_embed], dim=-1))
            else:  # "add"
                h = h + profile_embed





        # Compute relative position bias if using relative encoding
        attn_bias = None
        if self.rel_pos_bias is not None:
            attn_bias = self.rel_pos_bias(positions, key_padding_mask)


        h, attn_weights = self.transformer(h, key_padding_mask, attn_bias, need_weights=need_weights)

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
        mean, log_std, attn_weights = self.forward(obs, positions, key_padding_mask,
                                                   recep_profile, influence_profile,
                                                   need_weights=need_weights)
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

        # Mask out padded positions before summing
        if key_padding_mask is not None:
            # key_padding_mask: (batch, n_turbines), True = padding
            mask = ~key_padding_mask.unsqueeze(-1)  # (batch, n_turb, 1), True = real
            log_prob = log_prob * mask.float()

        # Sum over turbines and action dims -> (batch, 1)
        log_prob = log_prob.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)

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
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.pos_encoding_type = pos_encoding_type
        self.profile_encoding = profile_encoding
        self.profile_fusion_type = profile_fusion_type
        self.profile_embed_mode = profile_embed_mode

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

        # PyWake profile encoder (optional)
        # Use shared encoders if provided, otherwise create new ones
        if shared_recep_encoder is not None and shared_influence_encoder is not None:
            self.recep_encoder = shared_recep_encoder
            self.influence_encoder = shared_influence_encoder
        else:
            encoder_kwargs = json.loads(args.profile_encoder_kwargs)
            self.recep_encoder, self.influence_encoder = \
                create_profile_encoding(
                    profile_type=profile_encoding,
                    embed_dim=embed_dim,
                    hidden_channels=profile_encoder_hidden,
                    **encoder_kwargs,
                )


        # Observation + action encoder (no DroQ here — applied only in q_head per Hiraoka et al.)
        self.obs_action_encoder = nn.Sequential(
            nn.Linear(obs_dim_per_turbine + action_dim_per_turbine, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Input projection: only needed when concatenating position embedding
        if self.embedding_mode == "concat":
            self.input_proj = nn.Linear(embed_dim + pos_embed_dim, embed_dim)
        else:
            self.input_proj = nn.Identity()

        if profile_encoding is not None and profile_fusion_type == "joint":
            # self.profile_fusion = nn.Sequential(
            #     nn.Linear(2 * embed_dim, embed_dim),
            #     nn.LayerNorm(embed_dim),
            #     nn.GELU(),
            #     nn.Linear(embed_dim, embed_dim),
            # ))
            self.profile_fusion = nn.Linear(2 * embed_dim, embed_dim)

        if profile_encoding is not None and profile_embed_mode == "concat":
            self.profile_proj = nn.Linear(2 * embed_dim, embed_dim)

        # Transformer encoder (choose based on encoding type)
        self.transformer = TransformerEncoder(
            embed_dim, num_heads, num_layers, mlp_ratio, dropout
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

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
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
        batch_size = obs.shape[0]

        # Concatenate obs and action
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
        if self.recep_encoder and recep_profile is not None and influence_profile is not None:
            recep_embed = self.recep_encoder(recep_profile)  # (batch, n_turb, embed_dim)
            influence_embed = self.influence_encoder(influence_profile)  # (batch, n_turb, embed_dim)

            # Step 1: Fuse receptivity + influence
            if self.profile_fusion_type == "joint":
                profile_embed = self.profile_fusion(
                    torch.cat([recep_embed, influence_embed], dim=-1)
                )
            else:
                profile_embed = recep_embed + influence_embed

            # Step 2: Integrate into token representation
            if self.profile_embed_mode == "concat":
                h = self.profile_proj(torch.cat([h, profile_embed], dim=-1))
            else:
                h = h + profile_embed


        # Compute relative position bias if using relative encoding
        attn_bias = None
        if self.rel_pos_bias is not None:
            attn_bias = self.rel_pos_bias(positions, key_padding_mask)

        # Transformer (no need for attention weights in critic)
        h, _ = self.transformer(h, key_padding_mask, attn_bias, need_weights=False)


        # Masked mean pooling over turbines
        if key_padding_mask is not None:
            mask = ~key_padding_mask.unsqueeze(-1)  # (batch, n_turb, 1), True = real
            h = h * mask.float()
            h_sum = h.sum(dim=1)  # (batch, embed_dim)
            n_real = mask.float().sum(dim=1).clamp(min=1)  # (batch, 1)
            h_pooled = h_sum / n_real
        else:
            h_pooled = h.mean(dim=1)  # (batch, embed_dim)

        # Q-value
        q = self.q_head(h_pooled)  # (batch, 1)

        return q


# =============================================================================
# TQC CRITIC (Truncated Quantile Critics)
# =============================================================================

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
            embed_dim = critic.embed_dim
            q_head_layers: list[nn.Module] = [nn.Linear(embed_dim, embed_dim)]
            if droq_layer_norm:
                q_head_layers.append(nn.LayerNorm(embed_dim))
            q_head_layers.append(nn.ReLU())
            if droq_dropout > 0.0:
                q_head_layers.append(nn.Dropout(droq_dropout))
            q_head_layers.append(nn.Linear(embed_dim, n_quantiles))
            critic.q_head = nn.Sequential(*q_head_layers)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns (n_critics, batch, n_quantiles)."""
        return torch.stack([
            c(obs, action, positions, key_padding_mask,
              recep_profile, influence_profile)
            for c in self.critics
        ], dim=0)


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
