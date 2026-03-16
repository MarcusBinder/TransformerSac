"""
Transformer-based SAC for Wind Farm Control - V12

Changes in V12:
- Modular positional encoding system with multiple options
- Added Relative Positional Encoding (RPE) for physics-aware attention
- Attention bias directly encodes pairwise spatial relationships
- Scaffold for future encodings (sinusoidal, RoPE)
- NEW: Fine-tuning support via --resume_checkpoint

A clean implementation of transformer-based Soft Actor-Critic for wind farm
yaw control with the goal of generalizing across different farm layouts.

FINE-TUNING USAGE:
    # Fine-tune a model trained on layouts a-d on a new layout 'e':
    python transformer_sac_windfarm.py \
        --resume_checkpoint runs/my_experiment/checkpoints/step_100000.pt \
        --layouts e \
        --total_timesteps 50000 \
        --finetune_reset_optimizers True  # Fresh optimizers for fine-tuning

    # Resume training (keep optimizer states):
    python transformer_sac_windfarm.py \
        --resume_checkpoint runs/my_experiment/checkpoints/step_100000.pt \
        --layouts a,b,c,d \
        --total_timesteps 200000 \
        --finetune_reset_optimizers False

Key design principles:
1. Per-turbine tokenization: Each turbine is a token with local observations
2. Wind-relative positional encoding: Positions rotated so wind comes from 270°
3. Wind direction as DEVIATION from mean (not absolute) - rotation invariant
4. Shared actor/critic heads across turbines (permutation equivariant)
5. Adaptive target entropy based on actual turbine count (not max)
6. Optional farm-level token for global context
7. NEW: Modular positional encoding with absolute and relative options

Author: Marcus (DTU Wind Energy)
Based on discussions about transformer architectures for wind farm control.

================================================================================
TODO LIST - ACTIVE DEVELOPMENT
================================================================================

PHASE 1: CORE IMPLEMENTATION (COMPLETE)
[x] Basic transformer architecture (Actor, Critic)
[x] Wind-relative positional encoding
[x] Wind direction deviation calculation
[x] Enhanced per-turbine observation wrapper
[x] Combined wrapper applied in environment factory
[x] Replay buffer with raw position storage
[x] Adaptive target entropy for variable turbine counts
[x] Training loop with proper logging
[x] Test on single layout (test_layout: 2x1 grid)
[x] Verify learning signal (reward increasing)
[x] Verify wind direction index detection in EnhancedPerTurbineWrapper

PHASE 2: POSITIONAL ENCODING IMPROVEMENTS (CURRENT - V12)
[x] Modular positional encoding system with type selection
[x] Absolute MLP encoding (original, kept as default)
[x] Relative Positional Encoding (RPE) with attention bias
[x] Sinusoidal 2D encoding
[x] Wind-relative RoPE (Rotary Position Embeddings)
[x] Polar coordinate encoding (r, theta from wind)
[x] ALiBi and directional ALiBi
[ ] Attention pattern analysis: verify encodings show wake physics

POSITIONAL ENCODING OPTIONS (--pos_encoding_type):
- "absolute_mlp": (DEFAULT) MLP on absolute (x,y) → add to token embedding
- "sinusoidal_2d": Multi-frequency sin/cos encoding of 2D coordinates
- "polar_mlp": MLP on polar (r, θ) coordinates
- "relative_mlp": MLP on pairwise relative positions → attention bias (per-head)
- "relative_mlp_shared": Same as relative_mlp but heads share bias
- "relative_polar": MLP on pairwise polar (Δr, Δθ) → attention bias
- "alibi": Linear distance penalty (no learned params)
- "alibi_directional": ALiBi with upwind/downwind asymmetry (learned slopes)
- "absolute_plus_relative": Both absolute embedding AND relative bias
- "rope_2d": 2D Rotary Position Embeddings (modifies Q,K directly)

PHASE 3: VALIDATION & DEBUGGING
[ ] Attention visualization during evaluation
[x] Compare with baseline (greedy yaw controller)
[ ] Verify wind-relative encoding is working (test with different wind dirs)
[ ] Check that model attends to upwind turbines (physics validation)
[ ] Hyperparameter tuning (embed_dim, num_layers, learning rates)
[ ] We could 'split' farms into smaller sub-farms to augment data.

PHASE 4: MULTI-LAYOUT GENERALIZATION
[ ] Train on multiple layouts simultaneously
[ ] Test zero-shot transfer to unseen layouts
[ ] Analyze attention patterns across different farm sizes
[ ] Compare generalization vs. layout-specific training

PHASE 5: TEMPORAL EXTENSIONS (OPTION B)
[ ] Design spatio-temporal attention mechanism
[ ] Implement SpatioTemporalTransformer class
[ ] Add temporal attention masking (causal)
[ ] Consider GTrXL for very long-range dependencies

KNOWN ISSUES / NOTES:
- History length of 15 may need tuning based on wake propagation time
- Farm token is optional - test with and without
- Gradient clipping at 1.0 is not yet tested

================================================================================
"""

import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# WindGym imports (adjust path as needed for your setup)
from WindGym import WindFarmEnv
from WindGym.wrappers import RecordEpisodeVals, PerTurbineObservationWrapper
from MultiLayoutEnv import MultiLayoutEnv, LayoutConfig, create_layout_configs
from TurbineFailureEnv import TurbineFailureEnv, FailureConfig

# Logging utilities for multi-layout training
from multi_layout_debug import (
    MultiLayoutDebugLogger,
    create_debug_logger,
)

from helper_funcs import (
    get_layout_positions,
    get_env_wind_directions,
    get_env_raw_positions,
    get_env_attention_masks,
    save_checkpoint,
    load_checkpoint,
    make_env_config,
    transform_to_wind_relative,
    compute_wind_direction_deviation,
    EnhancedPerTurbineWrapper,
)

from encodings_helper import (
    AbsolutePositionalEncoding,
    RelativePositionalBias,
    Sinusoidal2DPositionalEncoding,
    PolarPositionalEncoding,
    RelativePolarBias,
    ALiBiPositionalBias,
    DirectionalALiBiPositionalBias,
    RoPE2DPositionalEncoding,
    RoPEMultiheadAttention
)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Args:
    """Command-line arguments for training."""
    
    # === Experiment Settings ===
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True  # Enable wandb tracking
    wandb_project_name: str = "failure_windfarm"
    wandb_entity: Optional[str] = None
    save_model: bool = True
    save_interval: int = 10000
    log_image: bool = False  # Log attention images to TensorBoard
    shuffle_turbs: bool = False  # Shuffle turbine order in obs/action

    failure_training: bool = False        # Enable dropout training
    failure_prob: float = 0.15            # Per-turbine failure prob
    failure_max: int = 2                  # Max simultaneous failures
    failure_min_active: int = 2           # Min active turbines to avoid episode end


    # === Environment Settings ===
    turbtype: str = "DTU10MW"  # Wind turbine type
    TI_type: str = "Random"   # Turbulence intensity sampling
    dt_sim: int = 5           # Simulation timestep (seconds)
    dt_env: int = 10          # Environment timestep (seconds)
    yaw_step: float = 5.0     # Max yaw change per sim step (degrees)
    max_eps: int = 20         # Number of flow passthroughs per episode
    num_envs: int = 1         # Number of parallel environments
    
    # === Layout Settings ===
    # Comma-separated list of layouts. Single = single-layout, Multiple = multi-layout
    layouts: str = "test_layout"  # e.g., "square_1,square_2,circular_1"
    
    # === Observation Settings ===
    history_length: int = 15      # Number of timesteps of history per feature
    # include_power: bool = True    # Include power in observations
    use_farm_token: bool = False  # Add learnable farm-level token
    wd_scale_range: float = 90.0  # Wind direction deviation range for scaling (±degrees → [-1,1])
    
    # === Transformer Architecture ===
    embed_dim: int = 128          # Transformer hidden dimension
    num_heads: int = 4            # Number of attention heads
    num_layers: int = 2           # Number of transformer layers
    mlp_ratio: float = 2.0        # FFN hidden dim = embed_dim * mlp_ratio
    dropout: float = 0.0          # Dropout rate (0 for RL typically)
    pos_embed_dim: int = 32       # Dimension for positional encoding
    
    # === Positional Encoding Settings ===
    # Options: "absolute_mlp", "relative_mlp", "relative_mlp_shared", 
    #          "sinusoidal_2d", "rope_2d"
    pos_encoding_type: str = "absolute_mlp"
    # For relative encoding: number of hidden units in the bias MLP
    rel_pos_hidden_dim: int = 64
    # For relative encoding: whether to use separate bias per head
    rel_pos_per_head: bool = True
    
    # === SAC Hyperparameters ===
    utd_ratio: float = 1.0           # Update-to-data ratio
    total_timesteps: int = 100_000
    buffer_size: int = int(1e6)
    gamma: float = 0.99           # Discount factor
    tau: float = 0.005            # Target network update rate
    batch_size: int = 256
    learning_starts: int = 5000   # Steps before training starts
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 2     # Policy update frequency
    target_network_frequency: int = 1
    alpha: float = 0.2            # Initial entropy coefficient
    autotune: bool = True         # Auto-tune entropy coefficient
    
    # === Gradient Clipping ===
    grad_clip: bool = True
    grad_clip_max_norm: float = 1.0
    
    # === Fine-tuning / Resume Settings ===
    resume_checkpoint: Optional[str] = None  # Path to checkpoint .pt file for fine-tuning or resuming
    finetune_reset_optimizers: int = 1   # If True, reset optimizers for fresh fine-tuning. If False, resume optimizer states too.
    finetune_reset_alpha: int = 1        # If True, reset entropy coefficient. If False, keep from checkpoint.


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

# Type alias for encoding type
VALID_POS_ENCODING_TYPES = [
    # === Additive (added to token embeddings) ===
    "absolute_mlp",         # Original: MLP on (x,y) → add to token
    "sinusoidal_2d",        # NeRF-style multi-frequency encoding
    "polar_mlp",            # MLP on (r, θ) polar coordinates
    
    # === Attention Bias (added to attention logits) ===
    "relative_mlp",         # MLP on pairwise rel pos → attention bias (per-head)
    "relative_mlp_shared",  # MLP on pairwise rel pos → attention bias (shared)
    "relative_polar",       # MLP on pairwise (Δr, Δθ) → attention bias
    "alibi",                # Linear distance penalty (no learned params)
    "alibi_directional",    # ALiBi with upwind/downwind asymmetry
    
    # === Rotary (modifies Q and K directly) ===
    "rope_2d",              # 2D Rotary Position Embeddings
    
    # === Combined ===
    "absolute_plus_relative",  # Both absolute embedding AND relative bias
]


def create_positional_encoding(
    encoding_type: str,
    embed_dim: int,
    pos_embed_dim: int,
    num_heads: int,
    rel_pos_hidden_dim: int = 64,
    rel_pos_per_head: bool = True,
) -> Tuple[Optional[nn.Module], Optional[nn.Module], bool, bool]:
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
        (pos_encoder, rel_pos_bias, uses_additive_embedding, uses_rope)
        - pos_encoder: Module for absolute position embedding (or None)
        - rel_pos_bias: Module for relative position bias (or None)
        - uses_additive_embedding: Whether pos embedding is added to tokens
        - uses_rope: Whether to use RoPE transformer (requires different encoder)
    """
    if encoding_type not in VALID_POS_ENCODING_TYPES:
        raise ValueError(
            f"Unknown pos_encoding_type: {encoding_type}. "
            f"Valid options: {VALID_POS_ENCODING_TYPES}"
        )
    
    uses_rope = False  # Default
    
    # =========================================================================
    # Additive Encodings (added to token embeddings)
    # =========================================================================
    
    if encoding_type == "absolute_mlp":
        # Original approach: MLP embedding added to tokens
        pos_encoder = AbsolutePositionalEncoding(pos_dim=2, embed_dim=pos_embed_dim)
        rel_pos_bias = None
        uses_additive_embedding = True
        
    elif encoding_type == "sinusoidal_2d":
        # Sinusoidal 2D encoding (frequency bands are fixed, projection is learned)
        pos_encoder = Sinusoidal2DPositionalEncoding(
            embed_dim=pos_embed_dim,
            num_frequencies=8,  # 8 frequency bands
            max_freq_log2=6,    # Max frequency 2^6 = 64
        )
        rel_pos_bias = None
        uses_additive_embedding = True
        
    elif encoding_type == "polar_mlp":
        # Polar coordinate encoding
        pos_encoder = PolarPositionalEncoding(embed_dim=pos_embed_dim)
        rel_pos_bias = None
        uses_additive_embedding = True
    
    # =========================================================================
    # Attention Bias Encodings (added to attention logits)
    # =========================================================================
    
    elif encoding_type == "relative_mlp":
        # Relative position bias added to attention (per-head)
        pos_encoder = None
        rel_pos_bias = RelativePositionalBias(
            num_heads=num_heads,
            hidden_dim=rel_pos_hidden_dim,
            per_head=True,
            pos_dim=2
        )
        uses_additive_embedding = False
        
    elif encoding_type == "relative_mlp_shared":
        # Relative position bias (shared across heads)
        pos_encoder = None
        rel_pos_bias = RelativePositionalBias(
            num_heads=num_heads,
            hidden_dim=rel_pos_hidden_dim,
            per_head=False,
            pos_dim=2
        )
        uses_additive_embedding = False
        
    elif encoding_type == "relative_polar":
        # Relative position bias using polar coordinates
        pos_encoder = None
        rel_pos_bias = RelativePolarBias(
            num_heads=num_heads,
            hidden_dim=rel_pos_hidden_dim,
            per_head=rel_pos_per_head,
        )
        uses_additive_embedding = False
        
    elif encoding_type == "alibi":
        # ALiBi: Simple linear distance penalty (no learned params)
        pos_encoder = None
        rel_pos_bias = ALiBiPositionalBias(num_heads=num_heads)
        uses_additive_embedding = False
        
    elif encoding_type == "alibi_directional":
        # Directional ALiBi with upwind/downwind asymmetry
        pos_encoder = None
        rel_pos_bias = DirectionalALiBiPositionalBias(num_heads=num_heads)
        uses_additive_embedding = False
    
    # =========================================================================
    # Rotary Position Embeddings (modifies Q and K)
    # =========================================================================
    
    elif encoding_type == "rope_2d":
        # RoPE: Rotary embeddings applied inside attention
        # No separate encoder or bias - handled by RoPETransformerEncoder
        pos_encoder = None
        rel_pos_bias = None
        uses_additive_embedding = False
        uses_rope = True
    
    # =========================================================================
    # Combined Encodings
    # =========================================================================
    
    elif encoding_type == "absolute_plus_relative":
        # Both absolute embedding AND relative bias
        pos_encoder = AbsolutePositionalEncoding(pos_dim=2, embed_dim=pos_embed_dim)
        rel_pos_bias = RelativePositionalBias(
            num_heads=num_heads,
            hidden_dim=rel_pos_hidden_dim,
            per_head=rel_pos_per_head,
            pos_dim=2
        )
        uses_additive_embedding = True
    
    else:
        raise ValueError(f"Encoding type '{encoding_type}' not implemented yet.")
    
    return pos_encoder, rel_pos_bias, uses_additive_embedding, uses_rope


# Backward compatibility alias
PositionalEncoding = AbsolutePositionalEncoding


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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_tokens, embed_dim)
            key_padding_mask: (batch, n_tokens) where True = ignore this position
            attn_bias: (batch, n_heads, n_tokens, n_tokens) optional bias to add
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
            average_attn_weights=False  # Return per-head weights
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
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (batch, n_tokens, embed_dim)
            key_padding_mask: (batch, n_tokens) where True = padding
            attn_bias: (batch, n_heads, n_tokens, n_tokens) optional attention bias
        
        Returns:
            x: Transformed tensor
            all_attn_weights: List of attention weights from each layer
        """
        all_attn_weights = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, key_padding_mask, attn_bias)
            all_attn_weights.append(attn_weights)
        
        x = self.norm(x)
        
        return x, all_attn_weights


# =============================================================================
# RoPE-ENABLED TRANSFORMER (separate implementation for RoPE)
# =============================================================================

class RoPETransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with 2D Rotary Position Embeddings.
    
    Uses RoPEMultiheadAttention instead of standard attention.
    Positions are passed through to the attention mechanism.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        max_position: float = 50.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Pre-norm layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # RoPE-enabled attention
        self.attn = RoPEMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_position=max_position,
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
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_tokens, embed_dim)
            positions: (batch, n_tokens, 2) wind-relative positions
            key_padding_mask: (batch, n_tokens) where True = padding
        
        Returns:
            x: Transformed tensor
            attn_weights: (batch, n_heads, n_tokens, n_tokens)
        """
        # Self-attention with pre-norm and RoPE
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm, positions, key_padding_mask=key_padding_mask
        )
        x = x + attn_out
        
        # FFN with pre-norm
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights


class RoPETransformerEncoder(nn.Module):
    """
    Stack of RoPE-enabled transformer encoder layers.
    
    Unlike the standard TransformerEncoder, this requires positions
    to be passed through for applying rotary embeddings.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        max_position: float = 50.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                embed_dim, num_heads, mlp_ratio, dropout, max_position
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (batch, n_tokens, embed_dim)
            positions: (batch, n_tokens, 2) wind-relative positions
            key_padding_mask: (batch, n_tokens) where True = padding
        
        Returns:
            x: Transformed tensor
            all_attn_weights: List of attention weights from each layer
        """
        all_attn_weights = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, positions, key_padding_mask)
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
       - "rope_2d": Position used to rotate Q and K in attention
    3. Optional: Prepend learnable farm token for global context
    4. Process through transformer (turbines attend to each other)
    5. Per-turbine action heads (shared weights across turbines)
    
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
        use_farm_token: bool = False,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
        # New positional encoding args
        pos_encoding_type: str = "absolute_mlp",
        rel_pos_hidden_dim: int = 64,
        rel_pos_per_head: bool = True,
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
            use_farm_token: Whether to use a learnable farm-level token
            action_scale: Scale for tanh output
            action_bias: Bias for tanh output
            pos_encoding_type: Type of positional encoding (see VALID_POS_ENCODING_TYPES)
            rel_pos_hidden_dim: Hidden dimension for relative position MLP
            rel_pos_per_head: Whether relative bias is per-head
        """
        super().__init__()
        
        self.obs_dim_per_turbine = obs_dim_per_turbine
        self.action_dim_per_turbine = action_dim_per_turbine
        self.embed_dim = embed_dim
        self.use_farm_token = use_farm_token
        self.pos_encoding_type = pos_encoding_type
        
        # Create positional encoding modules based on type
        self.pos_encoder, self.rel_pos_bias, self.uses_additive_embedding, self.uses_rope = \
            create_positional_encoding(
                encoding_type=pos_encoding_type,
                embed_dim=embed_dim,
                pos_embed_dim=pos_embed_dim,
                num_heads=num_heads,
                rel_pos_hidden_dim=rel_pos_hidden_dim,
                rel_pos_per_head=rel_pos_per_head,
            )
        
        # Observation encoder (shared across turbines)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim_per_turbine, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Input projection depends on encoding type
        if self.uses_additive_embedding:
            # Absolute encoding: obs_embed + pos_embed → project to embed_dim
            self.input_proj = nn.Linear(embed_dim + pos_embed_dim, embed_dim)
        else:
            # Relative/RoPE encoding: just obs_embed → embed_dim (already correct size)
            self.input_proj = nn.Linear(embed_dim, embed_dim)
        
        # Optional farm token (prepended to sequence)
        if use_farm_token:
            self.farm_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.farm_token, std=0.02)
        
        # Transformer encoder (choose based on encoding type)
        if self.uses_rope:
            # RoPE requires special transformer that passes positions through
            self.transformer = RoPETransformerEncoder(
                embed_dim, num_heads, num_layers, mlp_ratio, dropout,
                max_position=50.0  # Max expected position in rotor diameters
            )
        else:
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
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning action distribution parameters.
        
        Args:
            obs: (batch, n_turbines, obs_dim_per_turbine)
            positions: (batch, n_turbines, 2) wind-relative normalized positions
            key_padding_mask: (batch, n_turbines) where True = padding
        
        Returns:
            mean: (batch, n_turbines, action_dim) action means
            log_std: (batch, n_turbines, action_dim) action log stds
            attn_weights: List of attention weights from each layer
        """
        batch_size, n_turbines, _ = obs.shape
        
        # Encode observations
        h = self.obs_encoder(obs)  # (batch, n_turb, embed_dim)
        
        # Apply positional encoding based on type
        if self.uses_additive_embedding and self.pos_encoder is not None:
            # Absolute encoding: concatenate position embedding
            pos_embed = self.pos_encoder(positions)  # (batch, n_turb, pos_embed_dim)
            h = torch.cat([h, pos_embed], dim=-1)  # (batch, n_turb, embed_dim + pos_embed_dim)
        
        # Project to embed_dim
        h = self.input_proj(h)  # (batch, n_turb, embed_dim)
        
        # Compute relative position bias if using relative encoding (not RoPE)
        attn_bias = None
        if self.rel_pos_bias is not None:
            attn_bias = self.rel_pos_bias(positions, key_padding_mask)
        
        # Store positions for RoPE (will be extended if farm token is used)
        positions_for_rope = positions
        
        # Optionally prepend farm token
        if self.use_farm_token:
            farm_tokens = self.farm_token.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
            h = torch.cat([farm_tokens, h], dim=1)  # (batch, 1 + n_turb, embed_dim)
            
            # Extend padding mask for farm token (farm token is never masked)
            if key_padding_mask is not None:
                farm_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=h.device)
                key_padding_mask_extended = torch.cat([farm_mask, key_padding_mask], dim=1)
            else:
                key_padding_mask_extended = None
            
            # Extend attention bias for farm token
            if attn_bias is not None:
                # Add row/column for farm token with zero bias
                n_heads = attn_bias.shape[1]
                n_total = h.shape[1]  # 1 + n_turb
                
                # Create new bias tensor with farm token
                new_bias = torch.zeros(
                    batch_size, n_heads, n_total, n_total,
                    device=attn_bias.device, dtype=attn_bias.dtype
                )
                # Copy original bias to turbine-turbine portion
                new_bias[:, :, 1:, 1:] = attn_bias
                attn_bias = new_bias
            
            # Extend positions for RoPE (farm token at origin)
            if self.uses_rope:
                farm_pos = torch.zeros(batch_size, 1, 2, device=positions.device, dtype=positions.dtype)
                positions_for_rope = torch.cat([farm_pos, positions], dim=1)
        else:
            key_padding_mask_extended = key_padding_mask
        
        # Transformer (different call signature for RoPE vs standard)
        if self.uses_rope:
            h, attn_weights = self.transformer(h, positions_for_rope, key_padding_mask_extended)
        else:
            h, attn_weights = self.transformer(h, key_padding_mask_extended, attn_bias)
        
        # Remove farm token from output (we only need turbine actions)
        if self.use_farm_token:
            h = h[:, 1:, :]  # (batch, n_turb, embed_dim)
        
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Sample action from policy with log probability.
        
        Args:
            obs: (batch, n_turbines, obs_dim)
            positions: (batch, n_turbines, 2) wind-relative normalized positions
            key_padding_mask: (batch, n_turbines) where True = padding
            deterministic: If True, return mean action
        
        Returns:
            action: (batch, n_turbines, action_dim) sampled actions
            log_prob: (batch, 1) log probability of actions
            mean_action: (batch, n_turbines, action_dim) mean actions
            attn_weights: List of attention weights
        """
        mean, log_std, attn_weights = self.forward(obs, positions, key_padding_mask)
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
    3. Add positional encoding (method depends on pos_encoding_type)
    4. Optional: Prepend farm token
    5. Process through transformer
    6. Pool over turbines (masked mean) → single Q-value
    
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
        use_farm_token: bool = False,
        # New positional encoding args
        pos_encoding_type: str = "absolute_mlp",
        rel_pos_hidden_dim: int = 64,
        rel_pos_per_head: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_farm_token = use_farm_token
        self.pos_encoding_type = pos_encoding_type
        
        # Create positional encoding modules based on type
        self.pos_encoder, self.rel_pos_bias, self.uses_additive_embedding, self.uses_rope = \
            create_positional_encoding(
                encoding_type=pos_encoding_type,
                embed_dim=embed_dim,
                pos_embed_dim=pos_embed_dim,
                num_heads=num_heads,
                rel_pos_hidden_dim=rel_pos_hidden_dim,
                rel_pos_per_head=rel_pos_per_head,
            )
        
        # Observation + action encoder
        self.obs_action_encoder = nn.Sequential(
            nn.Linear(obs_dim_per_turbine + action_dim_per_turbine, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Input projection depends on encoding type
        if self.uses_additive_embedding:
            self.input_proj = nn.Linear(embed_dim + pos_embed_dim, embed_dim)
        else:
            self.input_proj = nn.Linear(embed_dim, embed_dim)
        
        # Optional farm token
        if use_farm_token:
            self.farm_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.farm_token, std=0.02)
        
        # Transformer encoder (choose based on encoding type)
        if self.uses_rope:
            self.transformer = RoPETransformerEncoder(
                embed_dim, num_heads, num_layers, mlp_ratio, dropout,
                max_position=50.0
            )
        else:
            self.transformer = TransformerEncoder(
                embed_dim, num_heads, num_layers, mlp_ratio, dropout
            )
        
        # Q-value head (after pooling)
        self.q_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
    
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Q-value for observation-action pair.
        
        Args:
            obs: (batch, n_turbines, obs_dim)
            action: (batch, n_turbines, action_dim)
            positions: (batch, n_turbines, 2) wind-relative normalized positions
            key_padding_mask: (batch, n_turbines) where True = padding
        
        Returns:
            q_value: (batch, 1) Q-value for the entire farm
        """
        batch_size = obs.shape[0]
        
        # Concatenate obs and action
        x = torch.cat([obs, action], dim=-1)
        
        # Encode
        h = self.obs_action_encoder(x)
        
        # Apply positional encoding based on type
        if self.uses_additive_embedding and self.pos_encoder is not None:
            pos_embed = self.pos_encoder(positions)
            h = torch.cat([h, pos_embed], dim=-1)
        
        # Project to embed_dim
        h = self.input_proj(h)
        
        # Compute relative position bias if using relative encoding (not RoPE)
        attn_bias = None
        if self.rel_pos_bias is not None:
            attn_bias = self.rel_pos_bias(positions, key_padding_mask)
        
        # Store positions for RoPE
        positions_for_rope = positions
        
        # Optional farm token
        if self.use_farm_token:
            farm_tokens = self.farm_token.expand(batch_size, -1, -1)
            h = torch.cat([farm_tokens, h], dim=1)
            
            if key_padding_mask is not None:
                farm_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=h.device)
                key_padding_mask_extended = torch.cat([farm_mask, key_padding_mask], dim=1)
            else:
                key_padding_mask_extended = None
            
            # Extend attention bias for farm token
            if attn_bias is not None:
                n_heads = attn_bias.shape[1]
                n_total = h.shape[1]
                new_bias = torch.zeros(
                    batch_size, n_heads, n_total, n_total,
                    device=attn_bias.device, dtype=attn_bias.dtype
                )
                new_bias[:, :, 1:, 1:] = attn_bias
                attn_bias = new_bias
            
            # Extend positions for RoPE (farm token at origin)
            if self.uses_rope:
                farm_pos = torch.zeros(batch_size, 1, 2, device=positions.device, dtype=positions.dtype)
                positions_for_rope = torch.cat([farm_pos, positions], dim=1)
        else:
            key_padding_mask_extended = key_padding_mask
        
        # Transformer (different call signature for RoPE vs standard)
        if self.uses_rope:
            h, _ = self.transformer(h, positions_for_rope, key_padding_mask_extended)
        else:
            h, _ = self.transformer(h, key_padding_mask_extended, attn_bias)
        
        # Remove farm token if used (for consistent pooling)
        if self.use_farm_token:
            # Option: use farm token output directly as pooled representation
            h_pooled = h[:, 0, :]  # (batch, embed_dim)
        else:
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
# REPLAY BUFFER
# =============================================================================

class TransformerReplayBuffer:
    """
    Replay buffer that stores raw positions and wind direction.
    
    Wind-relative transformation is applied at sample time to ensure
    correct positional encoding regardless of when the transition was collected.
    
    This is important because:
    1. Wind direction may change within an episode
    2. Different episodes have different wind directions
    3. The model always sees positions in a canonical wind-relative frame
    
    Storage format per transition:
    - obs: (max_turbines, obs_dim)
    - next_obs: (max_turbines, obs_dim)
    - action: (max_turbines, action_dim)
    - reward: scalar
    - done: bool
    - raw_positions: (max_turbines, 2) - NOT transformed
    - attention_mask: (max_turbines,) - True = padding
    - wind_direction: scalar - for transformation at sample time
    """
    
    def __init__(
        self,
        capacity: int,
        device: torch.device,
        rotor_diameter: float
    ):
        """
        Args:
            capacity: Maximum number of transitions
            device: Torch device for sampled tensors
            rotor_diameter: For position normalization
        """
        self.capacity = capacity
        self.device = device
        self.rotor_diameter = rotor_diameter
        self.buffer = []
        self.position = 0
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        raw_positions: np.ndarray,
        attention_mask: np.ndarray,
        wind_direction: float
    ) -> None:
        """Store a transition."""
        data = (obs, next_obs, action, reward, done, raw_positions, attention_mask, wind_direction)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample batch and apply wind-relative transformation.
        
        Returns:
            Dict with keys:
            - observations: (batch, max_turb, obs_dim)
            - next_observations: (batch, max_turb, obs_dim)
            - actions: (batch, max_turb, action_dim)
            - positions: (batch, max_turb, 2) - transformed and normalized
            - attention_mask: (batch, max_turb)
            - rewards: (batch, 1)
            - dones: (batch, 1)
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Unpack batch
        obs_list, next_obs_list, action_list = [], [], []
        raw_positions_list, mask_list, wind_dirs = [], [], []
        rewards, dones = [], []
        
        for obs, next_obs, action, reward, done, raw_pos, mask, wind_dir in batch:
            obs_list.append(obs)
            next_obs_list.append(next_obs)
            action_list.append(action)
            raw_positions_list.append(raw_pos)
            mask_list.append(mask)
            wind_dirs.append(wind_dir)
            rewards.append(reward)
            dones.append(done)
        
        # Stack to arrays
        raw_positions = np.stack(raw_positions_list)  # (batch, max_turb, 2)
        wind_directions = np.array(wind_dirs)  # (batch,)
        
        # Normalize positions by rotor diameter
        positions_norm = raw_positions / self.rotor_diameter
        
        # Convert to tensors
        positions_tensor = torch.tensor(positions_norm, device=self.device, dtype=torch.float32)
        wind_dir_tensor = torch.tensor(wind_directions, device=self.device, dtype=torch.float32)
        
        # Apply wind-relative transformation
        positions_transformed = transform_to_wind_relative(positions_tensor, wind_dir_tensor)
        
        return {
            "observations": torch.tensor(np.stack(obs_list), device=self.device, dtype=torch.float32),
            "next_observations": torch.tensor(np.stack(next_obs_list), device=self.device, dtype=torch.float32),
            "actions": torch.tensor(np.stack(action_list), device=self.device, dtype=torch.float32),
            "positions": positions_transformed,
            "attention_mask": torch.tensor(np.stack(mask_list), device=self.device, dtype=torch.bool),
            "rewards": torch.tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(-1),
            "dones": torch.tensor(dones, device=self.device, dtype=torch.float32).unsqueeze(-1),
        }
    
    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def compute_adaptive_target_entropy(
    attention_mask: torch.Tensor,
    action_dim_per_turbine: int = 1
) -> torch.Tensor:
    """
    Compute target entropy adapted to actual turbine count per sample.
    
    This fixes a bug where using max_turbines for all samples causes
    incorrect entropy targeting when training on variable-size farms.
    
    Args:
        attention_mask: (batch, max_turbines) where True = padding
        action_dim_per_turbine: Actions per turbine (typically 1 for yaw)
    
    Returns:
        target_entropy: (batch, 1) tensor of per-sample target entropies
    """
    # Count real turbines per sample
    n_real_turbines = (~attention_mask).sum(dim=1, keepdim=True).float()
    
    # Target entropy scales with turbine count
    # Convention: -1 per action dimension
    target_entropy = -action_dim_per_turbine * n_real_turbines
    
    return target_entropy





def get_env_current_layout(envs) -> List[str]:
    '''Get the current layout name for each environment.'''
    names_tuple = envs.env.get_attr('current_layout')
    names_list = [names_tuple[x].name for x in range(len(names_tuple))]
    return names_list



# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_attention_weights(
    attention_weights: List[torch.Tensor],
    positions: np.ndarray,
    attention_mask: np.ndarray,
    wind_direction: float,
    turbine_names: Optional[List[str]] = None,
    layer_idx: int = -1,
    head_idx: int = 0,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize attention patterns overlaid on farm layout.
    
    This helps verify that the model learns physically meaningful
    wake interaction patterns (upwind turbines should attend to
    downwind turbines that they affect).
    
    Args:
        attention_weights: List of attention weights from transformer layers
                          Each: (batch, n_heads, n_tokens, n_tokens)
        positions: (n_turbines, 2) raw positions
        attention_mask: (n_turbines,) where True = padding
        wind_direction: Wind direction in degrees
        turbine_names: Optional names for turbines
        layer_idx: Which layer to visualize (-1 = last)
        head_idx: Which attention head to visualize
        save_path: Path to save figure (None = display)
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available for visualization")
        return
    
    # Get attention from specified layer (take first sample in batch)
    attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()  # (n_tokens, n_tokens)
    
    # Filter out padded positions
    n_real = (~attention_mask).sum()
    attn = attn[:n_real, :n_real]
    positions = positions[:n_real]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Draw attention as edges between turbines
    for i in range(n_real):
        for j in range(n_real):
            if i != j:
                weight = attn[i, j]
                if weight > 0.1:  # Only show significant attention
                    ax.annotate(
                        "",
                        xy=positions[j],
                        xytext=positions[i],
                        arrowprops=dict(
                            arrowstyle="->",
                            alpha=min(weight * 2, 1.0),
                            color="blue",
                            lw=weight * 3,
                        ),
                    )
    
    # Draw turbines
    ax.scatter(positions[:, 0], positions[:, 1], s=200, c='red', zorder=5)
    
    # Label turbines
    if turbine_names is None:
        turbine_names = [f"T{i}" for i in range(n_real)]
    for i, (x, y) in enumerate(positions):
        ax.annotate(turbine_names[i], (x, y), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=10)
    
    # Draw wind direction arrow
    center = positions.mean(axis=0)
    wind_rad = np.deg2rad(270 - wind_direction)  # Convert to standard math convention
    wind_length = np.max(np.ptp(positions, axis=0)) * 0.3
    dx = wind_length * np.cos(wind_rad)
    dy = wind_length * np.sin(wind_rad)
    ax.annotate(
        "", xy=(center[0] + dx, center[1] + dy),
        xytext=(center[0] - dx, center[1] - dy),
        arrowprops=dict(arrowstyle="->", color="green", lw=3),
    )
    ax.text(center[0], center[1] + wind_length * 1.2,
            f"Wind: {wind_direction:.0f}°", ha='center', fontsize=12, color='green')
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Attention Weights (Layer {layer_idx}, Head {head_idx})")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Main training function."""
    
    # Parse arguments
    args = tyro.cli(Args)
    
    # Parse layouts
    # layout_names = [l.strip() for l in args.layouts.split(",")]
    layout_name = args.layouts
    # is_multi_layout = len(layout_names) > 1
    
    # Create run name
    run_name = f"{args.exp_name}_{layout_name}_{args.seed}"
    
    print("=" * 60)
    print(f"Transformer SAC for Wind Farm Control - Failure version")
    print("=" * 60)
    # if is_multi_layout:
    #     print(f"Mode: Multi-layout training with layouts: {layout_names}")
    # else:
    #     print(f"Mode: Single-layout training: {layout_names[0]}")
    print(f"Run name: {run_name}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    os.makedirs(f"runs/{run_name}/attention_plots", exist_ok=True)
    

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    
    # =========================================================================
    # ENVIRONMENT SETUP
    # =========================================================================
    
    # Import WindGym components
    from WindGym import WindFarmEnv
    from WindGym.wrappers import RecordEpisodeVals, PerTurbineObservationWrapper
    from MultiLayoutEnv import MultiLayoutEnv, LayoutConfig
    
    # Wind turbine
    if args.turbtype == "DTU10MW":
        from py_wake.examples.data.dtu10mw import DTU10MW as WT
    elif args.turbtype == "V80":
        from py_wake.examples.data.hornsrev1 import V80 as WT
    else:
        raise ValueError(f"Unknown turbine type: {args.turbtype}")
    
    wind_turbine = WT()
    
    # Create layout configurations
    # layouts = []
    # for name in layout_names:
    #     x_pos, y_pos = get_layout_positions(name, wind_turbine)
    #     layouts.append(LayoutConfig(name=name, x_pos=x_pos, y_pos=y_pos))
    
    x_pos, y_pos = get_layout_positions(layout_name, wind_turbine)
    layout = LayoutConfig(name=layout_name, x_pos=x_pos, y_pos=y_pos)

    # Environment configuration
    config = make_env_config()
    
    mes_prefixes = {
        "ws_mes": "ws",
        "wd_mes": "wd",
        "yaw_mes": "yaw",
        "power_mes": "power",
    }

    for mes_type, prefix in mes_prefixes.items():
        config[mes_type][f"{prefix}_history_N"] = args.history_length
        config[mes_type][f"{prefix}_history_length"] = args.history_length

    
    base_env_kwargs = {
        "turbine": wind_turbine,
        "n_passthrough": args.max_eps,
        "TurbBox": "/work/users/manils/rl_timestep/Boxes/V80env/",  # Adjust path as needed
        "config": config,
        "turbtype": args.TI_type,
        "dt_sim": args.dt_sim,
        "dt_env": args.dt_env,
        "yaw_step_sim": args.yaw_step,
    }
    
    def env_factory(x_pos: np.ndarray, y_pos: np.ndarray) -> gym.Env:
        """Create a base WindFarmEnv with given positions."""
        env = WindFarmEnv(x_pos=x_pos, y_pos=y_pos, **base_env_kwargs)
        env.action_space.seed(args.seed)
        return env
    
    def combined_wrapper(env: gym.Env) -> gym.Env:
        """
        Combined wrapper that:
        1. Applies PerTurbineObservationWrapper (reshapes obs to per-turbine)
        2. Applies EnhancedPerTurbineWrapper (converts wind direction to deviation)
        """
        env = PerTurbineObservationWrapper(env)
        env = EnhancedPerTurbineWrapper(env, wd_scale_range=args.wd_scale_range)
        return env
    
    def make_env_fn(seed, layout: LayoutConfig):
        """Factory function for vectorized environments."""
        def _init():


            failure_config = FailureConfig(
                 mode='random' if args.failure_training else 'none',  # Use the flag!
                failure_prob=args.failure_prob,
                max_failures=args.failure_max,
                min_active=args.failure_min_active,
            )
            env = TurbineFailureEnv(
                full_x_pos=layout.x_pos,
                full_y_pos=layout.y_pos,
                env_factory=env_factory,
                per_turbine_wrapper=combined_wrapper,
                failure_config=failure_config,
                seed=seed,
                layout_name=layout.name,  # Pass the name
            )
            return env
        return _init
    
    # Create vectorized environments
    print(f"Creating {args.num_envs} parallel environment(s)...")
    envs = gym.vector.AsyncVectorEnv(
        [make_env_fn(args.seed + i, layout) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    envs = RecordEpisodeVals(envs)
    
    # Get dimensions from sample env
    sample_env = make_env_fn(args.seed, layout)()
    sample_obs, sample_info = sample_env.reset()
    
    n_turbines_max = sample_env.max_turbines
    obs_dim_per_turbine = sample_obs.shape[1]
    action_dim_per_turbine = 1
    rotor_diameter = sample_env.rotor_diameter
    sample_env.close()
    
    print(f"Max turbines: {n_turbines_max}")
    print(f"Obs dim per turbine: {obs_dim_per_turbine}")
    print(f"Action dim per turbine: {action_dim_per_turbine}")
    print(f"Rotor diameter: {rotor_diameter:.1f} m")
    
    # Action scaling
    action_high = envs.single_action_space.high[0]
    action_low = envs.single_action_space.low[0]
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0
    
    # =========================================================================
    # DEBUG LOGGER AND TRACKING SETUP
    # =========================================================================

    # Initialize debug logger with configurable frequencies
    debug_logger = create_debug_logger(
        layout_names=[layout_name],
        log_every=250,  # Base frequency - others are multiples of this
    )
    # Frequencies will be:
    #   - summary metrics: every 100 steps
    #   - attention analysis: every 500 steps  
    #   - gradient norms: every 100 steps
    #   - q-value stats: every 50 steps
    #   - diagnostic print: every 2000 steps

    print(f"Debug logger initialized for layouts: {layout_name}")
    print(f"  Attention logging every {debug_logger.attention_log_frequency} steps")
    print(f"  Gradient logging every {debug_logger.gradient_log_frequency} steps")
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args) | {
                # Debug/multi-layout config
                "debug/n_layouts": len(layout_name),
                "debug/layout_names": layout_name,
                # "debug/is_multi_layout": is_multi_layout,
                "debug/max_turbines": n_turbines_max,
                "debug/log_frequency": debug_logger.log_frequency,
                "debug/attention_log_frequency": debug_logger.attention_log_frequency,
                "debug/gradient_log_frequency": debug_logger.gradient_log_frequency,
            },
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])
    )

    # =========================================================================
    # NETWORK SETUP
    # =========================================================================
    
    print("\nCreating networks...")
    print(f"Positional encoding type: {args.pos_encoding_type}")
    
    actor = TransformerActor(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=action_dim_per_turbine,
        embed_dim=args.embed_dim,
        pos_embed_dim=args.pos_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_farm_token=args.use_farm_token,
        action_scale=action_scale,
        action_bias=action_bias,
        # Positional encoding settings
        pos_encoding_type=args.pos_encoding_type,
        rel_pos_hidden_dim=args.rel_pos_hidden_dim,
        rel_pos_per_head=args.rel_pos_per_head,
    ).to(device)
    
    qf1 = TransformerCritic(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=action_dim_per_turbine,
        embed_dim=args.embed_dim,
        pos_embed_dim=args.pos_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_farm_token=args.use_farm_token,
        # Positional encoding settings
        pos_encoding_type=args.pos_encoding_type,
        rel_pos_hidden_dim=args.rel_pos_hidden_dim,
        rel_pos_per_head=args.rel_pos_per_head,
    ).to(device)
    
    qf2 = TransformerCritic(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=action_dim_per_turbine,
        embed_dim=args.embed_dim,
        pos_embed_dim=args.pos_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_farm_token=args.use_farm_token,
        # Positional encoding settings
        pos_encoding_type=args.pos_encoding_type,
        rel_pos_hidden_dim=args.rel_pos_hidden_dim,
        rel_pos_per_head=args.rel_pos_per_head,
    ).to(device)
    
    # Target networks
    qf1_target = TransformerCritic(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=action_dim_per_turbine,
        embed_dim=args.embed_dim,
        pos_embed_dim=args.pos_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_farm_token=args.use_farm_token,
        # Positional encoding settings
        pos_encoding_type=args.pos_encoding_type,
        rel_pos_hidden_dim=args.rel_pos_hidden_dim,
        rel_pos_per_head=args.rel_pos_per_head,
    ).to(device)
    
    qf2_target = TransformerCritic(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=action_dim_per_turbine,
        embed_dim=args.embed_dim,
        pos_embed_dim=args.pos_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_farm_token=args.use_farm_token,
        # Positional encoding settings
        pos_encoding_type=args.pos_encoding_type,
        rel_pos_hidden_dim=args.rel_pos_hidden_dim,
        rel_pos_per_head=args.rel_pos_per_head,
    ).to(device)
    
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    # Count parameters
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in qf1.parameters())
    print(f"Actor parameters: {actor_params:,}")
    print(f"Critic parameters: {critic_params:,} (x2)")
    
    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()),
        lr=args.q_lr
    )
    
    # Entropy tuning
    if args.autotune:
        # Initial target entropy (will be adapted per-batch)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
        log_alpha = None
        alpha_optimizer = None
    
    # =========================================================================
    # LOAD CHECKPOINT (for fine-tuning or resuming)
    # =========================================================================
    
    start_step = 0
    if args.resume_checkpoint is not None:
        print(f"\n{'='*60}")
        print(f"LOADING CHECKPOINT FOR FINE-TUNING")
        print(f"{'='*60}")
        print(f"Checkpoint path: {args.resume_checkpoint}")
        
        if not os.path.exists(args.resume_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_checkpoint}")
        
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        
        # Load network weights
        actor.load_state_dict(checkpoint["actor_state_dict"])
        qf1.load_state_dict(checkpoint["qf1_state_dict"])
        qf2.load_state_dict(checkpoint["qf2_state_dict"])
        qf1_target.load_state_dict(checkpoint["qf1_state_dict"])
        qf2_target.load_state_dict(checkpoint["qf2_state_dict"])
        
        print(f"✓ Loaded network weights from step {checkpoint['step']}")
        
        # Optionally load optimizer states
        if not args.finetune_reset_optimizers:
            actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
            start_step = checkpoint["step"]
            print(f"✓ Loaded optimizer states (resuming from step {start_step})")
        else:
            print(f"✓ Reset optimizers for fresh fine-tuning")
        
        # Optionally load entropy coefficient
        if args.autotune and not args.finetune_reset_alpha:
            if "log_alpha" in checkpoint:
                log_alpha.data = checkpoint["log_alpha"].to(device)
                alpha = log_alpha.exp().item()
                print(f"✓ Loaded entropy coefficient: alpha={alpha:.4f}")
            if "alpha_optimizer_state_dict" in checkpoint and not args.finetune_reset_optimizers:
                alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
                print(f"✓ Loaded alpha optimizer state")
        else:
            print(f"✓ Reset entropy coefficient (alpha={alpha:.4f})")
        
        # Log checkpoint info
        if "args" in checkpoint:
            ckpt_args = checkpoint["args"]
            print(f"\nOriginal training config:")
            print(f"  - Layouts: {ckpt_args.get('layouts', 'unknown')}")
            print(f"  - Total timesteps: {ckpt_args.get('total_timesteps', 'unknown')}")
            print(f"  - Pos encoding: {ckpt_args.get('pos_encoding_type', 'unknown')}")
        
        print(f"\nFine-tuning on layouts: {args.layouts}")
        print(f"{'='*60}\n")
    
    # =========================================================================
    # REPLAY BUFFER
    # =========================================================================
    
    rb = TransformerReplayBuffer(args.buffer_size, device, rotor_diameter)
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    
    print(f"\nStarting training for {args.total_timesteps} timesteps...")
    print(f"UTD ratio: {args.utd_ratio} (gradient updates per env step)")
    print(f"With {args.num_envs} envs: {int(args.num_envs * args.utd_ratio)} gradient updates per iteration")
    print("=" * 60)
    

    save_checkpoint(
        actor, qf1, qf2, actor_optimizer, q_optimizer,
        0, run_name, args, log_alpha, alpha_optimizer
    )


    start_time = time.time()
    global_step = start_step  # Start from checkpoint step if resuming, else 0
    total_gradient_steps = 0  # Track total gradient updates for logging
    # Reset environments
    obs, infos = envs.reset(seed=args.seed)
    
    # Tracking
    step_reward_window = deque(maxlen=1000)
    # next_save_step = ((start_step // args.save_interval) + 1) * args.save_interval  # Account for resumed step
    next_save_step = start_step + args.save_interval 
    # For logging losses (we'll average over the UTD updates)
    loss_accumulator = {
        'qf1_loss': [], 'qf2_loss': [], 'actor_loss': [], 'alpha_loss': []
    }

    # Calculate remaining updates if resuming
    remaining_timesteps = args.total_timesteps - start_step
    num_updates = max(0, remaining_timesteps // args.num_envs)
    
    if start_step > 0:
        print(f"Resuming from step {start_step}, {remaining_timesteps} timesteps remaining")
        print(f"Will run {num_updates} more updates")
    
    for update in range(num_updates + 2):
        global_step += args.num_envs
        
        # Get environment info
        wind_dirs = get_env_wind_directions(envs, args.num_envs)
        raw_positions = get_env_raw_positions(envs, args.num_envs, n_turbines_max)
        current_masks = get_env_attention_masks(envs, args.num_envs, n_turbines_max)
        
        # Select action
        if global_step < args.learning_starts:
            # Random exploration
            actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                
                # Normalize and transform positions
                positions_norm = raw_positions / rotor_diameter
                positions_tensor = torch.tensor(positions_norm, dtype=torch.float32, device=device)
                wind_dir_tensor = torch.tensor(wind_dirs, dtype=torch.float32, device=device)
                positions_transformed = transform_to_wind_relative(positions_tensor, wind_dir_tensor)
                
                # Attention mask
                mask_tensor = torch.tensor(current_masks, dtype=torch.bool, device=device)
                
                # Get action
                action_tensor, _, _, _ = actor.get_action(
                    obs_tensor, positions_transformed, mask_tensor
                )
                actions = action_tensor.squeeze(-1).cpu().numpy()
        
        # Step environment
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)


        # Get current layout names for each env
        current_layouts = get_env_current_layout(envs)

        # Log per-step data to debug tracker (always - internal deques handle storage)
        for i in range(args.num_envs):
            debug_logger.log_layout_step(
                layout_name=current_layouts[i],
                reward=float(rewards[i]),
                power=float(infos.get("Power agent", [0.0] * args.num_envs)[i]) if "Power agent" in infos else None,
                actions=actions[i] if isinstance(actions, np.ndarray) else np.array(actions[i]),
            )
            debug_logger.log_wind_direction(float(wind_dirs[i]))


        # Track rewards
        step_reward_window.extend(np.array(rewards).flatten().tolist())
        
        # Log episode stats
        if "final_info" in infos:
            ep_return = np.mean(envs.return_queue)
            ep_length = np.mean(envs.length_queue)
            ep_power = np.mean(envs.mean_power_queue)
            
            print(f"Step {global_step}: Episode return={ep_return:.2f}, power={ep_power:.2f}")
            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_length", ep_length, global_step)
            writer.add_scalar("charts/episodic_power", ep_power, global_step)

            # REMOVED DUE TO SOME BUG. 
            # ep_return = infos.get("final_return", [None])[idx]
            # IndexError: list index out of range
            
            # # NEW: Per-layout episode tracking
            # for i, final_info in enumerate(infos.get("final_info", [])):
            #     if final_info is not None and i < len(current_layouts):
            #         layout_name = current_layouts[i]
            #         ep_ret = final_info.get("episode", {}).get("r", ep_return)
                    
            #         debug_logger.log_layout_episode(layout_name, float(ep_ret))
            #         writer.add_scalar(f"charts/layout/{layout_name}/episodic_return", ep_ret, global_step)

        # Handle final observations
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_obs"][idx]
        
        # Store in replay buffer
        for i in range(args.num_envs):
            done = terminations[i] or truncations[i]
            action_reshaped = actions[i].reshape(-1, action_dim_per_turbine)
            rb.add(
                obs[i],
                real_next_obs[i],
                action_reshaped,
                rewards[i],
                done,
                raw_positions[i],
                current_masks[i],
                wind_dirs[i]
            )
        
        obs = next_obs
        
        # =====================================================================
        # TRAINING
        # =====================================================================
        
        if global_step > args.learning_starts and len(rb) >= args.batch_size:

            # Calculate number of gradient updates for this iteration
            # This scales with num_envs to maintain consistent sample efficiency
            num_gradient_updates = max(1, int(args.num_envs * args.utd_ratio))
            
            # Clear loss accumulator for this iteration
            loss_accumulator = {k: [] for k in loss_accumulator}


            for grad_step in range(num_gradient_updates):
                # Sample a fresh batch for each gradient update
                data = rb.sample(args.batch_size)
                
                batch_mask = data["attention_mask"]
                

                # -----------------------------------------------------------------
                # Update Critics
                # -----------------------------------------------------------------
                with torch.no_grad():
                    # Get next actions from current policy
                    next_actions, next_log_pi, _, _ = actor.get_action(
                        data["next_observations"],
                        data["positions"],
                        batch_mask
                    )
                    
                    # Compute target Q-values
                    qf1_next = qf1_target(
                        data["next_observations"], next_actions, data["positions"], batch_mask
                    )
                    qf2_next = qf2_target(
                        data["next_observations"], next_actions, data["positions"], batch_mask
                    )
                    min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
                    
                    # Bellman target
                    target_q = data["rewards"] + (1 - data["dones"]) * args.gamma * min_qf_next
                
                # Current Q-values
                qf1_value = qf1(data["observations"], data["actions"], data["positions"], batch_mask)
                qf2_value = qf2(data["observations"], data["actions"], data["positions"], batch_mask)
                

                # Log Q-value statistics (frequency controlled by logger)
                if debug_logger.should_log_q_values(total_gradient_steps):
                    debug_logger.log_q_value_stats(
                        qf1_values=qf1_value,
                        qf2_values=qf2_value,
                        target_q=target_q,
                        writer=writer,
                        global_step=global_step,
                    )

                # Critic loss
                qf1_loss = F.mse_loss(qf1_value, target_q)
                qf2_loss = F.mse_loss(qf2_value, target_q)
                qf_loss = qf1_loss + qf2_loss
                
                # Update critics
                q_optimizer.zero_grad()
                qf_loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        list(qf1.parameters()) + list(qf2.parameters()),
                        max_norm=args.grad_clip_max_norm
                    )
                q_optimizer.step()
                
                if debug_logger.should_log_gradients(total_gradient_steps):
                    debug_logger.log_critic_gradient_norms(qf1, qf2, writer, global_step)

                # Accumulate losses for logging
                loss_accumulator['qf1_loss'].append(qf1_loss.item())
                loss_accumulator['qf2_loss'].append(qf2_loss.item())

                # -----------------------------------------------------------------
                # Update Actor (delayed based on total gradient steps)
                # -----------------------------------------------------------------
                if total_gradient_steps % args.policy_frequency == 0:
                    # Get actions from current policy
                    actions_pi, log_pi, _, _ = actor.get_action(
                        data["observations"], data["positions"], batch_mask
                    )
                    
                    # Q-values for policy actions
                    qf1_pi = qf1(data["observations"], actions_pi, data["positions"], batch_mask)
                    qf2_pi = qf2(data["observations"], actions_pi, data["positions"], batch_mask)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    
                    # Policy loss (maximize Q - alpha * entropy)
                    actor_loss = (alpha * log_pi - min_qf_pi).mean()
                    
                    # Update actor
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            actor.parameters(),
                            max_norm=args.grad_clip_max_norm
                        )
                    actor_optimizer.step()

                    if debug_logger.should_log_gradients(total_gradient_steps):
                        debug_logger.log_actor_gradient_norms(actor, writer, global_step)

                    loss_accumulator['actor_loss'].append(actor_loss.item())
                    
                    # -------------------------------------------------------------
                    # Update Alpha (entropy coefficient)
                    # -------------------------------------------------------------
                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _, _ = actor.get_action(
                                data["observations"], data["positions"], batch_mask
                            )
                        
                        # Adaptive target entropy per sample
                        target_entropy_batch = compute_adaptive_target_entropy(
                            data["attention_mask"],
                            action_dim_per_turbine
                        )
                        
                        # Alpha loss
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy_batch)).mean()
                        
                        alpha_optimizer.zero_grad()
                        alpha_loss.backward()
                        alpha_optimizer.step()
                        alpha = log_alpha.exp().item()
                        
                        loss_accumulator['alpha_loss'].append(alpha_loss.item())
                
                # -----------------------------------------------------------------
                # Update Target Networks
                # -----------------------------------------------------------------
                if total_gradient_steps % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
                
                # Attention physics analysis (frequency controlled by logger)
                if debug_logger.should_log_attention(total_gradient_steps):
                    with torch.no_grad():
                        # Get fresh attention weights from a small batch
                        sample_size = min(8, args.batch_size)
                        _, _, _, attn_weights = actor.get_action(
                            data["observations"][:sample_size],
                            data["positions"][:sample_size],
                            batch_mask[:sample_size] if batch_mask is not None else None
                        )
                        
                        # This logs both scalar metrics AND a visualization image!
                        debug_logger.log_attention_metrics(
                            attention_weights=attn_weights,
                            positions=data["positions"][:sample_size],
                            attention_mask=batch_mask[:sample_size] if batch_mask is not None else None,
                            writer=writer,
                            global_step=global_step,
                            log_image=args.log_image,  # Set False to disable image (faster)
                        )
                        
                        # Optional: Log per-head attention figure (more expensive)
                        if args.log_image:
                            # Useful for understanding what each head specializes in
                            if debug_logger.should_log_histograms(total_gradient_steps):  # Less frequent
                                fig = debug_logger.create_multi_head_attention_figure(
                                    attention_weights=attn_weights,
                                    positions=data["positions"][:1],  # Single sample
                                    attention_mask=batch_mask[:1] if batch_mask is not None else None,
                                    title=f"Step {global_step}",
                                )
                                if fig is not None:
                                    writer.add_figure("debug/attention/per_head", fig, global_step)
                                    import matplotlib.pyplot as plt
                                    plt.close(fig)


                total_gradient_steps += 1

            # -----------------------------------------------------------------
            # Logging
            # -----------------------------------------------------------------
            if update % 20 == 0:
                sps = int(global_step / (time.time() - start_time))
                mean_reward = float(np.mean(step_reward_window)) if step_reward_window else 0.0
                
                # Average losses over the UTD updates
                mean_qf1_loss = np.mean(loss_accumulator['qf1_loss']) if loss_accumulator['qf1_loss'] else 0
                mean_qf2_loss = np.mean(loss_accumulator['qf2_loss']) if loss_accumulator['qf2_loss'] else 0
                mean_actor_loss = np.mean(loss_accumulator['actor_loss']) if loss_accumulator['actor_loss'] else 0
                
                writer.add_scalar("losses/qf1_loss", mean_qf1_loss, global_step)
                writer.add_scalar("losses/qf2_loss", mean_qf2_loss, global_step)
                writer.add_scalar("losses/actor_loss", mean_actor_loss, global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/step_reward_mean_1000", mean_reward, global_step)
                writer.add_scalar("debug/mean_wind_direction", float(np.mean(wind_dirs)), global_step)
                writer.add_scalar("debug/total_gradient_steps", total_gradient_steps, global_step)
                writer.add_scalar("debug/gradient_updates_per_iter", num_gradient_updates, global_step)
                
                print(f"Step {global_step}: SPS={sps}, qf_loss={mean_qf1_loss + mean_qf2_loss:.4f}, "
                      f"actor_loss={mean_actor_loss:.4f}, alpha={alpha:.4f}, "
                      f"reward_mean={mean_reward:.4f}, grad_steps={total_gradient_steps}")
        
        
            # Log summary metrics (frequency controlled by logger)
            if debug_logger.should_log(global_step):
                debug_logger.log_summary_metrics(
                    writer=writer,
                    global_step=global_step,
                )

                # Print diagnostic summary to console (frequency controlled by logger)
                if debug_logger.should_print_diagnostics(global_step):
                    debug_logger.print_diagnostics(global_step)



        # =====================================================================
        # CHECKPOINTING
        # =====================================================================
        
        if args.save_model and global_step >= next_save_step:
            save_checkpoint(
                actor, qf1, qf2, actor_optimizer, q_optimizer,
                global_step, run_name, args, log_alpha, alpha_optimizer
            )
            next_save_step += args.save_interval
    # =========================================================================
    
    # FINAL SAVE AND CLEANUP
    # =========================================================================
    
    if args.save_model:
        save_checkpoint(
            actor, qf1, qf2, actor_optimizer, q_optimizer,
            global_step, run_name, args, log_alpha, alpha_optimizer
        )
    
    print("\n" + "=" * 60)
    print("Training finished!")
    print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")
    print("=" * 60)
    
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()