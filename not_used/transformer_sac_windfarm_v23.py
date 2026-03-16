"""
Transformer-based SAC for Wind Farm Control - V23

Changes in V23:
- Added geometric profiles as alternative to PyWake profiles
- Renamed the following:
    PyWakeProfileEncoder -> CNNProfileEncoder
    PyWakeProfileEncoderDilated ->  DilatedProfileEncoder
    PyWakeProfileEncoderWithAttention -> AttentionProfileEncoder

Changes in V22:
- Added the following positional encoders:
    GATPositionalEncoder: A Graph Attention Network encoder for turbine positions, allowing for more flexible spatial context encoding based on local neighborhoods.
    SpatialContextEmbedding: Embeds spatial context such as local turbine density or distance to nearest turbine, providing additional information about the environment.
    NeighborhoodAggregationEmbedding: Embeds local neighborhood information by aggregating features of nearby turbines, capturing local interactions more explicitly.
    WakeKernelBias: A positional bias based on physics-inspired wake interaction kernels, encoding expected influence patterns based on relative positions.
- Added pos_embedding_mode: Arguemnt to control how positional encodings are integrated (added vs concatenated), allowing for more flexible architectural choices.
    
Changes in V21:
- Reuse log_pi, for the alpha_loss calculation 
- Updated rotate_profiles_tensor
- Vectorized replay buffer: pre-allocated numpy arrays instead of list of tuples.
  Turns the per-sample unpack/permute/pad loop into vectorized array indexing.
  Profile registry is pre-padded to max_turbines at init; permutations are sanitized
  at add-time so that batch gather + np.take_along_axis replaces the Python loop.
- Added soft_update using torch lerp. 


Key design principles:
1. Per-turbine tokenization: Each turbine is a token with local observations
2. Wind-relative positional encoding: Positions rotated so wind comes from 270° (optional)
3. Wind direction as DEVIATION from mean (not absolute) - rotation invariant (optional)
4. Shared actor/critic heads across turbines (permutation equivariant)
5. Adaptive target entropy based on actual turbine count (not max)
6. Optional farm-level token for global context (removed)
7. NEW: Modular positional encoding with absolute and relative options

Author: Marcus (DTU Wind Energy)
Based on discussions about transformer architectures for wind farm control.

================================================================================
TODO LIST - ACTIVE DEVELOPMENT
================================================================================

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
- "RelativePositionalBiasAdvanced": Relative bias with distance and angle features
- "RelativePositionalBiasFactorized": Factorized relative bias for efficiency
- "RelativePositionalBiasWithWind": Relative bias incorporating wind direction

MAIN TASKS:
[ ] Consider separate profile encoder instances for target networks if profiles become wind-direction-dependent (e.g. FourierProfileEncoderWithContext). Currently shared = soft-update is a no-op for encoder params.
[ ] Implement FourierProfileEncoderWithContext that takes wind direction as input
[ ] Add RelativePositionalBiasWithWind that takes wind direction as input
[ ] Make weights shared between actor and critic for: pos_encoder, rel_pos_bias, input_proj, transformer



TEMPORAL EXTENSIONS (OPTION B)
[ ] Design spatio-temporal attention mechanism
[ ] Implement SpatioTemporalTransformer class
[ ] Add temporal attention masking (causal)
[ ] Consider GTrXL for very long-range dependencies

KNOWN ISSUES / NOTES:
- History length of 15 may need tuning based on wake propagation time
- Gradient clipping at 1.0 is not yet tested

================================================================================
"""

import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union
from collections import deque

# Set memory allocation config BEFORE importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


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
# from MultiLayoutEnv import MultiLayoutEnv, LayoutConfig, create_layout_configs

from agent import WindFarmAgent

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
    make_BIG_config,
    transform_to_wind_relative,
    compute_wind_direction_deviation,
    EnhancedPerTurbineWrapper,
    # get_env_profiles,
    get_env_receptivity_profiles,
    get_env_influence_profiles,
    rotate_profiles_tensor,
    get_env_layout_indices,
    get_env_permutations,
    soft_update,
)

# Receptivity profile computation
from receptivity_profiles import compute_layout_profiles

# Evaluation import
from eval_utils import PolicyEvaluator, run_evaluation

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
    FourierProfileEncoderWithContext, # Needs wind direction as input. Not yet implemented
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
    wandb_project_name: str = "transformer_windfarm"
    wandb_entity: Optional[str] = None
    save_model: bool = True
    save_interval: int = 10000
    log_image: bool = False  # Log attention images to TensorBoard

    shuffle_turbs: bool = False  # Shuffle turbine order in obs/action
    max_episode_steps: Optional[int] = None # Max steps per episode (None = use env default)

    # === Receptivity Profile Settings ===
    profile_source: str = "PyWake"  # "pywake" or "geometric"
    profile_encoding_type: Optional[str] = None  # Now Optional, use None for no pos encoding
    profile_encoder_hidden: int = 128       # Hidden dim in profile encoder MLP
    rotate_profiles: bool = True            # Rotate profiles to wind-relative frame
    n_profile_directions: int = 360         # Number of directions in profile
    profile_fusion_type: str = "add"       # "add" or "joint" fusion of receptivity and influence profiles
    share_profile_encoder: bool = False         # Whether to share weights between actor and critic for profile encoder

    # === Environment Settings ===
    turbtype: str = "DTU10MW"  # Wind turbine type
    TI_type: str = "Random"   # Turbulence intensity sampling
    dt_sim: int = 5           # Simulation timestep (seconds)
    dt_env: int = 10          # Environment timestep (seconds)
    yaw_step: float = 5.0     # Max yaw change per sim step (degrees)
    max_eps: int = 20         # Number of flow passthroughs per episode
    num_envs: int = 1         # Number of parallel environments

    # === Evaluation Settings ===
    eval_interval: int = 50000        # How often to evaluate (in env steps)
    eval_initial: bool = False        # Run evaluation before training starts
    num_eval_steps: int = 200         # Number of steps per evaluation episode
    num_eval_episodes: int = 1        # Number of episodes per evaluation
    eval_layouts: str = ""            # Comma-separated eval layouts (empty = use training layouts)
    eval_seed: int = 42               # Seed for evaluation environments

    # === Layout Settings ===
    # Comma-separated list of layouts. Single = single-layout, Multiple = multi-layout
    layouts: str = "test_layout"  # e.g., "square_1,square_2,circular_1"
    
    # === Observation Settings ===
    history_length: int = 15            # Number of timesteps of history per feature
    use_wd_deviation: bool = False      # If True, convert WD to deviation from mean
    use_wind_relative_pos: bool = True  # Transform positions to wind-relative frame
    wd_scale_range: float = 90.0        # Only used if use_wd_deviation=True. Wind direction deviation range for scaling (±degrees → [-1,1])

    # === Transformer Architecture ===
    embed_dim: int = 128          # Transformer hidden dimension
    num_heads: int = 4            # Number of attention heads
    num_layers: int = 2           # Number of transformer layers
    mlp_ratio: float = 2.0        # FFN hidden dim = embed_dim * mlp_ratio
    dropout: float = 0.0          # Dropout rate (0 for RL typically)
    pos_embed_dim: int = 32       # Dimension for positional encoding
    

    # === Positional Encoding Settings ===
    # Options: "absolute_mlp", "relative_mlp", "relative_mlp_shared", 
    #          "sinusoidal_2d",
    pos_encoding_type: Optional[str] = None  # Now Optional, use None for no pos encoding
    # For relative encoding: number of hidden units in the bias MLP
    rel_pos_hidden_dim: int = 64
    # For relative encoding: whether to use separate bias per head
    rel_pos_per_head: bool = True
    pos_embedding_mode: str = "concat"  # "add" or "concat" positional embedding to token (only for absolute types)
    
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
    finetune_reset_actor_optimizer: int = 0     # If True, reset optimizers for fresh fine-tuning. If False, resume optimizer states too.
    finetune_reset_critic_optimizer: int = 0    # If True, reset optimizers for fresh fine-tuning. If False, resume optimizer states too.
    finetune_reset_alpha: int = 0               # If True, reset entropy coefficient. If False, keep from checkpoint.


import gc


def clear_gpu_memory():
    """Clear GPU memory - works on both NVIDIA and AMD."""
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        # Report
        device_name = torch.cuda.get_device_name(0)
        allocated = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {device_name}")
        print(f"Memory: {allocated:.2f}GB used / {total:.2f}GB total")

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
    
    # === Fourier Based ===
    "FourierProfileEncoder",                # Encode circular profiles via Fourier decomposition.
    "FourierProfileEncoderWithContext",     # Needs wind direction as input. Not yet implemented
]

def create_profile_encoding(
    profile_type: Optional[str],  # Optional
    embed_dim: int,
    hidden_channels: int,
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
    
    elif profile_type == "CNNProfileEncoder":
        
        recep_encoder = CNNProfileEncoder(
                embed_dim=embed_dim,
                hidden_channels=hidden_channels,
            )
        influence_encoder = CNNProfileEncoder(
                embed_dim=embed_dim,
                hidden_channels=hidden_channels,
            )
        
    elif profile_type == "DilatedProfileEncoder":
        recep_encoder = DilatedProfileEncoder(
                embed_dim=embed_dim,
                hidden_channels=hidden_channels,
            )
        influence_encoder = DilatedProfileEncoder(
                embed_dim=embed_dim,
                hidden_channels=hidden_channels,
            )
        
    elif profile_type == "AttentionProfileEncoder":
        recep_encoder = AttentionProfileEncoder(
                embed_dim=embed_dim,
                hidden_channels=hidden_channels,
                n_attention_heads=4,
            )
        influence_encoder = AttentionProfileEncoder(
                embed_dim=embed_dim,
                hidden_channels=hidden_channels,
                n_attention_heads=4,
            )
       

    elif profile_type == "FourierProfileEncoder":
        recep_encoder = FourierProfileEncoder(
                embed_dim=embed_dim,
                n_harmonics=8,
                use_phase=False,
                learnable_weights=True,
        )
        influence_encoder = FourierProfileEncoder(
                embed_dim=embed_dim,
                n_harmonics=8,
                use_phase=False,
                learnable_weights=True,
        )
        
    elif profile_type == "FourierProfileEncoderWithContext":
        raise NotImplementedError(
            "FourierProfileEncoderWithContext requires wind direction as input. See TODO."
        )
    
    else:
        raise ValueError(f"profile type '{profile_type}' not implemented yet.")
    
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
        # Shared profile encoders (optional - if None, creates own)
        shared_recep_encoder: Optional[nn.Module] = None,
        shared_influence_encoder: Optional[nn.Module] = None,
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

        assert profile_fusion_type in ("add", "joint"), \
            f"Invalid profile_fusion_type: {profile_fusion_type}"

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
            self.recep_encoder, self.influence_encoder = \
                create_profile_encoding(
                    profile_type=profile_encoding,
                    embed_dim=embed_dim,
                    hidden_channels=profile_encoder_hidden,
                )



        if profile_encoding is not None and profile_fusion_type == "joint":
            self.profile_fusion = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

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
        
        # ADD profile encoding (after projection, like positional encoding in LLMs)
        if self.recep_encoder and recep_profile is not None and influence_profile is not None:
            recep_embed = self.recep_encoder(recep_profile)  # (batch, n_turb, embed_dim)
            influence_embed = self.influence_encoder(influence_profile)  # (batch, n_turb, embed_dim)
            
            if self.profile_fusion_type == "joint":
                profile_combined = torch.cat([recep_embed, influence_embed], dim=-1)
                h = h + self.profile_fusion(profile_combined)
            else:
                h = h + recep_embed + influence_embed


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
        # Shared profile encoders (optional - if None, creates own)
        shared_recep_encoder: Optional[nn.Module] = None,
        shared_influence_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.pos_encoding_type = pos_encoding_type
        self.profile_encoding = profile_encoding
        self.profile_fusion_type = profile_fusion_type

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
            self.recep_encoder, self.influence_encoder = \
                create_profile_encoding(
                    profile_type=profile_encoding,
                    embed_dim=embed_dim,
                    hidden_channels=profile_encoder_hidden,
                )


        # Observation + action encoder
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
            self.profile_fusion = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )

        # Transformer encoder (choose based on encoding type)
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
        
        # ADD profile encoding (after projection, like positional encoding in LLMs)
        if self.recep_encoder and recep_profile is not None and influence_profile is not None:
            recep_embed = self.recep_encoder(recep_profile)  # (batch, n_turb, embed_dim)
            influence_embed = self.influence_encoder(influence_profile)  # (batch, n_turb, embed_dim)
            
            if self.profile_fusion_type == "joint":
                profile_combined = torch.cat([recep_embed, influence_embed], dim=-1)
                h = h + self.profile_fusion(profile_combined)
            else:
                h = h + recep_embed + influence_embed

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
# REPLAY BUFFER
# =============================================================================

class TransformerReplayBuffer:
    """
    Replay buffer with pre-allocated numpy arrays for O(1) insertion and
    vectorized batch sampling (no per-sample Python loops).

    Wind-relative transformation is applied at sample time to ensure
    correct positional encoding regardless of when the transition was collected.

    Profiles are looked up at sample time via vectorized gather + permute
    on a pre-padded profile registry, rather than storing full profiles
    per-transition.

    Storage (pre-allocated numpy arrays):
    - _obs:            (capacity, max_turbines, obs_dim)         float32
    - _next_obs:       (capacity, max_turbines, obs_dim)         float32
    - _actions:        (capacity, max_turbines, action_dim)      float32
    - _rewards:        (capacity,)                               float32
    - _dones:          (capacity,)                               float32
    - _raw_positions:  (capacity, max_turbines, 2)               float32
    - _attention_mask: (capacity, max_turbines)                  bool
    - _wind_directions:(capacity,)                               float32
    - _layout_indices: (capacity,)                               int32   (profiles only)
    - _permutations:   (capacity, max_turbines)                  int64   (profiles only)
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        rotor_diameter: float,
        max_turbines: int,
        obs_dim: int,
        action_dim: int,
        use_wind_relative: bool = True,
        use_profiles: bool = False,
        rotate_profiles: bool = False,
        profile_registry: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        """
        Args:
            capacity: Maximum number of transitions
            device: Torch device for sampled tensors
            rotor_diameter: For position normalization
            max_turbines: Maximum number of turbines across all layouts
            obs_dim: Observation dimension per turbine
            action_dim: Action dimension per turbine
            use_wind_relative: Whether to transform positions to wind-relative frame
            use_profiles: Whether to store and return receptivity profiles
            rotate_profiles: Whether to rotate profiles to wind-relative frame at sample time
            profile_registry: List of (recep, influence) tuples per layout, each (n_turb, n_dirs)
        """
        self.capacity = capacity
        self.device = device
        self.rotor_diameter = rotor_diameter
        self.max_turbines = max_turbines
        self.use_wind_relative = use_wind_relative
        self.use_profiles = use_profiles
        self.rotate_profiles = rotate_profiles
        self.position = 0
        self.size = 0  # Current number of stored transitions

        # --- Pre-allocate storage arrays ---
        self._obs = np.zeros((capacity, max_turbines, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, max_turbines, obs_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, max_turbines, action_dim), dtype=np.float32)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._raw_positions = np.zeros((capacity, max_turbines, 2), dtype=np.float32)
        self._attention_mask = np.zeros((capacity, max_turbines), dtype=bool)
        self._wind_directions = np.zeros(capacity, dtype=np.float32)

        # --- Profile-specific storage ---
        if self.use_profiles:
            assert profile_registry is not None, "Must provide profile_registry when use_profiles=True"

            self._layout_indices = np.zeros(capacity, dtype=np.int32)
            self._permutations = np.zeros((capacity, max_turbines), dtype=np.int64)

            # Pre-pad registry profiles to max_turbines for vectorized gather.
            # _padded_recep[layout_idx] = (max_turbines, n_dirs), zero-padded
            # _padded_infl[layout_idx]  = (max_turbines, n_dirs), zero-padded
            n_dirs = profile_registry[0][0].shape[1]
            n_layouts = len(profile_registry)
            self._padded_recep = np.zeros((n_layouts, max_turbines, n_dirs), dtype=np.float32)
            self._padded_infl = np.zeros((n_layouts, max_turbines, n_dirs), dtype=np.float32)
            for li, (recep, infl) in enumerate(profile_registry):
                nt = recep.shape[0]
                self._padded_recep[li, :nt] = recep
                self._padded_infl[li, :nt] = infl
            self._n_dirs = n_dirs
        else:
            self._layout_indices = None
            self._permutations = None

        alloc_mb = (
            self._obs.nbytes + self._next_obs.nbytes + self._actions.nbytes
            + self._rewards.nbytes + self._dones.nbytes
            + self._raw_positions.nbytes + self._attention_mask.nbytes
            + self._wind_directions.nbytes
        ) / 1e6
        print(f"[ReplayBuffer] Pre-allocated {alloc_mb:.1f} MB for {capacity} transitions "
              f"(max_turb={max_turbines}, obs_dim={obs_dim}, act_dim={action_dim})")

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        raw_positions: np.ndarray,
        attention_mask: np.ndarray,
        wind_direction: float,
        layout_index: Optional[int] = None,
        permutation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Store a transition by writing directly into pre-allocated arrays.
        
        Args:
            obs: (max_turbines, obs_dim)
            next_obs: (max_turbines, obs_dim)
            action: (max_turbines, action_dim)
            reward: scalar
            done: bool
            raw_positions: (max_turbines, 2)
            attention_mask: (max_turbines,) - True = padding
            wind_direction: scalar
            layout_index: Index of current layout (for profile lookup)
            permutation: Turbine permutation array (for shuffled profiles)
        """
        i = self.position

        self._obs[i] = obs
        self._next_obs[i] = next_obs
        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = float(done)
        self._raw_positions[i] = raw_positions
        self._attention_mask[i] = attention_mask
        self._wind_directions[i] = wind_direction

        if self.use_profiles:
            assert layout_index is not None, "layout_index required when use_profiles=True"
            self._layout_indices[i] = layout_index

            if permutation is not None:
                # Sanitize permutation: padding positions get identity mapping
                # so that indexing into the zero-padded registry stays zero.
                safe_perm = permutation.copy()
                n_real = int((~attention_mask).sum())
                if n_real < self.max_turbines:
                    safe_perm[n_real:] = np.arange(n_real, self.max_turbines)
                self._permutations[i] = safe_perm
            else:
                self._permutations[i] = np.arange(self.max_turbines)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch using vectorized array indexing (no Python loop).
        
        Returns:
            Dict with keys:
            - observations: (batch, max_turb, obs_dim)
            - next_observations: (batch, max_turb, obs_dim)
            - actions: (batch, max_turb, action_dim)
            - positions: (batch, max_turb, 2) - transformed and normalized
            - attention_mask: (batch, max_turb)
            - rewards: (batch, 1)
            - dones: (batch, 1)
            - receptivity: (batch, max_turb, n_directions) - only if use_profiles=True
            - influence: (batch, max_turb, n_directions) - only if use_profiles=True
        """
        indices = np.random.choice(self.size, batch_size, replace=False)

        # --- Vectorized array indexing (the whole point) ---
        raw_positions = self._raw_positions[indices]            # (B, T, 2)
        wind_directions = self._wind_directions[indices]        # (B,)

        # Normalize positions by rotor diameter
        positions_norm = raw_positions / self.rotor_diameter

        # Convert to tensors
        positions_tensor = torch.tensor(positions_norm, device=self.device, dtype=torch.float32)

        # Conditionally apply wind-relative transformation
        if self.use_wind_relative:
            wind_dir_tensor = torch.tensor(wind_directions, device=self.device, dtype=torch.float32)
            positions_final = transform_to_wind_relative(positions_tensor, wind_dir_tensor)
        else:
            positions_final = positions_tensor

        result = {
            "observations": torch.tensor(self._obs[indices], device=self.device, dtype=torch.float32),
            "next_observations": torch.tensor(self._next_obs[indices], device=self.device, dtype=torch.float32),
            "actions": torch.tensor(self._actions[indices], device=self.device, dtype=torch.float32),
            "positions": positions_final,
            "attention_mask": torch.tensor(self._attention_mask[indices], device=self.device, dtype=torch.bool),
            "rewards": torch.tensor(self._rewards[indices], device=self.device, dtype=torch.float32).unsqueeze(-1),
            "dones": torch.tensor(self._dones[indices], device=self.device, dtype=torch.float32).unsqueeze(-1),
        }

        # --- Vectorized profile lookup + permutation ---
        if self.use_profiles:
            layout_idx_batch = self._layout_indices[indices]    # (B,)
            perm_batch = self._permutations[indices]            # (B, T)

            # Gather from pre-padded registry: (B, T, n_dirs)
            recep_batch = self._padded_recep[layout_idx_batch]  # (B, T, D)
            infl_batch = self._padded_infl[layout_idx_batch]    # (B, T, D)

            # Apply permutation via advanced indexing (vectorized, no loop)
            # perm_batch[:, :, None] broadcasts over the n_dirs axis
            recep_batch = np.take_along_axis(recep_batch, perm_batch[:, :, None], axis=1)
            infl_batch = np.take_along_axis(infl_batch, perm_batch[:, :, None], axis=1)

            # Optionally rotate profiles to wind-relative frame
            if self.rotate_profiles:
                recep_batch = self._rotate_profiles_batch(recep_batch, wind_directions)
                infl_batch = self._rotate_profiles_batch(infl_batch, wind_directions)

            result["receptivity"] = torch.tensor(recep_batch, device=self.device, dtype=torch.float32)
            result["influence"] = torch.tensor(infl_batch, device=self.device, dtype=torch.float32)

        return result

    def _rotate_profiles_batch(
        self,
        profiles: np.ndarray,
        wind_directions: np.ndarray
    ) -> np.ndarray:
        """
        Rotate profiles so current wind direction is at index 0 (vectorized).
        
        Args:
            profiles: (batch, max_turb, n_directions)
            wind_directions: (batch,) wind directions in degrees
        
        Returns:
            Rotated profiles with same shape
        """
        n_directions = profiles.shape[2]
        degrees_per_index = 360.0 / n_directions

        shifts = np.round(wind_directions / degrees_per_index).astype(int)
        # Build shifted index array: (batch, 1, n_directions)
        indices = (np.arange(n_directions)[None, None, :] + shifts[:, None, None]) % n_directions
        return np.take_along_axis(profiles, indices, axis=-1)

    def __len__(self) -> int:
        return self.size



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
# FINE-TUNING DIAGNOSTICS
# =============================================================================

def log_optimizer_effective_lr(optimizer, name: str, nominal_lr: float):
    """Print effective learning rate statistics for an optimizer."""
    effective_lrs = []
    momentum_norms = []
    
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                if 'exp_avg_sq' in state:
                    # Effective LR ≈ nominal_lr / sqrt(exp_avg_sq + eps)
                    second_moment = state['exp_avg_sq']
                    eff_lr = nominal_lr / (second_moment.sqrt().mean().item() + 1e-8)
                    effective_lrs.append(eff_lr)
                if 'exp_avg' in state:
                    momentum_norms.append(state['exp_avg'].norm().item())
    
    if effective_lrs:
        mean_eff_lr = np.mean(effective_lrs)
        ratio = mean_eff_lr / nominal_lr
        print(f"  {name}: nominal_lr={nominal_lr:.2e}, effective_lr={mean_eff_lr:.2e}, ratio={ratio:.4f}")
    else:
        print(f"  {name}: Fresh optimizer (no accumulated state)")


def compute_optimizer_diagnostics(optimizer, name: str, nominal_lr: float) -> dict:
    """Compute optimizer diagnostics for logging to wandb/tensorboard."""
    effective_lrs = []
    momentum_norms = []
    second_moments = []
    
    for group in optimizer.param_groups:
        for p in group['params']:
            if p in optimizer.state:
                state = optimizer.state[p]
                if 'exp_avg_sq' in state:
                    second_moment = state['exp_avg_sq'].mean().item()
                    second_moments.append(second_moment)
                    eff_lr = nominal_lr / (second_moment ** 0.5 + 1e-8)
                    effective_lrs.append(eff_lr)
                if 'exp_avg' in state:
                    momentum_norms.append(state['exp_avg'].norm().item())
    
    diagnostics = {}
    if effective_lrs:
        diagnostics[f"finetune/{name}_effective_lr"] = np.mean(effective_lrs)
        diagnostics[f"finetune/{name}_effective_lr_ratio"] = np.mean(effective_lrs) / nominal_lr
    if momentum_norms:
        diagnostics[f"finetune/{name}_momentum_norm"] = np.mean(momentum_norms)
    if second_moments:
        diagnostics[f"finetune/{name}_second_moment"] = np.mean(second_moments)
    
    return diagnostics


def log_finetune_diagnostics(
    writer,
    global_step: int,
    actor_optimizer,
    q_optimizer,
    policy_lr: float,
    q_lr: float,
    qf1_values: Optional[torch.Tensor] = None,
    qf2_values: Optional[torch.Tensor] = None,
    episode_returns: Optional[list] = None,
    alpha: Optional[float] = None,
    policy_entropy: Optional[float] = None,
):
    """Log fine-tuning specific diagnostics to tensorboard/wandb."""
    
    # Optimizer state diagnostics
    actor_diag = compute_optimizer_diagnostics(actor_optimizer, "actor", policy_lr)
    critic_diag = compute_optimizer_diagnostics(q_optimizer, "critic", q_lr)
    
    for key, value in {**actor_diag, **critic_diag}.items():
        writer.add_scalar(key, value, global_step)
    
    # Q-value diagnostics (detect overestimation)
    if qf1_values is not None and qf2_values is not None:
        q_mean = (qf1_values.mean().item() + qf2_values.mean().item()) / 2
        q_std = (qf1_values.std().item() + qf2_values.std().item()) / 2
        writer.add_scalar("finetune/q_mean", q_mean, global_step)
        writer.add_scalar("finetune/q_std", q_std, global_step)
        
        # Q-value vs actual returns (if we have episode data)
        if episode_returns and len(episode_returns) > 0:
            mean_return = np.mean(episode_returns)
            q_overestimation = q_mean - mean_return
            writer.add_scalar("finetune/q_overestimation", q_overestimation, global_step)
            writer.add_scalar("finetune/q_to_return_ratio", q_mean / (mean_return + 1e-8), global_step)
    
    # Entropy tracking
    if alpha is not None:
        writer.add_scalar("finetune/alpha", alpha, global_step)
    
    if policy_entropy is not None:
        writer.add_scalar("finetune/policy_entropy", policy_entropy, global_step)


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Main training function."""
    
    # Parse arguments
    args = tyro.cli(Args)
    
    # Parse layouts
    layout_names = [l.strip() for l in args.layouts.split(",")]
    is_multi_layout = len(layout_names) > 1
    
    # Parse evaluation layouts
    if args.eval_layouts.strip():
        eval_layout_names = [l.strip() for l in args.eval_layouts.split(",")]
    else:
        eval_layout_names = layout_names  # Use training layouts for evaluation
    
    print(f"Training layouts: {layout_names}")
    print(f"Evaluation layouts: {eval_layout_names}")


    # Create run name
    run_name = f"{args.exp_name}"
    
    print("=" * 60)
    print(f"Transformer SAC for Wind Farm Control")
    print("=" * 60)
    if is_multi_layout:
        print(f"Mode: Multi-layout training with layouts: {layout_names}")
    else:
        print(f"Mode: Single-layout training: {layout_names[0]}")
    print(f"Run name: {run_name}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    os.makedirs(f"runs/{run_name}/attention_plots", exist_ok=True)
    
    clear_gpu_memory()
    
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
    print("Setting up layouts...")
    layouts = []
    for name in layout_names:
        x_pos, y_pos = get_layout_positions(name, wind_turbine)
        layout = LayoutConfig(name=name, x_pos=x_pos, y_pos=y_pos)
        

        if args.profile_encoding_type is not None:
            if args.profile_source.lower() == "geometric":
                from geometric_profiles import compute_layout_profiles_vectorized
                
                # Get rotor diameter as a float (geometric version doesn't need the full WT object)
                D = wind_turbine.diameter()  # or however DTU10MW exposes this
                
                print(f"Computing GEOMETRIC profiles for layout: {name}")
                receptivity_profiles, influence_profiles = compute_layout_profiles_vectorized(
                    x_pos, y_pos,
                    rotor_diameter=D,
                    k_wake=0.04,
                    n_directions=args.n_profile_directions,
                    sigma_smooth=10.0,
                    scale_factor=15.0,
                )
            elif args.profile_source.lower() == "pywake":
                print(f"Computing PyWake profiles for layout: {name}")
                receptivity_profiles, influence_profiles = compute_layout_profiles(
                    x_pos, y_pos, wind_turbine,
                    n_directions=args.n_profile_directions,
                )
            else:
                raise ValueError(
                    f"Unknown profile_source: {args.profile_source}. "
                    f"Use 'pywake' or 'geometric'."
                )
            
            layout.receptivity_profiles = receptivity_profiles  # (n_turbines, n_directions
            layout.influence_profiles = influence_profiles      # (n_turbines, n_directions
            
        layouts.append(layout)

    if args.profile_encoding_type is not None:
        use_profiles = True
    else:
        use_profiles = False



    # Build profile registry from layouts
    if use_profiles:
        profile_registry = [
            (layout.receptivity_profiles, layout.influence_profiles)
            for layout in layouts
        ]
    else:
        profile_registry = None


    # Environment configuration
    config = make_env_config()
    # config = make_BIG_config()
    
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
        env = WindFarmEnv(x_pos=x_pos,
                          y_pos=y_pos,
                          reset_init=False,  # Defer reset to training loop
                          **base_env_kwargs)
        env.action_space.seed(args.seed)
        return env
    
    def combined_wrapper(env: gym.Env) -> gym.Env:
        """
        Combined wrapper that:
        1. Applies PerTurbineObservationWrapper (reshapes obs to per-turbine)
        2. Optionally applies EnhancedPerTurbineWrapper (converts WD to deviation)
        """
        env = PerTurbineObservationWrapper(env)
        if args.use_wd_deviation:
            env = EnhancedPerTurbineWrapper(env, wd_scale_range=args.wd_scale_range)
        return env
    
    def make_env_fn(seed):
        """Factory function for vectorized environments."""
        def _init():
            env = MultiLayoutEnv(
                layouts=layouts,
                env_factory=env_factory,
                per_turbine_wrapper=combined_wrapper,  # Use combined wrapper
                seed=seed,
                shuffle=args.shuffle_turbs,  # Shuffle turbines within each layout
                max_episode_steps=args.max_episode_steps,
            )
            return env
        return _init

    # Create vectorized environments
    print(f"Creating {args.num_envs} parallel environment(s)...")
    envs = gym.vector.AsyncVectorEnv(
        [make_env_fn(args.seed + i) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    envs = RecordEpisodeVals(envs)
       

    n_turbines_max = envs.env.get_attr('max_turbines')[0]
    obs_dim_per_turbine = envs.single_observation_space.shape[-1]
    action_dim_per_turbine = 1
    rotor_diameter = envs.env.get_attr('rotor_diameter')[0]

    print(f"Max turbines: {n_turbines_max}")
    print(f"Obs dim per turbine: {obs_dim_per_turbine}")
    print(f"Action dim per turbine: {action_dim_per_turbine}")
    print(f"Rotor diameter: {rotor_diameter:.1f} m")
    

    # Create policy evaluator
    evaluator = PolicyEvaluator(
        agent=None,  # Will be set after actor is created
        eval_layouts=eval_layout_names,
        env_factory=env_factory,
        combined_wrapper=combined_wrapper,
        num_envs=args.num_envs,
        num_eval_steps=args.num_eval_steps,
        num_eval_episodes=args.num_eval_episodes,
        device=device,
        rotor_diameter=rotor_diameter,
        wind_turbine=wind_turbine,
        seed=args.eval_seed,
        max_turbines=n_turbines_max,
        deterministic=False,
        use_profiles=use_profiles,  # NEW: Pass profile setting
        n_profile_directions=args.n_profile_directions,  # NEW: Pass profile resolution
        profile_source=args.profile_source,
    )


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
        layout_names=layout_names,
        log_every=250000,  # Base frequency - others are multiples of this
    )
    # Frequencies will be:
    #   - summary metrics: every 100 steps
    #   - attention analysis: every 500 steps  
    #   - gradient norms: every 100 steps
    #   - q-value stats: every 50 steps
    #   - diagnostic print: every 2000 steps

    print(f"Debug logger initialized for layouts: {layout_names}")
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
                "debug/n_layouts": len(layout_names),
                "debug/layout_names": layout_names,
                "debug/is_multi_layout": is_multi_layout,
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

    # ==========================================================================
    # Create SHARED profile encoders (if using profiles)
    # ==========================================================================
    if args.profile_encoding_type is not None:
        if args.share_profile_encoder:
            print(f"Creating shared profile encoders: {args.profile_encoding_type}")
            shared_recep_encoder, shared_influence_encoder = create_profile_encoding(
                profile_type=args.profile_encoding_type,
                embed_dim=args.embed_dim,
                hidden_channels=args.profile_encoder_hidden,
            )
            # Move to device
            shared_recep_encoder = shared_recep_encoder.to(device)
            shared_influence_encoder = shared_influence_encoder.to(device)
        
            # Count shared encoder parameters
            recep_params = sum(p.numel() for p in shared_recep_encoder.parameters())
            influence_params = sum(p.numel() for p in shared_influence_encoder.parameters())
            print(f"Shared receptivity encoder parameters: {recep_params:,}")
            print(f"Shared influence encoder parameters: {influence_params:,}")
        else:
            print(f"Using separate profile encoders for each network, handled internally in the critic and actor classes")
            shared_recep_encoder = None  # 
            shared_influence_encoder = None  # 
    else:
        shared_recep_encoder = None
        shared_influence_encoder = None


    # Common profile args (to avoid repetition)
    common_kwargs = {
        # Architecture
        "obs_dim_per_turbine": obs_dim_per_turbine,
        "action_dim_per_turbine": action_dim_per_turbine,
        "embed_dim": args.embed_dim,
        "pos_embed_dim": args.pos_embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        # Positional encoding
        "pos_encoding_type": args.pos_encoding_type,
        "rel_pos_hidden_dim": args.rel_pos_hidden_dim,
        "rel_pos_per_head": args.rel_pos_per_head,
        "pos_embedding_mode": args.pos_embedding_mode,
        # PyWake profiles
        "profile_encoding": args.profile_encoding_type,
        "profile_encoder_hidden": args.profile_encoder_hidden,
        "n_profile_directions": args.n_profile_directions,
        "profile_fusion_type": args.profile_fusion_type,
        # SHARED encoders
        "shared_recep_encoder": shared_recep_encoder,
        "shared_influence_encoder": shared_influence_encoder,
    }
        

    # Actor has additional action scaling params
    actor = TransformerActor(
        action_scale=action_scale,
        action_bias=action_bias,
        **common_kwargs,
    ).to(device)
    

    agent = WindFarmAgent(
        actor=actor,
        device=device,
        rotor_diameter=rotor_diameter,
        use_wind_relative=args.use_wind_relative_pos,
        use_profiles=use_profiles,
        rotate_profiles=args.rotate_profiles,
    )

    # Update evaluator with actor reference
    evaluator.agent = agent

    # Critics all use the same config
    qf1 = TransformerCritic(**common_kwargs).to(device)
    qf2 = TransformerCritic(**common_kwargs).to(device)
    qf1_target = TransformerCritic(**common_kwargs).to(device)
    qf2_target = TransformerCritic(**common_kwargs).to(device)
    
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    # Count parameters
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in qf1.parameters())
    print(f"Actor parameters: {actor_params:,}")
    print(f"Critic parameters: {critic_params:,} (x2)")
    
    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    # q_optimizer = optim.Adam(
    #     list(qf1.parameters()) + list(qf2.parameters()),
    #     lr=args.q_lr
    # )
    
    # Get critic parameters, excluding shared profile encoders
    def get_critic_params_excluding_shared(critic, shared_recep, shared_influence):
        '''Get critic parameters, excluding shared modules.'''
        shared_param_ids = set()
        if shared_recep is not None:
            shared_param_ids.update(id(p) for p in shared_recep.parameters())
        if shared_influence is not None:
            shared_param_ids.update(id(p) for p in shared_influence.parameters())
        
        return [p for p in critic.parameters() if id(p) not in shared_param_ids]
    
    qf1_params = get_critic_params_excluding_shared(qf1, shared_recep_encoder, shared_influence_encoder)
    qf2_params = get_critic_params_excluding_shared(qf2, shared_recep_encoder, shared_influence_encoder)
    
    q_optimizer = optim.Adam(
        qf1_params + qf2_params,
        lr=args.q_lr
    )
    
    # Verify parameter counts
    if shared_recep_encoder is not None:
        actor_unique = sum(p.numel() for p in actor.parameters())
        critic_unique = sum(p.numel() for p in qf1_params)
        shared_total = sum(p.numel() for p in shared_recep_encoder.parameters()) + \
                       sum(p.numel() for p in shared_influence_encoder.parameters())
        print(f"Actor parameters (includes shared): {actor_unique:,}")
        print(f"Critic parameters (excluding shared): {critic_unique:,} (x2)")
        print(f"Shared encoder parameters: {shared_total:,}")


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
        
        # === Actor optimizer ===
        if not args.finetune_reset_actor_optimizer:
            actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            print(f"✓ Loaded actor optimizer state")
        else:
            print(f"✓ Reset actor optimizer (fresh)")

        # === Critic optimizer ===
        if not args.finetune_reset_critic_optimizer:
            q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
            print(f"✓ Loaded critic optimizer state")
        else:
            print(f"✓ Reset critic optimizer (fresh)")
        
        # === Alpha (entropy coefficient) ===
        if args.autotune:
            if not args.finetune_reset_alpha:
                if "log_alpha" in checkpoint:
                    log_alpha.data = checkpoint["log_alpha"].to(device)
                    alpha = log_alpha.exp().item()
                    print(f"✓ Loaded entropy coefficient: alpha={alpha:.4f}")
                if "alpha_optimizer_state_dict" in checkpoint:
                    alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
                    print(f"✓ Loaded alpha optimizer state")
            else:
                print(f"✓ Reset entropy coefficient (alpha={alpha:.4f})")
       
        # === Resume step logic ===
        ## REMOVED FOR SIMPLICITY
        # Only resume from checkpoint step if keeping ALL optimizer states
        # if (not args.finetune_reset_actor_optimizer and 
        #     not args.finetune_reset_critic_optimizer and
        #     not args.finetune_reset_alpha):
        #     start_step = checkpoint["step"]
        #     print(f"✓ Resuming from step {start_step}")
        # else:
        #     print(f"✓ Starting from step 0 (fine-tuning mode)")

        # === Diagnostic: Check effective learning rates ===
        print(f"\n--- Optimizer State Diagnostics ---")
        log_optimizer_effective_lr(actor_optimizer, "Actor", args.policy_lr)
        log_optimizer_effective_lr(q_optimizer, "Critic", args.q_lr)
        
        # Log checkpoint info
        if "args" in checkpoint:
            ckpt_args = checkpoint["args"]
            print(f"\nOriginal training config:")
            print(f"  - Layouts: {ckpt_args.get('layouts', 'unknown')}")
            print(f"  - Total timesteps: {ckpt_args.get('total_timesteps', 'unknown')}")
            print(f"  - Pos encoding: {ckpt_args.get('pos_encoding_type', 'unknown')}")
        
        print(f"\nFine-tuning config:")
        print(f"  - Target layouts: {args.layouts}")
        print(f"  - Reset actor optimizer: {args.finetune_reset_actor_optimizer}")
        print(f"  - Reset critic optimizer: {args.finetune_reset_critic_optimizer}")
        print(f"  - Reset alpha: {args.finetune_reset_alpha}")
        print(f"{'='*60}\n")
    
    # =========================================================================
    # REPLAY BUFFER
    # =========================================================================

    rb = TransformerReplayBuffer(
        capacity=args.buffer_size,
        device=device,
        rotor_diameter=rotor_diameter,
        max_turbines=n_turbines_max,
        obs_dim=obs_dim_per_turbine,
        action_dim=action_dim_per_turbine,
        use_wind_relative=args.use_wind_relative_pos,
        use_profiles=use_profiles,
        rotate_profiles=args.rotate_profiles,
        profile_registry=profile_registry,
    )


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


    # Track evaluation timing
    next_eval_step = args.eval_interval
    
    # Initial evaluation
    if args.eval_initial:
        print("\nRunning initial evaluation before training...")
        eval_metrics = evaluator.evaluate()
        eval_dict = eval_metrics.to_dict()
        
        for name, value in eval_dict.items():
            writer.add_scalar(name, value, 0)
        
        print(f"Initial eval - Mean reward: {eval_metrics.mean_reward:.4f}, "
              f"Power ratio: {eval_metrics.power_ratio:.4f}")


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
        
        # Get environment info (needed for replay buffer)
        wind_dirs = get_env_wind_directions(envs)
        raw_positions = get_env_raw_positions(envs)
        current_masks = get_env_attention_masks(envs)

        # Get layout identifiers for replay buffer (lightweight)
        if args.profile_encoding_type is not None:
            current_layout_indices = get_env_layout_indices(envs)
            current_permutations = get_env_permutations(envs)
        else:
            current_layout_indices = None
            current_permutations = None


        # Select action
        if global_step < args.learning_starts:
            # Random exploration
            actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                actions = agent.act(envs, obs)
        
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


        # Handle final observations
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_obs"][idx]
        
        # Store in replay buffer
        for i in range(args.num_envs):
            done = terminations[i] or truncations[i]
            action_reshaped = actions[i].reshape(-1, action_dim_per_turbine)
        
            layout_idx_i = current_layout_indices[i] if current_layout_indices is not None else None
            perm_i = current_permutations[i] if current_permutations is not None else None

            rb.add(
                obs[i],
                real_next_obs[i],
                action_reshaped,
                rewards[i],
                done,
                raw_positions[i],
                current_masks[i],
                wind_dirs[i],
                layout_index=layout_idx_i,
                permutation=perm_i,
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
                
                # Get profiles from batch (will be None if not using profiles)
                batch_receptivity = data.get("receptivity", None)
                batch_influence = data.get("influence", None)


                # -----------------------------------------------------------------
                # Update Critics
                # -----------------------------------------------------------------
                with torch.no_grad():
                    # Get next actions from current policy
                    next_actions, next_log_pi, _, _ = actor.get_action(
                        data["next_observations"],
                        data["positions"],
                        batch_mask,
                        recep_profile=batch_receptivity,
                        influence_profile=batch_influence,
                    )
                    
                    # Compute target Q-values
                    qf1_next = qf1_target(
                        data["next_observations"], next_actions, 
                        data["positions"], batch_mask, 
                        recep_profile=batch_receptivity,
                        influence_profile=batch_influence,
                    )
                    qf2_next = qf2_target(
                        data["next_observations"], next_actions, 
                        data["positions"], batch_mask, 
                        recep_profile=batch_receptivity,
                        influence_profile=batch_influence,
                    )
                    min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
                    
                    # Bellman target
                    target_q = data["rewards"] + (1 - data["dones"]) * args.gamma * min_qf_next
                
                # Current Q-values
                qf1_value = qf1(data["observations"], data["actions"], 
                                data["positions"], batch_mask, 
                                recep_profile=batch_receptivity,
                                influence_profile=batch_influence)
                
                qf2_value = qf2(data["observations"], data["actions"], 
                                data["positions"], batch_mask, 
                                recep_profile=batch_receptivity,
                                influence_profile=batch_influence)
                
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
                q_optimizer.zero_grad(set_to_none=True)
                qf_loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        qf1_params + qf2_params,
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
                        data["observations"], data["positions"], batch_mask,
                        recep_profile=batch_receptivity,
                        influence_profile=batch_influence,
                    )
                    
                    # Q-values for policy actions
                    qf1_pi = qf1(data["observations"], actions_pi, data["positions"], 
                                 batch_mask, 
                                 recep_profile=batch_receptivity,
                                 influence_profile=batch_influence)
                    qf2_pi = qf2(data["observations"], actions_pi, data["positions"], 
                                 batch_mask, 
                                 recep_profile=batch_receptivity,
                                 influence_profile=batch_influence)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    
                    # Policy loss (maximize Q - alpha * entropy)
                    actor_loss = (alpha * log_pi - min_qf_pi).mean()
                    
                    # Update actor
                    actor_optimizer.zero_grad(set_to_none=True)
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
                        log_pi_detached = log_pi.detach()
                        
                        # Adaptive target entropy per sample
                        target_entropy_batch = compute_adaptive_target_entropy(
                            data["attention_mask"],
                            action_dim_per_turbine
                        )
                        
                        # Alpha loss
                        alpha_loss = (-log_alpha.exp() * (log_pi_detached + target_entropy_batch)).mean()
                        
                        alpha_optimizer.zero_grad(set_to_none=True)
                        alpha_loss.backward()
                        alpha_optimizer.step()
                        alpha = log_alpha.exp().item()
                        
                        loss_accumulator['alpha_loss'].append(alpha_loss.item())
                
                # -----------------------------------------------------------------
                # Update Target Networks
                # -----------------------------------------------------------------
                # NOTE: When share_profile_encoder=True, the shared encoder params appear in both
                 # qf1 and qf1_target, making the soft-update a no-op (x ← τx + (1-τ)x = x) for those params.
                if total_gradient_steps % args.target_network_frequency == 0:

                    soft_update(qf1, qf1_target, args.tau)
                    soft_update(qf2, qf2_target, args.tau)
                
                # Attention physics analysis (frequency controlled by logger)
                if debug_logger.should_log_attention(total_gradient_steps):
                    with torch.no_grad():
                        # Get fresh attention weights from a small batch
                        sample_size = min(8, args.batch_size)
                        _, _, _, attn_weights = actor.get_action(
                            data["observations"][:sample_size],
                            data["positions"][:sample_size],
                            batch_mask[:sample_size] if batch_mask is not None else None,
                            recep_profile=batch_receptivity[:sample_size] if batch_receptivity is not None else None,
                            influence_profile=batch_influence[:sample_size] if batch_influence is not None else None,
                            need_weights=True, # Need this if we actually want attention
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
        

                # === Fine-tuning diagnostics (when resuming from checkpoint) ===
                if args.resume_checkpoint is not None and update % 100 == 0:
                    # Collect recent episode returns for Q-value comparison
                    recent_returns = list(envs.return_queue)[-10:] if hasattr(envs, 'return_queue') else []
                    
                    # Compute policy entropy from recent actions
                    with torch.no_grad():
                        _, log_pi_diag, _, _ = actor.get_action(
                            data["observations"][:32],
                            data["positions"][:32],
                            data["attention_mask"][:32],
                            recep_profile=batch_receptivity[:32] if batch_receptivity is not None else None,
                            influence_profile=batch_influence[:32] if batch_influence is not None else None,
                        )
                        policy_entropy = -log_pi_diag.mean().item()
                    
                    log_finetune_diagnostics(
                        writer=writer,
                        global_step=global_step,
                        actor_optimizer=actor_optimizer,
                        q_optimizer=q_optimizer,
                        policy_lr=args.policy_lr,
                        q_lr=args.q_lr,
                        qf1_values=qf1_value,
                        qf2_values=qf2_value,
                        episode_returns=recent_returns,
                        alpha=alpha,
                        policy_entropy=policy_entropy,
                    )


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

        # =====================================================================
        # PERIODIC EVALUATION
        # =====================================================================
        
        if global_step >= next_eval_step:
            print(f"\nRunning evaluation at step {global_step}...")
            eval_metrics = evaluator.evaluate()
            eval_dict = eval_metrics.to_dict()
            
            # Log to tensorboard/wandb
            for name, value in eval_dict.items():
                writer.add_scalar(name, value, global_step)
            
            print(f"Eval step {global_step} - Mean reward: {eval_metrics.mean_reward:.4f}, "
                  f"Power ratio: {eval_metrics.power_ratio:.4f}")
            
            # Per-layout summary
            if len(eval_metrics.per_layout_rewards) > 1:
                print("  Per-layout power ratios:")
                for layout, ratio in eval_metrics.per_layout_power_ratios.items():
                    print(f"    {layout}: {ratio:.4f}")
            
            next_eval_step += args.eval_interval
        
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
    

    # Close evaluator
    evaluator.close()

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()