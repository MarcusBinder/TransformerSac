"""
Transformer-based SAC for Wind Farm Control - V12

Changes in V12:
- Modular positional encoding system with multiple options
- Added Relative Positional Encoding (RPE) for physics-aware attention
- Attention bias directly encodes pairwise spatial relationships
- Scaffold for future encodings (sinusoidal, RoPE)

A clean implementation of transformer-based Soft Actor-Critic for wind farm
yaw control with the goal of generalizing across different farm layouts.

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
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from collections import deque

# WindGym imports (adjust path as needed for your setup)
from WindGym import WindFarmEnv
from WindGym.wrappers import RecordEpisodeVals, PerTurbineObservationWrapper
from WindGym.utils.generate_layouts import generate_square_grid, generate_cirular_farm, generate_right_triangle_grid, generate_line_dots_multiple_thetas
from collections import deque
from MultiLayoutEnv import MultiLayoutEnv, LayoutConfig, create_layout_configs

# Logging utilities for multi-layout training
from multi_layout_debug import (
    MultiLayoutDebugLogger,
    create_debug_logger,
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
    save_interval: int = 25000
    log_image: bool = False  # Log attention images to TensorBoard

    shuffle_turbs: bool = False  # Shuffle turbine order in obs/action
    
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


# =============================================================================
# OBSERVATION PROCESSING
# =============================================================================

def compute_wind_direction_deviation(
    local_wd: np.ndarray, 
    mean_wd: float,
    scale_range: float = 90.0
) -> np.ndarray:
    """
    Compute wind direction deviation from mean, scaled to [-1, 1].
    
    The deviation captures local flow variations due to wakes and turbulence,
    while being invariant to the absolute wind direction (which is encoded
    in the positional encoding via wind-relative coordinates).
    
    Args:
        local_wd: Per-turbine local wind direction in degrees, shape (n_turbines,)
                  or (n_turbines, history_length) for history
        mean_wd: Farm-level mean wind direction in degrees
        scale_range: Deviation range for scaling. Deviations beyond this are clipped.
                     Default 90° means ±90° maps to [-1, 1].
    
    Returns:
        Scaled deviation in [-1, 1], same shape as input
    """
    # Compute raw deviation
    deviation = local_wd - mean_wd
    
    # Wrap to [-180, 180]
    deviation = ((deviation + 180) % 360) - 180
    
    # Scale to [-1, 1] based on scale_range (±90° -> [-1, 1])
    scaled = np.clip(deviation / scale_range, -1.0, 1.0)
    
    return scaled.astype(np.float32)


class EnhancedPerTurbineWrapper(gym.Wrapper):
    """
    Enhanced wrapper that transforms wind direction observations to DEVIATION from mean.
    
    This wrapper sits on top of PerTurbineObservationWrapper and transforms the 
    absolute wind direction values to deviations from the farm's mean wind direction.
    
    Why deviation instead of absolute?
    - With wind-relative positional encoding, absolute wind direction is redundant
    - Deviation captures local wake effects and turbulence
    - Makes observations invariant to global wind direction changes
    
    Observation order per turbine (from TurbMes.get_measurements):
        [probes..., ws..., wd..., yaw..., TI, power...]
    
    We detect the WD indices by reading the TurbMes configuration directly.
    """
    
    def __init__(
        self, 
        env: gym.Env,
        wd_scale_range: float = 90.0,
    ):
        """
        Args:
            env: Base environment (should be wrapped with PerTurbineObservationWrapper)
            wd_scale_range: Range for wind direction deviation scaling.
                           ±wd_scale_range degrees maps to [-1, 1].
                           Default 90° means ±90° deviation → [-1, 1].
        """
        super().__init__(env)
        self.wd_scale_range = wd_scale_range
        
        # Get the base WindFarmEnv to access farm_measurements config
        self._base_env = self._get_base_env()
        
        # Detect wind direction indices from the measurement configuration
        self._wd_indices, self._wd_min, self._wd_max = self._detect_wd_indices()
        
        if self._wd_indices is not None:
            print(f"EnhancedPerTurbineWrapper: Found WD at indices {self._wd_indices} "
                  f"(scaling from [{self._wd_min}, {self._wd_max}] to deviation)")
        else:
            print("EnhancedPerTurbineWrapper: No wind direction in observations, "
                  "wrapper will pass through unchanged")
    
    def _get_base_env(self):
        """Unwrap to get the base WindFarmEnv."""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env
    
    def _detect_wd_indices(self) -> Tuple[Optional[List[int]], float, float]:
        """
        Detect which observation indices correspond to wind direction.
        
        Reads directly from TurbMes configuration to determine:
        1. How many probe features come first
        2. How many WS features 
        3. Where WD features start and how many there are
        
        Returns:
            (wd_indices, wd_min, wd_max) or (None, 0, 360) if WD not in obs
        """
        try:
            fm = self._base_env.farm_measurements
            turb_mes = fm.turb_mes[0]  # All turbines have same structure
            
            # Check if wind direction is enabled
            if not fm.turb_wd:
                return None, 0, 360
            
            # Count features before wind direction
            offset = 0
            
            # 1. Probes (if any)
            if hasattr(turb_mes, 'n_probes') and turb_mes.n_probes > 0:
                offset += turb_mes.n_probes
            
            # 2. Wind speed features
            ws_mes = turb_mes.ws
            n_ws = (1 if ws_mes.current else 0) + (ws_mes.history_N if ws_mes.rolling_mean else 0)
            offset += n_ws
            
            # 3. Wind direction features start at offset
            wd_mes = turb_mes.wd
            n_wd = (1 if wd_mes.current else 0) + (wd_mes.history_N if wd_mes.rolling_mean else 0)
            
            if n_wd == 0:
                return None, 0, 360
            
            wd_indices = list(range(offset, offset + n_wd))
            wd_min = turb_mes.wd_min
            wd_max = turb_mes.wd_max
            
            return wd_indices, wd_min, wd_max
            
        except Exception as e:
            print(f"Warning: Could not detect WD indices: {e}")
            return None, 0, 360
    
    def _transform_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Transform wind direction values to deviations from mean.
        
        Args:
            obs: Per-turbine observations, shape (n_turbines, obs_dim)
        
        Returns:
            Transformed observations with same shape
        """
        if self._wd_indices is None:
            return obs
        
        # Get mean wind direction from environment
        mean_wd = self._base_env.wd
        
        # Transform wind direction columns to deviations
        obs_transformed = obs.copy()
        
        for idx in self._wd_indices:
            if idx < obs.shape[1]:
                # Current values are scaled to [-1, 1] from [wd_min, wd_max]
                # Unscale: val = (scaled + 1) / 2 * (max - min) + min
                wd_scaled = obs[:, idx]
                wd_degrees = (wd_scaled + 1) / 2 * (self._wd_max - self._wd_min) + self._wd_min
                
                # Compute deviation from mean and rescale
                wd_dev_scaled = compute_wind_direction_deviation(
                    wd_degrees, mean_wd, self.wd_scale_range
                )
                
                obs_transformed[:, idx] = wd_dev_scaled
        
        return obs_transformed
    
    # Pass through all properties from the wrapped env
    @property
    def n_turbines(self) -> int:
        return self.env.n_turbines
    
    @property 
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def action_space(self):
        return self.env.action_space
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        obs_transformed = self._transform_observation(obs)
        return obs_transformed, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_transformed = self._transform_observation(obs)
        return obs_transformed, reward, terminated, truncated, info


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


class AbsolutePositionalEncoding(nn.Module):
    """
    Original absolute positional encoding for turbine (x, y) coordinates.
    
    Transforms 2D positions into a higher-dimensional embedding space
    using a small MLP. This is ADDED to the token embedding.
    
    Input positions should be:
    1. Normalized by rotor diameter (physics-meaningful scale)
    2. Transformed to wind-relative coordinates (wind from 270°)
    """
    
    def __init__(self, pos_dim: int = 2, embed_dim: int = 32):
        """
        Args:
            pos_dim: Input position dimension (2 for x, y)
            embed_dim: Output embedding dimension
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(pos_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_turbines, 2) normalized wind-relative coordinates
        
        Returns:
            Position embeddings: (batch, n_turbines, embed_dim)
        """
        return self.encoder(positions)


class RelativePositionalBias(nn.Module):
    """
    Relative positional bias for attention.
    
    Computes a learned bias for each pair of positions based on their
    relative displacement. This bias is ADDED to attention logits.
    
    Physics intuition:
    - rel_pos[i,j] = pos[j] - pos[i] tells us "j is X upwind, Y lateral from i"
    - The learned bias can encode "pay more attention to upwind turbines"
    - Translation invariant: same relative geometry → same bias
    
    For wind farm control:
    - Positive x in wind-relative coords = upwind
    - The model can learn that upwind turbines are important (wake sources)
    """
    
    def __init__(
        self, 
        num_heads: int,
        hidden_dim: int = 64,
        per_head: bool = True,
        pos_dim: int = 2
    ):
        """
        Args:
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension of bias MLP
            per_head: If True, each head gets its own bias. If False, shared.
            pos_dim: Dimension of position vectors (2 for x, y)
        """
        super().__init__()
        self.num_heads = num_heads
        self.per_head = per_head
        
        output_dim = num_heads if per_head else 1
        
        # MLP: relative_position (2D) → bias value(s)
        self.bias_mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self, 
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute relative position bias matrix.
        
        Args:
            positions: (batch, n_tokens, 2) wind-relative positions
            key_padding_mask: (batch, n_tokens) where True = padding
        
        Returns:
            bias: (batch, num_heads, n_tokens, n_tokens) attention bias
                  Add this to attention logits before softmax.
        """
        batch_size, n_tokens, _ = positions.shape
        
        # Compute pairwise relative positions
        # pos_i: (batch, n, 1, 2), pos_j: (batch, 1, n, 2)
        pos_i = positions.unsqueeze(2)  # (batch, n, 1, 2)
        pos_j = positions.unsqueeze(1)  # (batch, 1, n, 2)
        
        # rel_pos[i,j] = pos[j] - pos[i]: "displacement from i to j"
        # If j is upwind of i (positive x), rel_pos has positive x component
        rel_pos = pos_j - pos_i  # (batch, n, n, 2)
        
        # Reshape for MLP: (batch * n * n, 2)
        rel_pos_flat = rel_pos.reshape(-1, 2)
        
        # Compute bias values
        bias_flat = self.bias_mlp(rel_pos_flat)  # (batch*n*n, num_heads or 1)
        
        # Reshape back
        if self.per_head:
            bias = bias_flat.reshape(batch_size, n_tokens, n_tokens, self.num_heads)
            bias = bias.permute(0, 3, 1, 2)  # (batch, num_heads, n, n)
        else:
            bias = bias_flat.reshape(batch_size, n_tokens, n_tokens, 1)
            bias = bias.permute(0, 3, 1, 2)  # (batch, 1, n, n)
            bias = bias.expand(-1, self.num_heads, -1, -1)  # (batch, num_heads, n, n)
        
        # Apply masking: set bias to large negative for padded positions
        if key_padding_mask is not None:
            # Expand mask: (batch, n) → (batch, 1, 1, n) for keys
            # and (batch, 1, n, 1) for queries
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n)
            mask_q = key_padding_mask.unsqueeze(1).unsqueeze(3)  # (batch, 1, n, 1)
            
            # Zero out bias for padded positions (they'll be masked in attention anyway)
            # This prevents any gradient flow through padded positions
            combined_mask = mask_k | mask_q  # (batch, 1, n, n)
            bias = bias.masked_fill(combined_mask, 0.0)
        
        return bias


class Sinusoidal2DPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding extended to 2D coordinates.
    
    Uses multiple frequencies for both x and y dimensions, similar to
    NeRF-style positional encoding. This captures both coarse and fine
    spatial structure without learning the frequency bands.
    
    A final linear layer projects to the desired output dimension.
    """
    
    def __init__(self, embed_dim: int = 32, num_frequencies: int = 8, max_freq_log2: int = 6):
        """
        Args:
            embed_dim: Output embedding dimension
            num_frequencies: Number of frequency bands
            max_freq_log2: Log2 of maximum frequency (default: 2^6 = 64 cycles per unit)
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.raw_dim = 4 * num_frequencies  # sin/cos for x and y
        self.embed_dim = embed_dim
        
        # Frequency bands: 2^0, 2^1, ..., 2^(max_freq_log2)
        freq_bands = 2.0 ** torch.linspace(0, max_freq_log2, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)
        
        # Project to desired dimension
        self.proj = nn.Linear(self.raw_dim, embed_dim)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_turbines, 2) normalized coordinates
        
        Returns:
            Sinusoidal embeddings: (batch, n_turbines, embed_dim)
        """
        # positions: (batch, n, 2)
        x = positions[..., 0:1]  # (batch, n, 1)
        y = positions[..., 1:2]  # (batch, n, 1)
        
        # Multiply by frequencies: (batch, n, num_freq)
        x_freq = x * self.freq_bands * math.pi
        y_freq = y * self.freq_bands * math.pi
        
        # Compute sin and cos
        raw_embeddings = torch.cat([
            torch.sin(x_freq),
            torch.cos(x_freq),
            torch.sin(y_freq),
            torch.cos(y_freq),
        ], dim=-1)
        
        return self.proj(raw_embeddings)


class PolarPositionalEncoding(nn.Module):
    """
    Positional encoding using polar coordinates (r, θ).
    
    In wind-relative coordinates (wind from 270°):
    - θ = 0° means directly upwind
    - θ = 180° means directly downwind
    - r = distance from farm centroid
    
    This naturally aligns with wake physics where effects depend on
    both distance and angle relative to wind.
    """
    
    def __init__(self, embed_dim: int = 32):
        """
        Args:
            embed_dim: Output embedding dimension
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # MLP on (r, θ, sin(θ), cos(θ)) for better angle representation
        self.encoder = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_turbines, 2) wind-relative Cartesian coordinates
        
        Returns:
            Polar embeddings: (batch, n_turbines, embed_dim)
        """
        x = positions[..., 0]
        y = positions[..., 1]
        
        # Convert to polar
        r = torch.sqrt(x**2 + y**2 + 1e-8)
        theta = torch.atan2(y, x)  # Angle from positive x-axis (upwind direction)
        
        # Create input features: (r, θ, sin(θ), cos(θ))
        polar_features = torch.stack([
            r,
            theta,
            torch.sin(theta),
            torch.cos(theta),
        ], dim=-1)
        
        return self.encoder(polar_features)


class RelativePolarBias(nn.Module):
    """
    Relative positional bias using polar coordinates.
    
    For each pair (i, j), computes:
    - Δr = distance from i to j
    - θ_ij = angle from i to j relative to wind direction
    
    The bias MLP learns how attention should depend on:
    - How far apart turbines are
    - Whether j is upwind/downwind/lateral from i
    """
    
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int = 64,
        per_head: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.per_head = per_head
        
        output_dim = num_heads if per_head else 1
        
        # Input: (Δr, θ, sin(θ), cos(θ))
        self.bias_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_tokens, 2) wind-relative positions
            key_padding_mask: (batch, n_tokens) where True = padding
        
        Returns:
            bias: (batch, num_heads, n_tokens, n_tokens)
        """
        batch_size, n_tokens, _ = positions.shape
        
        # Compute pairwise relative positions
        pos_i = positions.unsqueeze(2)  # (batch, n, 1, 2)
        pos_j = positions.unsqueeze(1)  # (batch, 1, n, 2)
        rel_pos = pos_j - pos_i  # (batch, n, n, 2)
        
        # Convert to polar
        dx = rel_pos[..., 0]
        dy = rel_pos[..., 1]
        
        r = torch.sqrt(dx**2 + dy**2 + 1e-8)
        theta = torch.atan2(dy, dx)
        
        # Stack polar features
        polar_features = torch.stack([
            r,
            theta,
            torch.sin(theta),
            torch.cos(theta),
        ], dim=-1)  # (batch, n, n, 4)
        
        # Apply MLP
        polar_flat = polar_features.reshape(-1, 4)
        bias_flat = self.bias_mlp(polar_flat)
        
        if self.per_head:
            bias = bias_flat.reshape(batch_size, n_tokens, n_tokens, self.num_heads)
            bias = bias.permute(0, 3, 1, 2)
        else:
            bias = bias_flat.reshape(batch_size, n_tokens, n_tokens, 1)
            bias = bias.permute(0, 3, 1, 2)
            bias = bias.expand(-1, self.num_heads, -1, -1)
        
        if key_padding_mask is not None:
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(2)
            mask_q = key_padding_mask.unsqueeze(1).unsqueeze(3)
            combined_mask = mask_k | mask_q
            bias = bias.masked_fill(combined_mask, 0.0)
        
        return bias


class ALiBiPositionalBias(nn.Module):
    """
    Attention with Linear Biases (ALiBi) for 2D spatial positions.
    
    Simple linear penalty based on distance:
        bias[i,j] = -slope * distance(i, j)
    
    No learned parameters! Just an inductive bias that nearby turbines
    should attend more to each other.
    
    Each attention head gets a different slope (geometric sequence),
    allowing different heads to focus on different distance scales.
    
    Reference: Press et al., "Train Short, Test Long" (2022)
    """
    
    def __init__(self, num_heads: int, max_distance: float = 20.0):
        """
        Args:
            num_heads: Number of attention heads
            max_distance: Expected maximum distance (in rotor diameters) for slope scaling
        """
        super().__init__()
        self.num_heads = num_heads
        
        # Geometric sequence of slopes (like original ALiBi)
        # Slopes: 2^(-8/n), 2^(-8*2/n), ..., 2^(-8)
        slopes = torch.tensor([
            2 ** (-8 * (i + 1) / num_heads) for i in range(num_heads)
        ])
        self.register_buffer("slopes", slopes.view(1, num_heads, 1, 1))
    
    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_tokens, 2) normalized positions
            key_padding_mask: (batch, n_tokens) where True = padding
        
        Returns:
            bias: (batch, num_heads, n_tokens, n_tokens)
        """
        batch_size, n_tokens, _ = positions.shape
        
        # Compute pairwise distances
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        diff = pos_j - pos_i
        distances = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # (batch, n, n)
        
        # Apply linear penalty with per-head slopes
        distances = distances.unsqueeze(1)  # (batch, 1, n, n)
        bias = -self.slopes * distances  # (batch, num_heads, n, n)
        
        if key_padding_mask is not None:
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(2)
            mask_q = key_padding_mask.unsqueeze(1).unsqueeze(3)
            combined_mask = mask_k | mask_q
            bias = bias.masked_fill(combined_mask, 0.0)
        
        return bias


class DirectionalALiBiPositionalBias(nn.Module):
    """
    Directional ALiBi: Different slopes for upwind vs downwind.
    
    In wind-relative coordinates (wind from negative x):
    - Upwind (positive x direction): Use upwind_slope
    - Downwind (negative x direction): Use downwind_slope
    
    This encodes the physical asymmetry: upwind turbines affect
    downwind ones, but not vice versa.
    
    Learned slopes allow the model to discover the right asymmetry.
    """
    
    def __init__(self, num_heads: int):
        """
        Args:
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        
        # Learnable slopes for upwind and downwind (per head)
        # Initialize with ALiBi-style geometric sequence
        init_slopes = torch.tensor([
            2 ** (-8 * (i + 1) / num_heads) for i in range(num_heads)
        ])
        
        self.upwind_slopes = nn.Parameter(init_slopes.clone())
        self.downwind_slopes = nn.Parameter(init_slopes.clone() * 0.5)  # Less penalty downwind
        self.lateral_slopes = nn.Parameter(init_slopes.clone() * 0.3)  # Even less for lateral
    
    def forward(
        self,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            positions: (batch, n_tokens, 2) wind-relative positions
            key_padding_mask: (batch, n_tokens) where True = padding
        
        Returns:
            bias: (batch, num_heads, n_tokens, n_tokens)
        """
        batch_size, n_tokens, _ = positions.shape
        
        # Compute pairwise relative positions
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        rel_pos = pos_j - pos_i  # (batch, n, n, 2)
        
        dx = rel_pos[..., 0]  # Positive = j is upwind of i
        dy = rel_pos[..., 1]
        
        # Distances
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
        
        # Classify direction: upwind (dx > 0), downwind (dx < 0), lateral (|dy| > |dx|)
        is_upwind = dx > torch.abs(dy)      # Predominantly upwind
        is_downwind = dx < -torch.abs(dy)   # Predominantly downwind
        is_lateral = ~is_upwind & ~is_downwind  # Lateral
        
        # Apply different slopes based on direction
        slopes_upwind = self.upwind_slopes.view(1, self.num_heads, 1, 1)
        slopes_downwind = self.downwind_slopes.view(1, self.num_heads, 1, 1)
        slopes_lateral = self.lateral_slopes.view(1, self.num_heads, 1, 1)
        
        dist = dist.unsqueeze(1)  # (batch, 1, n, n)
        is_upwind = is_upwind.unsqueeze(1).float()
        is_downwind = is_downwind.unsqueeze(1).float()
        is_lateral = is_lateral.unsqueeze(1).float()
        
        # Weighted combination of slopes
        bias = -(
            slopes_upwind * dist * is_upwind +
            slopes_downwind * dist * is_downwind +
            slopes_lateral * dist * is_lateral
        )
        
        if key_padding_mask is not None:
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(2)
            mask_q = key_padding_mask.unsqueeze(1).unsqueeze(3)
            combined_mask = mask_k | mask_q
            bias = bias.masked_fill(combined_mask, 0.0)
        
        return bias


class RoPE2DPositionalEncoding(nn.Module):
    """
    2D Rotary Position Embeddings for wind farm coordinates.
    
    RoPE encodes position by rotating query and key vectors. When Q·K is
    computed, the rotation angles subtract, naturally encoding relative position.
    
    For 2D positions, we split the head dimension into two halves:
    - First half: rotated by angle proportional to x-position
    - Second half: rotated by angle proportional to y-position
    
    This is applied INSIDE the attention mechanism by transforming Q and K
    before the dot product.
    
    Key properties:
    - Relative position encoded in dot product (no explicit bias)
    - Decays with distance (like a soft locality bias)
    - No learned parameters for position encoding itself
    
    Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    Extended to 2D following approaches from vision transformers.
    """
    
    def __init__(
        self, 
        head_dim: int,
        max_position: float = 50.0,
        base: float = 10000.0,
    ):
        """
        Args:
            head_dim: Dimension per attention head (must be divisible by 4)
            max_position: Maximum expected position value (in rotor diameters)
            base: Base for frequency computation (like in standard RoPE)
        """
        super().__init__()
        
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4 for 2D RoPE, got {head_dim}")
        
        self.head_dim = head_dim
        self.max_position = max_position
        
        # Each spatial dimension gets half the head_dim
        # Within each half, we have pairs for rotation (so divide by 2 again)
        dim_per_axis = head_dim // 2
        
        # Frequency bands for rotation (standard RoPE frequencies)
        # Lower frequencies = longer range interactions
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_per_axis, 2).float() / dim_per_axis))
        self.register_buffer("inv_freq", inv_freq)
    
    def _compute_rotation_angles(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotation angles for x and y positions.
        
        Args:
            positions: (batch, n_tokens, 2) normalized positions
        
        Returns:
            angles_x, angles_y: (batch, n_tokens, dim_per_axis/2) rotation angles
        """
        x = positions[..., 0]  # (batch, n)
        y = positions[..., 1]  # (batch, n)
        
        # Scale positions to reasonable angle range
        x_scaled = x / self.max_position * math.pi
        y_scaled = y / self.max_position * math.pi
        
        # Compute angles: position * frequency for each frequency band
        # (batch, n, 1) * (dim/4,) -> (batch, n, dim/4)
        angles_x = x_scaled.unsqueeze(-1) * self.inv_freq
        angles_y = y_scaled.unsqueeze(-1) * self.inv_freq
        
        return angles_x, angles_y
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of x."""
        x1 = x[..., ::2]   # Even indices
        x2 = x[..., 1::2]  # Odd indices
        return torch.stack((-x2, x1), dim=-1).flatten(-2)
    
    def apply_rotary_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D rotary embeddings to query and key tensors.
        
        Args:
            q: (batch, n_heads, n_tokens, head_dim) query tensor
            k: (batch, n_heads, n_tokens, head_dim) key tensor
            positions: (batch, n_tokens, 2) wind-relative positions
        
        Returns:
            q_rot, k_rot: Rotated query and key tensors, same shape as input
        """
        batch, n_heads, n_tokens, head_dim = q.shape
        half_dim = head_dim // 2
        
        # Split into x and y portions
        q_x, q_y = q[..., :half_dim], q[..., half_dim:]
        k_x, k_y = k[..., :half_dim], k[..., half_dim:]
        
        # Compute rotation angles
        angles_x, angles_y = self._compute_rotation_angles(positions)
        
        # Expand angles for all heads: (batch, n, dim/4) -> (batch, 1, n, dim/4)
        angles_x = angles_x.unsqueeze(1)
        angles_y = angles_y.unsqueeze(1)
        
        # Duplicate angles for sin/cos pairs: (batch, 1, n, dim/4) -> (batch, 1, n, dim/2)
        cos_x = torch.cos(angles_x).repeat(1, 1, 1, 2)
        sin_x = torch.sin(angles_x).repeat(1, 1, 1, 2)
        cos_y = torch.cos(angles_y).repeat(1, 1, 1, 2)
        sin_y = torch.sin(angles_y).repeat(1, 1, 1, 2)
        
        # Apply rotation: x' = x*cos - rotate(x)*sin
        q_x_rot = q_x * cos_x + self._rotate_half(q_x) * sin_x
        k_x_rot = k_x * cos_x + self._rotate_half(k_x) * sin_x
        q_y_rot = q_y * cos_y + self._rotate_half(q_y) * sin_y
        k_y_rot = k_y * cos_y + self._rotate_half(k_y) * sin_y
        
        # Concatenate back
        q_rot = torch.cat([q_x_rot, q_y_rot], dim=-1)
        k_rot = torch.cat([k_x_rot, k_y_rot], dim=-1)
        
        return q_rot, k_rot


class RoPEMultiheadAttention(nn.Module):
    """
    Multi-head attention with 2D Rotary Position Embeddings.
    
    This replaces nn.MultiheadAttention when using RoPE, because RoPE
    needs to be applied to Q and K before the attention computation.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        max_position: float = 50.0,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert self.head_dim % 4 == 0, f"head_dim ({self.head_dim}) must be divisible by 4 for 2D RoPE"
        
        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # RoPE module
        self.rope = RoPE2DPositionalEncoding(
            head_dim=self.head_dim,
            max_position=max_position,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, n_tokens, embed_dim) input tensor
            positions: (batch, n_tokens, 2) wind-relative positions
            key_padding_mask: (batch, n_tokens) where True = padding
            attn_mask: Optional additional attention mask
        
        Returns:
            output: (batch, n_tokens, embed_dim)
            attn_weights: (batch, n_heads, n_tokens, n_tokens)
        """
        batch, n_tokens, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to (batch, n_heads, n_tokens, head_dim)
        q = q.view(batch, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, n_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        q, k = self.rope.apply_rotary_emb(q, k, positions)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply masks
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        
        if key_padding_mask is not None:
            # (batch, n_tokens) -> (batch, 1, 1, n_tokens)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape back: (batch, n_heads, n_tokens, head_dim) -> (batch, n_tokens, embed_dim)
        output = output.transpose(1, 2).contiguous().view(batch, n_tokens, self.embed_dim)
        output = self.out_proj(output)
        
        return output, attn_weights


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


def transform_to_wind_relative(
    positions: torch.Tensor,
    wind_direction: torch.Tensor
) -> torch.Tensor:
    """
    Transform positions to wind-relative coordinates.
    
    Rotates the coordinate system so that wind effectively comes from 270°
    (negative x direction). This makes the learning problem invariant to
    absolute wind direction - the model only needs to learn wake patterns
    in a canonical reference frame.
    
    Physics intuition:
    - After transformation, "upwind" always means positive x direction
    - Wake effects always propagate in positive x direction
    - Lateral offset is always in y direction
    
    Args:
        positions: (batch, n_turbines, 2) raw positions in meters
        wind_direction: (batch,) wind direction in degrees (meteorological convention)
    
    Returns:
        Rotated positions with same shape as input
    """
    # Rotation angle: how much to rotate to align wind with 270°
    angle_offset = wind_direction - 270.0
    theta = angle_offset * (math.pi / 180.0)
    
    # Handle dimensions for broadcasting
    if theta.dim() == 1:
        theta = theta.unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
    
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    x = positions[..., 0:1]
    y = positions[..., 1:2]
    
    # Standard 2D rotation matrix
    x_rot = cos_theta * x - sin_theta * y
    y_rot = sin_theta * x + cos_theta * y
    
    return torch.cat([x_rot, y_rot], dim=-1)


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


def save_checkpoint(
    actor: nn.Module,
    qf1: nn.Module,
    qf2: nn.Module,
    actor_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    step: int,
    run_name: str,
    args: Args,
    log_alpha: Optional[torch.Tensor] = None,
    alpha_optimizer: Optional[optim.Optimizer] = None,
) -> str:
    """
    Save training checkpoint.
    
    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir = f"runs/{run_name}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = f"{checkpoint_dir}/step_{step}.pt"
    
    checkpoint = {
        "step": step,
        "actor_state_dict": actor.state_dict(),
        "qf1_state_dict": qf1.state_dict(),
        "qf2_state_dict": qf2.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "args": vars(args),
    }
    
    if log_alpha is not None:
        checkpoint["log_alpha"] = log_alpha.detach().cpu()
    if alpha_optimizer is not None:
        checkpoint["alpha_optimizer_state_dict"] = alpha_optimizer.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    actor: nn.Module,
    qf1: nn.Module,
    qf2: nn.Module,
    qf1_target: nn.Module,
    qf2_target: nn.Module,
    actor_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    device: torch.device,
    log_alpha: Optional[torch.Tensor] = None,
    alpha_optimizer: Optional[optim.Optimizer] = None,
) -> int:
    """
    Load training checkpoint.
    
    Returns:
        Step number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    actor.load_state_dict(checkpoint["actor_state_dict"])
    qf1.load_state_dict(checkpoint["qf1_state_dict"])
    qf2.load_state_dict(checkpoint["qf2_state_dict"])
    qf1_target.load_state_dict(checkpoint["qf1_state_dict"])
    qf2_target.load_state_dict(checkpoint["qf2_state_dict"])
    actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
    
    if log_alpha is not None and "log_alpha" in checkpoint:
        log_alpha.data = checkpoint["log_alpha"].to(device)
    if alpha_optimizer is not None and "alpha_optimizer_state_dict" in checkpoint:
        alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
    
    print(f"Loaded checkpoint from {checkpoint_path} at step {checkpoint['step']}")
    
    return checkpoint["step"]


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
# ENVIRONMENT FACTORY
# =============================================================================

def make_env_config() -> Dict[str, Any]:
    """
    Create base environment configuration.
    
    This configuration is designed for transformer-based control:
    - Per-turbine measurements enabled (ws, wd, yaw)
    - History stacking for temporal context
    - Baseline comparison for reward calculation
    """
    return {
        "yaw_init": "Random",
        "BaseController": "Local",
        "ActionMethod": "yaw",
        "Track_power": False,
        "farm": {"yaw_min": -30, "yaw_max": 30},
        "wind": {
            "ws_min": 10, "ws_max": 10,
            "TI_min": 0.07, "TI_max": 0.07,
            "wd_min": 260, "wd_max": 280,
        },
        "act_pen": {"action_penalty": 0.0, "action_penalty_type": "Change"},
        "power_def": {"Power_reward": "Baseline", "Power_avg": 1, "Power_scaling": 1.0},
        "mes_level": {
            "turb_ws": True,
            "turb_wd": True,  # Will be converted to deviation
            "turb_TI": False,
            "turb_power": True,  # Include power
            "farm_ws": False,
            "farm_wd": False,
            "farm_TI": False,
            "farm_power": False,
        },
        "ws_mes": {
            "ws_current": False,
            "ws_rolling_mean": True,
            "ws_history_N": 15,  # History length
            "ws_history_length": 15,
            "ws_window_length": 1,
        },
        "wd_mes": {
            "wd_current": False,
            "wd_rolling_mean": True,
            "wd_history_N": 15,
            "wd_history_length": 15,
            "wd_window_length": 1,
        },
        "yaw_mes": {
            "yaw_current": False,
            "yaw_rolling_mean": True,
            "yaw_history_N": 15,
            "yaw_history_length": 15,
            "yaw_window_length": 1,
        },
        "power_mes": {
            "power_current": False,
            "power_rolling_mean": True,
            "power_history_N": 15,
            "power_history_length": 15,
            "power_window_length": 1,
        },
    }


def get_layout_positions(layout_type: str, wind_turbine) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get turbine positions for a given layout type.
    
    Args:
        layout_type: Layout identifier string
        wind_turbine: PyWake wind turbine object
    
    Returns:
        x_pos, y_pos: Arrays of turbine positions
    """
    # Import here to avoid circular imports when not using WindGym
    # from WindGym.utils.generate_layouts import generate_square_grid, generate_cirular_farm
    
    layouts = {
        "test_layout": lambda: generate_square_grid(turbine=wind_turbine, nx=2, ny=1, xDist=5, yDist=5),
        "square_2x2": lambda: generate_square_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5),
        "small_triangle": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=3, xDist=5, yDist=5),
        "square_3x3": lambda: generate_square_grid(turbine=wind_turbine, nx=3, ny=3, xDist=5, yDist=5),
        "circular_6": lambda: generate_cirular_farm(n_list=[1, 5], turbine=wind_turbine, r_dist=5),
        "circular_10": lambda: generate_cirular_farm(n_list=[3, 7], turbine=wind_turbine, r_dist=5),
        "tri1": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='lower_left'),
        "tri2": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='lower_right'),
        "tri3": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='upper_left'),
        "tri4": lambda: generate_right_triangle_grid(turbine=wind_turbine, nx=2, ny=2, xDist=5, yDist=5, orientation='upper_right'),
        "5turb1": lambda: generate_line_dots_multiple_thetas(X=3, spacing=5, thetas=[0, 30], turbine=wind_turbine),
        "5turb2": lambda: generate_line_dots_multiple_thetas(X=3, spacing=5, thetas=[0, -30], turbine=wind_turbine),
        "5turb3": lambda: generate_line_dots_multiple_thetas(X=3, spacing=5, thetas=[-30, 30], turbine=wind_turbine),
    }
    
    if layout_type not in layouts:
        raise ValueError(f"Unknown layout type: {layout_type}. Available: {list(layouts.keys())}")
    
    return layouts[layout_type]()


# =============================================================================
# HELPER FUNCTIONS FOR ENVIRONMENT INTERACTION
# =============================================================================

def get_env_wind_directions(envs, num_envs: int) -> np.ndarray:
    """Get current wind direction from each environment."""
    return np.array(envs.env.get_attr('wd'), dtype=np.float32)


def get_env_raw_positions(envs, num_envs: int, n_turbines_max: int) -> np.ndarray:
    """Get raw (unnormalized) turbine positions from each environment."""
    return np.array(envs.env.get_attr('turbine_positions'), dtype=np.float32)


def get_env_attention_masks(envs, num_envs: int, n_turbines_max: int) -> np.ndarray:
    """Get attention masks from each environment."""
    return np.array(envs.env.get_attr('attention_mask'), dtype=bool)


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
    
    # Create run name
    run_name = f"{args.exp_name}_{'_'.join(layout_names)}_{args.seed}"
    
    print("=" * 60)
    print(f"Transformer SAC for Wind Farm Control - V9")
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
    
    # Tracking
    # if args.track:
    #     import wandb
    #     wandb.init(
    #         project=args.wandb_project_name,
    #         entity=args.wandb_entity,
    #         sync_tensorboard=True,
    #         config=vars(args),
    #         name=run_name,
    #         save_code=True,
    #     )
    
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
    layouts = []
    for name in layout_names:
        x_pos, y_pos = get_layout_positions(name, wind_turbine)
        layouts.append(LayoutConfig(name=name, x_pos=x_pos, y_pos=y_pos))
    
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
    
    def make_env_fn(seed):
        """Factory function for vectorized environments."""
        def _init():
            env = MultiLayoutEnv(
                layouts=layouts,
                env_factory=env_factory,
                per_turbine_wrapper=combined_wrapper,  # Use combined wrapper
                seed=seed,
                shuffle=args.shuffle_turbs, # Shuffle turbines within each layout
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
    
    # Get dimensions from sample env
    sample_env = make_env_fn(args.seed)()
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
        layout_names=layout_names,
        log_every=250,  # Base frequency - others are multiples of this
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
    
    start_time = time.time()
    global_step = 0
    total_gradient_steps = 0  # Track total gradient updates for logging
    # Reset environments
    obs, infos = envs.reset(seed=args.seed)
    
    # Tracking
    step_reward_window = deque(maxlen=1000)
    next_save_step = args.save_interval
    
    # For logging losses (we'll average over the UTD updates)
    loss_accumulator = {
        'qf1_loss': [], 'qf2_loss': [], 'actor_loss': [], 'alpha_loss': []
    }

    num_updates = args.total_timesteps // args.num_envs
    
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
                mask_tensor = torch.tensor(current_masks, dtype=torch.bool, device=device) if is_multi_layout else None
                
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
                
                batch_mask = data["attention_mask"] if is_multi_layout else None
                

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